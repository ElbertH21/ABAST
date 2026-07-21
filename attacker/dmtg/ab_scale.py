"""Data-scale A/B for A3 diffusion training.

A: current asinh normalization (train std ~0.318), DDIMScheduler(clip_sample=True).
B: standardized to ~unit std (asinh / train-std), DDIMScheduler(clip_sample=False)
   -- mandatory, since standardizing pushes the range to ~+-3 and the default
   clip_sample_range=1.0 would clip predicted x0 at sample time and break sampling.

Both runs: identical seeds, same session-disjoint spread split, full train set,
~2000 steps. epsilon-MSE is measured against the injected unit-variance noise, so
train/val losses are directly comparable between A and B despite the x0 rescale.

Note: clip_sample is a SAMPLING-time flag (reverse step); it does not enter
add_noise or the epsilon-MSE loss, so the training-curve difference here is driven
purely by the x0 scale / effective-SNR alignment. clip_sample is set correctly per
arm because it is the paired sampling config the winner will be sampled with.

Reports both curves and picks the winner on final val loss; ties -> keep A.

CORRECTION (post-hoc, do not re-run): this A/B is a NULL, not a win for A.
epsilon-MSE is NOT comparable across x0 scales, because the Bayes-optimal MSE floor
scales with x0 variance: floor ~= 0.121 for sigma=0.32 (arm A) vs ~= 0.276 for
sigma=1.0 (arm B). Both arms landed ~36.8% below their OWN floor -> identical model
quality; the raw val-loss gap (0.080 vs 0.174) is just the two different floors, not
a quality difference. We keep A for two scale-invariant reasons only:
  (1) simpler -- no standardization constant to store and invert at sample time;
  (2) A's data is 100% within [-1,1], so DDIMScheduler(clip_sample=True) is a no-op
      rather than a distortion (B's ~+-3 range would be clipped, hence clip=False).
The printed "WINNER: A / lower val loss" line below is retained for provenance but
is superseded by this note. See ab_scale_results.json["correction"].
"""
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler

import data as D
from model import build_model

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = 2000
BATCH = 256
LR = 1e-3
EVAL_EVERY = 200
SEED = 0


@torch.no_grad()
def eval_loss(model, sched, x0v, condv, device, K=8, bs=512, seed=1234):
    """Deterministic val epsilon-MSE: fixed generator, K noise/t draws averaged."""
    model.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    tot, cnt = 0.0, 0
    for _ in range(K):
        for i in range(0, x0v.shape[0], bs):
            xb, cb = x0v[i:i + bs], condv[i:i + bs]
            noise = torch.randn(xb.shape, generator=g, device=device)
            t = torch.randint(0, sched.config.num_train_timesteps, (xb.shape[0],),
                              generator=g, device=device).long()
            noisy = sched.add_noise(xb, noise, t)
            out = model(torch.cat([noisy, cb], 1), t)
            pred = out.sample if hasattr(out, "sample") else out
            tot += F.mse_loss(pred, noise, reduction="sum").item()
            cnt += noise.numel()
    model.train()
    return tot / cnt


def run_arm(label, x0_full, cond_full, tr, va, clip_sample, device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model, name = build_model(kind="custom", verbose=False)
    model.to(device).train()
    sched = DDIMScheduler(num_train_timesteps=1000, clip_sample=clip_sample)

    x0 = torch.from_numpy(x0_full).to(device)
    cond = torch.from_numpy(cond_full).to(device)
    xtr, ctr = x0[tr], cond[tr]
    xva, cva = x0[va], cond[va]

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    g = torch.Generator(device=device).manual_seed(SEED)

    curve = []
    run_tr, run_n = 0.0, 0
    print(f"\n[{label}] {name}  clip_sample={clip_sample}  train std={float(xtr.std()):.4f}")
    print(f"    {'step':>5} {'train_eMSE':>12} {'val_eMSE':>12}")
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, xtr.shape[0], (BATCH,), generator=g, device=device)
        xb, cb = xtr[idx], ctr[idx]
        noise = torch.randn(xb.shape, generator=g, device=device)
        t = torch.randint(0, 1000, (BATCH,), generator=g, device=device).long()
        noisy = sched.add_noise(xb, noise, t)
        out = model(torch.cat([noisy, cb], 1), t)
        pred = out.sample if hasattr(out, "sample") else out
        loss = F.mse_loss(pred, noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        run_tr += float(loss.item())
        run_n += 1
        if step == 1 or step % EVAL_EVERY == 0:
            vl = eval_loss(model, sched, xva, cva, device)
            tl = run_tr / run_n
            run_tr, run_n = 0.0, 0
            curve.append({"step": step, "train": tl, "val": vl})
            print(f"    {step:>5} {tl:>12.4f} {vl:>12.4f}")
    return {"arm": label, "model": name, "clip_sample": clip_sample, "curve": curve,
            "final_train": curve[-1]["train"], "final_val": curve[-1]["val"]}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = D.load_corpus()
    x0, cond = D.to_tensors(corpus)
    tr, va, val_sessions = D.session_split(corpus["keys"], val_frac=0.10, seed=SEED,
                                           spread=True, min_users=5)
    std = D.compute_std(x0, tr)
    x0_std = D.standardize(x0, std)

    print("=" * 62)
    print("A3 DATA-SCALE A/B")
    print(f"device={device}  train={len(tr)}  val={len(va)}  "
          f"val_users={len({s.split('/')[0] for s in val_sessions})}")
    print(f"train-only asinh std (A) = {std:.4f}  ->  B divides by it (unit std)")
    print("=" * 62)

    A = run_arm("A: asinh (std~0.32), clip=True", x0, cond, tr, va,
                clip_sample=True, device=device)
    B = run_arm("B: standardized (unit std), clip=False", x0_std, cond, tr, va,
                clip_sample=False, device=device)

    # winner on final val loss; tie (within noise) -> keep A
    tie_tol = 0.01  # relative
    rel = (A["final_val"] - B["final_val"]) / A["final_val"]
    if abs(rel) < tie_tol:
        winner = "A"
        why = f"within noise ({rel*100:+.1f}% val diff < {tie_tol*100:.0f}%); keep A (fewer moving parts, clip stabilizer)"
    else:
        winner = "B" if B["final_val"] < A["final_val"] else "A"
        why = f"lower final val loss ({rel*100:+.1f}% A->B)"

    print("\n" + "=" * 62)
    print("RESULT")
    print(f"  A final: train {A['final_train']:.4f}  val {A['final_val']:.4f}")
    print(f"  B final: train {B['final_train']:.4f}  val {B['final_val']:.4f}")
    print(f"  WINNER: {winner}  ({why})")
    print("=" * 62)

    out = {"seed": SEED, "steps": STEPS, "batch": BATCH, "lr": LR,
           "train_std": std, "n_train": len(tr), "n_val": len(va),
           "val_sessions": val_sessions,
           "val_users": sorted({s.split('/')[0] for s in val_sessions}),
           "A": A, "B": B, "winner": winner, "why": why}
    with open(os.path.join(HERE, "ab_scale_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {os.path.join(HERE, 'ab_scale_results.json')}")


if __name__ == "__main__":
    main()
