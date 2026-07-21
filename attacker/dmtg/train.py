"""A3 diffusion training + STAGE 1-2 CHECKPOINT.

Default action (no flags) runs the checkpoint battery and STOPS — it never runs
full training. Full training is guarded behind --full and is intentionally not
invoked at this stage.

Checkpoint battery (all reported before any full training):
  1. Round-trip     : denorm(norm(r)) == r to float32 tolerance (max abs error).
  2. Reconstruction : equidistant(D) + residual == action for 100 sampled actions
                      (max px discrepancy; must be ~0 or the parameterization is broken).
  3. Shape/range    : final tensor shapes + min/max/mean/std of normalized residual.
  4. Overfit        : train on 32 samples for a few hundred steps; loss must fall
                      toward ~0 (else conditioning/model is wrong).
  5. Split          : train/val sizes + confirm the split is session-disjoint.

epsilon-prediction MSE, DDIMScheduler. eta is a SAMPLING-time entropy knob and is
NOT touched during training.
"""
import argparse
import copy
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler

import data as D
from model import build_model, count_params

NUM_TRAIN_TIMESTEPS = 1000
HERE = os.path.dirname(os.path.abspath(__file__))

# ---- winning config from the data-scale A/B (ab_scale.py) --------------------
# A: asinh normalization (no unit-std standardization), clip_sample=True.
# B (unit std, clip=False) lost on val eMSE 0.174 vs 0.080. See ab_scale_results.json.
WINNER = {"standardize": False, "clip_sample": True}


def make_scheduler(clip_sample=True):
    return DDIMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS, clip_sample=clip_sample)


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


def epsilon_loss(model, sched, x0, cond, device):
    """Standard epsilon-prediction MSE for one batch of (x0, cond) tensors."""
    B = x0.shape[0]
    noise = torch.randn_like(x0)
    t = torch.randint(0, sched.config.num_train_timesteps, (B,), device=device).long()
    noisy = sched.add_noise(x0, noise, t)
    inp = torch.cat([noisy, cond], dim=1)          # (B,4,128)
    out = model(inp, t)
    pred = out.sample if hasattr(out, "sample") else out
    return F.mse_loss(pred, noise)


def overfit_test(x0, cond, device, n=32, steps=300, lr=1e-3, seed=0):
    """Overfit a tiny subset; return (model_name, loss_history)."""
    torch.manual_seed(seed)
    model, name = build_model()
    model.to(device).train()
    sched = make_scheduler()
    xb = torch.from_numpy(x0[:n]).to(device)
    cb = torch.from_numpy(cond[:n]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = epsilon_loss(model, sched, xb, cb, device)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return name, losses, count_params(model)


# =============================== CHECKPOINT ===================================
def run_checkpoint(seed=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 74)
    print("A3 STAGE 1-2 CHECKPOINT  (no full training)")
    print(f"device: {device}")
    print("=" * 74)

    # ---- load corpus / tensors ------------------------------------------------
    corpus = D.load_corpus()
    x0, cond = D.to_tensors(corpus)
    print(f"corpus: {corpus['n_total']} total -> dropped {corpus['n_dropped_degenerate']} "
          f"degenerate (||D||<{corpus['min_disp']:g}) -> {corpus['n_kept']} kept "
          f"across {corpus['meta_n_users']} users")

    # ---- 1. round-trip -------------------------------------------------------
    r = corpus["residual"]                                   # pixels (N,128,2)
    rt = D.denorm_residual(D.norm_residual(r).astype(np.float32)).astype(np.float32)
    rt_err = float(np.abs(rt - r).max())
    print("\n[1] ROUND-TRIP  denorm(norm(r)) == r")
    print(f"    max abs error (px): {rt_err:.3e}   "
          f"({'PASS' if rt_err < 1e-2 else 'FAIL'} @ float32 tol)")

    # ---- 2. reconstruction ---------------------------------------------------
    rng = np.random.default_rng(seed)
    idx = rng.choice(corpus["n_kept"], size=min(100, corpus["n_kept"]), replace=False)
    recon = corpus["equidistant"][idx] + corpus["residual"][idx]
    disc = float(np.abs(recon - corpus["actions"][idx]).max())
    # also confirm net displacement is honored exactly
    D_recon = recon.sum(axis=1)
    D_true = corpus["actions"][idx].sum(axis=1)
    D_disc = float(np.abs(D_recon - D_true).max())
    print("\n[2] RECONSTRUCTION  equidistant(D) + residual == action  (100 sampled)")
    print(f"    max px discrepancy      : {disc:.3e}   "
          f"({'PASS' if disc < 1e-3 else 'FAIL'})")
    print(f"    max ||D|| discrepancy   : {D_disc:.3e}  (net displacement honored)")

    # ---- 3. shapes / range ---------------------------------------------------
    print("\n[3] SHAPES / RANGE")
    print(f"    x0   (norm residual) : {x0.shape}  dtype {x0.dtype}")
    print(f"    cond (norm D bcast)  : {cond.shape}  dtype {cond.dtype}")
    print(f"    normalized residual  : min {x0.min():+.4f}  max {x0.max():+.4f}  "
          f"mean {x0.mean():+.4f}  std {x0.std():.4f}")
    frac_in = float(np.mean(np.abs(x0) <= 1.0))
    print(f"    fraction within [-1,1]: {frac_in*100:.2f}%")
    print(f"    cond range           : min {cond.min():+.4f}  max {cond.max():+.4f}")
    range_ok = (x0.mean() > -0.2 and x0.mean() < 0.2 and 0.05 < x0.std() < 1.5
                and frac_in > 0.90)
    print(f"    spread sane          : {'PASS' if range_ok else 'FAIL — inspect'}")

    # ---- 5. split (computed before overfit so we can report sizes) -----------
    tr_idx, va_idx, va_sessions = D.session_split(corpus["keys"], val_frac=0.10, seed=seed)
    tr_keys = set(np.unique(corpus["keys"][tr_idx]))
    va_keys = set(va_sessions)
    overlap = tr_keys & va_keys
    n_all_sessions = len(set(corpus["keys"]))
    print("\n[5] SESSION-DISJOINT SPLIT (val_frac=0.10)")
    print(f"    train actions : {len(tr_idx):,}  ({len(tr_keys)} sessions)")
    print(f"    val   actions : {len(va_idx):,}  ({len(va_keys)} sessions)")
    print(f"    val fraction  : {len(va_idx)/corpus['n_kept']*100:.2f}%")
    print(f"    total sessions: {n_all_sessions}")
    print(f"    session overlap train∩val : {len(overlap)}  "
          f"({'PASS — disjoint' if not overlap else 'FAIL — LEAK'})")

    # ---- 4. overfit ----------------------------------------------------------
    print("\n[4] OVERFIT TEST  (32 samples, 300 steps)")
    name, losses, nparams = overfit_test(x0, cond, device, n=32, steps=300, seed=seed)
    l0 = np.mean(losses[:10])
    lend = np.mean(losses[-10:])
    lmin = min(losses)
    print(f"    model                 : {name}  ({nparams:,} params)")
    print(f"    loss  start(~10 avg)  : {l0:.4f}")
    print(f"    loss  end  (~10 avg)  : {lend:.4f}")
    print(f"    loss  min             : {lmin:.4f}")
    drop = 1.0 - lend / l0 if l0 > 0 else 0.0
    # Require a low ABSOLUTE floor, not just a relative drop: a timestep-blind
    # denoiser can post a big relative drop yet plateau ~0.55 (see model.py).
    ok = lend < 0.20
    print(f"    relative drop         : {drop*100:.1f}%")
    print(f"    verdict               : "
          f"{'PASS — collapses toward ~0' if ok else 'FAIL — plateaus high (model/conditioning wrong)'}")

    print("\n" + "=" * 74)
    print("CHECKPOINT COMPLETE — stopping before full training (use --full to train).")
    print("=" * 74)


def run_full(seed=0, max_epochs=400, batch=256, lr=1e-3, patience=30,
             min_delta=1e-4, out_dir=None):
    """Full A3 training with the winning config (A: asinh, clip_sample=True).

    Logs per-epoch train/val epsilon-MSE, saves the best-val checkpoint, and stops
    early when val fails to improve for `patience` epochs (val/train divergence on
    this modest 17k corpus). Restores best weights before the final save.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = out_dir or os.path.join(HERE, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    corpus = D.load_corpus()
    x0, cond = D.to_tensors(corpus)                         # config A: asinh only
    tr, va, val_sessions = D.session_split(corpus["keys"], val_frac=0.10, seed=seed,
                                           spread=True, min_users=5)
    val_users = sorted({s.split("/")[0] for s in val_sessions})

    torch.manual_seed(seed)
    np.random.seed(seed)
    model, name = build_model(kind="custom")
    model.to(device).train()
    sched = make_scheduler(clip_sample=WINNER["clip_sample"])

    x0t = torch.from_numpy(x0).to(device)
    condt = torch.from_numpy(cond).to(device)
    xtr, ctr = x0t[tr], condt[tr]
    xva, cva = x0t[va], condt[va]

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator(device=device).manual_seed(seed)
    ntr = xtr.shape[0]

    print("=" * 70)
    print(f"A3 FULL TRAINING  config=A(asinh, clip_sample={WINNER['clip_sample']})")
    print(f"device={device}  model={name} ({count_params(model):,} params)")
    print(f"train={ntr}  val={len(va)}  val_users={len(val_users)} {val_users}")
    print(f"batch={batch}  lr={lr}  max_epochs={max_epochs}  patience={patience}")
    print("=" * 70)
    print(f"{'epoch':>5} {'train_eMSE':>11} {'val_eMSE':>10} {'best':>7} {'bad':>4}")

    best_val, best_state, best_epoch, bad = float("inf"), None, -1, 0
    history = []
    ckpt_path = os.path.join(out_dir, "a3_dmtg_best.pt")
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        perm = torch.randperm(ntr, generator=g, device=device)
        model.train()
        ep_sum, ep_n = 0.0, 0
        for i in range(0, ntr, batch):
            idx = perm[i:i + batch]
            xb, cb = xtr[idx], ctr[idx]
            noise = torch.randn(xb.shape, generator=g, device=device)
            t = torch.randint(0, NUM_TRAIN_TIMESTEPS, (xb.shape[0],),
                              generator=g, device=device).long()
            noisy = sched.add_noise(xb, noise, t)
            out = model(torch.cat([noisy, cb], 1), t)
            pred = out.sample if hasattr(out, "sample") else out
            loss = F.mse_loss(pred, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_sum += float(loss.item()) * xb.shape[0]
            ep_n += xb.shape[0]
        tr_loss = ep_sum / ep_n
        val_loss = eval_loss(model, sched, xva, cva, device)
        history.append({"epoch": epoch, "train": tr_loss, "val": val_loss})

        improved = val_loss < best_val - min_delta
        if improved:
            best_val, best_epoch, bad = val_loss, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
        if epoch <= 5 or epoch % 5 == 0 or improved or bad >= patience:
            print(f"{epoch:>5} {tr_loss:>11.4f} {val_loss:>10.4f} "
                  f"{best_val:>7.4f} {bad:>4}")
        if bad >= patience:
            print(f"-- early stop: val no-improve for {patience} epochs "
                  f"(best {best_val:.4f} @ epoch {best_epoch}) --")
            break
    elapsed = time.time() - t0

    # restore best weights and save checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    final_train = history[best_epoch - 1]["train"] if best_epoch > 0 else history[-1]["train"]
    ckpt = {
        "model_state": {k: v for k, v in model.state_dict().items()},
        "model_name": name,
        "in_channels": 4, "out_channels": 2, "sample_size": 128,
        "config": "A_asinh_clip",
        "scheduler": {"type": "DDIMScheduler",
                      "num_train_timesteps": NUM_TRAIN_TIMESTEPS,
                      "clip_sample": WINNER["clip_sample"]},
        "normalization": {  # everything the emitter needs to invert to pixels
            "asinh_s": D.ASINH_S, "asinh_ref": D.ASINH_REF,
            "asinh_denom": D.ASINH_DENOM, "d_divisor": D.D_DIVISOR,
            "standardize": WINNER["standardize"], "std": None,
            "min_disp": D.MIN_DISP, "window": D.WINDOW,
        },
        "split": {"seed": seed, "val_frac": 0.10, "spread": True,
                  "val_sessions": val_sessions, "val_users": val_users,
                  "n_train": int(ntr), "n_val": int(len(va))},
        "best_epoch": best_epoch, "best_val": best_val, "best_train": final_train,
        "epochs_run": len(history), "elapsed_sec": elapsed,
    }
    torch.save(ckpt, ckpt_path)
    with open(os.path.join(out_dir, "a3_train_history.json"), "w") as f:
        json.dump({"history": history, "best_epoch": best_epoch,
                   "best_val": best_val, "elapsed_sec": elapsed}, f, indent=2)

    print("=" * 70)
    print(f"DONE  best val eMSE {best_val:.4f} @ epoch {best_epoch}  "
          f"(train {final_train:.4f})")
    print(f"epochs run: {len(history)}   wall-clock: {elapsed:.1f}s "
          f"({elapsed/60:.1f} min)")
    print(f"checkpoint -> {ckpt_path}")
    print("=" * 70)
    return ckpt_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="run full training (config A)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=30)
    args = ap.parse_args()
    if args.full:
        run_full(seed=args.seed, max_epochs=args.max_epochs, patience=args.patience)
    else:
        run_checkpoint(seed=args.seed)
