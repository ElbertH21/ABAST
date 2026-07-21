"""A3 DMTG sampler + target-blind eta sweep (Stage 3).

Loads checkpoints/a3_dmtg_best.pt (which carries every normalization constant),
samples humanizing residuals via DDIM reverse (num_inference_steps=50, given eta),
denormalizes the asinh, zero-mean-projects the residual so the requested net
displacement D is honored EXACTLY, and reconstructs actions as
build_equidistant_inputs(D) + residual.

The eta sweep is chosen on REALISM against the pooled genuine corpus ONLY -- never
on bypass rate. A3 is target-blind and zero-query; tuning eta on defender feedback
would destroy that and invalidate the A1/A2/A3 comparison. Realism is measured
against pooled human data the attacker is already allowed to have, so it stays blind.

Does not modify balabit_actions.py. Does not write the emitter or generate sessions.
"""
import json
import os
import numpy as np
import torch
from diffusers import DDIMScheduler

import data as D
from model import build_model

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(HERE, "checkpoints", "a3_dmtg_best.pt")
ETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
N_INFERENCE = 50
SEED = 0


# ----- checkpoint / model -----------------------------------------------------
def load_a3(ckpt_path=CKPT_PATH, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model, _ = build_model(kind="custom", verbose=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    sched = DDIMScheduler(num_train_timesteps=ckpt["scheduler"]["num_train_timesteps"],
                          clip_sample=ckpt["scheduler"]["clip_sample"])
    return model, sched, ckpt, device


# ----- residual (de)normalization from checkpoint constants -------------------
def denorm_residual(x_norm, norm):
    """Invert asinh (and standardization if present) using checkpoint constants."""
    x = np.asarray(x_norm, dtype=np.float64)
    if norm.get("standardize") and norm.get("std"):
        x = x * float(norm["std"])
    return np.sign(x) * norm["asinh_s"] * np.sinh(np.abs(x) * norm["asinh_denom"])


# ----- equidistant line from a raw net displacement D -------------------------
def equidistant_from_D(Dvec, length=128):
    """Equidistant deltas for displacement D, reusing the repo builder (no reimpl).

    build_equidistant_inputs consumes an (N,length,2) target and internally uses
    D = target.sum(1); we hand it a container whose sum is exactly D so it returns
    the same straight line the model was trained against. D integer => sum == D exact.
    """
    Dvec = np.asarray(Dvec, dtype=np.float32)
    container = np.zeros((Dvec.shape[0], length, 2), dtype=np.float32)
    container[:, 0, :] = Dvec
    return D.build_equidistant_inputs(container, length=length)


# ----- core sampler -----------------------------------------------------------
@torch.no_grad()
def sample_residual(Dvec, eta, model, sched, device, seed=SEED, chunk=1024):
    """DDIM reverse from x_T ~ N(0,I) conditioned on D; return residual pixels
    (M,128,2), zero-mean-projected so residual.sum(axis=1) == 0 exactly-ish."""
    Dvec = np.asarray(Dvec, dtype=np.float32)
    M = Dvec.shape[0]
    out = np.empty((M, 128, 2), dtype=np.float32)
    sched.set_timesteps(N_INFERENCE)
    for s in range(0, M, chunk):
        Dc = Dvec[s:s + chunk]
        B = Dc.shape[0]
        cond = np.broadcast_to((Dc / D.D_DIVISOR)[:, :, None], (B, 2, 128))
        cond = torch.from_numpy(np.ascontiguousarray(cond, dtype=np.float32)).to(device)
        g = torch.Generator(device=device).manual_seed(seed)      # same x_T across etas
        x = torch.randn(B, 2, 128, generator=g, device=device)
        for t in sched.timesteps:
            eps = model(torch.cat([x, cond], dim=1), t)
            eps = eps.sample if hasattr(eps, "sample") else eps
            x = sched.step(eps, t, x, eta=eta, generator=g).prev_sample
        out[s:s + chunk] = x.cpu().numpy().transpose(0, 2, 1)             # (B,128,2) normalized
    return out  # normalized residual, channels-last; denorm handled by caller


@torch.no_grad()
def reconstruct(Dvec, eta, model, sched, ckpt, device, seed=SEED):
    """Full path: sample -> denorm asinh -> zero-mean project -> equidistant(D)+residual."""
    norm = ckpt["normalization"]
    res_norm = sample_residual(Dvec, eta, model, sched, device, seed=seed)   # (M,128,2) normalized
    residual = denorm_residual(res_norm, norm).astype(np.float32)            # pixels
    # zero-mean project so sum(residual)==0 => achieved D == equidistant sum == requested D
    residual = residual - residual.mean(axis=1, keepdims=True)
    equi = equidistant_from_D(Dvec).astype(np.float32)
    actions = equi + residual
    return actions.astype(np.float32), residual, equi


# ----- geometric metrics (geometry-only; no timing) --------------------------
def geom_metrics(actions):
    """Metrics on (M,128,2) dx/dy delta actions.
      step distance p50/p90/p99 (over all steps),
      turning-angle variance (var of wrapped d-heading),
      curvature_filtered = mean|d-heading / dist| over steps with dist>=2px only
        (the project's honest metric; raw curvature is inflated by ~0-length steps),
      path_efficiency = mean(||D|| / total path length).
    """
    dx = actions[:, :, 0]
    dy = actions[:, :, 1]
    dist = np.hypot(dx, dy)                                   # (M,128)
    heading = np.arctan2(dy, dx)                             # (M,128)
    dhead = np.diff(heading, axis=1)                         # (M,127)
    dhead = (dhead + np.pi) % (2 * np.pi) - np.pi            # wrap to [-pi,pi]

    alldist = dist.reshape(-1)
    p50, p90, p99 = np.percentile(alldist, [50, 90, 99])

    turn_var = float(np.var(dhead.reshape(-1)))

    dist_pair = dist[:, 1:]                                  # dist aligned to dhead
    mask = dist_pair >= 2.0
    curv_filt = float(np.mean(np.abs(dhead[mask] / dist_pair[mask]))) if mask.any() else float("nan")

    pathlen = dist.sum(axis=1)                               # (M,)
    Dnet = actions.sum(axis=1)                               # (M,2)
    Dn = np.hypot(Dnet[:, 0], Dnet[:, 1])
    eff = float(np.mean(Dn / np.maximum(pathlen, 1e-9)))

    return {
        "step_p50": float(p50), "step_p90": float(p90), "step_p99": float(p99),
        "turn_angle_var": turn_var,
        "curvature_filtered_ge2px": curv_filt,
        "path_efficiency": eff,
    }


# ----- driver -----------------------------------------------------------------
def main():
    model, sched, ckpt, device = load_a3()
    corpus = D.load_corpus()
    tr, va, val_sessions = D.session_split(corpus["keys"], val_frac=0.10,
                                           seed=ckpt["split"]["seed"], spread=True,
                                           min_users=5)
    assert sorted(val_sessions) == sorted(ckpt["split"]["val_sessions"]), \
        "val split does not match checkpoint — refusing to compare on a different split"
    val_actions = corpus["actions"][va]
    D_val = corpus["D"][va]                                   # requested conditioning (integer)
    print("=" * 78)
    print(f"A3 SAMPLER + ETA SWEEP  device={device}")
    print(f"val actions={len(va)} across {len({s.split('/')[0] for s in val_sessions})} users; "
          f"DDIM steps={N_INFERENCE}")
    print("=" * 78)

    # --- target-honoring verification (500 samples) --------------------------
    n_check = min(500, len(D_val))
    acts_chk, res_chk, _ = reconstruct(D_val[:n_check], eta=0.0, model=model, sched=sched,
                                       ckpt=ckpt, device=device)
    achieved = acts_chk.sum(axis=1)
    err = np.linalg.norm(achieved - D_val[:n_check], axis=1)
    print(f"\n[target-honoring] max ||achieved D - requested D|| over {n_check}: "
          f"{err.max():.3e} px   ({'PASS' if err.max() < 1e-2 else 'FAIL'})")
    print(f"                  mean {err.mean():.3e}  |  residual.sum max "
          f"{np.abs(res_chk.sum(axis=1)).max():.3e}")

    # --- genuine reference ---------------------------------------------------
    gen = geom_metrics(val_actions)

    # --- eta sweep -----------------------------------------------------------
    sweep = {}
    for eta in ETAS:
        acts, _, _ = reconstruct(D_val, eta=eta, model=model, sched=sched,
                                 ckpt=ckpt, device=device)
        sweep[eta] = geom_metrics(acts)
        print(f"  sampled eta={eta:.2f}")

    # --- table ---------------------------------------------------------------
    rows = [
        ("step dist p50 (px)", "step_p50"),
        ("step dist p90 (px)", "step_p90"),
        ("step dist p99 (px)", "step_p99"),
        ("turning-angle var", "turn_angle_var"),
        ("curvature (dist>=2px)", "curvature_filtered_ge2px"),
        ("path efficiency", "path_efficiency"),
    ]
    print("\n" + "=" * 78)
    print("ETA SWEEP  (metric x eta, genuine val alongside)")
    print("=" * 78)
    hdr = f"{'metric':<24}{'genuine':>10}" + "".join(f"{'eta='+format(e,'.2f'):>10}" for e in ETAS)
    print(hdr)
    print("-" * len(hdr))
    for label, key in rows:
        line = f"{label:<24}{gen[key]:>10.4f}"
        for e in ETAS:
            line += f"{sweep[e][key]:>10.4f}"
        print(line)

    # --- pick eta on realism vs genuine (blind) ------------------------------
    # normalized absolute deviation from genuine, averaged across metrics (equal weight)
    keys = [k for _, k in rows]
    scores = {}
    for e in ETAS:
        devs = []
        for k in keys:
            gv = gen[k]
            denom = abs(gv) if abs(gv) > 1e-9 else 1.0
            devs.append(abs(sweep[e][k] - gv) / denom)
        scores[e] = float(np.mean(devs))
    best_eta = min(scores, key=scores.get)

    print("\nrealism distance to genuine (mean normalized |dev|, lower=closer):")
    for e in ETAS:
        mark = "  <-- closest" if e == best_eta else ""
        print(f"   eta={e:.2f}: {scores[e]:.4f}{mark}")

    out = {"n_val": int(len(va)), "ddim_steps": N_INFERENCE,
           "val_users": sorted({s.split('/')[0] for s in val_sessions}),
           "target_honoring_max_px": float(err.max()),
           "genuine": gen, "sweep": {str(e): sweep[e] for e in ETAS},
           "realism_distance": {str(e): scores[e] for e in ETAS},
           "recommended_eta": best_eta,
           "selection_rule": "realism vs pooled genuine only; never bypass rate (target-blind, zero-query)"}
    with open(os.path.join(HERE, "eta_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {os.path.join(HERE, 'eta_sweep.json')}")
    print(f"recommended eta = {best_eta}")


if __name__ == "__main__":
    main()
