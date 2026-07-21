"""A3 (DMTG) attack-session emitter -> Balabit format for the defender notebook.

Contract (matches A1/A2 so A3 is scoreable head-to-head):
  * 20 sessions/seed, seeds {0,1,2} = 60 sessions, 12 segments each.
  * Output <root>/dmtg_sessions/seed{0,1,2}/session_NN.csv with EXACTLY the three
    columns "record timestamp","x","y" (the defender's extract_features derives
    everything else from these; A3 contract asks for these three only).
  * Endpoints sampled uniformly over 1920x1080 (rng.integers), exactly like A1/A2.
  * Per-step timing + inter-segment pause REUSED BY IMPORT from sapiagent_emitter
    (make_empirical_dt_sampler, segment_timing) -- not reimplemented. rng is reset
    per seed and shared with the dt sampler, identical to realism_step8's A1 driver.
  * eta=0.25, num_inference_steps=50, checkpoint checkpoints/a3_dmtg_best.pt.

Per segment: D = dest - cur; sample a residual conditioned on D; reconstruct
action = equidistant(D) + residual (zero-mean projected so D is honored exactly);
advance cur += cumsum(action)[-1] (== dest). Off-screen intermediate points are NOT
clamped -- extract_features is delta-only, so OOB is non-confounding and clamping
would inject artifacts A1/A2 lack.

Does NOT score against defenders (that is the Colab notebook's job).
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ATTACKER_DIR = os.path.dirname(HERE)
if ATTACKER_DIR not in sys.path:
    sys.path.insert(0, ATTACKER_DIR)

from sapiagent_emitter import make_empirical_dt_sampler, segment_timing  # reuse by import
from balabit_actions import list_training_sessions
from sample import load_a3, reconstruct

SCREEN = (1920, 1080)
SEEDS = [0, 1, 2]
N_SESSIONS = 20
N_SEGMENTS = 12
ETA = 0.25
COLS = ["record timestamp", "x", "y"]
DEFAULT_ROOT = os.path.join(ATTACKER_DIR, os.pardir, "a2", "handoff")  # beside A1/A2 handoff


def _load_sessions_dt():
    """Same Balabit inter-event pool A1/A2 use for the empirical dt sampler."""
    return [pd.read_csv(p, usecols=["record timestamp"]) for _, p in list_training_sessions()]


def generate_dmtg_session(session_id, rng, model, sched, ckpt, device, dt_sampler,
                          eta=ETA, n_segments=N_SEGMENTS, screen=SCREEN, torch_seed=0,
                          return_debug=False):
    """One A3 session as a 3-column Balabit-shaped DataFrame.

    dest is sampled uniformly (rng.integers, like A1/A2). Because D is honored
    exactly, cur advances to dest each segment, so all 12 conditioning vectors are
    known up front and generated in one batched DDIM pass; timing is then assembled
    per segment via the shared segment_timing (identical dt draw + pause)."""
    W, H = screen
    start = np.array([rng.integers(0, W), rng.integers(0, H)], dtype=float)
    dests = np.array([[rng.integers(0, W), rng.integers(0, H)] for _ in range(n_segments)],
                     dtype=float)
    curs = np.vstack([start, dests[:-1]])                 # cur before each segment
    Dseg = dests - curs                                   # (n_segments, 2) requested D

    actions, _, _ = reconstruct(Dseg, eta, model, sched, ckpt, device, seed=torch_seed)

    xs, ys, ts = [], [], []
    t = 0.0
    cur = start.copy()
    achieved = np.empty_like(Dseg)
    for k in range(n_segments):
        action = actions[k]                               # (128,2) dx/dy deltas honoring Dseg[k]
        pts = cur + np.cumsum(action, axis=0)             # segment path (never clamped)
        achieved[k] = action.sum(axis=0)                  # == cumsum(action)[-1]
        seg_t, t = segment_timing(pts, t, rng, dt_sampler)  # SHARED timing + pause
        xs.extend(np.round(pts[:, 0]).astype(int))
        ys.extend(np.round(pts[:, 1]).astype(int))
        ts.extend(seg_t)
        cur = cur + achieved[k]                            # advance to dest (D honored)

    df = pd.DataFrame({
        "record timestamp": np.asarray(ts, dtype=float),
        "x": np.asarray(xs, dtype=int),
        "y": np.asarray(ys, dtype=int),
    })
    if return_debug:
        return df, {"Dreq": Dseg, "Dach": achieved, "start": start, "dests": dests}
    return df


# ----- session geometry (for reporting; matches step10's definitions) --------
def session_geometry(df):
    x = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)
    t = df["record timestamp"].to_numpy(float)
    step = np.hypot(np.diff(x), np.diff(y))
    path = float(step.sum())
    dur = float(t[-1] - t[0])
    vel = path / dur if dur > 0 else float("nan")
    net = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    return {"path": path, "duration": dur, "velocity": vel, "net": net}


def _make_seed_context(seed, model, ckpt, device):
    """Fresh rng per seed, dt sampler sharing that rng (identical to A1 driver)."""
    rng = np.random.default_rng(seed)
    dt_sampler = make_empirical_dt_sampler(_SESSIONS_DT, rng=rng)
    return rng, dt_sampler


# ----- checkpoint: one session ------------------------------------------------
def run_one(root):
    model, sched, ckpt, device = load_a3()
    print("=" * 74)
    print("A3 EMITTER CHECKPOINT — one session (seed 0, session 00)")
    print("=" * 74)
    rng = np.random.default_rng(0)
    dt_sampler = make_empirical_dt_sampler(_SESSIONS_DT, rng=rng)
    df, dbg = generate_dmtg_session(0, rng, model, sched, ckpt, device, dt_sampler,
                                    torch_seed=0, return_debug=True)

    print(f"\ncolumns      : {list(df.columns)}")
    print(f"dtypes       : {[str(df[c].dtype) for c in df.columns]}")
    print(f"row count    : {len(df)}  (expect {N_SEGMENTS} x 128 = {N_SEGMENTS*128})")
    print(f"\nfirst 5 rows:\n{df.head().to_string(index=False)}")

    derr = np.linalg.norm(dbg["Dach"] - dbg["Dreq"], axis=1)
    print(f"\nper-segment achieved vs requested D: max err {derr.max():.3e} px  "
          f"(mean {derr.mean():.3e})  ({'PASS' if derr.max() < 1e-2 else 'FAIL'})")

    g = session_geometry(df)
    print(f"\nduration        : {g['duration']:.3f} s")
    print(f"total path len  : {g['path']:.1f} px")
    print(f"mean velocity   : {g['velocity']:.1f} px/s")

    print(f"\ndt sampler module   : {make_empirical_dt_sampler.__module__}  "
          f"(imported, not a copy: {'YES' if make_empirical_dt_sampler.__module__ == 'sapiagent_emitter' else 'NO'})")
    print(f"segment_timing module: {segment_timing.__module__}  "
          f"(imported: {'YES' if segment_timing.__module__ == 'sapiagent_emitter' else 'NO'})")
    print("=" * 74)
    print("Checkpoint only — 60-session generation NOT run (use --all).")


# ----- full: all 60 -----------------------------------------------------------
def run_all(root):
    model, sched, ckpt, device = load_a3()
    out_root = os.path.join(root, "dmtg_sessions")
    print("=" * 74)
    print(f"A3 EMITTER — generating {len(SEEDS)*N_SESSIONS} sessions -> {out_root}")
    print("=" * 74)
    per_seed = {}
    geoms = []
    reqs = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)                 # reset per seed
        dt_sampler = make_empirical_dt_sampler(_SESSIONS_DT, rng=rng)  # shares rng
        d = os.path.join(out_root, f"seed{seed}")
        os.makedirs(d, exist_ok=True)
        for sid in range(N_SESSIONS):
            df, dbg = generate_dmtg_session(sid, rng, model, sched, ckpt, device,
                                            dt_sampler, torch_seed=seed * 1000 + sid,
                                            return_debug=True)
            df.to_csv(os.path.join(d, f"session_{sid:02d}.csv"), index=False)
            per_seed[seed] = per_seed.get(seed, 0) + 1
            geoms.append(session_geometry(df))
            reqs.append(float(np.hypot(dbg["Dreq"][:, 0], dbg["Dreq"][:, 1]).sum()))
        print(f"  seed {seed}: wrote {per_seed[seed]} sessions")

    path = np.array([g["path"] for g in geoms])
    dur = np.array([g["duration"] for g in geoms])
    vel = np.array([g["velocity"] for g in geoms])
    req = np.array(reqs)                                   # per-session sum of ||D_seg||
    ratio = path / req

    print("\n" + "=" * 74)
    print("AGGREGATE over 60 sessions (compare to step10: WM path/req 0.981, SA 0.776)")
    print("=" * 74)
    print(f"  per-seed counts      : {dict(sorted(per_seed.items()))}")
    print(f"  path length (px)     : {path.mean():8.1f} +/- {path.std():.1f}")
    print(f"  requested straight   : {req.mean():8.1f} +/- {req.std():.1f}  (sum ||D_seg||)")
    print(f"  path / requested     : {ratio.mean():8.3f} +/- {ratio.std():.3f}")
    print(f"  duration (s)         : {dur.mean():8.1f} +/- {dur.std():.1f}")
    print(f"  mean velocity (px/s) : {vel.mean():8.1f} +/- {vel.std():.1f}")
    print("=" * 74)
    print(f"done -> {out_root}/seed{{0,1,2}}/session_NN.csv")


_SESSIONS_DT = None

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="generate all 60 (default: one-session checkpoint)")
    ap.add_argument("--root", default=os.path.normpath(DEFAULT_ROOT),
                    help="output root; sessions go to <root>/dmtg_sessions/seed*/")
    args = ap.parse_args()
    _SESSIONS_DT = _load_sessions_dt()
    if args.all:
        run_all(args.root)
    else:
        run_one(args.root)
