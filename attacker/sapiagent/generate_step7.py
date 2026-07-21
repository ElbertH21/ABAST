"""Step 7: produce the full SapiAgent attack set — 20 sessions/seed x seeds [0,1,2]
= 60 sessions. Emitter used UNCHANGED (no cur-clamp). No realism metrics (step 8)."""
import os, sys, glob, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from balabit_actions import list_training_sessions
from sapiagent_emitter import (make_empirical_dt_sampler, make_action_popper,
                               generate_sapiagent_session)

ART = os.path.join(os.path.dirname(__file__), "artifacts")
SCREEN = (1920, 1080)
W, H = SCREEN
SEEDS = [0, 1, 2]
N_SESSIONS = 20
N_SEGMENTS = 12                       # same as the step-6 smoke session
WM_COLS = ["record timestamp", "x", "y", "user", "session", "label"]

# ---- inputs -------------------------------------------------------------
fcn_gen = np.load(os.path.join(ART, "fcn_gen.npy"))
sessions_dt = [pd.read_csv(p, usecols=["record timestamp"]) for _, p in list_training_sessions()]
print(f"fcn actions {fcn_gen.shape}; dt pool from {len(sessions_dt)} Balabit sessions")

# ---- generate 20 sessions/seed, reset pointer AND rng per seed -----------
all_sessions = {}   # seed -> list[DataFrame]
for seed in SEEDS:
    rng = np.random.default_rng(seed)                      # rng reset per seed
    dt_sampler = make_empirical_dt_sampler(sessions_dt, rng=rng)
    popper = make_action_popper(fcn_gen, start=0)          # action pointer reset per seed
    lst = []
    for sid in range(N_SESSIONS):
        df = generate_sapiagent_session(
            session_id=sid, target_user="sapiagent", rng=rng, action_popper=popper,
            n_segments=N_SEGMENTS, screen=SCREEN, dt_sampler=dt_sampler)
        lst.append(df)
    all_sessions[seed] = lst
    # save pkl (primary, as named) + portable per-session CSVs
    with open(os.path.join(ART, f"sapiagent_sessions_seed{seed}.pkl"), "wb") as f:
        pickle.dump(lst, f)
    csv_dir = os.path.join(ART, "sessions", f"seed{seed}")
    os.makedirs(csv_dir, exist_ok=True)
    for sid, df in enumerate(lst):
        df.to_csv(os.path.join(csv_dir, f"session_{sid:02d}.csv"), index=False)
    print(f"seed {seed}: {len(lst)} sessions saved (pkl + {csv_dir}/*.csv)")

# ---- structural validation across all 60 + OOB instrumentation ----------
rows = []
checks = {"columns": 0, "dtypes": 0, "label1": 0, "time_nondec": 0, "no_nan": 0, "cont_lt500": 0}
global_max_delta = 0
total_points = 0
per_seed_points = {}
for seed in SEEDS:
    sp = 0
    for sid, df in enumerate(all_sessions[seed]):
        x = df["x"].to_numpy(); y = df["y"].to_numpy(); t = df["record timestamp"].to_numpy()
        n = len(df); total_points += n; sp += n
        # structural
        cols_ok = list(df.columns) == WM_COLS
        dt_ok = (np.issubdtype(df["record timestamp"].dtype, np.floating)
                 and np.issubdtype(df["x"].dtype, np.integer)
                 and np.issubdtype(df["y"].dtype, np.integer))
        label_ok = bool((df["label"] == 1).all())
        nondec = bool(np.all(np.diff(t) >= 0))
        no_nan = not bool(df.isna().to_numpy().any())
        dmax = int(max(np.abs(np.diff(x)).max(), np.abs(np.diff(y)).max())) if n > 1 else 0
        cont_ok = dmax < 500
        global_max_delta = max(global_max_delta, dmax)
        checks["columns"] += cols_ok; checks["dtypes"] += dt_ok; checks["label1"] += label_ok
        checks["time_nondec"] += nondec; checks["no_nan"] += no_nan; checks["cont_lt500"] += cont_ok
        # OOB
        oob = ((x < 0) | (x >= W) | (y < 0) | (y >= H))
        rows.append({
            "seed": seed, "session": sid, "n_points": n,
            "oob_fraction": float(oob.mean()),
            "x_low_over": float(max(0, 0 - x.min())),
            "x_high_over": float(max(0, x.max() - (W - 1))),
            "y_low_over": float(max(0, 0 - y.min())),
            "y_high_over": float(max(0, y.max() - (H - 1))),
            "max_per_step_delta": dmax,
        })
    per_seed_points[seed] = sp

oob_df = pd.DataFrame(rows)
oob_df.to_csv(os.path.join(ART, "step7_oob_summary.csv"), index=False)

N = len(rows)
print("\n================= STRUCTURAL VALIDATION (all %d sessions) =================" % N)
for k, v in checks.items():
    print(f"  {k:14s}: {v}/{N}  {'ALL PASS' if v == N else '*** FAIL ***'}")
print(f"  global max per-step delta across all 60: {global_max_delta} px  "
      f"({'<500 OK' if global_max_delta < 500 else 'EXCEEDS 500'})")

print("\n================= OOB INSTRUMENTATION =================")
print(f"  OOB fraction: mean={oob_df['oob_fraction'].mean()*100:.3f}%  "
      f"max={oob_df['oob_fraction'].max()*100:.3f}%")
worst = {e: oob_df[e].max() for e in ["x_low_over", "x_high_over", "y_low_over", "y_high_over"]}
print(f"  max overshoot per edge (px): {worst}")
print(f"  single worst overshoot (any edge): {max(worst.values()):.1f} px")
print(f"  WindMouse OOB symmetry: NOT available locally (no wind_mouse code/frames "
      f"in this env) -> compare in Colab.")

# ---- rotation-invariance sanity: rotation orients, never reshapes --------
from sapiagent_emitter import _rotate_toward
def _plen(d):
    return float(np.linalg.norm(d, axis=1).sum())
def _curv(d):
    v1, v2 = d[:-1], d[1:]
    cross = v1[:, 0]*v2[:, 1] - v1[:, 1]*v2[:, 0]
    dot = (v1*v2).sum(1)
    return float(np.abs(np.arctan2(cross, dot)).mean())
print("\n================= ROTATION-INVARIANCE SANITY =================")
rs = np.random.default_rng(123)
inv_ok = True
for k in rs.choice(fcn_gen.shape[0], 3, replace=False):
    a = fcn_gen[k]
    dest = np.array([rs.integers(0, W), rs.integers(0, H)], float)
    r = _rotate_toward(a, np.array([W/2, H/2], float), dest, 5.0)
    dpl, dcv = abs(_plen(a)-_plen(r)), abs(_curv(a)-_curv(r))
    inv_ok = inv_ok and dpl < 1e-3 and dcv < 1e-6
    print(f"  action #{k}: path_len raw={_plen(a):.3f} rot={_plen(r):.3f} (Δ={dpl:.2e}) | "
          f"curvature raw={_curv(a):.5f} rot={_curv(r):.5f} (Δ={dcv:.2e})")
print(f"  rotation-invariant (shape unchanged, only oriented): {inv_ok}")

print("\n================= SUMMARY =================")
print(f"  sessions: {N}  ({len(SEEDS)} seeds x {N_SESSIONS})")
print(f"  total points: {total_points}   per-seed: {per_seed_points}")
print(f"  max per-step delta (all 60): {global_max_delta} px")
print(f"  saved: sapiagent_sessions_seed[0,1,2].pkl + sessions/seedN/*.csv + step7_oob_summary.csv")
print("STEP7_DONE")
