"""Step 9 prep (local): package the SapiAgent attack set for Colab handoff.
Copy CSVs (not pkls), portability-check them, copy realism tables, and VERIFY the
corrected dist>=2px curvature finding before documenting it. No bypass scoring."""
import os, sys, glob, shutil
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from balabit_actions import list_training_sessions
from sapiagent_emitter import make_empirical_dt_sampler

A2 = os.path.dirname(__file__)
ART = os.path.join(A2, "artifacts")
HAND = os.path.join(A2, "handoff")
WM_COLS = ["record timestamp", "x", "y", "user", "session", "label"]
SCREEN = (1920, 1080); SEEDS = [0, 1, 2]; N_SESSIONS = 20

# ---- 1. copy 60 per-session CSVs, preserve seed grouping (no pkls) -------
n_copied = 0
for seed in SEEDS:
    src_dir = os.path.join(ART, "sessions", f"seed{seed}")
    dst_dir = os.path.join(HAND, "sapiagent_sessions", f"seed{seed}")
    os.makedirs(dst_dir, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(src_dir, "session_*.csv"))):
        shutil.copy2(f, os.path.join(dst_dir, os.path.basename(f)))
        n_copied += 1
print(f"copied {n_copied} CSVs into handoff/sapiagent_sessions/")

# ---- 2. portability check: bare pd.read_csv, assert structure -----------
checks = {"columns": 0, "dtypes": 0, "label1": 0, "time_nondec": 0, "no_nan": 0}
n = 0; failures = []
for f in sorted(glob.glob(os.path.join(HAND, "sapiagent_sessions", "seed*", "session_*.csv"))):
    d = pd.read_csv(f)                    # bare read, exactly as Colab would
    n += 1
    cols_ok = list(d.columns) == WM_COLS
    dt_ok = (np.issubdtype(d["record timestamp"].dtype, np.floating)
             and np.issubdtype(d["x"].dtype, np.integer)
             and np.issubdtype(d["y"].dtype, np.integer)
             and np.issubdtype(d["label"].dtype, np.integer))
    label_ok = bool((d["label"] == 1).all())
    nondec = bool(np.all(np.diff(d["record timestamp"].to_numpy()) >= 0))
    no_nan = not bool(d.isna().to_numpy().any())
    for k, v in [("columns", cols_ok), ("dtypes", dt_ok), ("label1", label_ok),
                 ("time_nondec", nondec), ("no_nan", no_nan)]:
        checks[k] += v
        if not v:
            failures.append((os.path.relpath(f, HAND), k))
print(f"\nPortability check on {n} bare-reloaded CSVs:")
for k, v in checks.items():
    print(f"  {k:12s}: {v}/{n}  {'PASS' if v == n else '*** FAIL ***'}")
if failures:
    print("  failures:", failures)

# ---- 3. verify the corrected dist>=2px curvature finding ----------------
def mean_abs_curv(df, min_dist=None):
    dx = df["x"].diff().fillna(0); dy = df["y"].diff().fillna(0)
    dist = np.sqrt(dx**2 + dy**2); dist_safe = dist.replace(0, 1e-6)
    angle = np.arctan2(dy, dx)
    curv = (angle.diff().fillna(0) / dist_safe).replace([np.inf, -np.inf], np.nan).fillna(0).clip(-100, 100)
    if min_dist is None:
        return float(curv.abs().mean())
    m = (dist >= min_dist).to_numpy()
    return float(curv[m].abs().mean()) if m.sum() else np.nan

# genuine (65 raw sessions)
gen = [pd.read_csv(p, usecols=["record timestamp", "x", "y"]) for _, p in list_training_sessions()]
# sapiagent (60 handoff CSVs)
sa = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(HAND, "sapiagent_sessions", "seed*", "session_*.csv")))]
# windmouse (regenerate 60 deterministically, reset rng per seed) — needs the emitter
_sqrt3 = np.sqrt(3); _sqrt5 = np.sqrt(5)
def wind_mouse(sx, sy, dx_, dy_, G_0=9, W_0=3, M_0=15, D_0=12, rng=None):
    if rng is None: rng = np.random.default_rng()
    sx, sy = float(sx), float(sy); cx, cy = sx, sy
    v_x = v_y = W_x = W_y = 0.0; pts = [(cx, cy)]
    dist = np.hypot(dx_-sx, dy_-sy)
    while dist >= 1:
        W_mag = min(W_0, dist)
        if dist >= D_0:
            W_x = W_x/_sqrt3 + (2*rng.random()-1)*W_mag/_sqrt5
            W_y = W_y/_sqrt3 + (2*rng.random()-1)*W_mag/_sqrt5
        else:
            W_x /= _sqrt3; W_y /= _sqrt3
            if M_0 < 3: M_0 = rng.random()*3 + 3
            else: M_0 /= _sqrt5
        v_x += W_x + G_0*(dx_-sx)/dist; v_y += W_y + G_0*(dy_-sy)/dist
        v_mag = np.hypot(v_x, v_y)
        if v_mag > M_0:
            v_clip = M_0/2 + rng.random()*M_0/2
            v_x = (v_x/v_mag)*v_clip; v_y = (v_y/v_mag)*v_clip
        sx += v_x; sy += v_y
        mx, my = int(round(sx)), int(round(sy))
        if cx != mx or cy != my: cx, cy = mx, my; pts.append((cx, cy))
        dist = np.hypot(dx_-sx, dy_-sy)
    return np.array(pts)
sessions_dt = [pd.read_csv(p, usecols=["record timestamp"]) for _, p in list_training_sessions()]
wm = []
for seed in SEEDS:
    rng = np.random.default_rng(seed); dts = make_empirical_dt_sampler(sessions_dt, rng=rng)
    W, H = SCREEN
    for sid in range(N_SESSIONS):
        cur = np.array([rng.integers(0, W), rng.integers(0, H)], float)
        xs, ys, ts = [], [], []; t = 0.0
        for _ in range(12):
            dest = np.array([rng.integers(0, W), rng.integers(0, H)], float)
            pts = wind_mouse(cur[0], cur[1], dest[0], dest[1], rng=rng); nn = len(pts)
            dd = np.asarray(dts(max(nn-1, 1)), float)[:nn-1]; dd = np.clip(dd, 1e-4, None)
            seg_t = t + np.concatenate([[0.0], np.cumsum(dd)]) if nn > 1 else np.array([t])
            xs.extend(np.round(pts[:, 0]).astype(int)); ys.extend(np.round(pts[:, 1]).astype(int)); ts.extend(seg_t)
            t = seg_t[-1] + rng.uniform(0.05, 0.6); cur = dest
        wm.append(pd.DataFrame({"record timestamp": ts, "x": xs, "y": ys}))

print("\nCurvature verification (mean |curvature|):")
print(f"  {'source':10s} {'unfiltered':>12s} {'dist>=2px':>12s}")
for name, lst in [("genuine", gen), ("windmouse", wm), ("sapiagent", sa)]:
    unf = np.mean([mean_abs_curv(d) for d in lst])
    flt = np.nanmean([mean_abs_curv(d, 2.0) for d in lst])
    print(f"  {name:10s} {unf:12.3f} {flt:12.3f}")

# ---- copy realism tables into handoff -----------------------------------
for fn in ["realism_metrics.csv", "realism_per_session.csv"]:
    shutil.copy2(os.path.join(ART, fn), os.path.join(HAND, fn))
print("\ncopied realism_metrics.csv, realism_per_session.csv into handoff/")
print("STEP9PREP_DONE")
