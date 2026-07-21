"""Step 6 smoke test: build ONE SapiAgent session and validate it is a well-formed
Balabit frame. No 20x3 set (step 7), no realism metrics (step 8)."""
import os, sys, importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from balabit_actions import list_training_sessions
from sapiagent_emitter import (make_empirical_dt_sampler, make_action_popper,
                               generate_sapiagent_session)

ART = os.path.join(os.path.dirname(__file__), "artifacts")
SCREEN = (1920, 1080)

# ---- inputs -------------------------------------------------------------
fcn_gen = np.load(os.path.join(ART, "fcn_gen.npy"))
print(f"fcn actions: {fcn_gen.shape}")

# dt pool from the SAME real Balabit sessions used in step 3 (65 training sessions)
sessions = [pd.read_csv(p, usecols=["record timestamp"]) for _, p in list_training_sessions()]
print(f"dt pool built from {len(sessions)} real Balabit sessions")

# ---- build ONE session (seed 0, 12 segments) ----------------------------
rng = np.random.default_rng(0)          # one seed-0 stream drives dest, pause, and dt
dt_sampler = make_empirical_dt_sampler(sessions, rng=rng)
popper = make_action_popper(fcn_gen, start=0)
df = generate_sapiagent_session(session_id=0, target_user="user9", rng=rng,
                                action_popper=popper, n_segments=12,
                                screen=SCREEN, dt_sampler=dt_sampler)

# reference: the exact columns/dtypes a WindMouse frame produces
wm_cols = ["record timestamp", "x", "y", "user", "session", "label"]

print("\n================= STRUCTURAL VALIDATION =================")
# columns exactly
cols_ok = list(df.columns) == wm_cols
print(f"[columns] {list(df.columns)}  exact-match={cols_ok}")
# dtypes
print(f"[dtypes] ts={df['record timestamp'].dtype} x={df['x'].dtype} y={df['y'].dtype} "
      f"label={df['label'].dtype}")
ts_float = np.issubdtype(df['record timestamp'].dtype, np.floating)
xy_int = np.issubdtype(df['x'].dtype, np.integer) and np.issubdtype(df['y'].dtype, np.integer)
label_ok = bool((df['label'] == 1).all())
print(f"[dtypes] timestamp-is-float={ts_float}  x/y-are-int={xy_int}  label-all-1={label_ok}")
# timestamps non-decreasing
t = df['record timestamp'].to_numpy()
nondec = bool(np.all(np.diff(t) >= 0))
print(f"[time] non-decreasing={nondec}  t0={t[0]:.4f}  tN={t[-1]:.4f}")
# bounds
xmin, xmax = int(df['x'].min()), int(df['x'].max())
ymin, ymax = int(df['y'].min()), int(df['y'].max())
x_in = 0 <= xmin and xmax < SCREEN[0]
y_in = 0 <= ymin and ymax < SCREEN[1]
print(f"[bounds] x[{xmin},{xmax}] in[0,{SCREEN[0]})={x_in}   y[{ymin},{ymax}] in[0,{SCREEN[1]})={y_in}")
# NaN
nan_any = bool(df.isna().to_numpy().any())
print(f"[nan] any-nan={nan_any}")
# per-step deltas: action-scale not screen-scale
dx = np.abs(np.diff(df['x'].to_numpy()))
dy = np.abs(np.diff(df['y'].to_numpy()))
max_dx, max_dy = int(dx.max()), int(dy.max())
teleport = (dx > 500).any() or (dy > 500).any()
print(f"[continuity] max|dx|={max_dx}  max|dy|={max_dy}  "
      f"any-step>500px(teleport)={bool(teleport)}")

# ---- extract_features status --------------------------------------------
print("\n================= extract_features =================")
has_ef = importlib.util.find_spec("extract_features") is not None
print("Defender's extract_features is NOT importable in this environment "
      "(it lives in the Colab defender notebook). Relying on structural validation. "
      f"(local find_spec('extract_features')={has_ef})")

# ---- plot ---------------------------------------------------------------
x, y = df['x'].to_numpy(), df['y'].to_numpy()
seg_len = 128
fig, ax = plt.subplots(figsize=(11, 6.5))
n_seg = len(x) // seg_len
cmap = plt.get_cmap("viridis", n_seg)
for k in range(n_seg):
    sl = slice(k*seg_len, (k+1)*seg_len)
    ax.plot(x[sl], y[sl], "-", color=cmap(k), lw=1.0)
    ax.scatter(x[k*seg_len], y[k*seg_len], color=cmap(k), s=28, zorder=3)
ax.axvline(0, c="r", lw=0.6, ls=":"); ax.axvline(SCREEN[0], c="r", lw=0.6, ls=":")
ax.axhline(0, c="r", lw=0.6, ls=":"); ax.axhline(SCREEN[1], c="r", lw=0.6, ls=":")
ax.set_xlim(-100, SCREEN[0]+100); ax.set_ylim(SCREEN[1]+100, -100)  # y inverted (screen)
ax.set_aspect("equal", adjustable="box")
ax.set_title("Step 6 smoke: one SapiAgent session (12 segments, colored; dots=segment starts)")
ax.set_xlabel("x"); ax.set_ylabel("y")
fig.tight_layout(); fig.savefig(os.path.join(ART, "step6_session_smoke.png"), dpi=110)
print(f"\nSaved plot -> {ART}/step6_session_smoke.png")

# ---- save session -------------------------------------------------------
df.to_csv(os.path.join(ART, "step6_session_smoke.csv"), index=False)
print(f"Saved session -> {ART}/step6_session_smoke.csv")

# ---- summary ------------------------------------------------------------
dts = np.diff(t)
print("\n================= SUMMARY =================")
print(f"total points   : {len(df)}")
print(f"n_segments     : {n_seg}")
print(f"duration (s)   : {t[-1]-t[0]:.3f}")
print(f"dt mean/std (s): {dts.mean():.4f} / {dts.std():.4f}")
print(f"max per-step dx/dy (px): {max_dx} / {max_dy}")
print("STEP6_SMOKE_DONE")
