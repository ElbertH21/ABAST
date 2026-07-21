"""Step 8: realism comparison — genuine vs windmouse vs sapiagent, all scored with
the defender's own extract_features (verbatim, read-only). No bypass scoring, no SHAP."""
import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from balabit_actions import list_training_sessions
from sapiagent_emitter import make_empirical_dt_sampler   # reuse verbatim (same timing)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ART = os.path.join(_REPO, "a2", "artifacts")                    # SapiAgent pkls (scratch)
DMTG_DIR = os.path.join(_REPO, "a2", "handoff", "dmtg_sessions")     # A3 saved sessions
ART = os.path.join(_REPO, "results")                                 # tracked deliverables (outputs)
os.makedirs(ART, exist_ok=True)
SCREEN = (1920, 1080); W, H = SCREEN
SEEDS = [0, 1, 2]; N_SESSIONS = 20; N_SEGMENTS = 12


# ---- defender's extract_features (VERBATIM — do not modify) --------------
def extract_features(df):
    df = df.copy().reset_index(drop=True)
    dt = df['record timestamp'].diff().fillna(0)
    dt = dt.replace(0, 1e-6)
    dx = df['x'].diff().fillna(0)
    dy = df['y'].diff().fillna(0)
    dist = np.sqrt(dx**2 + dy**2)
    dist_safe = dist.replace(0, 1e-6)
    vx = dx / dt; vy = dy / dt
    velocity = dist / dt
    acceleration = velocity.diff().fillna(0) / dt
    jerk = acceleration.diff().fillna(0) / dt
    angle = np.arctan2(dy, dx)
    curvature = angle.diff().fillna(0) / dist_safe
    pause = (dt > 0.1).astype(int)
    result = pd.DataFrame({'vx': vx, 'vy': vy, 'velocity': velocity,
        'acceleration': acceleration, 'jerk': jerk, 'angle': angle,
        'curvature': curvature, 'pause': pause})
    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
    result['vx'] = result['vx'].clip(-5000, 5000)
    result['vy'] = result['vy'].clip(-5000, 5000)
    result['velocity'] = result['velocity'].clip(0, 5000)
    result['acceleration'] = result['acceleration'].clip(-1e6, 1e6)
    result['jerk'] = result['jerk'].clip(-1e8, 1e8)
    result['curvature'] = result['curvature'].clip(-100, 100)
    return result


# ---- exact wind_mouse from the defender notebook ------------------------
_sqrt3 = np.sqrt(3); _sqrt5 = np.sqrt(5)
def wind_mouse(start_x, start_y, dest_x, dest_y, G_0=9, W_0=3, M_0=15, D_0=12, rng=None):
    if rng is None: rng = np.random.default_rng()
    start_x, start_y = float(start_x), float(start_y)
    cx, cy = start_x, start_y
    v_x = v_y = W_x = W_y = 0.0
    pts = [(cx, cy)]
    dist = np.hypot(dest_x-start_x, dest_y-start_y)
    while dist >= 1:
        W_mag = min(W_0, dist)
        if dist >= D_0:
            W_x = W_x/_sqrt3 + (2*rng.random()-1)*W_mag/_sqrt5
            W_y = W_y/_sqrt3 + (2*rng.random()-1)*W_mag/_sqrt5
        else:
            W_x /= _sqrt3; W_y /= _sqrt3
            if M_0 < 3: M_0 = rng.random()*3 + 3
            else: M_0 /= _sqrt5
        v_x += W_x + G_0*(dest_x-start_x)/dist
        v_y += W_y + G_0*(dest_y-start_y)/dist
        v_mag = np.hypot(v_x, v_y)
        if v_mag > M_0:
            v_clip = M_0/2 + rng.random()*M_0/2
            v_x = (v_x/v_mag)*v_clip; v_y = (v_y/v_mag)*v_clip
        start_x += v_x; start_y += v_y
        mx, my = int(round(start_x)), int(round(start_y))
        if cx != mx or cy != my:
            cx, cy = mx, my; pts.append((cx, cy))
        dist = np.hypot(dest_x-start_x, dest_y-start_y)
    return np.array(pts)


# ---- original WindMouse emitter (brief verbatim; cur=dest, no constraint) -
def generate_attack_session(session_id, target_user, rng, n_segments=12, screen=(1920, 1080),
                            dt_sampler=None, fixed_dt=0.010, jitter=0.25,
                            pause_range=(0.05, 0.6), wm_kwargs=None):
    wm_kwargs = wm_kwargs or {}
    Wd, Hd = screen
    cur = np.array([rng.integers(0, Wd), rng.integers(0, Hd)], dtype=float)
    xs, ys, ts = [], [], []
    t = 0.0
    for _ in range(n_segments):
        dest = np.array([rng.integers(0, Wd), rng.integers(0, Hd)], dtype=float)
        pts = wind_mouse(cur[0], cur[1], dest[0], dest[1], rng=rng, **wm_kwargs)
        n = len(pts)
        if dt_sampler is not None:
            dts = np.asarray(dt_sampler(max(n - 1, 1)), dtype=float)[: n - 1]
        else:
            dts = fixed_dt * (1.0 + jitter * (2 * rng.random(n - 1) - 1))
        dts = np.clip(dts, 1e-4, None)
        seg_t = t + np.concatenate([[0.0], np.cumsum(dts)]) if n > 1 else np.array([t])
        xs.extend(np.round(pts[:, 0]).astype(int)); ys.extend(np.round(pts[:, 1]).astype(int))
        ts.extend(seg_t)
        t = seg_t[-1] + rng.uniform(*pause_range)
        cur = dest
    return pd.DataFrame({"record timestamp": np.asarray(ts, dtype=float),
        "x": np.asarray(xs, dtype=int), "y": np.asarray(ys, dtype=int),
        "user": target_user, "session": session_id, "label": 1})


def session_metrics(df):
    """Five per-session realism metrics + the per-event velocity vector."""
    f = extract_features(df)
    return {
        "velocity_mean": float(f["velocity"].mean()),
        "pause_rate": float(f["pause"].mean()),
        "curvature_mean": float(f["curvature"].abs().mean()),
        "angle_var": float(f["angle"].std()),
        "jerk_mean": float(f["jerk"].abs().mean()),
    }, f["velocity"].to_numpy()


# ---- assemble the three sources -----------------------------------------
sessions_dt = [pd.read_csv(p, usecols=["record timestamp"]) for _, p in list_training_sessions()]

sources = {}   # name -> list[(id, df)]
# 1. genuine: 65 raw Balabit sessions
gen = []
for user, p in list_training_sessions():
    df = pd.read_csv(p, usecols=["record timestamp", "x", "y"])
    gen.append((f"{user}/{os.path.basename(p)}", df))
sources["genuine"] = gen
# 2. windmouse: 60 sessions, original emitter, reset rng per seed
wm = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    dts = make_empirical_dt_sampler(sessions_dt, rng=rng)
    for sid in range(N_SESSIONS):
        df = generate_attack_session(sid, "windmouse", rng, n_segments=N_SEGMENTS,
                                     screen=SCREEN, dt_sampler=dts)
        wm.append((f"seed{seed}_s{sid}", df))
sources["windmouse"] = wm
# 3. sapiagent: existing 60 sessions
sa = []
for seed in SEEDS:
    lst = pickle.load(open(os.path.join(SRC_ART, f"sapiagent_sessions_seed{seed}.pkl"), "rb"))
    for sid, df in enumerate(lst):
        sa.append((f"seed{seed}_s{sid}", df))
sources["sapiagent"] = sa
# 4. dmtg (A3): 60 saved 3-col Balabit CSVs — SAME extract_features, same basis
dm = []
for seed in SEEDS:
    for sid in range(N_SESSIONS):
        df = pd.read_csv(os.path.join(DMTG_DIR, f"seed{seed}", f"session_{sid:02d}.csv"))
        dm.append((f"seed{seed}_s{sid}", df))
sources["dmtg"] = dm
print({k: len(v) for k, v in sources.items()})

# ---- metrics -------------------------------------------------------------
METRICS = ["velocity_mean", "pause_rate", "curvature_mean", "angle_var", "jerk_mean"]
per_session_rows = []
event_vel = {s: [] for s in sources}     # pooled per-event speeds for the velocity panel
for src, lst in sources.items():
    for sid, df in lst:
        m, vel = session_metrics(df)
        per_session_rows.append({"source": src, "id": sid, **m})
        event_vel[src].append(vel)
per_session = pd.DataFrame(per_session_rows)
per_session.to_csv(os.path.join(ART, "realism_per_session.csv"), index=False)

SOURCES4 = ["genuine", "windmouse", "sapiagent", "dmtg"]
agg_rows = []
for src in SOURCES4:
    sub = per_session[per_session["source"] == src]
    row = {"source": src, "n_sessions": len(sub)}
    for m in METRICS:
        row[f"{m}_mean"] = sub[m].mean()
        row[f"{m}_std"] = sub[m].std()
    agg_rows.append(row)
agg = pd.DataFrame(agg_rows)
agg.to_csv(os.path.join(ART, "realism_metrics.csv"), index=False)

# ---- displacement-filtered curvature ------------------------------------
# The raw curvature (extract_features) divides Δangle by dist_safe (0 -> 1e-6), so
# near-stationary steps inflate it and make SapiAgent look human. Restricting to
# steps with dist >= 2px is the honest version. Curvature comes VERBATIM from
# extract_features; only the step-distance mask is computed here (same dx/dy/hypot
# extract_features itself uses), purely to filter — no new metric, no reimplementation.
def _curv_and_dist(df):
    f = extract_features(df)                                  # verbatim: clipped |curvature|
    d = df.copy().reset_index(drop=True)
    dx = d["x"].diff().fillna(0); dy = d["y"].diff().fillna(0)
    dist = np.sqrt(dx**2 + dy**2).to_numpy()                  # identical to extract_features' dist
    return f["curvature"].abs().to_numpy(), dist

curv_rows = []
persess_filt = {s: [] for s in SOURCES4}     # per-session filtered mean -> table ± std
for src in SOURCES4:
    cabs_parts, dist_parts = [], []
    for sid, df in sources[src]:
        cabs, dist = _curv_and_dist(df)
        cabs_parts.append(cabs); dist_parts.append(dist)
        keep_s = dist >= 2.0
        persess_filt[src].append(float(cabs[keep_s].mean()) if keep_s.any() else np.nan)
    cabs_all = np.concatenate(cabs_parts); dist_all = np.concatenate(dist_parts)
    keep = dist_all >= 2.0
    curv_rows.append({"source": src,
                      "curvature_raw": float(cabs_all.mean()),
                      "curvature_dist_ge_2px": float(cabs_all[keep].mean()),
                      "n_steps_total": int(cabs_all.size),
                      "n_steps_kept": int(keep.sum())})
curv_df = pd.DataFrame(curv_rows)
curv_df.to_csv(os.path.join(ART, "realism_curvature_filtered.csv"), index=False)

pd.set_option("display.width", 240, "display.max_columns", 30)
print("\n=== realism_metrics (mean +/- std across sessions) ===")
show = agg.set_index("source")
for m in METRICS:
    show[m] = show[f"{m}_mean"].map(lambda v: f"{v:.3g}") + " ± " + show[f"{m}_std"].map(lambda v: f"{v:.2g}")
# filtered-curvature column (per-session mean ± std), inserted next to raw curvature
for src in show.index:
    vals = np.array(persess_filt[src], dtype=float); vals = vals[~np.isnan(vals)]
    show.loc[src, "curvature_filt"] = f"{vals.mean():.3g} ± {vals.std():.2g}"
cols = ["n_sessions", "velocity_mean", "pause_rate", "curvature_mean",
        "curvature_filt", "angle_var", "jerk_mean"]
print(show[cols].to_string())
print("\n=== realism_curvature_filtered.csv (pooled steps per source) ===")
print(curv_df.to_string(index=False))

# ---- WindMouse OOB freebie (closes step-7 deferral) ---------------------
def oob_frac(df):
    x = df["x"].to_numpy(); y = df["y"].to_numpy()
    return float(((x < 0) | (x >= W) | (y < 0) | (y >= H)).mean())
wm_oob = np.array([oob_frac(df) for _, df in sources["windmouse"]])
sa_oob = np.array([oob_frac(df) for _, df in sources["sapiagent"]])
print(f"\n=== OOB fraction ===\n  windmouse: mean={wm_oob.mean()*100:.3f}%  max={wm_oob.max()*100:.3f}%")
print(f"  sapiagent: mean={sa_oob.mean()*100:.3f}%  max={sa_oob.max()*100:.3f}%")

# ---- distribution figure ------------------------------------------------
COL = {"genuine": "#009E73", "windmouse": "#E69F00", "sapiagent": "#0072B2", "dmtg": "#CC79A7"}
plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 14, "legend.fontsize": 14})
fig, axes = plt.subplots(2, 3, figsize=(19, 11))
ax = axes.ravel()

# panel 0: velocity from pooled per-EVENT speeds (log-y density)
for src in SOURCES4:
    v = np.concatenate(event_vel[src])
    ax[0].hist(v, bins=np.linspace(0, 2000, 60), density=True, histtype="step",
               lw=2.4, color=COL[src], label=src)
ax[0].set_yscale("log"); ax[0].set_title("velocity (per-event speed)")
ax[0].set_xlabel("px / s"); ax[0].set_ylabel("density (log)")

# panels 1-4: per-session metric distributions
panel_cfg = [("pause_rate", "pause rate (frac dt>0.1)", False),
             ("curvature_mean", "curvature (mean |Δangle/dist|)", True),
             ("angle_var", "angle variation (std angle)", False),
             ("jerk_mean", "jerk (mean |jerk|, smoothness)", True)]
for i, (metric, title, logx) in enumerate(panel_cfg, start=1):
    vals = {src: per_session[per_session.source == src][metric].to_numpy() for src in COL}
    allv = np.concatenate(list(vals.values()))
    if logx:
        pos = allv[allv > 0]
        lo = max(pos.min() if pos.size else 1e-3, 1e-3)
        bins = np.logspace(np.log10(lo), np.log10(allv.max() + 1e-9), 30)
        ax[i].set_xscale("log")
    else:
        bins = np.linspace(allv.min(), allv.max(), 30)
    for src in SOURCES4:
        ax[i].hist(vals[src], bins=bins, density=True, histtype="stepfilled",
                   alpha=0.45, color=COL[src], label=src)
    if logx:  # clean decade ticks only — kill colliding minor-tick labels
        from matplotlib.ticker import LogLocator, NullFormatter
        ax[i].xaxis.set_major_locator(LogLocator(base=10))
        ax[i].xaxis.set_minor_formatter(NullFormatter())
    ax[i].set_title(title); ax[i].set_ylabel("density"); ax[i].set_xlabel(metric)

handles, labels = ax[0].get_legend_handles_labels()
ax[5].axis("off"); ax[5].legend(handles, labels, loc="center", fontsize=20, title="source",
                                title_fontsize=20, frameon=True)
fig.suptitle("Trajectory realism: genuine vs WindMouse vs SapiAgent vs DMTG (defender extract_features)",
             fontsize=20, y=0.995)
fig.tight_layout()
fig.savefig(os.path.join(ART, "realism_distributions.png"), dpi=200)
print(f"\nSaved -> realism_metrics.csv, realism_per_session.csv, realism_distributions.png")

# ---- RESULTS.md entry ----------------------------------------------------
g = show.loc["genuine"]; wmr = show.loc["windmouse"]; sar = show.loc["sapiagent"]; dmr = show.loc["dmtg"]
cf = curv_df.set_index("source")["curvature_dist_ge_2px"]
finding = (f"Four-source, one extract_features. Pause-rate genuine/WM/SA/DMTG = {g['pause_rate']} / "
           f"{wmr['pause_rate']} / {sar['pause_rate']} / {dmr['pause_rate']}: all three synthetics share the "
           f"empirical dt sampler so their pause rates match and sit far above genuine — acceptance is a "
           f"timing/threshold-gap effect, not mimicry. Geometry separates them: velocity genuine/WM/SA/DMTG = "
           f"{g['velocity_mean']} / {wmr['velocity_mean']} / {sar['velocity_mean']} / {dmr['velocity_mean']}. "
           f"Displacement-filtered curvature (dist>=2px) genuine/WM/SA/DMTG = {cf['genuine']:.3g} / "
           f"{cf['windmouse']:.3g} / {cf['sapiagent']:.3g} / {cf['dmtg']:.3g}: the raw curvature makes SapiAgent "
           f"look human, but filtering the near-stationary steps shows SapiAgent ~{cf['sapiagent']/cf['genuine']:.1f}x "
           f"genuine while DMTG stays close — correcting the 'SapiAgent matches human curvature' claim.")
line = ("\n## Step 8 — realism comparison (4-source: +DMTG/A3)\n"
        f"- [realism_metrics.csv](realism_metrics.csv), [realism_per_session.csv](realism_per_session.csv), "
        f"[realism_curvature_filtered.csv](realism_curvature_filtered.csv), "
        f"[realism_distributions.png](realism_distributions.png)\n"
        f"- Finding: {finding}\n")
with open(os.path.join(ART, "RESULTS.md"), "a") as f:
    f.write(line)
print("Appended entry to RESULTS.md")
print("STEP8_DONE")
