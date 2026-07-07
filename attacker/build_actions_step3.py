"""Step 3 runner: build the pooled Balabit human-action corpus, sanity-check
shapes/stats, flag degeneracy, save the array, and plot sample real actions."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from balabit_actions import build_pooled_actions

ART = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ART, exist_ok=True)

# ---- build ---------------------------------------------------------------
actions, meta = build_pooled_actions()
N = meta["n_actions"]
print(f"Pooled actions: {N}  shape={actions.shape}  dtype={actions.dtype}")
print(f"  (raw {meta['n_actions_raw']} - {meta['n_dropped_artifact']} 16-bit-sentinel "
      f"artifact actions dropped @ max_abs_delta={meta['max_abs_delta']:.0f}px)")
print(f"From {meta['n_sessions']} sessions across {meta['n_users']} users.")
print("Per-user action counts:")
for u in sorted(meta["per_user"]):
    print(f"  {u:8s} {meta['per_user'][u]:6d}")

assert actions.ndim == 3 and actions.shape[1:] == (128, 2), "action shape wrong!"

# ---- delta / geometry stats ---------------------------------------------
flat = actions.reshape(-1, 2)  # all (dx, dy) pairs
print("\nDelta stats over all points:")
print(f"  dx  min/max/mean/std: {flat[:,0].min():.0f} / {flat[:,0].max():.0f} / "
      f"{flat[:,0].mean():.3f} / {flat[:,0].std():.2f}")
print(f"  dy  min/max/mean/std: {flat[:,1].min():.0f} / {flat[:,1].max():.0f} / "
      f"{flat[:,1].mean():.3f} / {flat[:,1].std():.2f}")
zero_frac = float(((flat[:,0] == 0) & (flat[:,1] == 0)).mean())
print(f"  zero-movement steps (dx==0 & dy==0): {zero_frac*100:.2f}%")

# per-action geometry
step_len = np.linalg.norm(actions, axis=2)          # (N,128)
path_len = step_len.sum(axis=1)                       # total path length
net_disp = np.linalg.norm(actions.sum(axis=1), axis=1)  # start->end distance
straightness = net_disp / (path_len + 1e-9)          # 1.0 == perfectly straight

# degeneracy: essentially no movement, or an almost-perfectly-straight line
deg_flat = path_len < 10.0
deg_straight = (straightness > 0.999) & (path_len >= 10.0)
print("\nPer-action geometry (path length, px):")
for p in (5, 25, 50, 75, 95):
    print(f"  {p:2d}th pct: {np.percentile(path_len, p):8.1f}")
print(f"  degenerate (near-zero movement, path<10px): {deg_flat.sum()} "
      f"({100*deg_flat.mean():.2f}%)")
print(f"  near-perfect straight lines (straightness>0.999): {deg_straight.sum()} "
      f"({100*deg_straight.mean():.2f}%)")
print(f"  straightness  median={np.median(straightness):.3f} "
      f"(1.0=straight; lower=more curved)")

# ---- save corpus ---------------------------------------------------------
np.save(os.path.join(ART, "balabit_human_actions.npy"), actions)
summary = {k: v for k, v in meta.items() if k != "provenance"}
summary["delta_std"] = [float(flat[:,0].std()), float(flat[:,1].std())]
summary["zero_step_frac"] = zero_frac
summary["pathlen_pct"] = {str(p): float(np.percentile(path_len, p)) for p in (5,25,50,75,95)}
with open(os.path.join(ART, "corpus_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved -> {ART}/balabit_human_actions.npy  and  corpus_summary.json")

# ---- plot sample real actions -------------------------------------------
# Pick 6 non-degenerate actions spanning the path-length distribution so we can
# eyeball plausible (curved) movement rather than noise or straight lines.
rng = np.random.default_rng(11235)
good = np.where(~deg_flat)[0]
targets = [np.percentile(path_len[good], p) for p in (30, 50, 65, 80, 90, 97)]
picks = []
for tv in targets:
    idx = good[np.argmin(np.abs(path_len[good] - tv))]
    picks.append(int(idx))

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for ax, idx in zip(axes.ravel(), picks):
    pts = np.cumsum(actions[idx], axis=0)  # reconstruct positions from origin
    ax.plot(pts[:, 0], pts[:, 1], "-", lw=1.3, color="#1f77b4")
    ax.scatter([pts[0, 0]], [pts[0, 1]], c="green", s=40, zorder=3, label="start")
    ax.scatter([pts[-1, 0]], [pts[-1, 1]], c="red", s=40, zorder=3, label="end")
    ax.set_title(f"action #{idx}\npath={path_len[idx]:.0f}px  "
                 f"net={net_disp[idx]:.0f}px  straight={straightness[idx]:.2f}",
                 fontsize=9)
    ax.invert_yaxis()  # screen coords: y grows downward
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=7, loc="best")
fig.suptitle("Sample real Balabit human actions (128-step dx/dy, reconstructed from origin)",
             fontsize=12)
fig.tight_layout()
out_png = os.path.join(ART, "sample_actions.png")
fig.savefig(out_png, dpi=110)
print(f"Saved plot -> {out_png}")
print(f"Plotted action indices: {picks}")
