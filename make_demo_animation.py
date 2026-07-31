#!/usr/bin/env python3
"""
ABAST demo animation for the 3-minute REU video.

Renders the 0:52-1:52 system demonstration: three passes against the same
enrolled user, showing a genuine session accepted, the diffusion bot
rejected, and the crude bot admitted then caught by the anomaly layer.

Two required inputs, both scoped to a single user so they are cheap to export:

  demo_scores_user9.csv
      source, session_id, score, threshold, accepted, anomaly_flagged
      source in {genuine, dmtg, windmouse}
      accepted / anomaly_flagged are 0/1 or True/False

  demo_trajectories_user9.csv
      source, session_id, step, x, y

Run with --preview to render with placeholder data and a watermark, so pacing
and layout can be checked before the real exports exist.

Usage:
    python make_demo_animation.py --input ./results --output ./video
    python make_demo_animation.py --preview --output ./video
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import FancyBboxPatch

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

USER = "user9"                   # Balabit id; used in filenames AND on screen.
                                 # Do NOT render this as a bare "user 9" -- the
                                 # poster/slide 6 relabels users 1-10, where
                                 # "User 9" is Balabit user16, a different person.
                                 # This user is "User 5" under that scheme.
FPS = 24
SECONDS_PER_PASS = 20.0          # three passes -> 60 s total, matches the shipped mp4
WIDTH_PX, HEIGHT_PX = 1920, 1080

COL = {
    "genuine": "#999999",
    "windmouse": "#0072B2",
    "sapiagent": "#E69F00",
    "dmtg": "#009E73",
    "accept": "#2E7D32",     # decision green, distinct from DMTG's #009E73
    "reject": "#A32D2D",
    "flagged": "#D55E00",
    "bad": "#A32D2D",
    "ink": "#1A1A1A",
    "muted": "#4D4D4D",
    "rule": "#333333",
}

PRETTY = {
    "genuine": "Genuine session",
    "windmouse": "WindMouse",
    "sapiagent": "SapiAgent",
    "dmtg": "DMTG",
}

# Pass order and the sentence shown under each. The third pass is the
# "challenging case" the video guidelines require.
PASSES = [
    ("genuine", "A real session from the enrolled user"),
    ("dmtg", "The diffusion bot, the most human-like one we built"),
    ("windmouse", "The crudest bot, a hand-written physics rule"),
]

# Phase boundaries as a fraction of one pass.
PH_LABEL = 0.07     # title appears
PH_DRAW = 0.47      # trajectory draws
PH_FEAT = 0.60      # feature extraction highlight
PH_SCORE = 0.80     # score meter fills
PH_VERDICT = 0.84   # verdict lands
# remainder: hold

plt.rcParams.update(
    {
        "font.size": 15,
        "text.color": COL["ink"],
        "axes.edgecolor": COL["rule"],
        "axes.labelcolor": COL["ink"],
        "xtick.color": COL["ink"],
        "ytick.color": COL["ink"],
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# --------------------------------------------------------------------------
# DATA
# --------------------------------------------------------------------------


def load_real(indir):
    spath = os.path.join(indir, f"demo_scores_{USER}.csv")
    tpath = os.path.join(indir, f"demo_trajectories_{USER}.csv")
    for p in (spath, tpath):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"missing {os.path.basename(p)}. Export it from the notebook, "
                "or run with --preview to check pacing first."
            )

    scores = pd.read_csv(spath)
    traj = pd.read_csv(tpath)
    for df in (scores, traj):
        df["source"] = df.source.str.lower().str.strip()
    for c in ("accepted", "anomaly_flagged"):
        if c in scores.columns:
            scores[c] = scores[c].astype(str).str.lower().isin(
                ["1", "true", "t", "yes"]
            )

    picked = {}
    for src, _ in PASSES:
        rows = scores[scores.source == src]
        if rows.empty:
            raise ValueError(f"no rows for source '{src}' in {os.path.basename(spath)}")
        # For bots pick the session that best illustrates the pass:
        # the highest-scoring one, i.e. the bot's best attempt.
        if src == "genuine":
            ok = rows[rows.accepted] if "accepted" in rows.columns else rows
            # Pass one must read as a clean accept. A genuine session that the
            # anomaly layer also flags would render ACCEPTED and FLAGGED at once,
            # contradicting the narration.
            if "anomaly_flagged" in ok.columns:
                clean = ok[~ok.anomaly_flagged]
                if not clean.empty:
                    ok = clean
                else:
                    print("  WARNING: every accepted genuine session is anomaly-"
                          "flagged. Pass one will show ACCEPTED and FLAGGED "
                          "together. Export more genuine sessions.")
            if ok.empty:
                raise ValueError(
                    "no accepted genuine session for this user. Pick a different "
                    "user or widen the export."
                )
            row = ok.loc[ok.score.idxmax()]
        else:
            row = rows.loc[rows.score.idxmax()]
        path = traj[(traj.source == src) & (traj.session_id == row.session_id)]
        path = path.sort_values("step")
        if path.empty:
            raise ValueError(f"no trajectory for {src} session {row.session_id}")
        picked[src] = {
            "x": path.x.to_numpy(float),
            "y": path.y.to_numpy(float),
            "score": float(row.score),
            "threshold": float(row.threshold),
            "accepted": bool(row.accepted),
            "flagged": bool(row.get("anomaly_flagged", False)),
        }
    return picked, False


def load_preview():
    """Placeholder data so pacing and layout can be reviewed early.

    These numbers are invented. The watermark makes that unmistakable.
    """
    rng = np.random.default_rng(7)

    def wander(n=128, jitter=1.0, smooth=0.85):
        vx = vy = 0.0
        xs, ys = [0.0], [0.0]
        for _ in range(n - 1):
            vx = smooth * vx + rng.normal(0, jitter)
            vy = smooth * vy + rng.normal(0, jitter)
            xs.append(xs[-1] + vx * 6)
            ys.append(ys[-1] + vy * 6)
        return np.array(xs), np.array(ys)

    gx, gy = wander(jitter=1.1)
    dx, dy = wander(jitter=1.0)
    wx, wy = wander(jitter=0.25, smooth=0.95)

    thr = 0.50
    return (
        {
            "genuine": dict(x=gx, y=gy, score=0.83, threshold=thr,
                            accepted=True, flagged=False),
            "dmtg": dict(x=dx, y=dy, score=0.15, threshold=thr,
                         accepted=False, flagged=False),
            "windmouse": dict(x=wx, y=wy, score=0.62, threshold=thr,
                              accepted=True, flagged=True),
        },
        True,
    )


# --------------------------------------------------------------------------
# FIGURE
# --------------------------------------------------------------------------


def build_figure(preview):
    fig = plt.figure(figsize=(WIDTH_PX / 120, HEIGHT_PX / 120), dpi=120)
    fig.patch.set_facecolor("white")

    ax_traj = fig.add_axes([0.05, 0.16, 0.52, 0.66])
    ax_traj.set_xticks([])
    ax_traj.set_yticks([])
    for s in ax_traj.spines.values():
        s.set_edgecolor("#CCCCCC")
    ax_traj.spines["top"].set_visible(True)
    ax_traj.spines["right"].set_visible(True)

    ax_score = fig.add_axes([0.66, 0.16, 0.12, 0.66])
    ax_score.set_xlim(0, 1)
    ax_score.set_ylim(0, 1)
    ax_score.set_xticks([])
    ax_score.spines["bottom"].set_visible(False)
    ax_score.set_ylabel("Share of session windows accepted")

    ax_txt = fig.add_axes([0.80, 0.16, 0.17, 0.66])
    ax_txt.axis("off")

    if preview:
        fig.text(
            0.5, 0.5, "PREVIEW  •  PLACEHOLDER DATA",
            ha="center", va="center", fontsize=54, color="#D0D0D0",
            rotation=24, zorder=0, alpha=0.85,
        )
    return fig, ax_traj, ax_score, ax_txt


def render(data, preview, outdir):
    fig, ax_traj, ax_score, ax_txt = build_figure(preview)

    frames_per_pass = int(FPS * SECONDS_PER_PASS)
    total = frames_per_pass * len(PASSES)

    title = fig.text(0.05, 0.90, "", fontsize=26, fontweight="bold",
                     color=COL["ink"])
    subtitle = fig.text(0.05, 0.855, "", fontsize=17, color=COL["muted"])
    caption = fig.text(0.05, 0.06, "", fontsize=17, color=COL["muted"])
    featline = fig.text(0.05, 0.115, "", fontsize=16, color=COL["ink"])

    def draw_frame(i):
        idx = min(i // frames_per_pass, len(PASSES) - 1)
        local = (i % frames_per_pass) / frames_per_pass
        src, blurb = PASSES[idx]
        d = data[src]
        color = COL[src]

        ax_traj.clear()
        ax_traj.set_xticks([])
        ax_traj.set_yticks([])
        for s in ax_traj.spines.values():
            s.set_edgecolor("#CCCCCC")
            s.set_visible(True)

        pad_x = (d["x"].max() - d["x"].min()) * 0.08 + 1
        pad_y = (d["y"].max() - d["y"].min()) * 0.08 + 1
        ax_traj.set_xlim(d["x"].min() - pad_x, d["x"].max() + pad_x)
        ax_traj.set_ylim(d["y"].max() + pad_y, d["y"].min() - pad_y)  # screen coords

        title.set_text(PRETTY[src])
        subtitle.set_text(blurb)

        # --- trajectory ---------------------------------------------------
        if local <= PH_LABEL:
            n = 0
        else:
            frac = min((local - PH_LABEL) / (PH_DRAW - PH_LABEL), 1.0)
            n = max(2, int(frac * len(d["x"])))
        if n:
            ax_traj.plot(d["x"][:n], d["y"][:n], color=color, lw=2.4,
                         solid_capstyle="round")
            ax_traj.scatter(d["x"][n - 1], d["y"][n - 1], s=60, color=color,
                            zorder=3, edgecolors="white", linewidths=1.2)

        # --- feature extraction -------------------------------------------
        if local > PH_DRAW:
            featline.set_text(
                "Full session  ·  scored in 128-step windows  ·  10 features per step"
            )
        else:
            featline.set_text("")

        # --- score meter ---------------------------------------------------
        ax_score.clear()
        ax_score.set_xlim(0, 1)
        ax_score.set_ylim(0, 1)
        ax_score.set_xticks([])
        ax_score.spines["top"].set_visible(False)
        ax_score.spines["right"].set_visible(False)
        ax_score.spines["bottom"].set_visible(False)
        ax_score.set_ylabel("Share of session windows accepted")

        if local > PH_FEAT:
            frac = min((local - PH_FEAT) / (PH_SCORE - PH_FEAT), 1.0)
            val = d["score"] * frac
            ax_score.bar([0.5], [val], width=0.62, color=color)
            ax_score.text(0.5, val + 0.025, f"{val:.2f}", ha="center",
                          fontsize=17, color=COL["ink"])
        ax_score.axhline(d["threshold"], color=COL["rule"], ls="--", lw=1.6)
        ax_score.text(1.02, d["threshold"], "majority rule", va="center",
                      fontsize=13, color=COL["rule"])

        # --- verdict ---------------------------------------------------------
        ax_txt.clear()
        ax_txt.axis("off")
        if local > PH_VERDICT:
            accepted = d["accepted"]
            verdict = "ACCEPTED" if accepted else "REJECTED"

            badge = COL["accept"] if accepted else COL["reject"]

            if src == "genuine":
                meaning, tone = "real user, correct", COL["muted"]
            elif accepted:
                meaning, tone = "bot admitted", COL["reject"]
            else:
                meaning, tone = "attack blocked", COL["muted"]

            box = FancyBboxPatch(
                (0.02, 0.60), 0.94, 0.16,
                boxstyle="round,pad=0.02",
                linewidth=2.4,
                edgecolor=badge,
                facecolor="white",
                transform=ax_txt.transAxes,
            )
            ax_txt.add_patch(box)
            ax_txt.text(0.49, 0.68, verdict, transform=ax_txt.transAxes,
                        ha="center", va="center", fontsize=24,
                        fontweight="bold", color=badge)
            ax_txt.text(0.49, 0.55, meaning, transform=ax_txt.transAxes,
                        ha="center", va="top", fontsize=15, color=tone)

            # The anomaly layer only evaluates sessions the gate admitted, so
            # a rejected session never reaches it. Gate the display on accepted.
            if accepted and d["flagged"] and local > (PH_VERDICT + 0.04):
                ax_txt.text(0.49, 0.40, "anomaly layer:",
                            transform=ax_txt.transAxes, ha="center",
                            fontsize=14, color=COL["muted"])
                ax_txt.text(0.49, 0.33, "FLAGGED",
                            transform=ax_txt.transAxes, ha="center",
                            fontsize=20, fontweight="bold", color=COL["flagged"])
                ax_txt.text(
                    0.49, 0.24,
                    "but it also flags\n58% of real users",
                    transform=ax_txt.transAxes, ha="center", va="top",
                    fontsize=13, color=COL["muted"],
                )

        caption.set_text(f"Enrolled user: {USER}  ·  T2 Random Forest detector")
        return []

    anim = FuncAnimation(fig, draw_frame, frames=total, interval=1000 / FPS,
                         blit=False)

    os.makedirs(outdir, exist_ok=True)
    stem = "abast_demo_preview" if preview else "abast_demo"
    out = os.path.join(outdir, f"{stem}.mp4")
    writer = FFMpegWriter(fps=FPS, bitrate=6000,
                          metadata={"title": "ABAST system demonstration"})
    anim.save(out, writer=writer, dpi=120)
    plt.close(fig)

    print(f"wrote {out}")
    print(f"  {total} frames at {FPS} fps  ->  {total / FPS:.1f} s")
    if preview:
        print("  PREVIEW: numbers are placeholders, do not use in the video")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=".", help="directory with the demo CSVs")
    p.add_argument("--output", default="./video")
    p.add_argument("--preview", action="store_true",
                   help="render with placeholder data and a watermark")
    p.add_argument("--seconds-per-pass", type=float, default=SECONDS_PER_PASS)
    args = p.parse_args()

    globals()["SECONDS_PER_PASS"] = args.seconds_per_pass

    data, preview = load_preview() if args.preview else load_real(args.input)
    render(data, preview, args.output)


if __name__ == "__main__":
    main()