#!/usr/bin/env python3
"""
ABAST poster/paper figure generation.

Reads the canonical leakage-free CSV artifacts and emits four figures plus an
audit file listing every number that appears in them.

Nothing is hardcoded. If a number changes in a CSV, rerun this and both the
figures and the audit change together.

Usage:
    python make_poster_figures.py --input ./results --output ./figures
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

POLICY = "calibration_FAR<=0.10"
FLOOR_BASE = 0.00
FLOOR_APPLIED = 0.05
COLLAPSE_CUTOFF = 0.05

# Panel 3 aggregation. "case" = flat mean over model-user-seed cases.
# "tier" = tier-equal-weighted. See README note in the audit output.
PANEL3_AGG = "case"

COL = {
    "genuine": "#999999",
    "windmouse": "#0072B2",
    "sapiagent": "#E69F00",
    "dmtg": "#009E73",
    "a0": "#333333",
    "bypass": "#A32D2D",
    "bypass_light": "#F09595",
    "gray_dark": "#888780",
    "gray_light": "#B4B2A9",
}

PRETTY = {"windmouse": "WindMouse", "sapiagent": "SapiAgent", "dmtg": "DMTG"}

# Figure 4 encodes session source by color and defense stage by shade.
SOURCE_COL = {
    "Genuine user": "#999999",       # same grey used for genuine elsewhere
    "Human impostor": "#CC79A7",     # Okabe-Ito, unused by any attacker
    "Mouse bot": "#A32D2D", # same attack red as figure 3
}


def lighten(hex_color, amount=0.55):
    """Blend a hex color toward white. amount=0 keeps it, 1 makes it white."""
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    out = [int(c + (255 - c) * amount) for c in rgb]
    return "#{:02X}{:02X}{:02X}".format(*out)
ATTACK_ORDER = ["windmouse", "sapiagent", "dmtg"]

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#1A1A1A",
        "text.color": "#1A1A1A",
        "xtick.color": "#1A1A1A",
        "ytick.color": "#1A1A1A",
        "axes.linewidth": 1.0,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "svg.fonttype": "path",
    }
)

AUDIT = []

# Every figure is written twice: PNG for quick review, SVG for the poster.
# SVG is resolution independent, so a 12-inch placement stays sharp at any dpi.
SAVE_FORMATS = ("png", "svg")


def save(fig, outdir, stem):
    """Write one figure in every configured format and log the paths."""
    written = []
    for ext in SAVE_FORMATS:
        path = os.path.join(outdir, f"{stem}.{ext}")
        fig.savefig(path)
        written.append(path)
    plt.close(fig)
    log(f"  wrote {stem}." + "/".join(SAVE_FORMATS))
    return written


def log(msg):
    AUDIT.append(msg)
    print(msg)


def section(title):
    log("")
    log("=" * 72)
    log(title)
    log("=" * 72)


# --------------------------------------------------------------------------
# AGGREGATION HELPERS
# --------------------------------------------------------------------------


def tier_equal_weighted(df, value_col, by=None):
    """Mean within each tier, then unweighted mean across tiers.

    This is the canonical aggregation for headline bypass numbers. It prevents
    tiers with more models (T1, T2) from dominating tiers with one model
    (T0, T3).
    """
    by = by or []
    inner = df.groupby(by + ["tier"])[value_col].mean()
    if not by:
        return inner.mean()
    return inner.groupby(by).mean()


def require(path, what):
    if not os.path.exists(path):
        log(f"  MISSING: {os.path.basename(path)}  ({what})")
        return None
    return path


def pct(x):
    return f"{x * 100:.2f}%"


# --------------------------------------------------------------------------
# REALISM LOADER
# --------------------------------------------------------------------------


def load_realism(indir):
    """Read the corrected realism artifacts and return mean |log ratio| distance.

    Curvature has two published variants. The filtered one (steps of at least
    2px) is what the poster uses; the raw one is kept as a cross-check because
    it changes the distances even though it does not change the ordering.
    """
    mpath = os.path.join(indir, "realism_metrics_corrected.csv")
    if not os.path.exists(mpath):
        log("  MISSING: realism_metrics_corrected.csv")
        return None, None, None

    m = pd.read_csv(mpath)
    m["source"] = m.source.str.lower()
    m = m.set_index("source")

    cpath = os.path.join(indir, "realism_curvature_filtered_corrected.csv")
    if os.path.exists(cpath):
        c = pd.read_csv(cpath)
        c["source"] = c.source.str.lower()
        c = c.set_index("source")
        curv = c["curvature_dist_ge_2px"]
        curv_note = "filtered (steps >= 2px)"
        curv_alt = c["curvature_raw"]
    else:
        curv = m["curvature_mean_mean"]
        curv_note = "raw (curvature_mean_mean)"
        curv_alt = None

    realism = pd.DataFrame(
        {
            "velocity": m["velocity_mean_mean"],
            "pause_rate": m["pause_rate_mean"],
            "curvature": curv,
        }
    )
    log(f"  curvature variant used: {curv_note}")

    gen = realism.loc["genuine"]
    distances = {}
    contributions = {}
    for src in realism.index:
        if src == "genuine":
            continue
        parts = {c: abs(np.log(realism.loc[src, c] / gen[c])) for c in realism.columns}
        d = float(np.mean(list(parts.values())))
        distances[src] = d
        contributions[src] = parts
        detail = "  ".join(f"{k} {v:.3f}" for k, v in parts.items())
        log(f"  {PRETTY.get(src, src):<11} distance {d:.4f}   ({detail})")

    if curv_alt is not None:
        alt = {}
        for src in realism.index:
            if src == "genuine":
                continue
            alt[src] = float(
                np.mean(
                    [
                        abs(np.log(realism.loc[src, "velocity"] / gen["velocity"])),
                        abs(np.log(realism.loc[src, "pause_rate"] / gen["pause_rate"])),
                        abs(np.log(curv_alt[src] / curv_alt["genuine"])),
                    ]
                )
            )
        log(
            "  [cross-check] distances with RAW curvature: "
            + ", ".join(f"{PRETTY.get(k, k)} {v:.4f}" for k, v in alt.items())
        )

    return distances, realism, contributions


# --------------------------------------------------------------------------
# FIGURE 1 - bot realism vs effectiveness
# --------------------------------------------------------------------------


def figure_1(indir, outdir):
    section("FIGURE 1 - bot realism strip")

    attacks = pd.read_csv(os.path.join(indir, "all_attack_results_corrected.csv"))
    attacks = attacks[attacks.threshold_policy == POLICY]
    bypass = tier_equal_weighted(attacks, "bypass_rate_session", ["attack_type"])

    a0 = pd.read_csv(os.path.join(indir, "a0_baseline_session_corrected.csv"))
    a0 = a0[a0.threshold_policy == POLICY]
    a0_rate = tier_equal_weighted(a0, "FAR_session")

    log(f"  A0 human impostor baseline (tier-equal-weighted): {pct(a0_rate)}")
    for a in ATTACK_ORDER:
        verdict = "ABOVE baseline" if bypass[a] > a0_rate else "BELOW baseline"
        log(f"  {PRETTY[a]:<11} bypass {pct(bypass[a])}   {verdict}")
    log(f"  A0 by tier: {(a0.groupby('tier').FAR_session.mean() * 100).round(2).to_dict()}")

    # --- realism inputs -----------------------------------------------------
    distances, realism, contributions = load_realism(indir)
    tpath = require(
        os.path.join(indir, "trajectory_samples.csv"),
        "columns: source, step, x, y  (needed only for the trajectory strip)",
    )

    # --- build ------------------------------------------------------------
    if tpath and distances:
        traj = pd.read_csv(tpath)
        sources = ["Genuine"] + [PRETTY[a] for a in ATTACK_ORDER]
        fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.9))
        for ax, src in zip(axes, sources):
            sub = traj[traj.source.str.lower() == src.lower()].sort_values("step")
            key = src.lower()
            ax.plot(
                sub.x,
                sub.y,
                color=COL.get(key, COL["genuine"]),
                lw=1.4,
                solid_capstyle="round",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(src, fontsize=10, pad=6)
            if src == "Genuine":
                sub1, sub2 = "reference", ""
            else:
                sub1 = f"distance {distances[key]:.2f}"
                sub2 = f"bypass {bypass[key] * 100:.0f}%"
            ax.set_xlabel(f"{sub1}\n{sub2}", fontsize=8.5, labelpad=6)
            for s in ax.spines.values():
                s.set_edgecolor("#CCCCCC")
        fig.suptitle(
            "The most human-like bot was the least effective",
            fontsize=11,
            fontweight="bold",
            x=0.02,
            ha="left",
            y=1.02,
        )
        save(fig, outdir, "fig1_attacker_realism")
    if distances and contributions:
        # Realism only. Bar length is the distance from genuine; segments show
        # which metric drives it. Segments are each contribution / 3, so they
        # sum exactly to the mean absolute log ratio.
        fig, ax = plt.subplots(figsize=(6.2, 2.6))
        metric_col = {
            "pause_rate": "#B4B2A9",
            "velocity": "#6E9BC5",
            "curvature": "#3A5A78",
        }
        metric_lab = {
            "pause_rate": "pause",
            "velocity": "velocity",
            "curvature": "curvature",
        }
        order = sorted(ATTACK_ORDER, key=lambda a: distances[a])
        ys = np.arange(len(order))
        for y, a in zip(ys, order):
            left = 0.0
            for m in ["pause_rate", "velocity", "curvature"]:
                w = contributions[a][m] / 3.0
                ax.barh(y, w, left=left, height=0.55, color=metric_col[m])
                left += w
            ax.text(left + 0.015, y, f"{distances[a]:.3f}", va="center",
                    fontsize=9)
        ax.set_yticks(ys)
        ax.set_yticklabels([PRETTY[a] for a in order])
        ax.set_xlim(0, max(distances.values()) * 1.18)
        ax.set_xlabel("Distance from genuine movement  \u2192")
        ax.grid(axis="x", color="#D9D9D9", lw=0.7)
        ax.set_axisbelow(True)
        ax.set_title(
            "How human-like each bot is, on the step-level\nfeatures the detectors actually read",
            fontsize=11, fontweight="bold", loc="left", pad=10)
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=metric_col[m], label=metric_lab[m])
            for m in ["pause_rate", "velocity", "curvature"]
        ]
        ax.legend(handles=handles, loc="lower right", frameon=False,
                  fontsize=8, ncol=3)
        save(fig, outdir, "fig1c_realism_distance")

    if distances:
        # Slopegraph: realism ordering on the left, effectiveness on the right.
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        lx, rx = 0.0, 1.0
        dmax = max(distances.values()) * 1.15
        bmax = max(bypass[a] for a in ATTACK_ORDER) * 100 * 1.15
        styles = {"windmouse": "-", "sapiagent": (0, (6, 3)), "dmtg": (0, (1, 3))}
        # Draw the lines first, then place right-hand labels with collision
        # avoidance. WindMouse and SapiAgent land within 0.2 points of each
        # other, so their labels must be pushed apart and leadered back.
        right = []
        for a in ATTACK_ORDER:
            ly = 1 - distances[a] / dmax
            ry = bypass[a] * 100 / bmax
            ax.plot([lx, rx], [ly, ry], color=COL[a], lw=2, ls=styles[a], zorder=2)
            ax.scatter([lx, rx], [ly, ry], color=COL[a], s=36, zorder=3)
            ax.text(lx - 0.03, ly, f"{PRETTY[a]}  {distances[a]:.2f}",
                    ha="right", va="center", fontsize=8.5)
            right.append([ry, f"{PRETTY[a]}  {bypass[a] * 100:.1f}%", COL[a]])

        a0_y = a0_rate * 100 / bmax
        ax.plot([rx - 0.30, rx], [a0_y, a0_y], color=COL["a0"], ls="--", lw=0.9,
                zorder=1)
        right.append([a0_y, f"human impostor  {a0_rate * 100:.1f}%", COL["a0"]])

        # spread label positions downward until each clears the one above
        right.sort(key=lambda r: -r[0])
        min_gap = 0.075
        placed = []
        for anchor, label, color in right:
            y = anchor
            if placed and placed[-1] - y < min_gap:
                y = placed[-1] - min_gap
            placed.append(y)
            ax.plot([rx + 0.015, rx + 0.075], [anchor, y], color=color, lw=0.7,
                    zorder=1)
            ax.text(rx + 0.09, y, label, ha="left", va="center", fontsize=8.5,
                    color="#1A1A1A" if color != COL["a0"] else COL["a0"])

        ax.set_xlim(-0.55, 1.75)
        ax.text(lx, 1.10, "Realism distance", ha="center", fontsize=9.5,
                fontweight="bold")
        ax.text(lx, 1.03, "lower is more human", ha="center", fontsize=8,
                color="#333333")
        ax.text(rx, 1.10, "Bypass rate", ha="center", fontsize=9.5,
                fontweight="bold")
        ax.text(rx, 1.03, "higher is more effective", ha="center", fontsize=8,
                color="#333333")
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(-0.08, 1.18)
        ax.axis("off")
        ax.set_title("Realism and effectiveness run in opposite directions",
                     fontsize=11, fontweight="bold", loc="left", pad=18)
        save(fig, outdir, "fig1b_realism_slopegraph")

    if True:
        # Always emit: bypass against the human baseline, fully data-driven.
        fig, ax = plt.subplots(figsize=(5.2, 3.0))
        names = [PRETTY[a] for a in ATTACK_ORDER]
        vals = [bypass[a] * 100 for a in ATTACK_ORDER]
        cols = [COL[a] for a in ATTACK_ORDER]
        ax.bar(names, vals, color=cols, width=0.55)
        ax.axhline(a0_rate * 100, color=COL["a0"], ls="--", lw=1)
        ax.text(
            2.46,
            a0_rate * 100 - 0.9,
            f"human impostor\nbaseline {a0_rate * 100:.1f}%",
            ha="right",
            va="top",
            fontsize=8.5,
            linespacing=1.35,
            color=COL["a0"],
        )
        for i, v in enumerate(vals):
            ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)
        ax.set_ylabel("Session bypass rate")
        ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
        ax.set_ylim(0, max(vals) * 1.35)
        ax.set_title(
            "Only two of three bots beat a human impostor",
            fontsize=11,
            fontweight="bold",
            loc="left",
            pad=10,
        )
        save(fig, outdir, "fig1_bypass_vs_baseline")

    return bypass, a0_rate


# --------------------------------------------------------------------------
# FIGURE 2 - per-user vulnerability
# --------------------------------------------------------------------------


def figure_2(indir, outdir):
    section("FIGURE 2 - per-user bypass dot plot")

    d = pd.read_csv(os.path.join(indir, "all_attack_results_corrected.csv"))
    d = d[d.threshold_policy == POLICY]
    per_user = tier_equal_weighted(
        d, "bypass_rate_session", ["user", "attack_type"]
    ).unstack()
    per_user = per_user[ATTACK_ORDER]
    per_user["max"] = per_user.max(axis=1)
    per_user = per_user.sort_values("max", ascending=False)

    log("  tier-equal-weighted, sorted by worst case")
    log("  poster label -> balabit id (FOOTNOTE THIS ON THE POSTER)")
    mapping = {}
    for i, u in enumerate(per_user.index, start=1):
        mapping[f"User {i}"] = u
        row = "  ".join(
            f"{PRETTY[a]} {per_user.loc[u, a] * 100:.1f}%" for a in ATTACK_ORDER
        )
        log(f"    User {i:<2} = {u:<7}  {row}")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ys = np.arange(len(per_user))[::-1]
    markers = {"windmouse": "o", "sapiagent": "s", "dmtg": "^"}

    for y, (u, row) in zip(ys, per_user.iterrows()):
        vals = [row[a] * 100 for a in ATTACK_ORDER]
        ax.plot([min(vals), max(vals)], [y, y], color="#666666", lw=0.9, zorder=1)
        for a in ATTACK_ORDER:
            ax.scatter(
                row[a] * 100,
                y,
                marker=markers[a],
                s=42,
                color=COL[a],
                zorder=3,
                edgecolors="white",
                linewidths=0.5,
            )

    ax.set_yticks(ys)
    ax.set_yticklabels([f"User {i}" for i in range(1, len(per_user) + 1)])
    ax.set_xlim(0, 50)
    ax.set_xlabel("Attack sessions accepted as the real user  \u2192")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
    ax.grid(axis="x", color="#D9D9D9", lw=0.7)
    ax.set_axisbelow(True)
    ax.set_title(
        "The same user can be immune to one bot\nand wide open to another",
        fontsize=11,
        fontweight="bold",
        loc="left",
        pad=12,
    )
    handles = [
        Line2D(
            [],
            [],
            marker=markers[a],
            color="none",
            markerfacecolor=COL[a],
            markersize=7,
            label=PRETTY[a],
        )
        for a in ATTACK_ORDER
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        frameon=False,
        fontsize=8.5,
        ncol=1,
    )
    save(fig, outdir, "fig2_per_user_bypass")
    return mapping


# --------------------------------------------------------------------------
# FIGURE 3 - threshold floor
# --------------------------------------------------------------------------


def figure_3(indir, outdir):
    section("FIGURE 3 - threshold floor, two-group split")

    d = pd.read_csv(os.path.join(indir, "threshold_floor_sensitivity.csv"))
    bcols = ["bypass_windmouse", "bypass_sapiagent", "bypass_dmtg"]
    d["bypass_mean"] = d[bcols].mean(axis=1)

    base = d[d.floor == FLOOR_BASE].copy()
    applied = d[d.floor == FLOOR_APPLIED].copy()
    key = ["user", "seed", "model"]

    base["collapsed"] = base.accept_threshold < COLLAPSE_CUTOFF
    flags = base.set_index(key)["collapsed"]
    applied = applied.set_index(key)
    applied["collapsed"] = flags
    applied = applied.reset_index()

    n_total = len(base)
    n_col = int(base.collapsed.sum())
    log(f"  cases total: {n_total}   collapsed (<{COLLAPSE_CUTOFF}): {n_col}")
    log(f"  collapsed by tier: {base[base.collapsed].groupby('tier').size().to_dict()}")

    def agg(frame, collapsed):
        sub = frame[frame.collapsed == collapsed]
        if PANEL3_AGG == "tier":
            return (
                sub.groupby("tier").TAR.mean().mean(),
                sub.groupby("tier").bypass_mean.mean().mean(),
            )
        return sub.TAR.mean(), sub.bypass_mean.mean()

    groups = []
    for collapsed, label in [
        (False, f"{n_total - n_col} normal cases"),
        (True, f"{n_col} collapsed-threshold cases"),
    ]:
        tar_b, byp_b = agg(base, collapsed)
        tar_a, byp_a = agg(applied, collapsed)
        groups.append((label, tar_b, tar_a, byp_b, byp_a))
        log(f"  {label}")
        log(f"    genuine accepted : {pct(tar_b)} -> {pct(tar_a)}")
        log(f"    mean bypass      : {pct(byp_b)} -> {pct(byp_a)}")

    # cross-check both aggregations so the choice is visible in the audit
    for mode in ("case", "tier"):
        globals()["PANEL3_AGG"] = mode
        t, b = agg(base, False)
        log(f"  [cross-check] normal group, {mode} agg: TAR {pct(t)} bypass {pct(b)}")
    globals()["PANEL3_AGG"] = "case"

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    w = 0.16
    tick_pos, tick_lab, group_pos, group_lab = [], [], [], []
    for gi, (label, tb, ta, bb, ba) in enumerate(groups):
        base_x = gi * 1.15
        xs = [base_x - 0.30, base_x - 0.13, base_x + 0.17, base_x + 0.34]
        vals = [tb * 100, ta * 100, bb * 100, ba * 100]
        cols = [
            COL["gray_dark"],
            COL["gray_light"],
            COL["bypass"],
            COL["bypass_light"],
        ]
        ax.bar(xs, vals, width=w, color=cols)
        ax.hlines(tb * 100, xs[0] - 0.12, xs[-1] + 0.12, color="#333333", ls="--", lw=0.8)
        for x, v in zip(xs, vals):
            ax.text(x, v + 1.2, f"{v:.1f}", ha="center", fontsize=8)
        tick_pos += [(xs[0] + xs[1]) / 2, (xs[2] + xs[3]) / 2]
        tick_lab += ["Genuine\naccepted", "Attack\nbypass"]
        group_pos.append(base_x)
        group_lab.append(label)

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, fontsize=8)
    ax.tick_params(axis="x", length=0, pad=4)
    for gx, gl in zip(group_pos, group_lab):
        ax.text(
            gx,
            -0.20,
            gl,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9.5,
            fontweight="bold",
        )
    ax.set_ylim(0, 105)
    ax.set_ylabel("Rate")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
    ax.set_title(
        "The threshold floor is a targeted fix, not a general improvement",
        fontsize=11,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COL["gray_dark"], label="Genuine accepted"),
        plt.Rectangle((0, 0), 1, 1, color=COL["bypass"], label="Attack bypass"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        frameon=False,
        fontsize=8.5,
        ncol=2,
    )
    ax.text(
        0.5,
        -0.42,
        f"paler bar = threshold floor at {FLOOR_APPLIED}",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="#1A1A1A",
    )
    save(fig, outdir, "fig3_threshold_floor")


# --------------------------------------------------------------------------
# FIGURE 4 - layered defense
# --------------------------------------------------------------------------


def figure_4(indir, outdir):
    section("FIGURE 4 - layered defense tradeoff")

    d = pd.read_csv(os.path.join(indir, "layered_defense_session_level.csv"))
    log(f"  gatekeeper(s): {sorted(d.gatekeeper.unique())}  (single model, session level)")

    synth = ["WindMouse", "SapiAgent", "DMTG_Inspired"]
    d["group"] = np.where(
        d.source == "Genuine",
        "Genuine user",
        np.where(d.source == "HumanImpostor", "Human impostor", "Mouse bot"),
    )

    # per-user mean, then mean across users (reproduces the locked 53.7/22.6)
    def per_user(frame, col):
        return frame.groupby("user")[col].mean().mean()

    rows = []
    for g in ["Genuine user", "Human impostor", "Mouse bot"]:
        sub = d[d.group == g]
        gate = per_user(sub, "gate_accepted")
        final = per_user(sub, "final_accepted")
        rows.append((g, gate, final))
        log(f"  {g:<20} gate {pct(gate)}  ->  after anomaly {pct(final)}")

    gen_acc = d[(d.group == "Genuine user") & (d.gate_accepted)]
    flagged = gen_acc.groupby("user").anomaly_flagged.mean().mean()
    log(f"  of gate-accepted genuine sessions, flagged: {pct(flagged)}")

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    labels = [r[0] for r in rows]
    xs = np.arange(len(rows))
    gate_v = [r[1] * 100 for r in rows]
    final_v = [r[2] * 100 for r in rows]
    gate_cols = [SOURCE_COL[r[0]] for r in rows]
    final_cols = [lighten(c) for c in gate_cols]
    ax.bar(xs - 0.17, gate_v, width=0.30, color=gate_cols)
    ax.bar(xs + 0.17, final_v, width=0.30, color=final_cols)
    for x, v in zip(xs - 0.17, gate_v):
        ax.text(x, v + 1.0, f"{v:.1f}%", ha="center", fontsize=8.5)
    for x, v in zip(xs + 0.17, final_v):
        ax.text(x, v + 1.0, f"{v:.1f}%", ha="center", fontsize=8.5)
    ax.hlines(gate_v[0], -0.35, 2.35, color="#333333", ls="--", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 74)
    ax.set_ylabel("Sessions accepted")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
    stage_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#6E6E6E", label="Gate only"),
        plt.Rectangle((0, 0), 1, 1, color=lighten("#6E6E6E"), label="Gate + anomaly"),
    ]
    ax.legend(
        handles=stage_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        frameon=False,
        fontsize=8.5,
    )
    ax.text(
        0.995, 0.80, "color indicates session source",
        transform=ax.transAxes, ha="right", fontsize=8.5, color="#1A1A1A",
    )
    ax.set_title(
        "The anomaly layer stops every synthetic session,\nat a cost to real users",
        fontsize=11,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    ax.text(
        0.985,
        0.52,
        f"{flagged * 100:.1f}%",
        transform=ax.transAxes,
        ha="right",
        fontsize=17,
        color="#333333",
    )
    ax.text(
        0.985,
        0.42,
        "of gate-accepted genuine\nsessions flagged",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color="#333333",
    )
    save(fig, outdir, "fig4_layered_defense")



# --------------------------------------------------------------------------
# FIGURE 5 - bypass by defender tier (for the Methods pipeline panel)
# --------------------------------------------------------------------------


def figure_tiers(indir, outdir):
    section("FIGURE 5 - bypass by defender tier")

    d = pd.read_csv(os.path.join(indir, "all_attack_results_corrected.csv"))
    d = d[d.threshold_policy == POLICY]
    p = d.groupby(["tier", "attack_type"]).bypass_rate_session.mean().unstack()
    p = p[ATTACK_ORDER] * 100

    a0 = pd.read_csv(os.path.join(indir, "a0_baseline_session_corrected.csv"))
    a0 = a0[a0.threshold_policy == POLICY]
    a0_tier = a0.groupby("tier").FAR_session.mean() * 100

    log("  session bypass % by tier:")
    for t in p.index:
        row = "  ".join(f"{PRETTY[a]} {p.loc[t, a]:.2f}" for a in ATTACK_ORDER)
        log(f"    {t}  {row}   |  A0 human {a0_tier[t]:.2f}")

    tiers = list(p.index)
    x = np.arange(len(tiers))
    w = 0.25

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for k, a in enumerate(ATTACK_ORDER):
        ax.bar(x + (k - 1) * w, p[a].values, width=w, color=COL[a],
               label=PRETTY[a])

    # human impostor reference, one dash per tier group
    for xi, t in zip(x, tiers):
        ax.hlines(a0_tier[t], xi - 1.62 * w, xi + 1.62 * w, color=COL["a0"],
                  ls="--", lw=1.3, zorder=4)
    ax.plot([], [], color=COL["a0"], ls="--", lw=1.3, label="human impostor")

    # the only number worth stating here: nothing synthetic passes T0
    ax.text(x[0], 0.7, "0%", ha="center", fontsize=9, color="#1A1A1A")

    top = max(p.values.max(), a0_tier.max())
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_xlim(-0.55, len(tiers) - 0.45)
    ax.set_ylabel("Session bypass rate")
    ax.set_yticks(np.arange(0, 25, 5))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_ylim(0, 22)
    ax.grid(axis="y", color="#D9D9D9", lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper left",
              bbox_to_anchor=(0.0, 1.03), handlelength=1.6, columnspacing=1.4)
    ax.set_title(
        "Bypass rises with defender complexity,\nexcept for the diffusion bot",
        fontsize=11, fontweight="bold", loc="left", pad=26)
    ax.set_xlabel("Defender tier, simplest to most complex  \u2192")
    save(fig, outdir, "fig5_bypass_by_tier")


# --------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=".", help="directory holding the canonical CSVs")
    p.add_argument("--output", default="./figures")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    log("ABAST figure generation")
    log(f"input : {args.input}")
    log(f"output: {args.output}")
    log(f"threshold policy: {POLICY}")
    log(f"panel 3 aggregation: {PANEL3_AGG}")

    figure_1(args.input, args.output)
    mapping = figure_2(args.input, args.output)
    figure_3(args.input, args.output)
    figure_4(args.input, args.output)
    figure_tiers(args.input, args.output)

    section("USER LABEL MAPPING (must appear as a poster footnote)")
    for k, v in mapping.items():
        log(f"  {k} = {v}")

    audit_path = os.path.join(args.output, "figure_numbers_audit.txt")
    with open(audit_path, "w") as f:
        f.write("\n".join(AUDIT) + "\n")
    print(f"\naudit written to {audit_path}")


if __name__ == "__main__":
    main()