"""
step10_displacement_fidelity.py

Test whether SapiAgent (A2) under-produces displacement relative to what the
emitter asked for, which would mechanically explain its velocity collapse
(realism: genuine 1845 px/s, WindMouse 369, SapiAgent 221).

THE HYPOTHESIS
--------------
The pooled Balabit action corpus that A2 trained on has median ||D|| = 292 px
(p90 = 750). But the emitter samples both endpoints uniformly on 1920x1080,
which gives a median requested ||D|| of ~754 px: half of every request lands
above the training p90, and 12% above the training p99.

The fcn autoencoder is not constrained to preserve net displacement. It maps
equidistant(D) -> humanized(D'), and nothing forces D' == D. If it under-produces
D when extrapolating past its training range, each segment travels short. Timing
comes from a shared empirical dt sampler, so less distance over the same sampled
time is exactly a velocity collapse.

WHY THIS IS A FAIR TEST
-----------------------
WindMouse and SapiAgent sessions are emitted by the same emitter with the same
seeds, so session i of seed s has IDENTICAL requested start/target points for
both attackers. WindMouse is algorithmic and walks to the target by construction,
so it acts as the "requested displacement was honored" reference. Any pairwise
path-length deficit is attributable to the generator, not to the endpoint sampler.

This reads only the emitted session CSVs. It trains nothing and needs no emitter
internals, so it cannot drift from what was actually scored.

READING THE RESULT
------------------
  ratio ~= 1.0  -> no collapse; SapiAgent's slowness has another cause, and this
                   hypothesis is ruled out (still worth reporting).
  ratio << 1.0  -> SapiAgent systematically under-travels. If the deficit also
                   worsens as requested distance grows, that is the extrapolation
                   signature and the mechanism is confirmed.

Usage:
    python step10_displacement_fidelity.py \
        --windmouse-root <path>/windmouse_sessions \
        --sapiagent-root <path>/sapiagent_sessions
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

SEEDS = (0, 1, 2)
T_COL = "record timestamp"

# The emitter's own constants, restated here only to compute the expected
# request. Uniform endpoints on this screen give E[||D||] ~= 798 px per segment.
SCREEN_W, SCREEN_H = 1920, 1080
N_SEGMENTS = 12


def load_sessions(root: str, seeds=SEEDS) -> dict[int, list[tuple[str, pd.DataFrame]]]:
    """Mirror the notebook's load_attack_sessions_by_seed: {root}/seed{s}/*.csv."""
    out: dict[int, list[tuple[str, pd.DataFrame]]] = {}
    for s in seeds:
        paths = sorted(glob.glob(os.path.join(root, f"seed{s}", "*.csv")))
        sessions = []
        for p in paths:
            try:
                df = pd.read_csv(p)
                if T_COL in df.columns:
                    df = df.sort_values(T_COL, kind="stable")
                sessions.append((os.path.basename(p), df))
            except Exception as exc:  # noqa: BLE001
                print(f"  skipped {p}: {exc}", file=sys.stderr)
        out[s] = sessions
    return out


def session_geometry(df: pd.DataFrame) -> dict[str, float]:
    """Path length, net displacement, and straightness for one emitted session."""
    xy = df[["x", "y"]].to_numpy(dtype=float)
    if xy.shape[0] < 2:
        return {"n_events": xy.shape[0], "path_len": 0.0, "net_disp": 0.0,
                "straightness": 0.0, "duration": 0.0, "velocity": 0.0}

    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    path_len = float(steps.sum())
    net_disp = float(np.linalg.norm(xy[-1] - xy[0]))

    duration = 0.0
    if T_COL in df.columns:
        ts = df[T_COL].to_numpy(dtype=float)
        duration = float(ts[-1] - ts[0])

    return {
        "n_events": int(xy.shape[0]),
        "path_len": path_len,
        "net_disp": net_disp,
        "straightness": net_disp / path_len if path_len > 0 else 0.0,
        "duration": duration,
        "velocity": path_len / duration if duration > 0 else 0.0,
    }


def summarize(rows: list[dict], label: str) -> dict:
    """Per-attacker aggregate over sessions."""
    arr = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0] if k != "name"}
    print(f"\n  {label}  (n = {len(rows)} sessions)")
    print(f"    events/session   : {arr['n_events'].mean():8.0f}  +/- {arr['n_events'].std():.0f}")
    print(f"    path length (px) : {arr['path_len'].mean():8.0f}  +/- {arr['path_len'].std():.0f}")
    print(f"    net displacement : {arr['net_disp'].mean():8.0f}  +/- {arr['net_disp'].std():.0f}")
    print(f"    straightness     : {arr['straightness'].mean():8.3f}")
    print(f"    duration (s)     : {arr['duration'].mean():8.1f}")
    print(f"    velocity (px/s)  : {arr['velocity'].mean():8.0f}  +/- {arr['velocity'].std():.0f}")
    return {k: float(v.mean()) for k, v in arr.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windmouse-root", required=True)
    ap.add_argument("--sapiagent-root", required=True)
    ap.add_argument("--out", default="attacker/dmtg/displacement_fidelity.json")
    args = ap.parse_args()

    wm = load_sessions(args.windmouse_root)
    sa = load_sessions(args.sapiagent_root)

    wm_rows, sa_rows = [], []
    for s in SEEDS:
        for name, df in wm[s]:
            wm_rows.append({"name": f"seed{s}/{name}", **session_geometry(df)})
        for name, df in sa[s]:
            sa_rows.append({"name": f"seed{s}/{name}", **session_geometry(df)})

    if not wm_rows or not sa_rows:
        print("No sessions loaded. Check --windmouse-root / --sapiagent-root.", file=sys.stderr)
        return 1

    expected_path = N_SEGMENTS * 798.0  # 12 segments x E[||D||] under uniform endpoints

    print("=" * 66)
    print("PER-SESSION GEOMETRY")
    print("=" * 66)
    print(f"  Expected path length if every requested segment is fully")
    print(f"  traveled: {N_SEGMENTS} segments x ~798 px = ~{expected_path:.0f} px")
    wm_agg = summarize(wm_rows, "WindMouse (A1, algorithmic reference)")
    sa_agg = summarize(sa_rows, "SapiAgent (A2, learned)")

    print("\n" + "=" * 66)
    print("TRAVEL FIDELITY vs THE ALGORITHMIC REFERENCE")
    print("=" * 66)
    for label, agg in (("WindMouse", wm_agg), ("SapiAgent", sa_agg)):
        frac = agg["path_len"] / expected_path
        print(f"  {label:10s} path / expected request : {frac:6.3f}")
    ratio = sa_agg["path_len"] / wm_agg["path_len"] if wm_agg["path_len"] else float("nan")
    print(f"\n  SapiAgent / WindMouse path length  : {ratio:6.3f}")
    print(f"  SapiAgent / WindMouse velocity     : "
          f"{sa_agg['velocity'] / wm_agg['velocity'] if wm_agg['velocity'] else float('nan'):6.3f}")

    # Paired comparison. Both attackers were emitted with the same seeds and the
    # same endpoint sampler, so index i within a seed shares requested targets.
    print("\n" + "=" * 66)
    print("PAIRED BY (seed, session index) - same requested targets")
    print("=" * 66)
    pair_ratios = []
    for s in SEEDS:
        n = min(len(wm[s]), len(sa[s]))
        if len(wm[s]) != len(sa[s]):
            print(f"  seed {s}: count mismatch (wm {len(wm[s])}, sa {len(sa[s])}), "
                  f"pairing first {n}")
        for i in range(n):
            w = session_geometry(wm[s][i][1])["path_len"]
            a = session_geometry(sa[s][i][1])["path_len"]
            if w > 0:
                pair_ratios.append(a / w)
    if pair_ratios:
        pr = np.array(pair_ratios)
        print(f"  paired sessions      : {pr.size}")
        print(f"  path ratio (SA / WM) : mean {pr.mean():.3f}  median {np.median(pr):.3f}  "
              f"std {pr.std():.3f}")
        print(f"  sessions where SA travels less : {(pr < 1).sum()} / {pr.size}")

    print("\n" + "=" * 66)
    print("VERDICT")
    print("=" * 66)
    if ratio < 0.85:
        print(f"  SapiAgent travels {(1 - ratio) * 100:.0f}% less than WindMouse on identical")
        print("  requested targets. Displacement collapse is REAL and mechanically")
        print("  explains the velocity gap: same sampled timing, less distance.")
        print("  Report this as the cause of A2's slowness rather than leaving it")
        print("  as an unexplained property.")
    elif ratio > 1.15:
        print("  SapiAgent travels FURTHER than WindMouse. Its low velocity is then")
        print("  a timing/event-count effect, not a displacement effect. Check the")
        print("  events/session and duration rows above.")
    else:
        print(f"  Path lengths are comparable (ratio {ratio:.3f}). Displacement")
        print("  collapse is NOT the mechanism; A2's velocity gap comes from")
        print("  event count or timing instead. Hypothesis ruled out - still worth")
        print("  a sentence, since it removes a confound for A3.")
    print()
    print("  Note: this measures whole sessions. If the emitter snaps the cursor")
    print("  to each target between segments, a per-segment deficit could be")
    print("  masked at session level. If the ratio comes back ~1.0, check whether")
    print("  the emitter re-anchors cur = dest after each segment before")
    print("  concluding no collapse.")

    out = {
        "windmouse": wm_agg,
        "sapiagent": sa_agg,
        "expected_path_len": expected_path,
        "path_ratio_sa_over_wm": ratio,
        "paired_ratio_mean": float(np.mean(pair_ratios)) if pair_ratios else None,
        "paired_ratio_median": float(np.median(pair_ratios)) if pair_ratios else None,
        "n_paired": len(pair_ratios),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
