"""
step9_dmtg_data_scale_check.py
 
Decide whether pooled Balabit has enough human movement data to train the A3
diffusion generator (DMTG-style alpha-DDIM), before writing any model code.
 
Trains nothing. Reads the 65 Balabit training sessions ONCE, counts the units a
diffusion model would actually consume, and prints a verdict.
 
Two candidate training units:
  1. Fixed 128-step blocks   - matches the defender's window granularity and
                               SapiMouse's own segmentation (blocks of 128).
  2. Point-to-point segments - matches what DMTG actually models: a single
                               movement from a start point to a target point.
                               This is the unit A3 must emit, so it governs the
                               go/no-go decision.
 
Usage:
    python step9_dmtg_data_scale_check.py --data-root <path to Balabit checkout>
 
Needs only numpy + pandas. Safe to run in the dmtg venv or any env with those.
 
Reference points for reading the output:
  - SapiAgent (A2) trained successfully on SapiMouse: 120 subjects x ~4 min
    each, roughly 8 hours of mouse data total.
  - Balabit training: 65 sessions, each 40 min to 7 hours -> far more wall-clock
    data than SapiMouse, but only 10 users (less behavioral diversity).
  - DMTG's 1M figure pooled SapiMouse WITH Open Images V7 traces; it is not a
    floor for the method.
"""
 
from __future__ import annotations
 
import argparse
import glob
import os
import sys
 
import numpy as np
import pandas as pd
 
# A segment shorter than this has too few points to denoise into a trajectory.
MIN_SEGMENT_POINTS = 8
 
# Idle gap (seconds) marking the boundary between two distinct movements.
# 0.5s is the default; SWEEP_BREAKS tests whether the verdict depends on it.
SEGMENT_BREAK_SECONDS = 0.5
SWEEP_BREAKS = (0.2, 0.3, 0.5, 1.0, 2.0)
 
# Defender window size, reused as the fixed-block unit.
WINDOW = 128
 
# Rough floor for training a small UNet1DModel on 2-channel resampled paths.
SEGMENT_TARGET = 20_000
 
 
def find_training_sessions(data_root: str) -> list[str]:
    """Locate Balabit training session files (note: they have no file extension)."""
    patterns = [
        os.path.join(data_root, "training_files", "user*", "session_*"),
        os.path.join(data_root, "Mouse-Dynamics-Challenge", "training_files", "user*", "session_*"),
        os.path.join(data_root, "**", "training_files", "user*", "session_*"),
    ]
    for pattern in patterns:
        paths = sorted(glob.glob(pattern, recursive=True))
        paths = [p for p in paths if os.path.isfile(p)]
        if paths:
            return paths
    return []
 
 
def load_timestamps(path: str) -> tuple[np.ndarray, int] | None:
    """
    Read one session and return (sorted timestamps, row count).
 
    Only timestamps are retained: the segment analysis needs nothing else, and
    holding full frames for 65 long sessions is pointless memory pressure.
    Mirrors the defender notebook's load_session (sort by time, drop null x/y).
    """
    try:
        df = pd.read_csv(path)
        df = df.sort_values("record timestamp")
        df = df.dropna(subset=["x", "y"])
        if not len(df):
            return None
        return df["record timestamp"].to_numpy(dtype=float), len(df)
    except Exception as exc:  # noqa: BLE001 - report and skip, don't abort the scan
        print(f"  skipped {os.path.basename(path)}: {exc}", file=sys.stderr)
        return None
 
 
def segment_lengths(ts: np.ndarray, break_seconds: float) -> np.ndarray:
    """Split a timestamp array into movement segments on idle gaps."""
    if ts.size < 2:
        return np.array([], dtype=int)
    breaks = np.flatnonzero(np.diff(ts) > break_seconds) + 1
    bounds = np.concatenate([[0], breaks, [ts.size]])
    return np.diff(bounds)
 
 
def count_usable(cache: list[np.ndarray], break_seconds: float) -> int:
    """Total usable segments across all cached sessions at a given threshold."""
    total = 0
    for ts in cache:
        lens = segment_lengths(ts, break_seconds)
        total += int((lens >= MIN_SEGMENT_POINTS).sum())
    return total
 
 
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="data/Mouse-Dynamics-Challenge",
        help="Path to the Balabit checkout",
    )
    args = parser.parse_args()
 
    paths = find_training_sessions(args.data_root)
    if not paths:
        print(f"No training sessions found under {args.data_root!r}.", file=sys.stderr)
        print("Point --data-root at your Balabit checkout.", file=sys.stderr)
        return 1
 
    print(f"Found {len(paths)} session files under {args.data_root}")
    print("Loading (one pass, timestamps cached for the sweep)...\n")
 
    ts_cache: list[np.ndarray] = []
    per_user_sessions: dict[str, int] = {}
    total_events = 0
    total_blocks = 0
    total_seconds = 0.0
 
    for path in paths:
        result = load_timestamps(path)
        if result is None:
            continue
        ts, n_rows = result
        ts_cache.append(ts)
 
        user = os.path.basename(os.path.dirname(path))
        per_user_sessions[user] = per_user_sessions.get(user, 0) + 1
 
        total_events += n_rows
        total_blocks += n_rows // WINDOW
        if ts.size >= 2:
            total_seconds += float(ts[-1] - ts[0])
 
    if not ts_cache:
        print("No sessions loaded successfully.", file=sys.stderr)
        return 1
 
    all_lens = [segment_lengths(ts, SEGMENT_BREAK_SECONDS) for ts in ts_cache]
    segments = np.concatenate(all_lens) if all_lens else np.array([], dtype=int)
    usable = segments[segments >= MIN_SEGMENT_POINTS]
 
    print("=" * 62)
    print("POOLED BALABIT TRAINING CORPUS")
    print("=" * 62)
    print(f"  sessions loaded      : {len(ts_cache)}")
    print(f"  users                : {len(per_user_sessions)}")
    print(f"  wall-clock duration  : {total_seconds / 3600:.1f} hours")
    print(f"  total mouse events   : {total_events:,}")
    print()
    print(f"  fixed {WINDOW}-step blocks : {total_blocks:,}")
    print()
    print(f"  movement segments (gap > {SEGMENT_BREAK_SECONDS}s)")
    print(f"    total              : {segments.size:,}")
    print(f"    usable (>= {MIN_SEGMENT_POINTS} pts)  : {usable.size:,}")
    if usable.size:
        print(f"    length median      : {np.median(usable):.0f} pts")
        print(f"    length p25 / p75   : {np.percentile(usable, 25):.0f} / {np.percentile(usable, 75):.0f}")
        print(f"    length max         : {usable.max():,} pts")
    print()
 
    print("  sessions per user:")
    for user in sorted(per_user_sessions):
        print(f"    {user:10s} {per_user_sessions[user]}")
    print()
 
    # If the verdict flips across this sweep, the gap threshold is doing the
    # work and needs justifying in the paper rather than being assumed.
    print("  sensitivity to the segment-break threshold:")
    for break_s in SWEEP_BREAKS:
        n_usable = count_usable(ts_cache, break_s)
        marker = "  <- default" if break_s == SEGMENT_BREAK_SECONDS else ""
        print(f"    gap > {break_s:>4.1f}s -> {n_usable:>8,} usable segments{marker}")
    print()
 
    print("=" * 62)
    print("VERDICT")
    print("=" * 62)
    n = usable.size
    if n >= SEGMENT_TARGET:
        print(f"  {n:,} usable segments >= {SEGMENT_TARGET:,} target.")
        print("  Balabit is sufficient. Train A3 on pooled Balabit, matching A2's")
        print("  corpus so the A2-vs-A3 comparison stays geometry-only.")
    elif n >= SEGMENT_TARGET // 4:
        print(f"  {n:,} usable segments is workable but thin.")
        print("  Train on pooled Balabit with a small UNet and augmentation")
        print("  (rotation, reflection, time-warp). Reassess if it underfits.")
    else:
        print(f"  {n:,} usable segments is below the practical floor.")
        print("  Consider supplementing with SapiMouse (what DMTG used, same lab")
        print("  as SapiAgent). Note this confounds architecture with training")
        print("  corpus in the A2-vs-A3 comparison and must be stated in the paper.")
    print()
    print("  Caveat independent of the count above: Balabit has only")
    print(f"  {len(per_user_sessions)} users, so behavioral diversity is thin regardless of volume.")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())