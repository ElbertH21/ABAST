"""Balabit action pipeline for the A2 (SapiAgent) attacker.

Turns Balabit free-movement sessions into pooled, target-blind human "actions":
fixed-length 128-step (dx, dy) integer pixel-delta trajectories, shape (N, 128, 2)
with point i = (dx_i, dy_i) — the interleaved representation the fcn autoencoder
consumes (X.reshape(-1, 128, 2)).

Delta convention matches the repo's create_sapimouse_actions.action2rawfeatures:
first-order difference of consecutive positions, drop the leading NaN. Applied to
the whole session position stream, then windowed into non-overlapping 128-delta
chunks (remainder dropped). Pooled across ALL users => target-blind, matching the
WindMouse generator.
"""
import os
import glob
import numpy as np
import pandas as pd

BALABIT_ROOT = os.environ.get("BALABIT_ROOT", "./data/Mouse-Dynamics-Challenge")
WINDOW = 128  # deltas per action (settings.FEATURES)
T_COL = "record timestamp"


def list_training_sessions(root=BALABIT_ROOT):
    """Return [(user, session_path), ...] for every known-genuine training session."""
    out = []
    for user_dir in sorted(glob.glob(os.path.join(root, "training_files", "user*"))):
        user = os.path.basename(user_dir)
        for sess in sorted(glob.glob(os.path.join(user_dir, "session_*"))):
            out.append((user, sess))
    return out


def session_to_deltas(path):
    """Load one session, return its (M-1, 2) stream of consecutive (dx, dy) deltas.

    Reads only record-timestamp / x / y (the fields the defender uses), sorts by
    time, and diffs consecutive positions. No state/button filtering: the corpus
    is the raw free-movement position stream, matching the brief's recipe and
    WindMouse's domain-agnostic movement.
    """
    df = pd.read_csv(path, usecols=[T_COL, "x", "y"])
    df = df.sort_values(T_COL, kind="stable")
    xy = df[["x", "y"]].to_numpy(dtype=np.int64)
    if xy.shape[0] < 2:
        return np.empty((0, 2), dtype=np.float32)
    deltas = np.diff(xy, axis=0).astype(np.float32)  # (M-1, 2); == diff().dropna()
    return deltas


def window_actions(deltas, window=WINDOW):
    """Chop a (K, 2) delta stream into non-overlapping (window, 2) actions."""
    n = (deltas.shape[0] // window) * window
    if n == 0:
        return np.empty((0, window, 2), dtype=np.float32)
    return deltas[:n].reshape(-1, window, 2)


# Balabit records an unknown/off-screen cursor as the 16-bit sentinel 65535
# (0xFFFF), which yields ~65,500 px pseudo-deltas. Real single-step moves are
# bounded by the screen (<=~1920 px). max_abs_delta=2000 cleanly separates the
# two populations (nothing legitimate lands between ~1919 and ~65484 px), so any
# 128-window containing such a spike is a corrupted action and is dropped.
MAX_ABS_DELTA = 2000.0


def build_equidistant_inputs(targets, length=WINDOW):
    """Equidistant straight-line INPUT for each target action (repo logic).

    For each (length, 2) target, the input is the equidistant straight line
    spanning that action's OWN net displacement D = sum(target deltas). Replicates
    create_equidistant_actions: points p_i = i * D / length for i=0..length,
    integer-truncated (toward zero, like Python int()), then diffed -> `length`
    integer deltas. Anchored at the origin (representation is translation-invariant
    / target-blind). Because D is integer, sum(input) == sum(target) exactly, so
    input and target share start (anchor) and end (net displacement) and differ
    only in the between-shape (straight vs curved). Output interleaved (N, length, 2).
    """
    targets = np.asarray(targets, dtype=np.float32)
    D = targets.sum(axis=1)                       # (N, 2) net displacement (integer-valued)
    i = np.arange(length + 1)                      # (length+1,)
    pts = i[None, :, None] * (D[:, None, :] / length)   # (N, length+1, 2) float
    ipts = np.trunc(pts).astype(np.int64)          # int() truncation toward zero
    deltas = np.diff(ipts, axis=1).astype(np.float32)   # (N, length, 2)
    return deltas


def build_pooled_actions(root=BALABIT_ROOT, window=WINDOW, max_abs_delta=MAX_ABS_DELTA):
    """Build the pooled human-action corpus across all training sessions.

    Drops actions containing a 16-bit-sentinel pseudo-delta (|dx| or |dy| >
    max_abs_delta). Returns (actions, meta) where actions is (N, window, 2)
    float32 and meta is per-source bookkeeping (counts by user/session, drops).
    """
    sessions = list_training_sessions(root)
    all_actions = []
    per_user = {}
    per_session = {}
    provenance = []  # (user, session_basename) per action, for auditing only
    n_raw = 0
    n_dropped = 0
    for user, path in sessions:
        deltas = session_to_deltas(path)
        acts = window_actions(deltas, window)
        n_raw += acts.shape[0]
        if acts.shape[0] and max_abs_delta is not None:
            keep = np.abs(acts).reshape(acts.shape[0], -1).max(axis=1) <= max_abs_delta
            n_dropped += int((~keep).sum())
            acts = acts[keep]
        k = acts.shape[0]
        if k:
            all_actions.append(acts)
            provenance.extend([(user, os.path.basename(path))] * k)
        per_user[user] = per_user.get(user, 0) + k
        per_session[f"{user}/{os.path.basename(path)}"] = k
    actions = (np.concatenate(all_actions, axis=0)
               if all_actions else np.empty((0, window, 2), dtype=np.float32))
    meta = {
        "n_actions": int(actions.shape[0]),
        "n_actions_raw": int(n_raw),
        "n_dropped_artifact": int(n_dropped),
        "max_abs_delta": max_abs_delta,
        "n_sessions": len(sessions),
        "n_users": len(per_user),
        "per_user": per_user,
        "per_session": per_session,
        "provenance": provenance,
        "window": window,
    }
    return actions, meta
