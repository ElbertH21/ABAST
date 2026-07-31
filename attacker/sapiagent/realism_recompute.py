"""Recompute realism metrics using the CORRECTED defender feature extractor.

Companion to (never a replacement for) realism_step8.py. That script used the older
8-feature extractor (vx, vy, velocity, acceleration, jerk, angle, curvature, pause)
and recomputed step distance by hand. This one uses the corrected notebook's
`extract_features` + `collapse_duplicate_timestamps` VERBATIM, so the realism table
rests on the same preprocessing as the corrected (leakage-free) defender run:
deterministic tie ordering, duplicate-timestamp collapse, and the extractor's own
`dt` / `step_distance` columns rather than locally re-derived ones.

Loads existing sessions only. It never generates a trajectory and never writes the
July-16 artifacts (realism_metrics.csv / realism_curvature_filtered.csv); output goes
to the *_corrected.csv names.

The two verbatim functions are lifted directly out of the corrected notebook at run
time (--notebook), rather than transcribed here, so there is no copy drift. If the
notebook cannot be found the script fails loudly; it never silently falls back to the
old 8-feature extractor.

Usage:
    python attacker/sapiagent/realism_recompute.py --notebook path/to/corrected.ipynb
"""
import argparse
import ast
import glob
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from balabit_actions import list_training_sessions  # same 65 genuine sessions as step 8

# NOTE: realism_step8.py computes _REPO with two dirname() calls, which was correct
# when it lived at attacker/realism_step8.py. It now lives one level deeper, so the
# same expression resolves to attacker/ instead of the repo root. This script uses
# three, which is correct for its own location. See the report accompanying this file.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SRC_ART = os.path.join(_REPO, "a2", "artifacts")            # sapiagent session pickles
HANDOFF = os.path.join(_REPO, "a2", "handoff")              # saved per-attacker CSVs
RESULTS = os.path.join(_REPO, "results")

SEEDS = [0, 1, 2]
N_SESSIONS = 20
SOURCES4 = ["genuine", "windmouse", "sapiagent", "dmtg"]
T_COL = "record timestamp"

# Outputs (never the July-16 names)
OUT_METRICS = os.path.join(RESULTS, "realism_metrics_corrected.csv")
OUT_CURV = os.path.join(RESULTS, "realism_curvature_filtered_corrected.csv")
# Baselines for the before/after table
OLD_METRICS = os.path.join(RESULTS, "realism_metrics.csv")
OLD_CURV = os.path.join(RESULTS, "realism_curvature_filtered.csv")

PROTECTED = {os.path.abspath(OLD_METRICS), os.path.abspath(OLD_CURV)}
VERBATIM_FUNCS = ("extract_features", "collapse_duplicate_timestamps")


# ---- verbatim import of the corrected extractor --------------------------
def _strip_magics(src):
    return "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith(("%", "!", "?")))


def _notebook_code(nb_path):
    """Concatenated source of every code cell, IPython magics stripped.

    A plain .py path is read as-is, so the two functions can also be handed over in
    a file extracted from the notebook.
    """
    if nb_path.endswith(".py"):
        with open(nb_path, "r", encoding="utf-8") as fh:
            return _strip_magics(fh.read())
    with open(nb_path, "r", encoding="utf-8") as fh:
        nb = json.load(fh)
    chunks = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        # drop magics / shell escapes so the cell text is parseable Python
        chunks.append(_strip_magics(src))
    return "\n\n".join(chunks)


def load_verbatim(nb_path):
    """Lift the two functions (plus the ALL-CAPS constants they lean on) verbatim.

    Only function definitions and simple upper-case module constants are executed —
    never the whole notebook — so no training, Drive mount, or scoring runs here.
    """
    code = _notebook_code(nb_path)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise SystemExit(
            f"Could not parse code cells of {nb_path}: {exc}\n"
            "Extract the two functions into a .py file and pass that instead."
        )

    # (kind, source) in notebook order. Imports and constants are best-effort context;
    # only the two target functions are mandatory.
    segments, found = [], set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in VERBATIM_FUNCS:
            segments.append(("func", ast.get_source_segment(code, node)))
            found.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            segments.append(("import", ast.get_source_segment(code, node)))
        elif isinstance(node, ast.Assign) and all(
            isinstance(t, ast.Name) and t.id.isupper() for t in node.targets
        ):
            segments.append(("const", ast.get_source_segment(code, node)))

    missing = [f for f in VERBATIM_FUNCS if f not in found]
    if missing:
        raise SystemExit(
            f"{nb_path} does not define: {', '.join(missing)}.\n"
            "This is not the corrected notebook. Expected the run whose to_csv "
            "targets are *_corrected.csv and whose FEATURE_COLS include 'dt' and "
            "'step_distance'. Refusing to fall back to the old 8-feature extractor."
        )

    ns = {"np": np, "numpy": np, "pd": pd, "pandas": pd, "os": os, "Path": Path}
    origin = f"<verbatim:{os.path.basename(nb_path)}>"
    skipped = 0
    for kind, src in segments:
        try:
            exec(compile(src, origin, "exec"), ns)
        except Exception as exc:                      # noqa: BLE001
            if kind == "func":                        # the verbatim copy must land
                raise SystemExit(f"could not load {kind} from notebook: {exc}")
            skipped += 1                              # Drive paths, absent libs, etc.
    if skipped:
        print(f"  (skipped {skipped} notebook import/constant line(s) not needed here)")
    cols = ns.get("FEATURE_COLS")
    if cols is not None and not {"dt", "step_distance"} <= set(cols):
        print(f"  WARNING: FEATURE_COLS lacks dt/step_distance -> {cols}", file=sys.stderr)
    return ns[VERBATIM_FUNCS[0]], ns[VERBATIM_FUNCS[1]], cols


# ---- session loading (load only; nothing is generated) -------------------
def _read_csv_session(path):
    return pd.read_csv(path, usecols=[T_COL, "x", "y"])


def _seeded_csvs(kind):
    """a2/handoff/<kind>/seed{0,1,2}/session_{00..19}.csv -> 60 frames."""
    root = os.path.join(HANDOFF, kind)
    out = []
    for seed in SEEDS:
        for sid in range(N_SESSIONS):
            p = os.path.join(root, f"seed{seed}", f"session_{sid:02d}.csv")
            out.append((f"seed{seed}_s{sid}", _read_csv_session(p)))
    return out


def load_sources():
    """The same sessions realism_step8.py scored, read from disk.

    genuine / sapiagent / dmtg come from exactly the paths step 8 used. WindMouse is
    the one deviation: step 8 rebuilt those 60 sessions in-memory from a seeded RNG
    instead of reading files. Generating here is off the table, so we read the saved
    copies that save_windmouse_csvs.py wrote for this purpose.
    """
    sources = {}

    sources["genuine"] = [
        (f"{user}/{os.path.basename(p)}", _read_csv_session(p))
        for user, p in list_training_sessions()
    ]

    sources["windmouse"] = _seeded_csvs("windmouse_sessions")

    sa = []
    for seed in SEEDS:
        with open(os.path.join(SRC_ART, f"sapiagent_sessions_seed{seed}.pkl"), "rb") as fh:
            for sid, df in enumerate(pickle.load(fh)):
                sa.append((f"seed{seed}_s{sid}", df))
    sources["sapiagent"] = sa

    sources["dmtg"] = _seeded_csvs("dmtg_sessions")
    return sources


# ---- metrics -------------------------------------------------------------
def session_frame(df, extract_features, collapse_duplicate_timestamps, stable_sort=True):
    """One session -> the corrected extractor's feature frame."""
    d = df.copy()
    if stable_sort:
        # corrected fix #1: deterministic tie ordering before anything reads dt
        d = d.sort_values(T_COL, kind="stable").reset_index(drop=True)
    d = collapse_duplicate_timestamps(d)   # corrected fix #3
    return extract_features(d)


def compute(sources, extract_features, collapse_duplicate_timestamps, stable_sort=True):
    per_session_rows, curv_rows = [], []

    for src in SOURCES4:
        cabs_parts, dist_parts = [], []
        for sid, df in sources[src]:
            f = session_frame(df, extract_features, collapse_duplicate_timestamps, stable_sort)
            for col in ("dt", "velocity", "curvature", "step_distance"):
                if col not in f.columns:
                    raise SystemExit(
                        f"corrected extract_features returned no '{col}' column "
                        f"(got {list(f.columns)})."
                    )
            cabs = f["curvature"].abs().to_numpy()
            per_session_rows.append({
                "source": src,
                "id": sid,
                "pause_rate": float((f["dt"] > 0.1).mean()),
                "velocity_mean": float(f["velocity"].mean()),
                "curvature_mean": float(cabs.mean()),
            })
            cabs_parts.append(cabs)
            dist_parts.append(f["step_distance"].to_numpy())

        # pooled over every step of every session in this source
        cabs_all = np.concatenate(cabs_parts)
        dist_all = np.concatenate(dist_parts)
        keep = dist_all >= 2.0
        curv_rows.append({
            "source": src,
            "curvature_raw": float(cabs_all.mean()),
            "curvature_dist_ge_2px": float(cabs_all[keep].mean()) if keep.any() else np.nan,
            "n_steps_total": int(cabs_all.size),
            "n_steps_kept": int(keep.sum()),
        })

    per_session = pd.DataFrame(per_session_rows)
    agg_rows = []
    for src in SOURCES4:
        sub = per_session[per_session["source"] == src]
        row = {"source": src, "n_sessions": len(sub)}
        for m in ("velocity_mean", "pause_rate", "curvature_mean"):
            row[f"{m}_mean"] = sub[m].mean()
            row[f"{m}_std"] = sub[m].std()
        agg_rows.append(row)
    return pd.DataFrame(agg_rows), pd.DataFrame(curv_rows), per_session


# ---- before / after ------------------------------------------------------
def _pct(old, new):
    if old is None or new is None or (isinstance(old, float) and np.isnan(old)):
        return np.nan
    if old == 0:
        return np.nan if new == 0 else np.inf
    return (new - old) / abs(old) * 100.0


def before_after(agg, curv, threshold=10.0):
    """Print old-vs-new for every source/metric, flagging moves beyond ±threshold%."""
    if not (os.path.exists(OLD_METRICS) and os.path.exists(OLD_CURV)):
        print("  (baseline CSVs missing — skipping before/after)")
        return
    old_m = pd.read_csv(OLD_METRICS).set_index("source")
    old_c = pd.read_csv(OLD_CURV).set_index("source")
    new_m = agg.set_index("source")
    new_c = curv.set_index("source")

    fields = [
        (old_m, new_m, "velocity_mean_mean"), (old_m, new_m, "velocity_mean_std"),
        (old_m, new_m, "pause_rate_mean"), (old_m, new_m, "pause_rate_std"),
        (old_m, new_m, "curvature_mean_mean"), (old_m, new_m, "curvature_mean_std"),
        (old_c, new_c, "curvature_raw"), (old_c, new_c, "curvature_dist_ge_2px"),
        (old_c, new_c, "n_steps_total"), (old_c, new_c, "n_steps_kept"),
    ]

    print(f"\n=== before / after (old = July-16 files; flag = |Δ| > {threshold:g}%) ===")
    print(f"{'source':<10} {'metric':<24} {'old':>16} {'new':>16} {'Δ%':>10}  flag")
    print("-" * 88)
    flagged = 0
    for src in SOURCES4:
        for old_df, new_df, col in fields:
            if src not in old_df.index or col not in old_df.columns:
                continue
            if src not in new_df.index or col not in new_df.columns:
                continue
            old_v = float(old_df.loc[src, col])
            new_v = float(new_df.loc[src, col])
            d = _pct(old_v, new_v)
            hit = (not np.isnan(d)) and abs(d) > threshold
            flagged += bool(hit)
            dtxt = "n/a" if np.isnan(d) else f"{d:+.1f}%"
            print(f"{src:<10} {col:<24} {old_v:>16.6g} {new_v:>16.6g} {dtxt:>10}  "
                  f"{'<== CHANGED' if hit else ''}")
        print("-" * 88)
    print(f"{flagged} value(s) moved by more than {threshold:g}%.")


# ---- main ----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notebook", required=True,
                    help="corrected notebook (*_corrected.csv targets; FEATURE_COLS "
                         "with dt + step_distance), or a .py holding the two functions")
    ap.add_argument("--threshold", type=float, default=10.0,
                    help="flag before/after moves beyond this percent (default 10)")
    ap.add_argument("--no-sort", action="store_true",
                    help="skip the stable timestamp sort before collapsing duplicates")
    args = ap.parse_args()

    if not os.path.exists(args.notebook):
        raise SystemExit(f"notebook not found: {args.notebook}")

    print(f"Loading verbatim {' + '.join(VERBATIM_FUNCS)} from {args.notebook}")
    extract_features, collapse_dupes, cols = load_verbatim(args.notebook)
    if cols is not None:
        print(f"  FEATURE_COLS: {list(cols)}")

    print("Loading sessions (load-only; nothing generated)")
    sources = load_sources()
    print("  " + ", ".join(f"{k}={len(v)}" for k, v in sources.items()))

    agg, curv, per_session = compute(sources, extract_features, collapse_dupes,
                                     stable_sort=not args.no_sort)

    for path in (OUT_METRICS, OUT_CURV):
        if os.path.abspath(path) in PROTECTED:      # belt-and-braces
            raise SystemExit(f"refusing to overwrite protected baseline: {path}")
    agg.to_csv(OUT_METRICS, index=False)
    curv.to_csv(OUT_CURV, index=False)

    pd.set_option("display.width", 240, "display.max_columns", 30)
    print("\n=== realism_metrics_corrected.csv ===")
    print(agg.to_string(index=False))
    print("\n=== realism_curvature_filtered_corrected.csv (pooled steps) ===")
    print(curv.to_string(index=False))

    before_after(agg, curv, threshold=args.threshold)

    print(f"\nWrote {os.path.relpath(OUT_METRICS, _REPO)} and "
          f"{os.path.relpath(OUT_CURV, _REPO)}")
    print("July-16 realism_metrics.csv / realism_curvature_filtered.csv untouched.")
    print("RECOMPUTE_DONE")


if __name__ == "__main__":
    main()
