"""Reuse the EXACT WindMouse generation in realism_step8.py to save the 60 sessions
as CSVs for Colab handoff. It executes realism_step8's own source VERBATIM up through
`sources["windmouse"] = wm` (definitions + the windmouse block), then writes CSVs.

Why exec-a-slice rather than import: realism_step8.py has no __main__ guard, so a
normal import runs its whole body — which loads SapiAgent pkls from attacker/artifacts
(they live under a2/, so it would crash) and rewrites realism_metrics.csv / re-appends
RESULTS.md. Slicing at the windmouse block reuses the identical emitter + settings with
no reconstruction, no edits to that file, and none of those side effects."""
import os, re, glob
import numpy as np
import pandas as pd

SRC = "/home/junio/reu-sapiagent/attacker/realism_step8.py"
OUT = "/home/junio/reu-sapiagent/a2/handoff/windmouse_sessions"
WM_COLS = ["record timestamp", "x", "y", "user", "session", "label"]

# ---- 1. reuse the exact windmouse generation ----------------------------
src = open(SRC).read()
head = src.split("# 3. sapiagent")[0]          # up to & incl. `sources["windmouse"] = wm`
assert 'sources["windmouse"] = wm' in head, "windmouse block not found in slice"
ns = {"__file__": SRC}                           # exec sets no __file__; realism_step8 needs it
exec(compile(head, SRC, "exec"), ns)

wm = ns["wm"]                                     # list[(id, df)] — the 60 windmouse sessions
print(f"reused {len(wm)} WindMouse sessions from {os.path.basename(SRC)} (verbatim)")
print(f"  settings: N_SEGMENTS={ns['N_SEGMENTS']}  SEEDS={ns['SEEDS']}  "
      f"N_SESSIONS={ns['N_SESSIONS']}  SCREEN={ns['SCREEN']}")
print(f"  dt sampler: make_empirical_dt_sampler over {len(ns['sessions_dt'])} Balabit "
      f"sessions, rng reset per seed  (same as SapiAgent)")

# ---- 2. save CSVs mirroring the SapiAgent handoff layout ----------------
n_saved = 0
for _id, df in wm:
    m = re.match(r"seed(\d+)_s(\d+)", _id)
    seed, sid = int(m.group(1)), int(m.group(2))
    assert list(df.columns) == WM_COLS, f"bad columns for {_id}"
    d = os.path.join(OUT, f"seed{seed}")
    os.makedirs(d, exist_ok=True)
    df.to_csv(os.path.join(d, f"session_{sid:02d}.csv"), index=False)
    n_saved += 1
print(f"saved {n_saved} CSVs -> {OUT}/seed{{0,1,2}}/session_NN.csv")

# ---- 3. validate all 60 (bare pd.read_csv, as Colab would) --------------
checks = {"columns": 0, "dtypes": 0, "label1": 0, "time_nondec": 0, "no_nan": 0}
per_seed = {}; n = 0; max_delta = 0
for f in sorted(glob.glob(os.path.join(OUT, "seed*", "session_*.csv"))):
    d = pd.read_csv(f); n += 1
    seed = f.split(os.sep + "seed")[1].split(os.sep)[0]
    per_seed[seed] = per_seed.get(seed, 0) + 1
    checks["columns"] += list(d.columns) == WM_COLS
    checks["dtypes"] += (np.issubdtype(d["record timestamp"].dtype, np.floating)
                         and np.issubdtype(d["x"].dtype, np.integer)
                         and np.issubdtype(d["y"].dtype, np.integer)
                         and np.issubdtype(d["label"].dtype, np.integer))
    checks["label1"] += bool((d["label"] == 1).all())
    checks["time_nondec"] += bool(np.all(np.diff(d["record timestamp"].to_numpy()) >= 0))
    checks["no_nan"] += (not bool(d.isna().to_numpy().any()))
    if len(d) > 1:
        max_delta = max(max_delta, int(max(np.abs(np.diff(d["x"].to_numpy())).max(),
                                            np.abs(np.diff(d["y"].to_numpy())).max())))
print(f"\nvalidation on {n} bare-reloaded CSVs:")
for k, v in checks.items():
    print(f"  {k:12s}: {v}/{n}  {'PASS' if v == n else '*** FAIL ***'}")
print(f"per-seed counts: {dict(sorted(per_seed.items()))}")
print(f"max per-step delta across all 60: {max_delta} px (continuity sanity)")
print("WINDMOUSE_CSVS_DONE")
