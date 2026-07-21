"""Read-only characterization of the pooled Balabit human-action corpus.

A3 (DMTG diffusion attacker) must train on the *identical* corpus that A2
(SapiAgent) trained on, so any A2-vs-A3 difference is attributable to the
generative mechanism, not corpus construction. This script imports the existing
corpus builders from attacker/balabit_actions.py (does NOT reimplement them) and
reports the statistics needed to (a) confirm the corpus, (b) choose a diffusion
normalization, and (c) understand the conditioning signal (net displacement D).

Nothing here modifies balabit_actions.py or writes model code. Output goes to
stdout and to attacker/dmtg/corpus_stats.json as a traceable artifact.
"""
import os
import sys
import json
import numpy as np

# Import the existing builders from attacker/ — do not reimplement.
HERE = os.path.dirname(os.path.abspath(__file__))
ATTACKER_DIR = os.path.dirname(HERE)
sys.path.insert(0, ATTACKER_DIR)
from balabit_actions import build_pooled_actions, build_equidistant_inputs  # noqa: E402

PCTLS = [50, 90, 99, 99.9]  # 'max' reported separately alongside these
NEAR_ZERO_PX = 5.0          # ||D|| below this = near-stationary action


def pctl_dict(arr, pctls=PCTLS):
    """Return {'p50':..., 'p90':..., ..., 'max':...} for a 1-D array."""
    arr = np.asarray(arr, dtype=np.float64)
    out = {}
    for p in pctls:
        key = f"p{p:g}".replace(".", "_")  # p99.9 -> p99_9
        out[key] = float(np.percentile(arr, p))
    out["max"] = float(arr.max()) if arr.size else float("nan")
    return out


def fmt_row(label, d, width=14):
    """One aligned table row from a pctl_dict."""
    cells = "".join(f"{d[k]:>{width}.3f}" for k in d)
    return f"{label:<16}{cells}"


def main():
    print("=" * 78)
    print("A3 corpus characterization — pooled Balabit human actions")
    print("=" * 78)
    print("Building corpus via balabit_actions.build_pooled_actions() ...")

    actions, meta = build_pooled_actions()   # (N, 128, 2) float32
    N, W, Dim = actions.shape
    assert Dim == 2, f"expected 2D actions, got last dim {Dim}"

    summary = {}

    # ---------------------------------------------------------------- corpus size
    n_raw = meta["n_actions_raw"]
    n_drop = meta["n_dropped_artifact"]
    drop_pct = (100.0 * n_drop / n_raw) if n_raw else 0.0
    size = {
        "n_actions": meta["n_actions"],
        "n_actions_raw": n_raw,
        "n_dropped_artifact": n_drop,
        "drop_pct": drop_pct,
        "n_sessions": meta["n_sessions"],
        "n_users": meta["n_users"],
        "window": meta["window"],
        "shape": [int(N), int(W), int(Dim)],
    }
    summary["corpus_size"] = size

    print("\n" + "-" * 78)
    print("CORPUS SIZE")
    print("-" * 78)
    print(f"  actions kept (n_actions) : {size['n_actions']:,}")
    print(f"  actions raw              : {size['n_actions_raw']:,}")
    print(f"  dropped (artifact/16bit) : {size['n_dropped_artifact']:,}  ({drop_pct:.3f}%)")
    print(f"  sessions                 : {size['n_sessions']}")
    print(f"  users                    : {size['n_users']}")
    print(f"  actions array shape      : {tuple(size['shape'])}")

    # -------------------------------------------------------- per-user distribution
    per_user = meta["per_user"]
    # sort by count descending
    pu_sorted = sorted(per_user.items(), key=lambda kv: kv[1], reverse=True)
    counts = np.array([c for _, c in pu_sorted], dtype=np.int64)
    pu_stats = {
        "per_user_sorted": [[u, int(c)] for u, c in pu_sorted],
        "min": int(counts.min()) if counts.size else 0,
        "max": int(counts.max()) if counts.size else 0,
        "median": float(np.median(counts)) if counts.size else 0.0,
        "mean": float(counts.mean()) if counts.size else 0.0,
    }
    summary["per_user"] = pu_stats

    print("\n" + "-" * 78)
    print("PER-USER DISTRIBUTION (actions per user, sorted desc)")
    print("-" * 78)
    for u, c in pu_sorted:
        bar = "#" * int(40 * c / counts.max()) if counts.max() else ""
        print(f"  {u:<10}{c:>7,}  {bar}")
    print(f"  {'-'*10}")
    print(f"  min={pu_stats['min']:,}  max={pu_stats['max']:,}  "
          f"median={pu_stats['median']:,.1f}  mean={pu_stats['mean']:,.1f}")
    if pu_stats["min"]:
        print(f"  lopsidedness (max/min ratio): {pu_stats['max']/pu_stats['min']:.1f}x")

    # ------------------------------------------------- delta magnitude stats (|dx|,|dy|)
    dx = actions[:, :, 0].reshape(-1)   # all dx across all actions/steps
    dy = actions[:, :, 1].reshape(-1)
    dx_stats = pctl_dict(np.abs(dx))
    dy_stats = pctl_dict(np.abs(dy))
    summary["delta_magnitude"] = {"abs_dx": dx_stats, "abs_dy": dy_stats,
                                  "n_deltas": int(dx.size)}

    print("\n" + "-" * 78)
    print("DELTA MAGNITUDE  (per-step |dx|, |dy| across all actions; pixels)")
    print("-" * 78)
    hdr = "".join(f"{k:>14}" for k in dx_stats)
    print(f"{'':<16}{hdr}")
    print(fmt_row("|dx|", dx_stats))
    print(fmt_row("|dy|", dy_stats))
    print(f"  (n_deltas = {dx.size:,} per axis)")
    print("  -> normalization: raw deltas span well beyond [-1,1]; scale by a")
    print("     high percentile (e.g. p99/p99.9) or clip+divide before diffusion.")

    # ------------------------------------------------ net displacement D (conditioning)
    D = actions.sum(axis=1)             # (N, 2)  net displacement per action
    Dx, Dy = D[:, 0], D[:, 1]
    Dnorm = np.sqrt(Dx**2 + Dy**2)
    Dx_stats = pctl_dict(np.abs(Dx))
    Dy_stats = pctl_dict(np.abs(Dy))
    Dn_stats = pctl_dict(Dnorm)
    n_near_zero = int((Dnorm < NEAR_ZERO_PX).sum())
    near_zero_pct = 100.0 * n_near_zero / N if N else 0.0
    summary["net_displacement"] = {
        "abs_Dx": Dx_stats, "abs_Dy": Dy_stats, "norm": Dn_stats,
        "near_zero_px_threshold": NEAR_ZERO_PX,
        "n_near_zero": n_near_zero, "near_zero_pct": near_zero_pct,
    }

    print("\n" + "-" * 78)
    print("NET DISPLACEMENT D = actions.sum(axis=1)  (conditioning signal; pixels)")
    print("-" * 78)
    print(f"{'':<16}{hdr}")
    print(fmt_row("|Dx|", Dx_stats))
    print(fmt_row("|Dy|", Dy_stats))
    print(fmt_row("||D||", Dn_stats))
    print(f"  near-stationary (||D|| < {NEAR_ZERO_PX:g} px): "
          f"{n_near_zero:,} / {N:,}  ({near_zero_pct:.3f}%)")

    # ------------------------------------------------------- equidistant sanity check
    eq = build_equidistant_inputs(actions)     # (N, 128, 2)
    D_eq = eq.sum(axis=1)
    diff = np.abs(D_eq - D)
    max_disc = float(diff.max()) if diff.size else 0.0
    exact = bool(max_disc == 0.0)
    n_mismatch = int((diff.reshape(N, -1).max(axis=1) > 0).sum())
    summary["equidistant_sanity"] = {
        "claim": "build_equidistant_inputs(actions).sum(1) == actions.sum(1) exactly",
        "holds_exactly": exact,
        "max_discrepancy": max_disc,
        "n_actions_mismatch": n_mismatch,
    }

    print("\n" + "-" * 78)
    print("EQUIDISTANT SANITY CHECK")
    print("-" * 78)
    print("  claim: sum(build_equidistant_inputs(actions)) == sum(actions)  (D is integer)")
    print(f"  holds exactly            : {exact}")
    print(f"  max discrepancy (px)     : {max_disc:g}")
    print(f"  actions with any mismatch: {n_mismatch:,} / {N:,}")

    # ---------------------------------------------------------------- save artifact
    out_path = os.path.join(HERE, "corpus_stats.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + "=" * 78)
    print(f"Saved numeric summary -> {out_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
