"""A3 (DMTG diffusion attacker) — corpus -> normalized diffusion tensors.

Trains on the IDENTICAL corpus A2 used: build_pooled_actions() from
attacker/balabit_actions.py. All design decisions here are fixed from
attacker/dmtg/corpus_stats.json (step9) and must NOT be re-derived:

  * Corpus       : 17,505 actions (N,128,2) float32 dx/dy deltas. Drop the 273
                   with ||D|| < 5 px (128 steps going nowhere) -> ~17,232.
  * Parameterize : predict the humanizing RESIDUAL on the equidistant line,
                   residual = action - build_equidistant_inputs(action). Verified
                   (step9) to sum to exactly the same D, so equidistant(D)+residual
                   honors requested net displacement D exactly -> target-honoring.
  * Normalize    : signed asinh on the residual,
                   y = sign(r)*arcsinh(|r|/s)/arcsinh(1919/s), s=1.0. Exactly
                   invertible, no clipping. (Linear /max would crush p50 to 0.001.)
  * Condition    : D = actions.sum(1) (N,2), normalized D/1920.0 (same divisor for
                   x and y -> isotropy), broadcast across all 128 positions as 2
                   extra channels. Model in=(B,4,128), out=(B,2,128) epsilon.
  * Split        : 10% held out by SOURCE SESSION (meta["provenance"]), never by
                   random action index (adjacent 128-blocks are correlated).

Does not modify balabit_actions.py. No model code here.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ATTACKER_DIR = os.path.dirname(HERE)
if ATTACKER_DIR not in sys.path:
    sys.path.insert(0, ATTACKER_DIR)
from balabit_actions import build_pooled_actions, build_equidistant_inputs  # noqa: E402

# ----- fixed normalization constants (do not re-derive) -----------------------
ASINH_S = 1.0                                  # asinh scale
ASINH_REF = 1919.0                             # max |dx| in corpus -> maps to ~1.0
ASINH_DENOM = float(np.arcsinh(ASINH_REF / ASINH_S))  # arcsinh(1919) ~= 8.253
D_DIVISOR = 1920.0                             # isotropic D normalizer (x AND y)
MIN_DISP = 5.0                                 # drop actions with ||D|| < this
WINDOW = 128


# ----- residual normalization (signed asinh, exactly invertible) --------------
def norm_residual(r):
    """Pixels -> normalized. y = sign(r)*arcsinh(|r|/s)/arcsinh(1919/s)."""
    r = np.asarray(r, dtype=np.float64)
    return np.sign(r) * np.arcsinh(np.abs(r) / ASINH_S) / ASINH_DENOM


def denorm_residual(y):
    """Normalized -> pixels. Inverse of norm_residual."""
    y = np.asarray(y, dtype=np.float64)
    return np.sign(y) * ASINH_S * np.sinh(np.abs(y) * ASINH_DENOM)


# ----- corpus loading ---------------------------------------------------------
def load_corpus(min_disp=MIN_DISP):
    """Build the A2-identical corpus, drop degenerate actions, form residuals.

    Returns a dict with (all aligned along axis 0, after the degenerate drop):
      actions      (N,128,2) float32   real human dx/dy deltas
      equidistant  (N,128,2) float32   straight-line input (repo logic)
      residual     (N,128,2) float32   actions - equidistant  (the diffusion target)
      D            (N,2)     float32    net displacement = actions.sum(1) (conditioning)
      keys         (N,)      str        "user/session" source key per action
    plus bookkeeping counts.
    """
    actions, meta = build_pooled_actions()                # (N0,128,2)
    D_full = actions.sum(axis=1)                           # (N0,2)
    dnorm = np.hypot(D_full[:, 0], D_full[:, 1])
    keep = dnorm >= min_disp
    prov = meta["provenance"]                              # list[(user, session)]
    keys_full = np.array([f"{u}/{s}" for (u, s) in prov])

    actions_k = actions[keep].astype(np.float32)
    keys_k = keys_full[keep]
    equi_k = build_equidistant_inputs(actions_k).astype(np.float32)   # (N,128,2)
    residual_k = (actions_k - equi_k).astype(np.float32)
    D_k = actions_k.sum(axis=1).astype(np.float32)

    return {
        "actions": actions_k,
        "equidistant": equi_k,
        "residual": residual_k,
        "D": D_k,
        "keys": keys_k,
        "n_total": int(actions.shape[0]),
        "n_dropped_degenerate": int((~keep).sum()),
        "n_kept": int(actions_k.shape[0]),
        "meta_n_users": meta["n_users"],
        "min_disp": min_disp,
    }


# ----- corpus -> diffusion tensors -------------------------------------------
def to_tensors(corpus):
    """Return (x0, cond) float32 numpy, channels-first for Conv1d.

      x0   (N,2,128)  normalized residual (the epsilon-diffusion sample)
      cond (N,2,128)  normalized D broadcast across all 128 positions
    """
    residual = corpus["residual"]                 # (N,128,2) pixels
    D = corpus["D"]                               # (N,2) pixels
    N, L, _ = residual.shape

    x0 = norm_residual(residual).transpose(0, 2, 1).astype(np.float32)   # (N,2,128)
    Dn = (D / D_DIVISOR).astype(np.float32)                              # (N,2)
    cond = np.broadcast_to(Dn[:, :, None], (N, 2, L)).astype(np.float32) # (N,2,128)
    return x0, cond


# ----- standardization (optional, on top of asinh) ---------------------------
def compute_std(x0, train_idx):
    """Global std of the asinh-normalized residual over the TRAIN subset only.

    A single scalar (isotropic across both channels and all 128 positions), to be
    stored in the checkpoint so sampling can invert the standardization exactly.
    Never computed on val (would leak).
    """
    return float(np.asarray(x0)[train_idx].std())


def standardize(x0, std):
    """Divide asinh-normalized x0 by the train std to reach ~unit variance.
    Inverse at sample time: residual_norm = model_x0 * std."""
    return (np.asarray(x0, dtype=np.float32) / np.float32(std)).astype(np.float32)


# ----- session-disjoint split -------------------------------------------------
def session_split(keys, val_frac=0.10, seed=0, spread=True, min_users=5):
    """Split action indices into train/val by SOURCE SESSION (never by action index).

    Whole sessions go to validation, so zero session overlap => no correlated-block
    leakage. Two selection modes:

      spread=True (default): round-robin one session per user across a shuffled user
        order, accumulating until ~val_frac of actions are held out. On a lopsided
        corpus this spans many distinct users so val measures GENERIC human-likeness,
        not one heavy user's style.
      spread=False: shuffle sessions irrespective of user (legacy).

    Guarantees >= min_users distinct users in val (raises if impossible).
    Returns (train_idx, val_idx, val_sessions).
    """
    keys = np.asarray(keys)
    rng = np.random.default_rng(seed)
    uniq = np.unique(keys)
    counts = {k: int((keys == k).sum()) for k in uniq}
    target = val_frac * keys.shape[0]

    if spread:
        by_user = {}
        for k in uniq:
            by_user.setdefault(k.split("/")[0], []).append(k)
        users = list(by_user.keys())
        rng.shuffle(users)
        for u in users:
            rng.shuffle(by_user[u])
        val_sessions, cum, r = [], 0, 0
        stop = False
        while not stop:
            progressed = False
            for u in users:
                if r < len(by_user[u]):
                    k = by_user[u][r]
                    val_sessions.append(k)
                    cum += counts[k]
                    progressed = True
                    # only stop once we both hit the target AND span min_users
                    if cum >= target and len({s.split('/')[0] for s in val_sessions}) >= min_users:
                        stop = True
                        break
            r += 1
            if not progressed:
                break
        val_sessions = set(val_sessions)
    else:
        rng.shuffle(uniq)
        val_sessions, cum = set(), 0
        for k in uniq:
            if cum >= target:
                break
            val_sessions.add(k)
            cum += counts[k]

    n_users = len({s.split("/")[0] for s in val_sessions})
    if n_users < min_users:
        raise ValueError(f"val spans only {n_users} users (< min_users={min_users})")

    val_mask = np.array([k in val_sessions for k in keys])
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]
    return train_idx, val_idx, sorted(val_sessions)
