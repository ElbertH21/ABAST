"""SapiAgent (A2) session emitter — a drop-in twin of the WindMouse emitter.

Timing synthesis (make_empirical_dt_sampler), the session-assembly loop, the
inter-segment pause logic, the cumulative-time computation, and the Balabit-shaped
DataFrame construction are REUSED VERBATIM from the WindMouse emitter. The ONLY
difference is the per-segment path generator: instead of wind_mouse(cur, dest),
we pop the next SapiAgent action (128 dx/dy deltas), ROTATE it (no scaling) so its
net displacement points from cur toward a random screen dest, cumsum it from cur,
and advance cur to the action's real endpoint (continuous — never teleport to dest).
"""
import numpy as np
import pandas as pd


def make_empirical_dt_sampler(sessions, t_col="record timestamp", max_dt=1.0, rng=None):
    """VERBATIM from the WindMouse emitter: per-step dt drawn from real Balabit
    inter-event gaps (drops pauses > max_dt)."""
    if rng is None:
        rng = np.random.default_rng()
    pool = []
    for df in sessions:
        t = np.sort(df[t_col].to_numpy(dtype=float))
        dt = np.diff(t)
        pool.append(dt[(dt > 0) & (dt <= max_dt)])
    pool = np.concatenate(pool) if pool else np.array([0.010])
    if pool.size == 0:
        pool = np.array([0.010])
    def sample(n):
        return rng.choice(pool, size=n, replace=True)
    return sample


def make_action_popper(actions, start=0):
    """Sequential popper over a (M, 128, 2) action array; cycles when exhausted."""
    actions = np.asarray(actions, dtype=float)
    M = actions.shape[0]
    state = {"i": int(start)}
    def pop():
        a = actions[state["i"] % M]
        state["i"] += 1
        return a
    return pop


def _rotate_toward(action, cur, dest, min_disp=5.0):
    """Rotate a (128, 2) action so its net displacement (cumsum[-1] == sum) points
    from cur toward dest. Rotation ONLY — magnitude is preserved, no scaling. If the
    net displacement is < min_disp px its direction is ill-defined, so return the
    raw action unrotated."""
    net = action.sum(axis=0)                       # == cumsum(action, axis=0)[-1]
    if np.hypot(net[0], net[1]) < min_disp:
        return action
    ang = (np.arctan2(dest[1] - cur[1], dest[0] - cur[0])
           - np.arctan2(net[1], net[0]))
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s], [s, c]])
    return action @ R.T


def generate_sapiagent_session(session_id, target_user, rng, action_popper,
                               n_segments=12, screen=(1920, 1080),
                               dt_sampler=None, fixed_dt=0.010, jitter=0.25,
                               pause_range=(0.05, 0.6), min_disp=5.0, max_tries=20):
    """One SapiAgent attack session as a Balabit-shaped DataFrame. Drop-in twin of
    the WindMouse generate_attack_session: everything except the per-segment path
    generator is byte-for-byte the same assembly. The per-segment geometry keeps the
    segment ENDPOINT in-bounds (the WindMouse-parallel to cur=dest) by re-sampling
    dest until cur + rotated_net lands on-screen; cur is never clamped (no teleport)."""
    W, H = screen
    cur = np.array([rng.integers(0, W), rng.integers(0, H)], dtype=float)
    xs, ys, ts = [], [], []
    t = 0.0
    for _ in range(n_segments):
        # --- per-segment geometry (the ONLY swap vs WindMouse) --------------
        action = action_popper()                         # (128, 2) dx/dy deltas
        # Keep the endpoint in-bounds: re-sample dest / re-rotate (rotation only,
        # magnitude preserved) until cur + rotated_net is on-screen; else raw action.
        seg = action                                     # fallback if none land in-bounds
        for _try in range(max_tries):
            dest = np.array([rng.integers(0, W), rng.integers(0, H)], dtype=float)
            cand = _rotate_toward(action, cur, dest, min_disp)  # rotate-only toward dest
            endpoint = cur + cand.sum(axis=0)            # == cur + rotated net displacement
            if 0.0 <= endpoint[0] < W and 0.0 <= endpoint[1] < H:
                seg = cand
                break
        pts = cur + np.cumsum(seg, axis=0)               # (128, 2) segment path
        # -------------------------------------------------------------------
        n = len(pts)
        if dt_sampler is not None:
            dts = np.asarray(dt_sampler(max(n - 1, 1)), dtype=float)[: n - 1]
        else:
            dts = fixed_dt * (1.0 + jitter * (2 * rng.random(n - 1) - 1))
        dts = np.clip(dts, 1e-4, None)
        seg_t = t + np.concatenate([[0.0], np.cumsum(dts)]) if n > 1 else np.array([t])
        xs.extend(np.round(pts[:, 0]).astype(int))
        ys.extend(np.round(pts[:, 1]).astype(int))
        ts.extend(seg_t)
        t = seg_t[-1] + rng.uniform(*pause_range)        # inter-segment pause
        cur = pts[-1]                                    # continuous advance to the in-bounds endpoint (never clamped)
    return pd.DataFrame({
        "record timestamp": np.asarray(ts, dtype=float),
        "x": np.asarray(xs, dtype=int),
        "y": np.asarray(ys, dtype=int),
        "user": target_user,   # label only; target-blind
        "session": session_id,
        "label": 1,            # claims genuine; not used in scoring
    })
