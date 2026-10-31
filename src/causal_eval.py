"""
causal_eval.py  (revision v2, 2026-10)  --  rigorous shared evaluation core

Fixes the four audit findings so every re-run is apples-to-apples:

F0a  STRICTLY CAUSAL STATE.
     The cached obs_v / yaw_rate use np.gradient = CENTRAL difference, which
     reads x[i0+1] (one future sample). Here we recompute at each anchor using
     ONE-SIDED BACKWARD differences only (indices <= i0), and default the speed
     source to CAN speed (a real on-board sensor), never pose-differenced.

F0b  UNIFIED ANCHORS.
     anchors_for(df) returns ONE anchor list per segment (fixed stride, fixed
     history requirement). Every method -- kinematic, physics, MLP, residual --
     is scored on exactly these anchors, so "A beats B" is point-to-point.

F0c  PER-SUBSET CALIBRATION.
     gain_curv(items) fits the Model-C gain a from a given set of segments only;
     callers pass the SAME training subset used for the learner, so physics gets
     no extra calibration data (fixes the scarcity a_pool leak).

F0d  PAIRED GROUP-LEVEL STATS.
     paired_route_diff() aggregates per-anchor errors to per-route means for two
     methods on identical anchors, then bootstraps the paired route-level
     difference -> CI + sign-consistency, the correct unit of analysis.
"""
import re
import numpy as np
from scipy import stats

HIST = 10          # history samples required before an anchor (causal features)
MIN_HIST_S = 0.0   # extra guard


def route_of(seg):
    m = re.search(r"(\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2})", str(seg))
    return m.group(1) if m else str(seg)


def group_of(df, key, is_comma):
    return route_of(df["seg"].iloc[0]) if is_comma else str(key)


def anchors_for(df, hs, stride, hist=HIST):
    """Single shared anchor list for ALL methods on this segment."""
    n = len(df)
    return list(range(hist, n - hs, stride))


_CSTATE_CACHE = {}


def causal_state(df):
    """One-sided BACKWARD-difference kinematic state (no future sample).
    Memoized by object id (segments are reused across folds/seeds).
    Returns dict of arrays aligned to df index:
      v_can   : CAN speed sensor (real control-side signal)   -> causal speed
      v_bwd   : backward-difference pose speed (causal fallback)
      wr_bwd  : backward-difference yaw rate [rad/s]           -> causal
      yaw     : unwrapped heading [rad]  (reconstructed causally)
    """
    seg = str(df["seg"].iloc[0]) if "seg" in df else ""
    key = (seg, len(df), id(df))
    cached = _CSTATE_CACHE.get(key)
    if cached is not None and len(cached["x"]) == len(df):
        return cached
    x = df["x"].values; y = df["y"].values
    t = df["t"].values
    dtb = np.diff(t, prepend=t[0] - (t[1] - t[0]))      # backward dt, >0
    dtb[dtb <= 0] = np.median(dtb[dtb > 0]) if np.any(dtb > 0) else 1.0
    vx = np.diff(x, prepend=x[0]) / dtb
    vy = np.diff(y, prepend=y[0]) / dtb
    v_bwd = np.hypot(vx, vy)
    # CAUSAL heading: reconstruct from BACKWARD-difference velocity direction,
    # so yaw[i] depends only on positions <= i (no future sample). The cached
    # df["yaw_unwrap"] is central-differenced (leaks x[i+1]) and is NOT used.
    yaw = np.unwrap(np.arctan2(vy, vx))
    wr_bwd = np.diff(yaw, prepend=yaw[0]) / dtb         # rad/s, backward
    v_can = df["can_speed"].values if "can_speed" in df else v_bwd
    res = {"v_can": v_can, "v_bwd": v_bwd, "wr_bwd": wr_bwd, "yaw": yaw, "x": x, "y": y}
    _CSTATE_CACHE[key] = res
    if len(_CSTATE_CACHE) > 4000:
        _CSTATE_CACHE.clear(); _CSTATE_CACHE[key] = res
    return res


def oracle_path_local(df, i0, hs, dt, a):
    """ORACLE upper-bound path: future steer x future pose-diff speed (leaky).
    Rollout DYNAMICS use the central-difference cached heading/speed (the oracle
    part); the local-frame projection uses the SAME causal yaw[i0] as gt_path_local
    so FDE is computed in a consistent frame."""
    s = causal_state(df); yaw_c = s["yaw"]           # causal frame for projection
    x = df["x"].values; y = df["y"].values
    yaw0_dyn = np.radians(df["yaw_unwrap"].values[i0])  # oracle initial heading
    st = df["steer"].values; v = df["obs_v"].values; n = len(df)
    px, py, th = x[i0], y[i0], yaw0_dyn; path = []
    cc, ss = np.cos(-yaw_c[i0]), np.sin(-yaw_c[i0])
    for k in range(hs):
        j = min(i0 + k, n - 1)
        th += a * st[j] * v[j] * dt
        px += v[j] * np.cos(th) * dt; py += v[j] * np.sin(th) * dt
        path.append([cc * (px - x[i0]) - ss * (py - y[i0]), ss * (px - x[i0]) + cc * (py - y[i0])])
    return np.array(path)


def gain_curv(items, is_comma=True, min_v=None):
    """Fit Model-C gain a (kappa = a*steer) from CAUSAL backward state only.
    items: list of (key, df) OR list of df."""
    if min_v is None:
        min_v = 8.0 if is_comma else 1.0
    xs, ys = [], []
    for it in items:
        df = it[1] if isinstance(it, tuple) else it
        s = causal_state(df)
        v = s["v_can"]
        mask = v > min_v
        steer = df["steer"].values[mask]
        kap = (s["wr_bwd"] / (v + 1e-3))[mask]          # yaw_rate/v = curvature
        ok = np.isfinite(steer) & np.isfinite(kap)
        if ok.sum() > 5:
            hi = np.nanpercentile(np.abs(kap[ok]), 97)
            ok &= np.abs(kap) < hi
        xs.append(steer[ok]); ys.append(kap[ok])
    X = np.concatenate(xs); Y = np.concatenate(ys)
    a, b, r, p, se = stats.linregress(X, Y)
    return float(a)


def physics_endpoint_local(df, i0, hs, dt, a, steer_mode="hold", speed_mode="can_hold"):
    """Model-C rollout endpoint in anchor-local frame, CAUSAL state.
    speed_mode: can_hold (freeze CAN speed), bwd_hold (freeze backward speed),
                hist (history slope). steer_mode: hold / hist."""
    s = causal_state(df)
    st = df["steer"].values; yaw = s["yaw"]; x = s["x"]; y = s["y"]
    v_can = s["v_can"]; v_bwd = s["v_bwd"]; n = len(df)
    j0 = max(i0 - HIST, 0)
    vslope = (v_bwd[i0] - v_bwd[j0]) / ((i0 - j0) * dt) if i0 > j0 else 0.0
    sslope = (st[i0] - st[j0]) / ((i0 - j0) * dt) if i0 > j0 else 0.0
    px, py, th = x[i0], y[i0], yaw[i0]
    for k in range(hs):
        if speed_mode == "can_hold":
            v = v_can[i0]
        elif speed_mode == "bwd_hold":
            v = v_bwd[i0]
        elif speed_mode == "hist":
            v = max(v_bwd[i0] + vslope * k * dt, 0.0)
        else:
            v = v_can[i0]
        sd = st[i0] if steer_mode == "hold" else st[i0] + sslope * k * dt
        th += a * sd * v * dt
        px += v * np.cos(th) * dt; py += v * np.sin(th) * dt
    cc, ss = np.cos(-yaw[i0]), np.sin(-yaw[i0])
    return np.array([cc * (px - x[i0]) - ss * (py - y[i0]),
                     ss * (px - x[i0]) + cc * (py - y[i0])])


def features_causal(df, i0, hs, dt):
    """History-only causal feature vector (shared by MLP and residual)."""
    s = causal_state(df)
    st = df["steer"].values; v_can = s["v_can"]; v_bwd = s["v_bwd"]; wr = s["wr_bwd"]
    j0 = max(i0 - HIST, 0); h = slice(j0, i0 + 1)
    sslope = (st[i0] - st[j0]) / (max(i0 - j0, 1) * dt)
    return [v_can[i0], v_bwd[i0], wr[i0], st[i0],
            float(np.mean(st[h])), float(np.std(st[h])), float(np.mean(v_can[h])), sslope]


def target_local(df, i0, hs):
    s = causal_state(df); yaw = s["yaw"]; x = s["x"]; y = s["y"]
    cc, ss = np.cos(-yaw[i0]), np.sin(-yaw[i0])
    dx = x[i0 + hs] - x[i0]; dy = y[i0 + hs] - y[i0]
    return np.array([cc * dx - ss * dy, ss * dx + cc * dy])


def local_to_fde(loc, df, i0, hs):
    s = causal_state(df); yaw = s["yaw"]; x = s["x"]; y = s["y"]
    c, ss = np.cos(yaw[i0]), np.sin(yaw[i0])
    px = x[i0] + c * loc[0] - ss * loc[1]; py = y[i0] + ss * loc[0] + c * loc[1]
    return np.hypot(px - x[i0 + hs], py - y[i0 + hs])


def physics_path_local(df, i0, hs, dt, a, steer_mode="hold", speed_mode="can_hold"):
    """Full Model-C path (hs points) in anchor-local frame, causal state."""
    s = causal_state(df)
    st = df["steer"].values; yaw = s["yaw"]; x = s["x"]; y = s["y"]
    v_can = s["v_can"]; v_bwd = s["v_bwd"]
    j0 = max(i0 - HIST, 0)
    sslope = (st[i0] - st[j0]) / ((i0 - j0) * dt) if i0 > j0 else 0.0
    px, py, th = x[i0], y[i0], yaw[i0]; path = []
    for k in range(hs):
        v = v_can[i0] if speed_mode == "can_hold" else (v_bwd[i0] if speed_mode == "bwd_hold" else v_can[i0])
        sd = st[i0] if steer_mode == "hold" else st[i0] + sslope * k * dt
        th += a * sd * v * dt
        px += v * np.cos(th) * dt; py += v * np.sin(th) * dt
        cc, ss = np.cos(-yaw[i0]), np.sin(-yaw[i0])
        path.append([cc * (px - x[i0]) - ss * (py - y[i0]), ss * (px - x[i0]) + cc * (py - y[i0])])
    return np.array(path)


def kin_path_local(df, i0, hs, dt, model):
    """CV/CTRV/CTRA/EKF(=smoothed-CTRV) path in anchor-local frame, CAUSAL state.
    All use only backward-difference state at/<= i0."""
    s = causal_state(df)
    yaw = s["yaw"]; x = s["x"]; y = s["y"]; v_can = s["v_can"]; v_bwd = s["v_bwd"]; wr = s["wr_bwd"]
    j0 = max(i0 - 5, 0); acc = (v_bwd[i0] - v_bwd[j0]) / ((i0 - j0) * dt) if i0 > j0 else 0.0
    v0 = v_can[i0]; w0 = wr[i0]
    if model == "EKF":  # smoothed-CTRV: mean of short causal history
        k0 = max(i0 - HIST, 0); v0 = float(np.mean(v_can[k0:i0 + 1])); w0 = float(np.mean(wr[k0:i0 + 1]))
    px, py, th = x[i0], y[i0], yaw[i0]; vv = v0; path = []
    for k in range(hs):
        if model == "CV":
            dx, dy = vv * np.cos(th) * dt, vv * np.sin(th) * dt
        elif model == "CTRV":
            th += w0 * dt; dx, dy = vv * np.cos(th) * dt, vv * np.sin(th) * dt
        elif model == "CTRA":
            vv = max(vv + acc * dt, 0.0); th += w0 * dt; dx, dy = vv * np.cos(th) * dt, vv * np.sin(th) * dt
        else:  # EKF
            th += w0 * dt; dx, dy = vv * np.cos(th) * dt, vv * np.sin(th) * dt
        px += dx; py += dy
        cc, ss = np.cos(-yaw[i0]), np.sin(-yaw[i0])
        path.append([cc * (px - x[i0]) - ss * (py - y[i0]), ss * (px - x[i0]) + cc * (py - y[i0])])
    return np.array(path)


def gt_path_local(df, i0, hs):
    s = causal_state(df); yaw = s["yaw"]; x = s["x"]; y = s["y"]
    cc, ss = np.cos(-yaw[i0]), np.sin(-yaw[i0])
    dx = x[i0 + 1:i0 + hs + 1] - x[i0]; dy = y[i0 + 1:i0 + hs + 1] - y[i0]
    return np.stack([cc * dx - ss * dy, ss * dx + cc * dy], axis=1)


def ade_fde(path_local, df, i0, hs):
    gt = gt_path_local(df, i0, hs)
    d = np.hypot(path_local[:, 0] - gt[:, 0], path_local[:, 1] - gt[:, 1])
    return float(np.mean(d)), float(d[-1]), d


def paired_route_diff(err_a, err_b, groups, n_boot=2000, seed=0):
    """err_a, err_b: per-anchor errors for two methods on IDENTICAL anchors.
    groups: route id per anchor. Returns paired route-level diff (a-b):
      mean diff, CI, fraction of routes where a<b (a better), p (wilcoxon)."""
    err_a = np.asarray(err_a); err_b = np.asarray(err_b); groups = np.asarray(groups)
    uniq = np.unique(groups)
    da, db = [], []
    for g in uniq:
        m = groups == g
        da.append(err_a[m].mean()); db.append(err_b[m].mean())
    da = np.array(da); db = np.array(db); d = da - db
    rng = np.random.default_rng(seed)
    boots = [np.mean(rng.choice(d, size=len(d), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    try:
        w = stats.wilcoxon(da, db).pvalue
    except Exception:
        w = float("nan")
    return {"mean_diff_a_minus_b": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "frac_routes_a_better": float(np.mean(d < 0)), "wilcoxon_p": float(w),
            "n_routes": int(len(uniq)), "mean_a": float(da.mean()), "mean_b": float(db.mean())}
