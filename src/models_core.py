"""
models_core.py
Shared model components reused across experiments:
  - articulation/steering gain calibration
  - loader speed model (footboard/oil_door -> speed)
  - physics-prior rollout (loader & comma)
  - direct data-driven MLP dataset builders
  - bootstrap CI + paired Wilcoxon significance helpers
"""
import numpy as np
from scipy import stats
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# ---------------- calibration ----------------

def loader_gain(data, segs):
    xs, ys = [], []
    for s in segs:
        df = data[s]
        mask = (np.abs(df["yaw_rate"].values) < 3) & (df["obs_v"].values > 1.0)
        c = np.median(df["motor_angle"].values[mask]) if mask.sum() > 5 else np.median(df["motor_angle"].values)
        mv = df["obs_v"].values > 0.5
        x = (df["motor_angle"].values - c)[mv]
        y = np.radians(df["yaw_rate"].values)[mv]
        ok = np.abs(y) < np.nanpercentile(np.abs(y), 98)
        xs.append(x[ok]); ys.append(y[ok])
    g, b, r, p, se = stats.linregress(np.concatenate(xs), np.concatenate(ys))
    return g


def loader_center(df, warmup=True, dt=0.1):
    if warmup:
        sub = df.iloc[:int(3.0 / dt)]
        mask = (np.abs(sub["yaw_rate"].values) < 4) & (sub["obs_v"].values > 0.8)
        return np.median(sub["motor_angle"].values[mask]) if mask.sum() > 3 else np.median(sub["motor_angle"].values)
    mask = (np.abs(df["yaw_rate"].values) < 3) & (df["obs_v"].values > 1.0)
    return np.median(df["motor_angle"].values[mask]) if mask.sum() > 5 else np.median(df["motor_angle"].values)


def loader_speed_coef(data, segs):
    X, y = [], []
    for s in segs:
        df = data[s]; m = df["obs_v"].values > 0.2
        X.append(np.column_stack([df["footboard"].values[m], df["oil_door"].values[m]]))
        y.append(df["obs_v"].values[m])
    X = np.vstack(X); y = np.concatenate(y); A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None); return coef


def comma_gain(segs):
    xs, ys = [], []
    for df in segs:
        m = df["obs_v"].values > 8.0
        x = df["steer"].values[m]; y = np.radians(df["yaw_rate"].values)[m]
        ok = np.isfinite(x) & np.isfinite(y) & (np.abs(y) < np.nanpercentile(np.abs(y[np.isfinite(y)]), 97))
        xs.append(x[ok]); ys.append(y[ok])
    g, b, r, p, se = stats.linregress(np.concatenate(xs), np.concatenate(ys))
    return g


# ---------------- physics rollout (per-anchor FDE list) ----------------

def loader_phys_fde(df, hs, dt, g, c, coef, use_obs_speed=False, no_steer=False, no_warmup_c=None):
    x = df["x"].values; y = df["y"].values; yaw = np.radians(df["yaw_unwrap"].values)
    fb = df["footboard"].values; od = df["oil_door"].values; ma = df["motor_angle"].values
    obs_v = df["obs_v"].values
    n = len(df); out = []
    cc = c if no_warmup_c is None else no_warmup_c
    for i0 in range(0, n - hs, 2):
        px, py, th = x[i0], y[i0], yaw[i0]
        for k in range(hs):
            j = min(i0 + k, n - 1)
            v = obs_v[j] if use_obs_speed else max(coef[0] + coef[1] * fb[j] + coef[2] * od[j], 0.0)
            if not no_steer:
                th += g * (ma[j] - cc) * dt
            px += v * np.cos(th) * dt; py += v * np.sin(th) * dt
        out.append(np.hypot(px - x[i0 + hs], py - y[i0 + hs]))
    return np.array(out)


def comma_phys_fde(df, hs, dt, g, use_obs_speed=True, no_steer=False):
    x = df["x"].values; y = df["y"].values; yaw = np.radians(df["yaw_unwrap"].values)
    st = df["steer"].values; obs_v = df["obs_v"].values
    n = len(df); out = []
    for i0 in range(0, n - hs, 5):
        px, py, th = x[i0], y[i0], yaw[i0]
        for k in range(hs):
            j = min(i0 + k, n - 1)
            v = obs_v[j]
            if not no_steer:
                th += g * st[j] * dt
            px += v * np.cos(th) * dt; py += v * np.sin(th) * dt
        out.append(np.hypot(px - x[i0 + hs], py - y[i0 + hs]))
    return np.array(out)


# ---------------- direct MLP (loader) ----------------

def loader_direct_dataset(data, segs, hs, dt):
    X, Y = [], []
    for s in segs:
        df = data[s]; n = len(df); c = loader_center(df, warmup=False, dt=dt)
        for i0 in range(0, n - hs, 2):
            fut = slice(i0, i0 + hs)
            feat = [df["obs_v"].values[i0], np.radians(df["yaw_rate"].values[i0]),
                    np.mean(df["footboard"].values[fut]), np.mean(df["oil_door"].values[fut]),
                    np.mean(df["motor_angle"].values[fut] - c), np.std(df["motor_angle"].values[fut])]
            yaw = np.radians(df["yaw_unwrap"].values[i0])
            dx = df["x"].values[i0 + hs] - df["x"].values[i0]; dy = df["y"].values[i0 + hs] - df["y"].values[i0]
            cc, ss = np.cos(-yaw), np.sin(-yaw)
            X.append(feat); Y.append([cc * dx - ss * dy, ss * dx + cc * dy])
    return np.array(X), np.array(Y)


def train_mlp(X, Y, hidden=(48, 24), seed=0):
    sc = StandardScaler().fit(X)
    mdl = MLPRegressor(hidden_layer_sizes=hidden, max_iter=1000, alpha=1e-3,
                       random_state=seed, early_stopping=True).fit(sc.transform(X), Y)
    return sc, mdl


# ---------------- statistics ----------------

def bootstrap_ci(vals, n_boot=2000, alpha=0.05, seed=0):
    vals = np.asarray(vals); rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.mean(vals)), float(lo), float(hi)


def paired_wilcoxon(a, b):
    a = np.asarray(a); b = np.asarray(b)
    n = min(len(a), len(b))
    try:
        stat, p = stats.wilcoxon(a[:n], b[:n])
        return float(p)
    except Exception:
        return float("nan")


# ---------------- richer metrics (ADE/FDE + lat/lon decomposition) ----------------

def comma_phys_metrics(df, hs, dt, g, use_obs_speed=True):
    """Return per-anchor dict of arrays: ade, fde, lat, lon (physics prior,
    control-conditioned: uses future steer + observed speed)."""
    x = df["x"].values; y = df["y"].values; yaw = np.radians(df["yaw_unwrap"].values)
    st = df["steer"].values; obs_v = df["obs_v"].values
    n = len(df)
    ade, fde, lat, lon = [], [], [], []
    for i0 in range(0, n - hs, 5):
        px, py, th = x[i0], y[i0], yaw[i0]
        yaw0 = yaw[i0]
        ds = []
        ex = ey = 0.0
        for k in range(hs):
            j = min(i0 + k, n - 1)
            v = obs_v[j]
            th += g * st[j] * dt
            px += v * np.cos(th) * dt; py += v * np.sin(th) * dt
            gx = x[i0 + k + 1]; gy = y[i0 + k + 1]
            ds.append(np.hypot(px - gx, py - gy))
            ex, ey = px - gx, py - gy
        ade.append(np.mean(ds)); fde.append(ds[-1])
        c, s = np.cos(yaw0), np.sin(yaw0)
        lon.append(abs(c * ex + s * ey)); lat.append(abs(-s * ex + c * ey))
    return {"ade": np.array(ade), "fde": np.array(fde),
            "lat": np.array(lat), "lon": np.array(lon)}


def comma_kin_metrics(df, hs, dt, model):
    """Kinematic baseline (no future control) ADE/FDE per anchor via baselines."""
    import baselines as B
    x = df["x"].values; y = df["y"].values
    yaw = np.radians(df["yaw_unwrap"].values)
    v = df["obs_v"].values; w = np.radians(df["yaw_rate"].values)
    n = len(df); back = 5; win = 10
    ade, fde = [], []
    for i0 in range(0, n - hs, 5):
        j0 = max(i0 - back, 0)
        a0 = (v[i0] - v[j0]) / ((i0 - j0) * dt) if i0 > j0 else 0.0
        v0, w0 = v[i0], w[i0]
        if model == "EKF":
            k0 = max(i0 - win, 0); v0 = np.mean(v[k0:i0 + 1]); w0 = np.mean(w[k0:i0 + 1])
        px, py = B._rollout_paths(x[i0], y[i0], yaw[i0], v0, w0, a0, hs, dt, model)
        gx = x[i0 + 1:i0 + hs + 1]; gy = y[i0 + 1:i0 + hs + 1]
        d = np.hypot(px[:hs] - gx, py[:hs] - gy)
        ade.append(np.mean(d)); fde.append(d[-1])
    return {"ade": np.array(ade), "fde": np.array(fde)}
