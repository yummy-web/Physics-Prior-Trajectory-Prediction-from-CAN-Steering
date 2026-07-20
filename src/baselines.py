"""
baselines.py
Standard motion-model baselines for trajectory prediction under one unified,
vectorized eval interface. Returns ADE/FDE per horizon.

Baselines:
  CV    Constant Velocity
  CA    Constant Acceleration
  CTRV  Constant Turn Rate & Velocity
  CTRA  Constant Turn Rate & Acceleration
  EKF   trailing-window smoothed CTRV (measurement-noise-reduced)

Segment df columns used: x, y, yaw_unwrap(deg), obs_v(m/s), yaw_rate(deg/s)
"""
import numpy as np


def _rollout_paths(x0, y0, yaw0, v0, w0, a0, hs, dt, model):
    """Vectorized single-anchor rollout. Returns arrays px,py of length hs."""
    k = np.arange(1, hs + 1)
    if model == "CV":
        px = x0 + v0 * np.cos(yaw0) * dt * k
        py = y0 + v0 * np.sin(yaw0) * dt * k
        return px, py
    if model == "CA":
        v = np.maximum(v0 + a0 * dt * k, 0.0)
        s = np.cumsum(v * dt)
        px = x0 + np.cos(yaw0) * s
        py = y0 + np.sin(yaw0) * s
        return px, py
    if model in ("CTRV", "EKF"):
        th = yaw0 + w0 * dt * k
        # integrate cos/sin*v*dt
        px = x0 + np.cumsum(v0 * np.cos(th) * dt)
        py = y0 + np.cumsum(v0 * np.sin(th) * dt)
        return px, py
    if model == "CTRA":
        th = yaw0 + w0 * dt * k
        v = np.maximum(v0 + a0 * dt * k, 0.0)
        px = x0 + np.cumsum(v * np.cos(th) * dt)
        py = y0 + np.cumsum(v * np.sin(th) * dt)
        return px, py
    raise ValueError(model)


def eval_kinematic(df, horizons, dt, model, stride=3):
    x = df["x"].values; y = df["y"].values
    yaw = np.radians(df["yaw_unwrap"].values)
    v = df["obs_v"].values; w = np.radians(df["yaw_rate"].values)
    n = len(df)
    back = 5
    win = 10
    maxhs = int(round(max(horizons) / dt))
    res = {H: {"ade": [], "fde": []} for H in horizons}
    for i0 in range(0, n - 3, stride):
        # anchor kinematics
        v0, w0, yaw0 = v[i0], w[i0], yaw[i0]
        j0 = max(i0 - back, 0)
        a0 = (v[i0] - v[j0]) / ((i0 - j0) * dt) if i0 > j0 else 0.0
        if model == "EKF":
            k0 = max(i0 - win, 0)
            v0 = np.mean(v[k0:i0 + 1]); w0 = np.mean(w[k0:i0 + 1])
        px, py = _rollout_paths(x[i0], y[i0], yaw0, v0, w0, a0, maxhs, dt, model)
        for H in horizons:
            hs = int(round(H / dt))
            if i0 + hs >= n:
                continue
            gx = x[i0 + 1:i0 + hs + 1]; gy = y[i0 + 1:i0 + hs + 1]
            d = np.hypot(px[:hs] - gx, py[:hs] - gy)
            res[H]["ade"].append(np.mean(d))
            res[H]["fde"].append(d[-1])
    return {H: (float(np.mean(res[H]["ade"])), float(np.mean(res[H]["fde"]))) for H in horizons}
