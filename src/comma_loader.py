"""
comma_loader.py
comma2k19 data loader, aligned to the same unified schema as the loader data_loader.

comma2k19 layout (~60s per segment, 20Hz):
  processed_log/CAN/steering_angle/{t,value}   steering wheel angle (deg)
  processed_log/CAN/speed/{t,value}            vehicle speed (m/s)
  global_pose/frame_positions  (N,3) ECEF meters
  global_pose/frame_times      (N,)  monotonic time (s)

Unified output columns (matching the loader):
  t, x, y, yaw_unwrap(deg), obs_v(m/s), yaw_rate(deg/s), curvature(1/m),
  steer(deg), can_speed(m/s)
"""
import os
import glob
import pickle
import numpy as np
import pandas as pd

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "gongkai_placeholder")
# real dataset dir resolved below (Chinese folder name handled via glob)
_BASE = os.path.join(os.path.dirname(__file__), "..", "..")
CACHE_FILE = os.path.join(os.path.dirname(__file__), ".comma_cache.pkl")


def _dataset_dir():
    # find the folder that contains Chunk_1 / Chunk_2
    for d in glob.glob(os.path.join(_BASE, "*")):
        if os.path.isdir(os.path.join(d, "Chunk_1")) or os.path.isdir(os.path.join(d, "Chunk_2")):
            return d
    return os.path.join(_BASE, "gongkai")


def ecef_to_local(pos):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    a = 6378137.0
    e2 = 6.69437999014e-3
    x0, y0, z0 = pos[0]
    lon0 = np.arctan2(y0, x0)
    p = np.sqrt(x0**2 + y0**2)
    lat0 = np.arctan2(z0, p * (1 - e2))
    for _ in range(5):
        N = a / np.sqrt(1 - e2 * np.sin(lat0)**2)
        h = p / np.cos(lat0) - N
        lat0 = np.arctan2(z0, p * (1 - e2 * N / (N + h)))
    dx, dy, dz = x - x0, y - y0, z - z0
    slon, clon = np.sin(lon0), np.cos(lon0)
    slat, clat = np.sin(lat0), np.cos(lat0)
    east = -slon * dx + clon * dy
    north = -slat * clon * dx - slat * slon * dy + clat * dz
    return east, north


def list_segments(max_segments=None):
    dd = _dataset_dir()
    segs = sorted(glob.glob(os.path.join(dd, "Chunk_*", "*", "*", "global_pose")))
    segs = [os.path.dirname(s) for s in segs]
    if max_segments:
        segs = segs[:max_segments]
    return segs


def _nearest(t_src, v_src, t_q):
    idx = np.clip(np.searchsorted(t_src, t_q), 0, len(t_src) - 1)
    left = np.clip(idx - 1, 0, len(t_src) - 1)
    use_left = np.abs(t_src[left] - t_q) < np.abs(t_src[idx] - t_q)
    idx = np.where(use_left, left, idx)
    return v_src[idx]


def load_segment(seg_path):
    ft = np.load(os.path.join(seg_path, "global_pose", "frame_times"))
    pos = np.load(os.path.join(seg_path, "global_pose", "frame_positions"))
    east, north = ecef_to_local(pos)
    sa_t = np.load(os.path.join(seg_path, "processed_log", "CAN", "steering_angle", "t"))
    sa_v = np.load(os.path.join(seg_path, "processed_log", "CAN", "steering_angle", "value")).astype(float).ravel()
    sp_t = np.load(os.path.join(seg_path, "processed_log", "CAN", "speed", "t"))
    sp_v = np.load(os.path.join(seg_path, "processed_log", "CAN", "speed", "value")).astype(float).ravel()
    t = ft - ft[0]
    x, y = east, north
    dt = np.gradient(t)
    vx = np.gradient(x) / dt
    vy = np.gradient(y) / dt
    obs_v = np.hypot(vx, vy)
    heading = np.unwrap(np.arctan2(vy, vx))
    yaw_unwrap = np.degrees(heading)
    yaw_rate = np.degrees(np.gradient(heading) / dt)
    curvature = np.radians(yaw_rate) / (obs_v + 1e-3)
    steer = _nearest(sa_t - sa_t[0], sa_v, t)
    can_speed = _nearest(sp_t - sp_t[0], sp_v, t)
    seg_id = os.path.basename(seg_path) + "_" + os.path.basename(os.path.dirname(seg_path))
    return pd.DataFrame({
        "seg": seg_id, "t": t, "x": x, "y": y, "yaw_unwrap": yaw_unwrap,
        "obs_v": obs_v, "yaw_rate": yaw_rate, "curvature": curvature,
        "steer": steer, "can_speed": can_speed,
    })


def load_many(max_segments=40, min_speed=5.0, require_turning=True):
    out = {}
    for sp in list_segments():
        try:
            df = load_segment(sp)
        except Exception:
            continue
        if df["obs_v"].mean() < min_speed:
            continue
        if require_turning and np.nanstd(df["curvature"].values) < 1e-4:
            continue
        out[df["seg"].iloc[0]] = df
        if len(out) >= max_segments:
            break
    return out


def load_all_cached(min_speed=8.0, min_steer_std=1.0, use_cache=True, verbose=True):
    if use_cache and os.path.exists(CACHE_FILE):
        if verbose:
            print("  [cache] loading from .comma_cache.pkl")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    out = {}
    segs = list_segments()
    for sp in segs:
        try:
            df = load_segment(sp)
        except Exception:
            continue
        if df["obs_v"].mean() < min_speed:
            continue
        if df["steer"].std() < min_steer_std:
            continue
        m = df["obs_v"].values > 5.0
        if m.sum() < len(df) * 0.5:
            continue
        out[df["seg"].iloc[0]] = df
    if verbose:
        print("  [load] full load: %d/%d usable (v>%.0f, steer_std>%.1f)" % (len(out), len(segs), min_speed, min_steer_std))
    if use_cache:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(out, f)
    return out


def stratify(data):
    segs = list(data.keys())
    v = np.array([data[s]["obs_v"].mean() for s in segs])
    turn = np.array([data[s]["curvature"].abs().median() for s in segs])
    v_q = np.quantile(v, [1/3, 2/3])
    t_q = np.quantile(turn, [1/3, 2/3])
    labels = {}
    for i, s in enumerate(segs):
        vb = "low" if v[i] < v_q[0] else ("mid" if v[i] < v_q[1] else "high")
        tb = "straight" if turn[i] < t_q[0] else ("gentle" if turn[i] < t_q[1] else "curvy")
        labels[s] = {"speed_bin": vb, "turn_bin": tb, "mean_v": float(v[i]), "med_curv": float(turn[i])}
    return labels


if __name__ == "__main__":
    data = load_all_cached(use_cache=False)
    print("\nfull usable segments: %d" % len(data))
    total_dur = sum(df["t"].iloc[-1] for df in data.values())
    print("total duration: %.0fs (%.1f min)" % (total_dur, total_dur/60))
    labels = stratify(data)
    from collections import Counter
    print("speed bins:", dict(Counter(l["speed_bin"] for l in labels.values())))
    print("turn bins:", dict(Counter(l["turn_bin"] for l in labels.values())))
