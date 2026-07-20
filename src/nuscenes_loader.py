"""
nuscenes_loader.py
nuScenes CAN bus expansion loader -> unified schema (matches loader/comma).

Per scene (~20s):
  <scene>_vehicle_monitor.json  (2Hz): steering(deg), vehicle_speed(km/h),
                                       yaw_rate(deg/s), throttle, brake, rear rpms
  <scene>_pose.json            (~50Hz): pos[x,y,z](m), orientation(quat), vel
Unified columns: t, x, y, yaw_unwrap(deg), obs_v(m/s), yaw_rate(deg/s),
                 curvature(1/m), steer(deg), can_speed(m/s)
"""
import os
import glob
import json
import pickle
import numpy as np
import pandas as pd

_BASE = os.path.join(os.path.dirname(__file__), "..", "..")
CACHE_FILE = os.path.join(os.path.dirname(__file__), ".nusc_cache.pkl")


def _dataset_dir():
    for d in glob.glob(os.path.join(_BASE, "*")):
        # find folder that (recursively) contains *_vehicle_monitor.json
        hits = glob.glob(os.path.join(d, "**", "*_vehicle_monitor.json"), recursive=True)
        if hits:
            return os.path.dirname(hits[0])
    return None


def quat_yaw(q):
    w, x, y, z = q
    return np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def list_scenes(dd):
    files = sorted(glob.glob(os.path.join(dd, "scene-*_vehicle_monitor.json")))
    return [os.path.basename(f).replace("_vehicle_monitor.json", "") for f in files]


def _nearest(t_src, v_src, t_q):
    idx = np.clip(np.searchsorted(t_src, t_q), 0, len(t_src) - 1)
    left = np.clip(idx - 1, 0, len(t_src) - 1)
    use_left = np.abs(t_src[left] - t_q) < np.abs(t_src[idx] - t_q)
    idx = np.where(use_left, left, idx)
    return v_src[idx]


def load_scene(dd, scene):
    pose = json.load(open(os.path.join(dd, scene + "_pose.json")))
    vm = json.load(open(os.path.join(dd, scene + "_vehicle_monitor.json")))
    pt = np.array([r["utime"] for r in pose], float) / 1e6
    x = np.array([r["pos"][0] for r in pose], float)
    y = np.array([r["pos"][1] for r in pose], float)
    yaw = np.unwrap(np.radians([quat_yaw(r["orientation"]) for r in pose]))
    # downsample pose to ~20Hz to match other datasets and reduce noise
    t0 = pt[0]
    t = pt - t0
    # CAN
    vt = np.array([r["utime"] for r in vm], float) / 1e6 - t0
    steer = np.array([r["steering"] for r in vm], float)
    vspeed = np.array([r["vehicle_speed"] for r in vm], float) / 3.6  # km/h -> m/s
    a_steer = _nearest(vt, steer, t)
    a_vspeed = _nearest(vt, vspeed, t)

    dt = np.gradient(t)
    vx = np.gradient(x) / dt; vy = np.gradient(y) / dt
    obs_v = np.hypot(vx, vy)
    yaw_unwrap = np.degrees(yaw)
    yaw_rate = np.degrees(np.gradient(yaw) / dt)
    curvature = np.radians(yaw_rate) / (obs_v + 1e-3)
    return pd.DataFrame({
        "seg": scene, "t": t, "x": x, "y": y, "yaw_unwrap": yaw_unwrap,
        "obs_v": obs_v, "yaw_rate": yaw_rate, "curvature": curvature,
        "steer": a_steer, "can_speed": a_vspeed,
    })


def load_all_cached(min_speed=3.0, min_steer_std=5.0, use_cache=True, verbose=True, max_scenes=None):
    if use_cache and os.path.exists(CACHE_FILE):
        if verbose:
            print("  [cache] loading from .nusc_cache.pkl")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    dd = _dataset_dir()
    if dd is None:
        raise FileNotFoundError("nuScenes CAN expansion not found under 公开数据集/")
    scenes = list_scenes(dd)
    if max_scenes:
        scenes = scenes[:max_scenes]
    out = {}
    for sc in scenes:
        try:
            df = load_scene(dd, sc)
        except Exception:
            continue
        if df["obs_v"].mean() < min_speed:
            continue
        if df["steer"].std() < min_steer_std:
            continue
        if len(df) < 100:
            continue
        out[sc] = df
    if verbose:
        print("  [load] nuScenes: %d/%d usable scenes (v>%.0f, steer_std>%.0f)"
              % (len(out), len(scenes), min_speed, min_steer_std))
    if use_cache:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(out, f)
    return out


if __name__ == "__main__":
    import time
    dd = _dataset_dir()
    scenes = list_scenes(dd)
    print("found %d scenes in %s" % (len(scenes), dd))
    t0 = time.time()
    out = {}
    for sc in scenes:
        try:
            df = load_scene(dd, sc)
        except Exception:
            continue
        if df["obs_v"].mean() < 3.0 or df["steer"].std() < 5.0 or len(df) < 100:
            continue
        out[sc] = df
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(out, f)
    tot = sum(df["t"].iloc[-1] for df in out.values())
    print("usable %d/%d scenes, %.0fs (%.1f min); cache written." % (len(out), len(scenes), tot, tot / 60))
