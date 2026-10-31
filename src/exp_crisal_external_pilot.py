from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import warnings
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_BAG = Path(__file__).resolve().parents[1] / "external_data" / "CRISAL_2025" / "extracted" / "CR2" / "locsp_08_04_2024_anneau_cite_scientifique_2024-04-08-12-14-51.bag"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "results" / "crisal_external_validation"
HORIZONS = np.array([1.0, 2.0, 3.0], dtype=float)
ANCHOR_STRIDE_SAMPLES = 5  # approximately 0.5 s at the 10 Hz INSPVA rate
MAX_CAN_AGE_SEC = 0.15
MAX_INS_BACKDIFF_SEC = 0.5
MAX_LABEL_BRACKET_SEC = 0.5
SPLIT_FRACTIONS = (0.60, 0.80)
RANDOM_STATE = 20261013
SHRINK_KAPPAS = (100.0, 300.0, 800.0)


def stamp_seconds(msg: Any, bag_time_ns: int) -> tuple[float, bool]:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None) if header is not None else None
    sec = getattr(stamp, "sec", getattr(stamp, "secs", None)) if stamp is not None else None
    nsec = getattr(stamp, "nanosec", getattr(stamp, "nsecs", None)) if stamp is not None else None
    if sec is not None and nsec is not None:
        value = float(sec) + float(nsec) * 1e-9
        if math.isfinite(value) and value > 0:
            return value, False
    return float(bag_time_ns) * 1e-9, True


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def duplicate_timestamp_count(times: np.ndarray) -> int:
    ordered = np.sort(np.asarray(times, dtype=float))
    return int(np.count_nonzero(ordered[1:] == ordered[:-1])) if len(ordered) > 1 else 0


def quantile_summary(values: list[float] | np.ndarray) -> dict[str, float | int]:
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"n": 0}
    return {"n": int(len(a)), "min_sec": float(np.min(a)), "p05_sec": float(np.quantile(a, .05)),
            "median_sec": float(np.median(a)), "p95_sec": float(np.quantile(a, .95)), "max_sec": float(np.max(a))}


def stable_sort_unique(times: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    order = np.argsort(times, kind="stable")
    times = times[order]
    values = [np.asarray(a)[order] for a in arrays]
    # Keep the last message at a repeated timestamp; duplicate counts are reported separately.
    keep = np.r_[times[1:] != times[:-1], True] if len(times) else np.array([], dtype=bool)
    return (times[keep], *(a[keep] for a in values))


def read_topics(bag: Path) -> dict[str, Any]:
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise SystemExit("Missing optional dependency. Install with: python -m pip install -r requirements-crisal.txt") from exc

    topic_names = {"/novatel/oem7/inspva", "/can/speed", "/can/steering"}
    records: dict[str, list[tuple[float, int, Any]]] = {name: [] for name in topic_names}
    counts: dict[str, int] = {name: 0 for name in topic_names}
    header_fallbacks: dict[str, int] = {name: 0 for name in topic_names}
    header_minus_bag: dict[str, list[float]] = {name: [] for name in topic_names}
    total_decoded = 0
    with AnyReader([bag]) as reader:
        connections = [c for c in reader.connections if c.topic in topic_names]
        present = {c.topic for c in connections}
        missing = topic_names - present
        if missing:
            raise RuntimeError(f"Required topics missing from bag: {sorted(missing)}")
        for connection, bag_ns, raw in reader.messages(connections=connections):
            msg = reader.deserialize(raw, connection.msgtype)
            topic = connection.topic
            counts[topic] += 1
            total_decoded += 1
            t, fallback = stamp_seconds(msg, bag_ns)
            header_fallbacks[topic] += int(fallback)
            header_minus_bag[topic].append(t - float(bag_ns) * 1e-9)
            if total_decoded % 10000 == 0:
                print(f"Decoded {total_decoded:,} selected messages; topic counts={counts}", flush=True)
            if topic == "/novatel/oem7/inspva":
                status_obj = getattr(msg, "status", None)
                status = int(getattr(status_obj, "status", -999))
                records[topic].append((t, status, msg))
            elif topic == "/can/speed":
                records[topic].append((t, 0, msg))
            else:
                records[topic].append((t, 0, msg))

    inspva = records["/novatel/oem7/inspva"]
    status_counts: dict[str, int] = {}
    good = []
    for t, status, msg in inspva:
        status_counts[str(status)] = status_counts.get(str(status), 0) + 1
        if status == 3:
            lat = float(getattr(msg, "latitude", float("nan")))
            lon = float(getattr(msg, "longitude", float("nan")))
            height = float(getattr(msg, "height", float("nan")))
            az = float(getattr(msg, "azimuth", float("nan")))
            east_v = float(getattr(msg, "east_velocity", float("nan")))
            north_v = float(getattr(msg, "north_velocity", float("nan")))
            if all(math.isfinite(v) for v in (lat, lon, height, az, east_v, north_v)):
                good.append((t, lat, lon, height, az, east_v, north_v))
    if len(good) < 100:
        raise RuntimeError(f"Too few valid INS_SOLUTION_GOOD samples: {len(good)}")
    ins_arr = np.asarray(good, dtype=float)
    good_ins_duplicate_timestamps = duplicate_timestamp_count(ins_arr[:, 0])
    t_ins, lat, lon, height, az_deg, east_v, north_v = stable_sort_unique(
        ins_arr[:, 0], ins_arr[:, 1], ins_arr[:, 2], ins_arr[:, 3], ins_arr[:, 4], ins_arr[:, 5], ins_arr[:, 6]
    )

    speed_rows = records["/can/speed"]
    t_speed_raw, v_kmh_raw = [], []
    for t, _, msg in speed_rows:
        t_speed_raw.append(t)
        v_kmh_raw.append(float(getattr(msg, "speed", float("nan"))))
    t_speed_raw_a, v_kmh_raw_a = np.asarray(t_speed_raw), np.asarray(v_kmh_raw)
    speed_ok = np.isfinite(v_kmh_raw_a) & (v_kmh_raw_a >= 0.0)
    speed_invalid_count = int(len(speed_ok) - np.count_nonzero(speed_ok))
    speed_duplicate_timestamps = duplicate_timestamp_count(t_speed_raw_a)
    t_speed, v_kmh = stable_sort_unique(t_speed_raw_a[speed_ok], v_kmh_raw_a[speed_ok])

    steering_rows = records["/can/steering"]
    t_steer_raw, steer_angle_raw, steer_offset_raw = [], [], []
    for t, _, msg in steering_rows:
        t_steer_raw.append(t)
        steer_angle_raw.append(float(getattr(msg, "angle", float("nan"))))
        steer_offset_raw.append(float(getattr(msg, "angle_offset", 0.0)))
    t_steer_raw_a = np.asarray(t_steer_raw)
    angle_raw_a, offset_raw_a = np.asarray(steer_angle_raw), np.asarray(steer_offset_raw)
    steer_raw = angle_raw_a + offset_raw_a
    steer_ok = np.isfinite(steer_raw)
    steering_invalid_count = int(len(steer_ok) - np.count_nonzero(steer_ok))
    steering_duplicate_timestamps = duplicate_timestamp_count(t_steer_raw_a)
    t_steer, steer = stable_sort_unique(t_steer_raw_a[steer_ok], steer_raw[steer_ok])
    if min(len(t_speed), len(t_steer)) < 100:
        raise RuntimeError("Too few usable CAN samples after finite-value filtering")

    return {
        "t_ins_abs": t_ins, "lat": lat, "lon": lon, "height_m": height, "az_deg": az_deg,
        "east_v": east_v, "north_v": north_v,
        "t_speed_abs": t_speed, "speed_mps": v_kmh / 3.6,
        "t_steer_abs": t_steer, "steer_raw_corrected": steer,
        "raw_counts": counts, "header_fallbacks": header_fallbacks,
        "ins_status_counts": status_counts,
        "invalid_value_counts": {
            "inspva_good_nonfinite_position_heading_velocity_or_height": int(sum(1 for t, status, msg in inspva if status == 3) - len(good)),
            "speed": speed_invalid_count, "steering_angle_plus_offset": steering_invalid_count,
        },
        "duplicate_timestamp_counts": {
            "inspva_after_good_filter": good_ins_duplicate_timestamps,
            "speed": speed_duplicate_timestamps, "steering": steering_duplicate_timestamps,
        },
        "header_minus_bag_time_sec": {topic: quantile_summary(offsets) for topic, offsets in header_minus_bag.items()},
    }


def lla_to_enu(lat_deg: np.ndarray, lon_deg: np.ndarray, height_m: np.ndarray | float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """WGS84 geodetic coordinates to local ENU, with the first point as origin."""
    a = 6378137.0
    e2 = 6.69437999014e-3
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    h = np.broadcast_to(np.asarray(height_m, dtype=float), lat.shape)
    n = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    x = (n + h) * np.cos(lat) * np.cos(lon)
    y = (n + h) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - e2) + h) * np.sin(lat)
    lat0, lon0 = lat[0], lon[0]
    dx, dy, dz = x - x[0], y - y[0], z - z[0]
    east = -np.sin(lon0) * dx + np.cos(lon0) * dy
    north = -np.sin(lat0) * np.cos(lon0) * dx - np.sin(lat0) * np.sin(lon0) * dy + np.cos(lat0) * dz
    up = np.cos(lat0) * np.cos(lon0) * dx + np.cos(lat0) * np.sin(lon0) * dy + np.sin(lat0) * dz
    return east, north, up


def asof(times: np.ndarray, values: np.ndarray, query: float) -> tuple[float, float]:
    idx = int(np.searchsorted(times, query, side="right") - 1)
    if idx < 0:
        return float("nan"), float("inf")
    return float(values[idx]), float(query - times[idx])


def interpolated_xy(times: np.ndarray, x: np.ndarray, y: np.ndarray, query: float) -> tuple[float, float, float]:
    right = int(np.searchsorted(times, query, side="left"))
    if right == 0 or right >= len(times):
        return float("nan"), float("nan"), float("inf")
    left = right - 1
    gap = float(times[right] - times[left])
    if gap <= 0 or gap > MAX_LABEL_BRACKET_SEC:
        return float("nan"), float("nan"), gap
    alpha = (query - times[left]) / gap
    return float((1.0 - alpha) * x[left] + alpha * x[right]), float((1.0 - alpha) * y[left] + alpha * y[right]), gap


def build_samples(raw: dict[str, Any], chronological_split: bool = True) -> dict[str, Any]:
    t_abs = raw["t_ins_abs"]
    t0 = float(t_abs[0])
    t = t_abs - t0
    x, y, _ = lla_to_enu(raw["lat"], raw["lon"], raw["height_m"])
    az = np.unwrap(np.deg2rad(raw["az_deg"]))
    speed_t = raw["t_speed_abs"] - t0
    steer_t = raw["t_steer_abs"] - t0
    speed = raw["speed_mps"]
    steer = raw["steer_raw_corrected"]

    # Backward-only yaw rate at each INS anchor; large temporal gaps are rejected.
    omega_bwd = np.full(len(t), np.nan)
    ins_dt = np.diff(t)
    az_wrapped_delta = (np.diff(np.deg2rad(raw["az_deg"])) + np.pi) % (2.0 * np.pi) - np.pi
    good_dt = (ins_dt > 0.0) & (ins_dt <= MAX_INS_BACKDIFF_SEC)
    omega_bwd[1:][good_dt] = az_wrapped_delta[good_dt] / ins_dt[good_dt]

    t_start, t_end = float(t[0]), float(t[-1])
    cut1 = t_start + SPLIT_FRACTIONS[0] * (t_end - t_start)
    cut2 = t_start + SPLIT_FRACTIONS[1] * (t_end - t_start)
    anchor_indices = np.arange(0, len(t), ANCHOR_STRIDE_SAMPLES, dtype=int)
    features, labels, meta = [], [], []
    for i in anchor_indices:
        ti = float(t[i])
        if not math.isfinite(float(omega_bwd[i])):
            continue
        v, v_age = asof(speed_t, speed, ti)
        s, s_age = asof(steer_t, steer, ti)
        if not (math.isfinite(v) and math.isfinite(s) and v_age <= MAX_CAN_AGE_SEC and s_age <= MAX_CAN_AGE_SEC):
            continue
        v05, v05_age = asof(speed_t, speed, ti - 0.5)
        s05, s05_age = asof(steer_t, steer, ti - 0.5)
        s10, s10_age = asof(steer_t, steer, ti - 1.0)
        if not all(math.isfinite(q) for q in (v05, s05, s10)) or max(v05_age, s05_age, s10_age) > MAX_CAN_AGE_SEC:
            continue
        # All features are obtained from the current or past CAN sample; no future interpolation.
        feat = [v, s, v * s, (v - v05) / 0.5, (s - s05) / 0.5, s05, s10, v05]
        target_body = []
        label_gaps = []
        theta = float(az[i])
        valid_label = True
        for horizon in HORIZONS:
            xe, ye, gap = interpolated_xy(t, x, y, ti + float(horizon))
            label_gaps.append(gap)
            if not (math.isfinite(xe) and math.isfinite(ye) and gap <= MAX_LABEL_BRACKET_SEC):
                valid_label = False
                break
            dx, dy = xe - x[i], ye - y[i]
            # Body axes: forward and right, expressed using the anchor INS azimuth.
            forward = dx * math.sin(theta) + dy * math.cos(theta)
            right = dx * math.cos(theta) - dy * math.sin(theta)
            target_body.extend([forward, right])
        if not valid_label:
            continue
        if chronological_split:
            if ti + float(HORIZONS[-1]) <= cut1:
                split = "fit"
            elif ti >= cut1 + float(HORIZONS[-1]) and ti + float(HORIZONS[-1]) <= cut2:
                split = "validation"
            elif ti >= cut2 + float(HORIZONS[-1]) and ti + float(HORIZONS[-1]) <= t_end:
                split = "test"
            else:
                continue
        else:
            # In acquisition-grouped evaluation, the entire acquisition is assigned
            # to exactly one outer role (fit, validation, or test). Do not discard
            # within-acquisition time blocks or create a second overlapping split.
            split = "all"
        features.append(feat)
        labels.append(target_body)
        meta.append({
            "ins_index": int(i), "time_sec": ti, "split": split,
            "can_speed_age_sec": float(v_age), "can_steering_age_sec": float(s_age),
            "max_label_bracket_sec": float(max(label_gaps)),
            "speed_mps": float(v), "steering_raw_corrected": float(s),
            "steer_product": float(v * s), "yaw_rate_bwd_rad_s": float(omega_bwd[i]),
            "heading_rad": theta, "x_east_m": float(x[i]), "y_north_m": float(y[i]),
            "east_velocity_mps": float(raw["east_v"][i]), "north_velocity_mps": float(raw["north_v"][i]),
        })
    if not features:
        raise RuntimeError("No valid anchors after causal alignment, label-gap, and temporal split checks")
    return {
        "X": np.asarray(features, dtype=float), "Y_body": np.asarray(labels, dtype=float).reshape(-1, 3, 2),
        "meta": meta,
        "time_bounds": {"start_sec": t_start, "fit_validation_cut_sec": cut1, "validation_test_cut_sec": cut2, "end_sec": t_end},
        "ins_t_rel": t, "east": x, "north": y, "heading_unwrapped_rad": az,
        "omega_bwd": omega_bwd, "raw_speed_t_rel": speed_t, "raw_speed_mps": speed,
        "raw_steer_t_rel": steer_t, "raw_steer_corrected": steer,
    }


def fit_steering_yaw_gain(samples: dict[str, Any], fit_mask: np.ndarray) -> dict[str, Any]:
    from sklearn.linear_model import HuberRegressor

    rows = [m for m, keep in zip(samples["meta"], fit_mask) if keep]
    q = np.asarray([m["steer_product"] for m in rows], dtype=float)
    omega = np.asarray([m["yaw_rate_bwd_rad_s"] for m in rows], dtype=float)
    # A robust, fit-only empirical mapping. The CAN angle unit/steering ratio is not assumed.
    estimator = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=500)
    estimator.fit(q.reshape(-1, 1), omega)
    pred = estimator.predict(q.reshape(-1, 1))
    denom = float(np.sum((omega - np.mean(omega)) ** 2))
    r2 = 1.0 - float(np.sum((omega - pred) ** 2)) / denom if denom > 0 else None
    return {
        "intercept_rad_s": float(estimator.intercept_),
        "slope_rad_s_per_speed_times_raw_steer": float(estimator.coef_[0]),
        "fit_r2_descriptive": r2,
        "n_fit_anchors": int(len(rows)),
        "fit_yaw_rate_p01_p50_p99_rad_s": [float(v) for v in np.quantile(omega, [0.01, 0.5, 0.99])],
        "fit_steering_raw_p01_p50_p99": [float(v) for v in np.quantile(np.asarray([m["steering_raw_corrected"] for m in rows]), [0.01, 0.5, 0.99])],
    }


def make_physics_predictions(samples: dict[str, Any], indices: np.ndarray, gain: dict[str, Any]) -> dict[str, np.ndarray]:
    meta = [samples["meta"][int(i)] for i in indices]
    n = len(meta)
    cv_ins = np.zeros((n, 3, 2), dtype=float)
    cv_can = np.zeros_like(cv_ins)
    ctrv_ins = np.zeros_like(cv_ins)
    ctrv_steer = np.zeros_like(cv_ins)
    beta0 = gain["intercept_rad_s"]
    beta1 = gain["slope_rad_s_per_speed_times_raw_steer"]
    for j, row in enumerate(meta):
        theta = row["heading_rad"]
        v = row["speed_mps"]
        # Convert INSPVA EN velocity at t0 into anchor body axes.
        ve, vn = row["east_velocity_mps"], row["north_velocity_mps"]
        cv_ins_forward = ve * math.sin(theta) + vn * math.cos(theta)
        cv_ins_right = ve * math.cos(theta) - vn * math.sin(theta)
        omega_ins = row["yaw_rate_bwd_rad_s"]
        omega_can = beta0 + beta1 * row["steer_product"]
        omega_can = float(np.clip(omega_can, -1.5, 1.5))
        for k, h in enumerate(HORIZONS):
            cv_ins[j, k] = [cv_ins_forward * h, cv_ins_right * h]
            cv_can[j, k] = [v * h, 0.0]
            for dest, omega in ((ctrv_ins, omega_ins), (ctrv_steer, omega_can)):
                if abs(omega) < 1e-6:
                    dest[j, k] = [v * h, 0.0]
                else:
                    a = omega * h
                    dest[j, k] = [v / omega * math.sin(a), v / omega * (1.0 - math.cos(a))]
    return {
        "INS_velocity_ZOH": cv_ins,
        "CAN_speed_straight": cv_can,
        "CTRV_past_INS_yaw_rate_CAN_speed": ctrv_ins,
        "CTRV_fit_CAN_steering_gain": ctrv_steer,
    }


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, float | int]]:
    rows = []
    for h_idx, horizon in enumerate(HORIZONS):
        error = np.linalg.norm(y_true[:, h_idx, :] - y_pred[:, h_idx, :], axis=1)
        rows.append({
            "horizon_sec": int(horizon), "n_anchors": int(len(error)),
            "fde_mean_m": float(np.mean(error)), "fde_median_m": float(np.median(error)),
            "fde_p90_m": float(np.quantile(error, 0.90)),
        })
    return rows


def paired_test_diagnostics(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> list[dict[str, float | int]]:
    """Descriptive paired differences only; intentionally no p-values/CI for one run."""
    out = []
    for h_idx, horizon in enumerate(HORIZONS):
        ea = np.linalg.norm(y_true[:, h_idx] - pred_a[:, h_idx], axis=1)
        eb = np.linalg.norm(y_true[:, h_idx] - pred_b[:, h_idx], axis=1)
        d = ea - eb
        out.append({
            "horizon_sec": int(horizon), "mean_error_a_minus_b_m": float(np.mean(d)),
            "median_error_a_minus_b_m": float(np.median(d)),
            "fraction_b_lower_error": float(np.mean(d > 0)),
        })
    return out
def run_experiment(bag: Path, dataset_label: str = "CRISAL 2025, CR2 acquisition") -> dict[str, Any]:
    from sklearn.neural_network import MLPRegressor

    print(f"Reading selected topics from {bag} (18 GB bags can take a while)...", flush=True)
    raw = read_topics(bag)
    print("Selected-topic decoding complete; constructing causal samples...", flush=True)
    samples = build_samples(raw)
    meta = samples["meta"]
    split_masks = {name: np.asarray([row["split"] == name for row in meta], dtype=bool) for name in ("fit", "validation", "test")}
    split_indices = {name: np.flatnonzero(mask) for name, mask in split_masks.items()}
    split_counts = {name: int(len(idx)) for name, idx in split_indices.items()}
    if split_counts["fit"] < 60 or split_counts["validation"] < 20 or split_counts["test"] < 20:
        raise RuntimeError(f"Insufficient anchors for the predeclared temporal split: {split_counts}")

    fit_idx, val_idx, test_idx = (split_indices[name] for name in ("fit", "validation", "test"))
    print(f"Valid anchors by temporal split: {split_counts}; fitting models...", flush=True)
    gain = fit_steering_yaw_gain(samples, split_masks["fit"])
    physics_fit = make_physics_predictions(samples, fit_idx, gain)
    physics_val = make_physics_predictions(samples, val_idx, gain)
    physics_test = make_physics_predictions(samples, test_idx, gain)
    y = samples["Y_body"]
    X = samples["X"]
    base_fit = physics_fit["CTRV_fit_CAN_steering_gain"]
    base_val = physics_val["CTRV_fit_CAN_steering_gain"]
    base_test = physics_test["CTRV_fit_CAN_steering_gain"]

    # Residual MLP is fitted only on the first chronological 60% segment.
    x_mean, x_scale = X[fit_idx].mean(axis=0), X[fit_idx].std(axis=0)
    x_scale[x_scale < 1e-8] = 1.0
    y_fit_flat = (y[fit_idx] - base_fit).reshape(len(fit_idx), -1)
    y_mean, y_scale = y_fit_flat.mean(axis=0), y_fit_flat.std(axis=0)
    y_scale[y_scale < 1e-8] = 1.0
    mlp = MLPRegressor(
        hidden_layer_sizes=(24, 24), activation="relu", solver="adam", alpha=1.0,
        batch_size=min(64, len(fit_idx)), learning_rate_init=0.002,
        max_iter=800, tol=1e-4, early_stopping=False, random_state=RANDOM_STATE,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mlp.fit((X[fit_idx] - x_mean) / x_scale, (y_fit_flat - y_mean) / y_scale)
    mlp_warning_messages = [str(w.message) for w in caught]

    def residual_prediction(indices: np.ndarray, base: np.ndarray) -> np.ndarray:
        residual_flat = mlp.predict((X[indices] - x_mean) / x_scale) * y_scale + y_mean
        return base + residual_flat.reshape(len(indices), 3, 2)

    pred_mlp_val = residual_prediction(val_idx, base_val)
    val_resid = pred_mlp_val - base_val
    val_error = y[val_idx] - base_val
    denom = float(np.sum(val_resid ** 2))
    lambda_raw = float(np.sum(val_resid * val_error) / denom) if denom > 1e-12 else 0.0
    lambda_clipped = float(np.clip(lambda_raw, 0.0, 1.5))
    pred_mlp_test = residual_prediction(test_idx, base_test)

    predictions: dict[str, np.ndarray] = {
        "INS_velocity_ZOH": physics_test["INS_velocity_ZOH"],
        "CAN_speed_straight": physics_test["CAN_speed_straight"],
        "CTRV_past_INS_yaw_rate_CAN_speed": physics_test["CTRV_past_INS_yaw_rate_CAN_speed"],
        "CTRV_fit_CAN_steering_gain": base_test,
        "Residual_MLP_lambda_1": pred_mlp_test,
        "Residual_MLP_validation_lambda": base_test + lambda_clipped * (pred_mlp_test - base_test),
        "Residual_MLP_validation_lambda_0.5": base_test + 0.5 * (pred_mlp_test - base_test),
    }
    shrink_factors = {}
    n_val = len(val_idx)
    for kappa in SHRINK_KAPPAS:
        factor = n_val / (n_val + kappa)
        shrink_factors[str(int(kappa))] = float(factor)
        predictions[f"Residual_MLP_validation_shrunk_kappa_{int(kappa)}"] = base_test + (lambda_clipped * factor) * (pred_mlp_test - base_test)

    test_y = y[test_idx]
    model_results = {name: metric_summary(test_y, pred) for name, pred in predictions.items()}
    comparison_results = {
        "full_residual_minus_physics": paired_test_diagnostics(test_y, predictions["Residual_MLP_lambda_1"], predictions["CTRV_fit_CAN_steering_gain"]),
        "validation_lambda_minus_full_residual": paired_test_diagnostics(test_y, predictions["Residual_MLP_validation_lambda"], predictions["Residual_MLP_lambda_1"]),
        "validation_lambda_minus_physics": paired_test_diagnostics(test_y, predictions["Residual_MLP_validation_lambda"], predictions["CTRV_fit_CAN_steering_gain"]),
    }

    def qstats(values: np.ndarray) -> dict[str, float]:
        a = np.asarray(values, dtype=float)
        return {"p50": float(np.quantile(a, 0.50)), "p95": float(np.quantile(a, 0.95)), "max": float(np.max(a))}

    t_ins_abs = raw["t_ins_abs"]
    t_speed_abs = raw["t_speed_abs"]
    t_steer_abs = raw["t_steer_abs"]
    # Nearest header-time distance is diagnostic only; inputs themselves are as-of/ZOH.
    def nearest_skew(source: np.ndarray) -> dict[str, float | int]:
        idx = np.searchsorted(t_ins_abs, source)
        idx = np.clip(idx, 0, len(t_ins_abs) - 1)
        prev = np.maximum(idx - 1, 0)
        d = np.minimum(np.abs(source - t_ins_abs[idx]), np.abs(source - t_ins_abs[prev]))
        return {"n": int(len(d)), "p50_sec": float(np.quantile(d, 0.50)), "p95_sec": float(np.quantile(d, 0.95)), "max_sec": float(np.max(d))}

    can_steer_train = np.asarray([m["steering_raw_corrected"] for m, keep in zip(meta, split_masks["fit"]) if keep])
    test_meta = [meta[int(i)] for i in test_idx]
    return {
        "schema_version": 1,
        "experiment": f"{dataset_label} single-acquisition external feasibility pilot",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": dataset_label,
            "doi": "10.57745/YCXRWF",
            "bag_path": str(bag.resolve()), "bag_bytes": bag.stat().st_size,
            "ins_status_counts_all": raw["ins_status_counts"],
            "raw_topic_message_counts": raw["raw_counts"],
            "header_time_fallback_counts": raw["header_fallbacks"],
            "header_minus_bag_time_sec": raw["header_minus_bag_time_sec"],
            "invalid_value_counts": raw["invalid_value_counts"],
            "duplicate_timestamp_counts": raw["duplicate_timestamp_counts"],
            "usable_ins_solution_good": int(len(t_ins_abs)),
            "usable_can_speed": int(len(t_speed_abs)), "usable_can_steering": int(len(t_steer_abs)),
            "ins_good_duration_sec": float(t_ins_abs[-1] - t_ins_abs[0]),
            "can_to_ins_nearest_header_skew": {
                "speed": nearest_skew(t_speed_abs), "steering": nearest_skew(t_steer_abs),
            },
            "input_zoh_age_at_all_accepted_anchors_sec": {
                "speed": qstats(np.asarray([m["can_speed_age_sec"] for m in meta])),
                "steering": qstats(np.asarray([m["can_steering_age_sec"] for m in meta])),
            },
            "ins_interval_sec": qstats(np.diff(t_ins_abs)),
            "position_projection": "WGS84 geodetic latitude/longitude/ellipsoidal height to local ENU using first good INS sample as origin",
            "heading": "NovAtel INSPVA azimuth in degrees clockwise from North, converted to radians; only the anchor heading is used",
            "speed_unit_conversion": "CAN message definition labels speed in km/h; divided by 3.6 to m/s",
            "steering_unit": "angle + angle_offset retained in message-native raw units; no wheel-angle or steering-ratio interpretation",
        },
        "protocol": {
            "causality": "CAN speed/steering are previous-sample zero-order-held at or before each anchor; history features are backward as-of; yaw-rate diagnostic uses backward INS azimuth difference only",
            "prediction_horizons_sec": [int(h) for h in HORIZONS],
            "anchor_stride_ins_samples": ANCHOR_STRIDE_SAMPLES,
            "approx_anchor_stride_sec": 0.5,
            "max_can_age_sec": MAX_CAN_AGE_SEC,
            "max_ins_backdiff_sec": MAX_INS_BACKDIFF_SEC,
            "max_future_label_bracket_sec": MAX_LABEL_BRACKET_SEC,
            "split": "single acquisition, chronological 60/20/20 by INS-good time; 3 s label guard between segments; no random window split",
            "split_boundaries_relative_to_first_good_ins_sec": samples["time_bounds"],
            "anchor_counts": split_counts,
            "overlap_warning": "0.5 s spaced anchors have overlapping future windows and are not independent statistical units; no p-values, confidence intervals, or population-level inference are reported",
            "physics_prior": "CTRV with anchor CAN speed ZOH and a steering-to-yaw-rate mapping fit on the fit segment only; steering units/ratio are not assumed",
            "residual_model": "Standardized 8-feature MLP (24,24), fit-only, target is body-frame endpoint displacement residual at 1/2/3 s",
            "lambda_calibration": "single closed-form least-squares scalar estimated on the chronological validation segment by minimizing pooled squared 2D endpoint-displacement error, clipped to [0,1.5], then frozen for test",
            "reported_metric": "mean Euclidean endpoint displacement error (FDE); this differs from the squared-error objective used to estimate lambda",
            "shrinkage_sensitivity": "n_val/(n_val+kappa) with kappa 100/300/800 is included as a descriptive sensitivity only; n_val is overlapping anchor count and kappa is not transferable to this dataset",
            "random_state": RANDOM_STATE,
            "mlp_training_warnings": mlp_warning_messages,
        },
        "steering_to_yaw_gain_fit_only": gain,
        "validation_lambda": {
            "raw_unclipped": lambda_raw, "clipped_0_1.5": lambda_clipped,
            "validation_anchor_count_not_independent": int(n_val),
            "shrink_factors_by_kappa": shrink_factors,
        },
        "test_metrics_by_method": model_results,
        "test_paired_differences_descriptive": comparison_results,
        "test_anchor_summary": {
            "n_anchors": len(test_meta),
            "time_sec_from_start_p05_p50_p95": [float(v) for v in np.quantile([m["time_sec"] for m in test_meta], [0.05, 0.50, 0.95])],
            "speed_mps_p05_p50_p95": [float(v) for v in np.quantile([m["speed_mps"] for m in test_meta], [0.05, 0.50, 0.95])],
            "steering_raw_corrected_p05_p50_p95": [float(v) for v in np.quantile([m["steering_raw_corrected"] for m in test_meta], [0.05, 0.50, 0.95])],
        },
        "allowed_conclusion": "single-run exploratory feasibility only; this run can test whether parsing, causal anchor construction, and time-forward evaluation are operational, but cannot establish cross-acquisition generalization or confirmatory evidence for the lambda-shrinkage claim",
        "limitations": [
            "Only one approximately five-minute acquisition is evaluated; overlapping anchors do not increase the number of independent runs.",
            "INSPVA is a fused navigation solution and is not an independent ground-truth system; reported errors are conditional on that reference.",
            "CAN steering is a steering-wheel message in undocumented/native units here, not verified road-wheel angle; the fitted gain is an empirical mapping.",
            "No hardware synchronization certification is available; nearest header-time skew is descriptive, and all CAN inputs are causal as-of samples.",
            "The validation lambda and shrinkage sensitivity are exploratory and should not be pooled with the existing comma/nuScenes estimates without a protocol-matched multi-acquisition study.",
        ],
        "software": {"python": platform.python_version(), "rosbags": package_version("rosbags"), "numpy": package_version("numpy"), "scikit_learn": package_version("scikit-learn")},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def write_report(result: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        f"# {result['dataset']['name']} 单采集外部可行性试验", "",
        f"- 生成时间（UTC）：{result['generated_utc']}",
        f"- bag：`{result['dataset']['bag_path']}`",
        f"- DOI：`{result['dataset']['doi']}`；文件大小：{result['dataset']['bag_bytes']:,} bytes",
        "- **定位：单次采集探索性试验，不是跨采集确认性验证。** INSPVA 为融合导航参考，不称为独立真值。", "",
        "## 数据与因果对齐", "",
        f"- INSPVA GOOD：{result['dataset']['usable_ins_solution_good']:,}；CAN speed：{result['dataset']['usable_can_speed']:,}；CAN steering：{result['dataset']['usable_can_steering']:,}。",
        f"- 按时间顺序 60/20/20 切分，有 3 s 边界隔离；有效锚点 fit/validation/test = {result['protocol']['anchor_counts']}。锚点约每 0.5 s 一个，未来窗口重叠。",
        "- CAN 速度依据消息定义从 km/h 换为 m/s。方向盘角使用 `angle + angle_offset` 的消息原生单位；未假定其为前轮转角或应用转向比。",
        "- 每个锚点只取时间戳不晚于锚点的 CAN 样本（ZOH）；历史特征同样使用 as-of 对齐。未来位置只用于标签。", "",
        "## 测试段结果", "",
        "FDE 为测试锚点平均终点距离误差；表中无显著性检验或置信区间。",
        "", "| 方法 | 1 s FDE (m) | 2 s FDE (m) | 3 s FDE (m) |", "|---|---:|---:|---:|",
    ]
    for name, rows in result["test_metrics_by_method"].items():
        cells = [f"{row['fde_mean_m']:.3f}" for row in rows]
        lines.append(f"| `{name}` | {cells[0]} | {cells[1]} | {cells[2]} |")
    metric = result["test_metrics_by_method"]
    def fde(method: str, horizon_sec: int) -> float:
        return next(row["fde_mean_m"] for row in metric[method] if row["horizon_sec"] == horizon_sec)
    gains = [fde("CTRV_fit_CAN_steering_gain", h) - fde("Residual_MLP_lambda_1", h) for h in (1, 2, 3)]
    lam = result["validation_lambda"]
    test_lambda3 = fde("Residual_MLP_validation_lambda", 3)
    test_full3 = fde("Residual_MLP_lambda_1", 3)
    kappa3 = {k: fde(f"Residual_MLP_validation_shrunk_kappa_{k}", 3) for k in (100, 300, 800)}
    lines += [
        "", "## 结果解读", "",
        f"- 本次测试时段的点估计中，满残差 MLP 相对拟合转向增益 CTRV 的 1/2/3 s FDE 分别低 {gains[0]:.3f}/{gains[1]:.3f}/{gains[2]:.3f} m；这是单采集描述性结果，不作总体显著性推断。",
        f"- Validation 得到 λ={lam['clipped_0_1.5']:.4f}。3 s FDE 下，验证集 λ 方案为 {test_lambda3:.3f} m，较 λ=1 的 {test_full3:.3f} m 低 {test_full3-test_lambda3:.3f} m；这只是单次采集的描述性差异，不能单独确认校准收益。",
        f"- 按本试验的 n/(n+κ) 规则，κ=100/300/800 时 3 s FDE 分别为 {kappa3[100]:.3f}/{kappa3[300]:.3f}/{kappa3[800]:.3f} m；κ=100 与 λ=1 几乎持平，κ=300/800 更差。κ 仅作敏感性展示，不视为已迁移或已验证的超参数。", "",
        "## λ 校准与结果解读", "",
        f"- Validation 闭式 λ：raw={lam['raw_unclipped']:.4f}，截断后={lam['clipped_0_1.5']:.4f}；仅在 validation 估计，test 固定评估。",
        f"- κ 敏感性缩放因子：`{lam['shrink_factors_by_kappa']}`。由于它使用重叠锚点计数且 κ 来自其他数据协议，仅作敏感性展示，不解释为有效独立样本收缩。",
        "- 可配对误差差值为描述统计；锚点重叠，不将锚点当独立样本，也不报告 p 值、CI 或跨运行泛化结论。", "",
        "## 允许的结论", "",
        f"本结果仅表明：在 {result['dataset']['name']} 的这一段采集中，数据解析、因果锚点构造、按时间留出评估及模型代码能够运行。它不能证明跨采集泛化，也不能作为 λ 收缩主张的确认性证据。", "",
        "## 限制", "",
        "- 仅评估一次约五分钟的采集；重叠锚点不会增加独立采集次数。",
        "- INSPVA 是融合导航解算，不是独立真值系统；误差仅相对于该参考轨迹计算。",
        "- CAN 转向信号仍是此处未完成独立标定的原生方向盘角量；拟合增益只能解释为经验映射，不能视作已验证的前轮转角或转向比。",
        "- 没有硬件级同步认证；最近邻时钟偏差仅为描述统计。模型输入仍按锚点时刻之前的消息进行因果对齐。",
        "- λ 的闭式估计以验证集二维位移平方误差为目标，而报告 FDE 使用欧氏端点距离；两者优化目标不同。",
        "- 本数据的 validation λ 与 κ 敏感性为探索性结果；未与 comma/nuScenes 数值合并，也不能据此声称跨采集或跨平台收益。", "",
        f"- 脚本 SHA-256：`{result['script_sha256']}`",
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Exploratory, causally aligned CRISAL single-acquisition external feasibility pilot")
    parser.add_argument("--bag", type=Path, default=DEFAULT_BAG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset-label", default="CRISAL 2025, CR2 acquisition")
    parser.add_argument("--output-stem", default="crisal_cr2_lambda_pilot")
    args = parser.parse_args()
    bag = args.bag.expanduser().resolve()
    if not bag.is_file():
        raise SystemExit(f"ROS bag not found: {bag}")
    if Path(args.output_stem).name != args.output_stem:
        raise SystemExit("--output-stem must be a filename stem, not a path")
    result = run_experiment(bag, dataset_label=args.dataset_label)
    json_path = args.out_dir / f"{args.output_stem}_results.json"
    md_path = args.out_dir / f"{args.output_stem}_zh.md"
    write_report(result, json_path, md_path)
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    print(f"Anchors: {result['protocol']['anchor_counts']}; validation lambda: {result['validation_lambda']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
