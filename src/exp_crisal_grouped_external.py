from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import sys
import warnings
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
DATA_ROOT = REPO / "external_data" / "CRISAL_2025"
OUT_DIR = REPO / "results" / "crisal_external_validation"
CACHE_DIR = DATA_ROOT / "cache" / "grouped_samples"
PILOT_PATH = HERE.with_name("exp_crisal_external_pilot.py")
FIT_ACQUISITIONS = 4
VALIDATION_ACQUISITIONS = 2
MODEL_SEEDS = (29, 73, 2026)
KAPPAS_ACQUISITION_UNITS = (0.5, 1.0, 2.0)
MIN_VALID_ANCHORS = 80
BOOTSTRAP_REPLICATES = 20000
CACHE_SCHEMA = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def safe_id(text: str) -> str:
    keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    return "".join(c if c in keep else "_" for c in text).strip("_")


def discover_bags(data_root: Path, out_dir: Path) -> list[dict[str, Any]]:
    manifest = out_dir / "crisal_extraction_manifest.json"
    candidates: list[dict[str, Any]] = []
    manifest_paths: set[str] = set()
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        for row in payload.get("results", []):
            if row.get("status") not in {"extracted", "already_extracted"} or not row.get("bag_path"):
                continue
            bag = Path(row["bag_path"])
            if not bag.is_file():
                continue
            archive = Path(row["filename"]).stem
            acq = archive.removeprefix("locsp_")
            candidates.append({"acquisition_id": acq, "bag_path": bag})
            manifest_paths.add(str(bag.resolve()).casefold())
    # The extraction manifest may be incomplete or stale. Include every extracted bag
    # not already represented there so a valid acquisition is not silently omitted.
    for bag in (data_root / "extracted").rglob("*.bag"):
        if str(bag.resolve()).casefold() in manifest_paths:
            continue
        acq = bag.stem if bag.parent.name == "CR2" else bag.parent.name
        candidates.append({"acquisition_id": acq, "bag_path": bag})
    # Prefer manifest-selected canonical extracts; remove exact filename/size duplicates.
    seen: set[tuple[str, int]] = set()
    unique: list[dict[str, Any]] = []
    for row in sorted(candidates, key=lambda r: (str(r["bag_path"].parent.name == "CR2"), str(r["bag_path"]))):
        bag = Path(row["bag_path"])
        key = (bag.name, bag.stat().st_size)
        if key in seen:
            continue
        seen.add(key)
        unique.append({**row, "acquisition_id": safe_id(str(row["acquisition_id"]))})
    return sorted(unique, key=lambda r: r["acquisition_id"])

def load_or_build_samples(row: dict[str, Any], cache_dir: Path, pilot: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    bag = Path(row["bag_path"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{row['acquisition_id']}.npz"
    source = {"bag_name": bag.name, "bag_bytes": bag.stat().st_size,
              "bag_mtime_ns": bag.stat().st_mtime_ns, "pilot_sha256": file_sha256(PILOT_PATH),
              "cache_schema": CACHE_SCHEMA}
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as z:
                cached = json.loads(str(z["cache_meta"].item()))
                if cached == source:
                    samples = {"X": z["X"], "Y_body": z["Y_body"], "meta": json.loads(str(z["sample_meta"].item()))}
                    diag = json.loads(str(z["diagnostics"].item()))
                    return samples, diag
        except Exception:
            pass
    print(f"[decode] {row['acquisition_id']}: {bag}", flush=True)
    raw = pilot.read_topics(bag)
    samples = pilot.build_samples(raw, chronological_split=False)
    meta = samples["meta"]
    diagnostics = {
        "acquisition_id": row["acquisition_id"], "bag_path": str(bag),
        "bag_bytes": bag.stat().st_size, "usable_ins_solution_good": int(len(raw["t_ins_abs"])),
        "ins_good_duration_sec": float(raw["t_ins_abs"][-1] - raw["t_ins_abs"][0]),
        "raw_topic_message_counts": raw["raw_counts"], "ins_status_counts": raw["ins_status_counts"],
        "header_time_fallback_counts": raw["header_fallbacks"],
        "invalid_value_counts": raw["invalid_value_counts"],
        "duplicate_timestamp_counts": raw["duplicate_timestamp_counts"],
        "valid_anchor_count": int(len(meta)),
        "speed_zoh_age_p95_sec": float(np.quantile([m["can_speed_age_sec"] for m in meta], .95)),
        "steering_zoh_age_p95_sec": float(np.quantile([m["can_steering_age_sec"] for m in meta], .95)),
        "label_bracket_p95_sec": float(np.quantile([m["max_label_bracket_sec"] for m in meta], .95)),
        "all_features_finite": bool(np.isfinite(samples["X"]).all()),
        "all_labels_finite": bool(np.isfinite(samples["Y_body"]).all()),
        "pilot_split_mode": "whole_acquisition; no within-acquisition random or chronological train/test split",
    }
    np.savez_compressed(cache_path, X=samples["X"], Y_body=samples["Y_body"],
        cache_meta=np.asarray(json.dumps(source, sort_keys=True)),
        sample_meta=np.asarray(json.dumps(meta, ensure_ascii=False)),
        diagnostics=np.asarray(json.dumps(diagnostics, ensure_ascii=False)))
    return samples, diagnostics


def equal_group_weights(group_ids: list[str]) -> np.ndarray:
    groups = np.asarray(group_ids, dtype=str)
    unique = np.unique(groups)
    w = np.zeros(len(groups), dtype=float)
    for group in unique:
        mask = groups == group
        w[mask] = 1.0 / (len(unique) * int(mask.sum()))
    return w


def weighted_mean_scale(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    mean = np.sum(x * w[:, None], axis=0)
    var = np.sum(((x - mean) ** 2) * w[:, None], axis=0)
    scale = np.sqrt(np.maximum(var, 0.0))
    scale[scale < 1e-8] = 1.0
    return mean, scale


def fit_gain(acq_ids: list[str], samples_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from sklearn.linear_model import HuberRegressor
    q_parts, omega_parts, groups = [], [], []
    for aid in acq_ids:
        meta = samples_by_id[aid]["meta"]
        q_parts.append(np.asarray([m["steer_product"] for m in meta], dtype=float))
        omega_parts.append(np.asarray([m["yaw_rate_bwd_rad_s"] for m in meta], dtype=float))
        groups.extend([aid] * len(meta))
    q, omega = np.concatenate(q_parts), np.concatenate(omega_parts)
    model = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=500)
    model.fit(q.reshape(-1, 1), omega, sample_weight=equal_group_weights(groups))
    return {"intercept_rad_s": float(model.intercept_),
            "slope_rad_s_per_speed_times_raw_steer": float(model.coef_[0]),
            "n_fit_acquisitions": len(acq_ids),
            "fit_anchors_by_acquisition": {aid: len(samples_by_id[aid]["meta"]) for aid in acq_ids}}


def fit_residual_mlp(fit_ids: list[str], samples_by_id: dict[str, dict[str, Any]],
                     base_by_id: dict[str, np.ndarray], seed: int) -> tuple[Any, dict[str, np.ndarray], dict[str, Any]]:
    from sklearn.neural_network import MLPRegressor
    x_parts, target_parts, groups = [], [], []
    for aid in fit_ids:
        s = samples_by_id[aid]
        n = len(s["X"])
        x_parts.append(s["X"])
        target_parts.append((s["Y_body"] - base_by_id[aid]).reshape(n, -1))
        groups.extend([aid] * n)
    x, target = np.concatenate(x_parts), np.concatenate(target_parts)
    weights = equal_group_weights(groups)
    x_mean, x_scale = weighted_mean_scale(x, weights)
    y_mean, y_scale = weighted_mean_scale(target, weights)
    model = MLPRegressor(hidden_layer_sizes=(24, 24), activation="relu", solver="adam",
        alpha=1.0, batch_size=min(64, len(x)), learning_rate_init=0.002,
        max_iter=800, tol=1e-4, early_stopping=False, random_state=int(seed))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit((x - x_mean) / x_scale, (target - y_mean) / y_scale, sample_weight=weights)
    scaler = {"x_mean": x_mean, "x_scale": x_scale, "y_mean": y_mean, "y_scale": y_scale}
    diag = {"fit_anchor_count": int(len(x)), "fit_acquisitions_equal_weighted": list(fit_ids),
            "warning_messages": [str(w.message) for w in caught]}
    return model, scaler, diag


def predict_residual(model: Any, scaler: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    z = model.predict((x - scaler["x_mean"]) / scaler["x_scale"])
    return (z * scaler["y_scale"] + scaler["y_mean"]).reshape(len(x), 3, 2)


def equal_acquisition_lambda(val_ids: list[str], samples_by_id: dict[str, dict[str, Any]],
                             base_by_id: dict[str, np.ndarray], residual_by_id: dict[str, np.ndarray]) -> float:
    numerator = denominator = 0.0
    for aid in val_ids:
        y, base, residual = samples_by_id[aid]["Y_body"], base_by_id[aid], residual_by_id[aid]
        numerator += float(np.mean(np.sum(residual * (y - base), axis=(1, 2))))
        denominator += float(np.mean(np.sum(residual * residual, axis=(1, 2))))
    raw = numerator / denominator if denominator > 1e-12 else 0.0
    return float(np.clip(raw, 0.0, 1.5))


def per_horizon_fde(y: np.ndarray, pred: np.ndarray) -> list[float]:
    return [float(v) for v in np.mean(np.linalg.norm(y - pred, axis=2), axis=0)]


def bootstrap_mean_ci(values: list[float], rng: np.random.Generator) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    draws = rng.choice(x, size=(BOOTSTRAP_REPLICATES, len(x)), replace=True).mean(axis=1)
    return {"n_acquisitions": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)),
            "std_across_acquisitions": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "bootstrap_percentile_95_ci_mean": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]}


def run_grouped_experiment(acq_rows: list[dict[str, Any]], samples_by_id: dict[str, dict[str, Any]],
                           seed_list: tuple[int, ...]) -> dict[str, Any]:
    from itertools import combinations
    from exp_crisal_external_pilot import HORIZONS, make_physics_predictions

    ids = [r["acquisition_id"] for r in acq_rows]
    if len(ids) != 7:
        raise RuntimeError(f"This predeclared leave-one-acquisition-out protocol expects 7 eligible acquisitions; got {len(ids)}")
    n_train_val = len(ids) - 1
    n_fit = n_train_val - VALIDATION_ACQUISITIONS
    raw_draws: list[dict[str, Any]] = []
    warning_fit_count = 0
    for outer_index, test_id in enumerate(ids, 1):
        other_ids = [aid for aid in ids if aid != test_id]
        val_sets = list(combinations(other_ids, VALIDATION_ACQUISITIONS))
        print(f"[outer {outer_index}/7] test={test_id}; fit={n_fit}, validation={VALIDATION_ACQUISITIONS}; validation combinations={len(val_sets)}", flush=True)
        test_samples = samples_by_id[test_id]
        y_test = test_samples["Y_body"]
        test_idx = np.arange(len(test_samples["X"]), dtype=int)
        # Cache only gain-independent test baselines. The fitted-steering CTRV
        # prediction must be recomputed for every fit/validation split because
        # its gain is estimated from the current fit acquisitions.
        test_static_physics = make_physics_predictions(test_samples, test_idx, {
            "intercept_rad_s": 0.0,
            "slope_rad_s_per_speed_times_raw_steer": 0.0,
        })
        for split_index, val_tuple in enumerate(val_sets, 1):
            val_ids = list(val_tuple)
            fit_ids = [aid for aid in other_ids if aid not in val_tuple]
            gain = fit_gain(fit_ids, samples_by_id)
            base_by_id: dict[str, np.ndarray] = {}
            for aid in fit_ids + val_ids:
                si = samples_by_id[aid]
                base_by_id[aid] = make_physics_predictions(si, np.arange(len(si["X"])), gain)["CTRV_fit_CAN_steering_gain"]
            # Recompute the fitted-steering prior using this split's gain.
            # Reusing a prior fitted under another split would leak/deserialize
            # the wrong fit state into the held-out acquisition evaluation.
            test_physics_current = make_physics_predictions(test_samples, test_idx, gain)
            base_by_id[test_id] = test_physics_current["CTRV_fit_CAN_steering_gain"]
            for seed in seed_list:
                model, scaler, train_diag = fit_residual_mlp(fit_ids, samples_by_id, base_by_id, seed)
                residual_by_id = {aid: predict_residual(model, scaler, samples_by_id[aid]["X"]) for aid in val_ids}
                lam = equal_acquisition_lambda(val_ids, samples_by_id, base_by_id, residual_by_id)
                pred_full = base_by_id[test_id] + predict_residual(model, scaler, test_samples["X"])
                predictions: dict[str, np.ndarray] = {
                    "INS_velocity_ZOH": test_static_physics["INS_velocity_ZOH"],
                    "CAN_speed_straight": test_static_physics["CAN_speed_straight"],
                    "CTRV_past_INS_yaw_rate_CAN_speed": test_static_physics["CTRV_past_INS_yaw_rate_CAN_speed"],
                    "CTRV_fit_CAN_steering_gain": base_by_id[test_id],
                    "Residual_MLP_lambda_1": pred_full,
                    "Residual_MLP_validation_lambda": base_by_id[test_id] + lam * (pred_full - base_by_id[test_id]),
                }
                factors = {}
                for kappa in KAPPAS_ACQUISITION_UNITS:
                    factor = VALIDATION_ACQUISITIONS / (VALIDATION_ACQUISITIONS + kappa)
                    key = f"Residual_MLP_validation_lambda_shrunk_kappa_acq_{kappa:g}"
                    factors[str(kappa)] = factor
                    predictions[key] = base_by_id[test_id] + (lam * factor) * (pred_full - base_by_id[test_id])
                metrics = {name: per_horizon_fde(y_test, pred) for name, pred in predictions.items()}
                raw_draws.append({"test_acquisition": test_id, "fit_acquisitions": fit_ids,
                    "validation_acquisitions": val_ids, "validation_split_index": split_index,
                    "model_seed": int(seed), "validation_lambda_clipped": lam,
                    "lambda_shrink_factors": factors, "fit_gain": gain,
                    "fit_diagnostics": train_diag, "test_fde_by_method_horizons_m": metrics})
                warning_fit_count += int(bool(train_diag["warning_messages"]))
            if split_index % 5 == 0 or split_index == len(val_sets):
                print(f"  validation split {split_index}/{len(val_sets)} complete", flush=True)
    method_names = list(raw_draws[0]["test_fde_by_method_horizons_m"])
    by_acquisition: dict[str, Any] = {}
    for aid in ids:
        draws = [d for d in raw_draws if d["test_acquisition"] == aid]
        method_means = {}
        for method in method_names:
            matrix = np.asarray([d["test_fde_by_method_horizons_m"][method] for d in draws], dtype=float)
            method_means[method] = [float(v) for v in matrix.mean(axis=0)]
        by_acquisition[aid] = {"n_validation_seed_draws_averaged": len(draws),
            "mean_validation_lambda": float(np.mean([d["validation_lambda_clipped"] for d in draws])),
            "methods_fde_by_horizon_m": method_means}
    rng = np.random.default_rng(20261014)
    aggregate: dict[str, Any] = {}
    for method in method_names:
        aggregate[method] = {}
        for hi, horizon in enumerate(HORIZONS):
            vals = [by_acquisition[aid]["methods_fde_by_horizon_m"][method][hi] for aid in ids]
            aggregate[method][f"{float(horizon):g}s"] = bootstrap_mean_ci(vals, rng)
    paired: dict[str, Any] = {}
    full = "Residual_MLP_lambda_1"
    for method in method_names:
        if method == full:
            continue
        key = method + "_minus_full_residual"
        paired[key] = {}
        for hi, horizon in enumerate(HORIZONS):
            diffs = [by_acquisition[aid]["methods_fde_by_horizon_m"][method][hi] -
                     by_acquisition[aid]["methods_fde_by_horizon_m"][full][hi] for aid in ids]
            ci = bootstrap_mean_ci(diffs, rng)
            ci["acquisitions_better_than_full_residual"] = int(sum(d < 0 for d in diffs))
            ci["per_acquisition_differences_m"] = {aid: float(d) for aid, d in zip(ids, diffs)}
            paired[key][f"{float(horizon):g}s"] = ci
    return {"outer_test_acquisition_count": len(ids), "fit_acquisitions_per_outer_fold": n_fit,
        "validation_acquisitions_per_inner_split": VALIDATION_ACQUISITIONS,
        "validation_combinations_per_outer_fold": math.comb(n_train_val, VALIDATION_ACQUISITIONS),
        "model_seeds": list(seed_list), "raw_configuration_draw_count": len(raw_draws),
        "draws_with_mlp_convergence_warning": warning_fit_count, "method_names": method_names,
        "per_acquisition": by_acquisition, "aggregate_acquisition_level": aggregate,
        "paired_vs_full_residual_acquisition_level": paired, "raw_draws": raw_draws}


def write_qc(rows: list[dict[str, Any]], diagnostics: dict[str, dict[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    ids = [r["acquisition_id"] for r in rows]
    eligible = [aid for aid in ids if diagnostics[aid]["valid_anchor_count"] >= MIN_VALID_ANCHORS
                and diagnostics[aid]["all_features_finite"] and diagnostics[aid]["all_labels_finite"]]
    payload = {"schema_version": 1, "generated_utc": utc_now(), "dataset": "CRISAL 2025; seven acquisitions, same vehicle and collection day",
        "raw_data_root": str(DATA_ROOT), "expected_acquisition_count": 7,
        "discovered_acquisition_count": len(rows), "eligible_acquisition_count": len(eligible),
        "eligible_acquisition_ids": eligible, "excluded_acquisitions": [aid for aid in ids if aid not in eligible],
        "minimum_valid_anchors_per_acquisition": MIN_VALID_ANCHORS,
        "acquisitions": [diagnostics[aid] for aid in ids],
        "scope_limit": "Acquisition QC and sampleability only; INSPVA is fused navigation and this dataset does not establish independent ground-truth accuracy or cross-vehicle generalization."}
    out_dir.mkdir(parents=True, exist_ok=True)
    jp, mp = out_dir / "crisal_grouped_qc.json", out_dir / "CRISAL_GROUPED_QC_zh.md"
    jp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# CRISAL 多采集分组实验：采集级 QC", "", f"- 生成时间（UTC）：{payload['generated_utc']}",
        f"- 发现采集：{len(rows)}；达到预设 QC 门槛：{len(eligible)}；门槛：每采集有效因果锚点 ≥ {MIN_VALID_ANCHORS}",
        "- 外层统计单位：采集；不把重叠锚点、验证组合或模型 seed 作为独立统计单位。", "",
        "| 采集 ID | INS 时长 (s) | 有效锚点 | 95% CAN speed age (s) | 95% CAN steering age (s) | 标签括区 p95 (s) | QC |",
        "|---|---:|---:|---:|---:|---:|:---:|"]
    for aid in ids:
        d = diagnostics[aid]
        ok = aid in eligible
        if "decode_error" in d:
            lines.append(f"| `{aid}` | — | 0 | — | — | — | 排除（{d['decode_error']}） |")
        else:
            lines.append(f"| `{aid}` | {d['ins_good_duration_sec']:.1f} | {d['valid_anchor_count']} | {d['speed_zoh_age_p95_sec']:.3f} | {d['steering_zoh_age_p95_sec']:.3f} | {d['label_bracket_p95_sec']:.3f} | {'通过' if ok else '排除'} |")
    lines += ["", "## 边界", "", payload["scope_limit"]]
    mp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jp, mp


def write_results(result: dict[str, Any], out_dir: Path) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "crisal_grouped_external_results.json"
    md_path = out_dir / "CRISAL_GROUPED_EXTERNAL_zh.md"
    csv_path = out_dir / "crisal_grouped_per_acquisition.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ids = list(result["per_acquisition"])
    methods = result["method_names"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["acquisition_id", "method", "fde_1s_m", "fde_2s_m", "fde_3s_m", "mean_validation_lambda", "draws_averaged"])
        for aid in ids:
            row = result["per_acquisition"][aid]
            for method in methods:
                vals = row["methods_fde_by_horizon_m"][method]
                writer.writerow([aid, method, *[f"{v:.8f}" for v in vals],
                    f"{row['mean_validation_lambda']:.8f}", row["n_validation_seed_draws_averaged"]])
    lines = ["# CRISAL 多采集外层留一采集验证", "", f"- 生成时间（UTC）：{result['generated_utc']}",
        f"- 独立外层测试采集数：{result['outer_test_acquisition_count']}",
        f"- 每折拟合/验证采集数：{result['fit_acquisitions_per_outer_fold']}/{result['validation_acquisitions_per_inner_split']}",
        f"- 每外层折验证组合数：{result['validation_combinations_per_outer_fold']}；MLP seeds：{result['protocol']['model_seeds']}",
        f"- 每折验证组合×seed：{result['protocol']['draws_per_test_acquisition']}；共配置数：{result['raw_configuration_draw_count']}",
        "- **统计单位是采集**：每个测试采集先平均其所有内层验证组合与模型 seed，再在 7 个采集级结果上汇总。",
        "- 区间为以采集为单位的 percentile bootstrap 95% CI（20,000 次）；n=7，区间只作小样本描述，不代表跨车型总体保证。", "",
        "## 主要结果：每采集宏平均 FDE", "",
        "| 方法 | 1 s (m), 均值 [95% CI] | 2 s (m), 均值 [95% CI] | 3 s (m), 均值 [95% CI] |", "|---|---:|---:|---:|"]
    for method in methods:
        cells = []
        for horizon in ("1s", "2s", "3s"):
            s = result["aggregate_acquisition_level"][method][horizon]
            lo, hi = s["bootstrap_percentile_95_ci_mean"]
            cells.append(f"{s['mean']:.3f} [{lo:.3f}, {hi:.3f}]")
        lines.append(f"| `{method}` | " + " | ".join(cells) + " |")
    lines += ["", "## 各采集配对差值", "", "差值定义为对照方法 FDE − 满残差 FDE；负值表示对照更好，正值表示满残差更好。以下为每采集配对差值的均值与采集级 bootstrap 区间。", "",
        "| 对照 − 满残差 | 1 s (m) | 2 s (m) | 3 s (m) | 更优采集数 (1/2/3 s) |", "|---|---:|---:|---:|---:|"]
    for key, horizons in result["paired_vs_full_residual_acquisition_level"].items():
        cells, counts = [], []
        for horizon in ("1s", "2s", "3s"):
            s = horizons[horizon]
            lo, hi = s["bootstrap_percentile_95_ci_mean"]
            cells.append(f"{s['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            counts.append(str(s["acquisitions_better_than_full_residual"]))
        lines.append(f"| `{key}` | " + " | ".join(cells) + f" | {' / '.join(counts)} |")
    lines += ["", "## λ 与收缩敏感性", "", "`κ_acq` 按独立验证采集数为单位；它不是原先重叠锚点计数版本的 κ，二者不可直接比较。λ 在两个验证采集上按采集等权的平方位移目标估计。", "",
        "## 可支持的解释范围", "", result["allowed_conclusion"], "", "## 复现记录", "",
        f"- 脚本 SHA-256：`{result['script_sha256']}`", f"- pilot 数据解析脚本 SHA-256：`{result['pilot_script_sha256']}`",
        f"- Python：`{result['software']['python']}`；NumPy：`{result['software']['numpy']}`；scikit-learn：`{result['software']['scikit_learn']}`；rosbags：`{result['software']['rosbags']}`",
        f"- JSON：`{json_path.name}`；逐采集 CSV：`{csv_path.name}`", "", "## 限制", "",
        "CRISAL 的七次记录来自同日、同一辆 Renault Zoe，采集间条件仍可能相关。INSPVA 是融合导航输出而非独立真值；本文指标是相对于该参考轨迹的误差。CAN 转向角保持消息原生单位，增益是经验映射。结论不构成跨车型/跨平台或独立定位真值验证。"]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path, csv_path



def import_pilot_module() -> Any:
    """Import the shared CRISAL decoder/sampler from the repository source tree."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("exp_crisal_external_pilot", PILOT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load pilot module: {PILOT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp_crisal_external_pilot"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ap = argparse.ArgumentParser(description="CRISAL leave-one-acquisition-out grouped external validation.")
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--allow-nonseven", action="store_true", help="Allow QC discovery to report a non-seven set; the experiment still requires exactly seven eligible acquisitions.")
    ap.add_argument("--skip-experiment", action="store_true", help="Decode/cache and write QC only.")
    args = ap.parse_args()
    data_root = args.data_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    cache_dir = (args.cache_dir or (data_root / "cache" / "grouped_samples")).expanduser().resolve()
    if not data_root.exists():
        raise SystemExit(f"CRISAL data root does not exist: {data_root}")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Use the shared pilot parser without executing its command-line entry point.
    pilot = import_pilot_module()
    rows = discover_bags(data_root, out_dir)
    if not rows:
        raise SystemExit(f"No extracted CRISAL .bag files found below {data_root / 'extracted'}")
    print(f"[discover] bags={len(rows)}", flush=True)
    samples_by_id: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    kept_rows: list[dict[str, Any]] = []
    for row in rows:
        try:
            samples, diag = load_or_build_samples(row, cache_dir, pilot)
            samples_by_id[row["acquisition_id"]] = samples
            diagnostics[row["acquisition_id"]] = diag
            kept_rows.append(row)
            print(f"[qc] {row['acquisition_id']}: anchors={diag['valid_anchor_count']}, duration={diag['ins_good_duration_sec']:.1f}s", flush=True)
        except Exception as exc:
            diagnostics[row["acquisition_id"]] = {
                "acquisition_id": row["acquisition_id"], "bag_path": str(row["bag_path"]),
                "decode_error": f"{type(exc).__name__}: {exc}", "valid_anchor_count": 0,
                "all_features_finite": False, "all_labels_finite": False,
            }
            kept_rows.append(row)
            print(f"[qc] {row['acquisition_id']}: FAILED {type(exc).__name__}: {exc}", flush=True)
    write_qc(kept_rows, diagnostics, out_dir)
    eligible = [r for r in kept_rows if diagnostics[r["acquisition_id"]].get("valid_anchor_count", 0) >= MIN_VALID_ANCHORS
                and diagnostics[r["acquisition_id"]].get("all_features_finite", False)
                and diagnostics[r["acquisition_id"]].get("all_labels_finite", False)
                and r["acquisition_id"] in samples_by_id]
    if args.skip_experiment:
        print(json.dumps({"qc": str(out_dir / "crisal_grouped_qc.json"), "eligible": len(eligible)}, ensure_ascii=False))
        return 0
    if len(eligible) != 7:
        raise SystemExit(f"Formal grouped experiment not started: eligible acquisitions={len(eligible)}, required exactly 7. See {out_dir / 'CRISAL_GROUPED_QC_zh.md'}")
    # Preserve the predeclared acquisition order from the discovery manifest/path list.
    eligible_samples = {r["acquisition_id"]: samples_by_id[r["acquisition_id"]] for r in eligible}
    result = run_grouped_experiment(eligible, eligible_samples, tuple(MODEL_SEEDS))
    result.update({
        "schema_version": 1,
        "generated_utc": utc_now(),
        "dataset": "CRISAL 2025",
        "data_root": str(data_root),
        "protocol": {
            "outer_split": "leave_one_acquisition_out",
            "fit_acquisitions": FIT_ACQUISITIONS,
            "validation_acquisitions": VALIDATION_ACQUISITIONS,
            "validation_combinations_per_test_acquisition": math.comb(6, VALIDATION_ACQUISITIONS),
            "model_seeds": list(MODEL_SEEDS),
            "draws_per_test_acquisition": math.comb(6, VALIDATION_ACQUISITIONS) * len(MODEL_SEEDS),
            "total_expected_draws": 7 * math.comb(6, VALIDATION_ACQUISITIONS) * len(MODEL_SEEDS),
            "causal_protocol": "backward-difference yaw rate; anchor-time CAN speed/steering ZOH; no future samples",
            "statistics_unit": "acquisition after averaging inner validation combinations and model seeds",
            "kappa_unit": "independent validation acquisitions",
        },
        "qc": {"eligible_acquisition_ids": [r["acquisition_id"] for r in eligible], "minimum_valid_anchors": MIN_VALID_ANCHORS},
        "software": {"python": platform.python_version(), "numpy": package_version("numpy"), "scikit_learn": package_version("scikit-learn"), "rosbags": package_version("rosbags")},
        "script_sha256": file_sha256(HERE),
        "pilot_script_sha256": file_sha256(PILOT_PATH),
        "allowed_conclusion": "在同日、同一辆 Renault Zoe 的七次 CRISAL 采集中，本实验可检验按采集分组的时间外部验证与离线收缩系数的稳定性；不能据此宣称跨车型、跨平台或独立定位真值泛化。",
    })
    paths = write_results(result, out_dir)
    print(json.dumps({"results": [str(p) for p in paths], "draws": result["raw_configuration_draw_count"], "acquisitions": result["outer_test_acquisition_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
