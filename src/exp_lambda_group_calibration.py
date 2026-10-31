"""Grouped validation-calibration audit for the causal residual trust coefficient.

Compares, on identical nested group splits and identical fitted physics+MLP models:
  * lambda=0 and lambda=1 references;
  * closed-form squared endpoint-error lambda, clipped to [0, 1.5];
  * the same estimator with the pre-existing empirical anchor-count shrinkage;
  * lambda selected by validation group-mean FDE;
  * validation group-mean FDE plus a 1.96-SE penalty (descriptive conservative rule).

The held-out outer test groups are never used for calibration or model selection.
Primary uncertainty is paired bootstrap over routes (comma) or scenes (nuScenes),
after averaging repeated subset-seed predictions within each test group.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import spearmanr, wilcoxon
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
DEFAULT_OUT = RESULTS / "exp_lambda_group_calibration_results.json"
REPORT = RESULTS / "EXP_LAMBDA_GROUP_CALIBRATION_zh.md"

sys.path.insert(0, str(HERE))
import comma_loader as C
import causal_protocol as P
import causal_eval as E
import exp_causal_nuscenes as NU
import exp_lambda_nusc_causal as NC

HORIZON_S = 3.0
DT_COMMA = 0.05
DT_NUSC = 0.02
HS_COMMA = round(HORIZON_S / DT_COMMA)
HS_NUSC = round(HORIZON_S / DT_NUSC)
STRIDE_COMMA = 15
STRIDE_NUSC = 5
N_DEFAULT = [3, 5, 10, 20]
FOLDS_DEFAULT = 3
SEEDS_DEFAULT = 4
KAPPA = 300
LAMBDA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
BOOTSTRAPS = 5000

METHODS = [
    "prior_lambda_0",
    "full_lambda_1",
    "squared_analytic_clipped",
    "squared_analytic_shrunk_k300",
    "validation_group_fde_grid",
    "validation_group_fde_upper95",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def group_fde_values(pred: np.ndarray, target: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    err = np.linalg.norm(pred - target, axis=1)
    return {
        str(g): float(np.mean(err[groups == g]))
        for g in np.unique(groups)
    }


def score_group_fde(pred: np.ndarray, residual: np.ndarray,
                    target: np.ndarray, groups: np.ndarray,
                    lam: float) -> dict[str, float]:
    return group_fde_values(pred + lam * residual, target, groups)


def group_summary(values: list[float]) -> dict[str, float | int]:
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "sd": float("nan"), "q25": float("nan"), "q75": float("nan")}
    return {
        "n": int(len(a)), "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "sd": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "q25": float(np.percentile(a, 25)), "q75": float(np.percentile(a, 75)),
    }


def choose_group_fde_lambda(ph: np.ndarray, residual: np.ndarray,
                            gt: np.ndarray, grp: np.ndarray,
                            conservative: bool) -> tuple[float, dict]:
    rows = []
    for lam in LAMBDA_GRID:
        by_group = group_fde_values(ph + lam * residual, gt, grp)
        vals = np.asarray(list(by_group.values()), dtype=float)
        mean = float(vals.mean())
        se_available = len(vals) >= 2
        se = float(vals.std(ddof=1) / np.sqrt(len(vals))) if se_available else None
        if conservative and se_available:
            objective = mean + 1.96 * se
        else:
            objective = mean
        rows.append({
            "lambda": float(lam), "mean_group_fde": mean,
            "se_group_mean": se, "se_available": bool(se_available),
            "selection_score": float(objective),
            "n_validation_groups": int(len(vals)),
        })
    # Grid is ascending; the secondary key explicitly breaks ties toward smaller lambda.
    selected = min(rows, key=lambda r: (r["selection_score"], r["lambda"]))
    if conservative and selected["n_validation_groups"] < 2:
        selection_note = "only one validation group; upper95 rule unavailable, fell back to group-mean FDE"
    elif conservative:
        selection_note = "minimize mean validation group FDE + 1.96 * SE; empirical selection rule, not a guarantee"
    else:
        selection_note = "minimize mean validation group FDE"
    return float(selected["lambda"]), {
        "selected_lambda": float(selected["lambda"]),
        "n_validation_groups": int(selected["n_validation_groups"]),
        "uncertainty_penalty": bool(conservative and selected["se_available"]),
        "fallback_used": bool(conservative and not selected["se_available"]),
        "selection_note": selection_note,
        "candidate_scores": rows,
    }


def paired_group_inference(ref: dict[str, float], candidate: dict[str, float],
                           seed: int) -> dict:
    ids = sorted(set(ref) & set(candidate))
    if not ids:
        return {"n_groups": 0}
    # Positive means candidate has lower FDE than reference.
    d = np.asarray([ref[g] - candidate[g] for g in ids], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(BOOTSTRAPS, len(d)))
    boot = d[idx].mean(axis=1)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = float(wilcoxon(d, alternative="greater", zero_method="wilcox").pvalue)
    except (ValueError, FloatingPointError):
        p = 1.0
    return {
        "n_groups": int(len(d)),
        "mean_gain_ref_minus_candidate_m": float(d.mean()),
        "paired_group_bootstrap_CI95_m": [float(np.percentile(boot, 2.5)),
                                             float(np.percentile(boot, 97.5))],
        "fraction_groups_candidate_better": float(np.mean(d > 0)),
        "fraction_groups_tied": float(np.mean(d == 0)),
        "wilcoxon_one_sided_p_candidate_better": p,
        "statistical_unit": "route" if len(ids) == 42 else "scene/group",
    }


def summarize_lambdas(records: list[dict]) -> dict:
    fields = [
        "lambda_sq_raw", "lambda_sq_clipped", "lambda_sq_shrunk",
        "lambda_validation_group_fde", "lambda_validation_group_fde_upper95",
    ]
    result = {field: group_summary([float(r[field]) for r in records]) for field in fields}
    result["n_draws_descriptive_only"] = int(len(records))
    if len(records) >= 3:
        a = np.asarray([r["lambda_sq_clipped"] for r in records])
        b = np.asarray([r["lambda_validation_group_fde"] for r in records])
        rho = spearmanr(a, b)
        result["clipped_squared_vs_fde_selected_lambda_spearman_descriptive"] = {
            "rho": float(rho.statistic), "p_value_not_inferential": float(rho.pvalue),
        }
    else:
        result["clipped_squared_vs_fde_selected_lambda_spearman_descriptive"] = None
    return result


def build_comma(data: dict, keys: list[str], gain: float):
    features, physics, target, groups = [], [], [], []
    for key in keys:
        df = data[key]
        route = E.route_of(df["seg"].iloc[0])
        for i0 in E.anchors_for(df, HS_COMMA, STRIDE_COMMA):
            features.append(E.features_causal(df, i0, HS_COMMA, DT_COMMA))
            physics.append(E.physics_endpoint_local(df, i0, HS_COMMA, DT_COMMA, gain))
            target.append(E.target_local(df, i0, HS_COMMA))
            groups.append(route)
    return (np.asarray(features, dtype=float), np.asarray(physics, dtype=float),
            np.asarray(target, dtype=float), np.asarray(groups, dtype=str))


def load_dataset(name: str):
    if name == "comma":
        data = C.load_all_cached(verbose=False)
        group_map = P.group_keys(data, "comma")
        return data, group_map, "comma"
    if name == "nuscenes":
        raw = NU.load()
        data = {}
        for k, df in raw.items():
            key = NU.key_of(df, k)
            if key in data:
                raise RuntimeError(f"Duplicate nuScenes group key: {key}")
            data[key] = df
        group_map = {k: [k] for k in data}
        return data, group_map, "nuscenes"
    raise ValueError(name)


def build_arrays(dataset: str, data: dict, keys: list[str], gain: float):
    if dataset == "comma":
        return build_comma(data, keys, gain)
    items = [(key, data[key]) for key in keys]
    return NC.collect(items, gain)


def fit_gain(dataset: str, data: dict, keys: list[str]) -> float:
    if dataset == "comma":
        return float(E.gain_curv([data[k] for k in keys], is_comma=True))
    return float(NC.gain_causal([(k, data[k]) for k in keys]))


def run_dataset(dataset: str, Ns: list[int], fold_indices: list[int], seeds: list[int],
                smoke: bool, output_path: Path) -> dict:
    data, group_map, protocol_dataset = load_dataset(dataset)
    gids = list(group_map.keys())
    rng = np.random.default_rng(0)
    rng.shuffle(gids)
    folds = [gids[i::FOLDS_DEFAULT] for i in range(FOLDS_DEFAULT)]
    all_group_predictions = {n: {m: defaultdict(list) for m in METHODS} for n in Ns}
    lambda_records_by_n = {n: [] for n in Ns}
    draw_manifests_by_n = {n: [] for n in Ns}
    actual_test_draw_counts = {n: defaultdict(int) for n in Ns}

    print(f"[{dataset}] groups={len(gids)} n_segments={len(data)}", flush=True)
    for n in Ns:
        for fold_i in fold_indices:
            test_groups = list(folds[fold_i])
            test_set = set(test_groups)
            pool = [g for g in gids if g not in test_set]
            if len(pool) < n:
                continue
            test_keys = [key for g in test_groups for key in group_map[g]]
            for subset_seed in seeds:
                local_rng = np.random.default_rng(100 * fold_i + subset_seed)
                subset = list(local_rng.choice(pool, size=n, replace=False))
                n_val = max(1, int(np.ceil(n / 3.0)))
                val_groups = list(local_rng.choice(subset, size=n_val, replace=False))
                fit_groups = [g for g in subset if g not in set(val_groups)]
                if len(fit_groups) + len(val_groups) != n:
                    raise AssertionError("fit/validation groups do not partition subset")
                if (set(fit_groups) & set(val_groups) or set(fit_groups) & test_set
                        or set(val_groups) & test_set):
                    raise AssertionError("group leakage between fit/validation/test")
                fit_keys = [key for g in fit_groups for key in group_map[g]]
                val_keys = [key for g in val_groups for key in group_map[g]]

                gain = fit_gain(dataset, data, fit_keys)
                Ff, PHf, GTf, _ = build_arrays(dataset, data, fit_keys, gain)
                Fv, PHv, GTv, GRPv = build_arrays(dataset, data, val_keys, gain)
                Ft, PHt, GTt, GRPt = build_arrays(dataset, data, test_keys, gain)
                min_fit = 50 if dataset == "nuscenes" else 20
                if len(Ff) < min_fit or len(Fv) < 5 or len(Ft) < 20:
                    print(f"[{dataset} N={n} fold={fold_i} seed={subset_seed}] skipped: "
                          f"anchors fit/val/test={len(Ff)}/{len(Fv)}/{len(Ft)}", flush=True)
                    continue

                scaler = StandardScaler().fit(Ff)
                model = MLPRegressor(
                    hidden_layer_sizes=(48, 24), max_iter=300 if dataset == "nuscenes" else 250,
                    alpha=1e-3, random_state=0, early_stopping=True,
                )
                model.fit(scaler.transform(Ff), GTf - PHf)
                Rv = model.predict(scaler.transform(Fv))
                Rt = model.predict(scaler.transform(Ft))

                numerator = float(np.sum(Rv * (GTv - PHv)))
                denominator = float(np.sum(Rv * Rv)) + 1e-9
                lambda_sq_raw = numerator / denominator
                lambda_sq = float(np.clip(lambda_sq_raw, 0.0, 1.5))
                shrink_n = int(len(Fv))
                lambda_sq_shrunk = float(shrink_n / (shrink_n + KAPPA) * lambda_sq)
                lambda_fde, fde_selection = choose_group_fde_lambda(
                    PHv, Rv, GTv, GRPv, conservative=False)
                lambda_upper, upper_selection = choose_group_fde_lambda(
                    PHv, Rv, GTv, GRPv, conservative=True)

                lambdas = {
                    "prior_lambda_0": 0.0,
                    "full_lambda_1": 1.0,
                    "squared_analytic_clipped": lambda_sq,
                    "squared_analytic_shrunk_k300": lambda_sq_shrunk,
                    "validation_group_fde_grid": lambda_fde,
                    "validation_group_fde_upper95": lambda_upper,
                }
                per_draw_test = {}
                for method, lam in lambdas.items():
                    gm = score_group_fde(PHt, Rt, GTt, GRPt, lam)
                    per_draw_test[method] = gm
                    for group, value in gm.items():
                        all_group_predictions[n][method][group].append(float(value))
                        actual_test_draw_counts[n][group] += 1 if method == "full_lambda_1" else 0

                record = {
                    "outer_fold": int(fold_i), "subset_seed": int(subset_seed),
                    "training_group_budget_N": int(n),
                    "fit_groups": sorted(map(str, fit_groups)),
                    "validation_groups": sorted(map(str, val_groups)),
                    "test_groups": sorted(map(str, test_groups)),
                    "fit_group_count": int(len(fit_groups)),
                    "validation_group_count": int(len(val_groups)),
                    "test_group_count": int(len(test_groups)),
                    "fit_anchor_count": int(len(Ff)),
                    "validation_anchor_count": int(len(Fv)),
                    "test_anchor_count": int(len(Ft)),
                    "physics_gain_fit_only": gain,
                    "lambda_sq_raw_unclipped": float(lambda_sq_raw),
                    "lambda_sq_raw": float(lambda_sq_raw),
                    "lambda_sq_clipped": lambda_sq,
                    "lambda_sq_shrunk": lambda_sq_shrunk,
                    "shrinkage_anchor_count_n": shrink_n,
                    "shrinkage_kappa": KAPPA,
                    "lambda_validation_group_fde": lambda_fde,
                    "lambda_validation_group_fde_upper95": lambda_upper,
                    "validation_fde_selection": fde_selection,
                    "validation_fde_upper95_selection": upper_selection,
                    "test_group_fde_by_method": per_draw_test,
                }
                lambda_records_by_n[n].append(record)
                draw_manifests_by_n[n].append({
                    k: record[k] for k in (
                        "outer_fold", "subset_seed", "training_group_budget_N",
                        "fit_groups", "validation_groups", "test_groups",
                        "fit_group_count", "validation_group_count", "test_group_count",
                        "fit_anchor_count", "validation_anchor_count", "test_anchor_count",
                    )
                })
                print(f"[{dataset} N={n} fold={fold_i} seed={subset_seed}] "
                      f"lam_sq={lambda_sq:.3f} shr={lambda_sq_shrunk:.3f} "
                      f"fde={lambda_fde:.2f} upper={lambda_upper:.2f} "
                      f"val_groups={len(val_groups)} test_groups={len(test_groups)}", flush=True)

        # Collapse repeated subset-seed evaluations within each independent test group.
        collapsed = {
            method: {g: float(np.mean(vals)) for g, vals in all_group_predictions[n][method].items()}
            for method in METHODS
        }
        if not collapsed["full_lambda_1"]:
            raise RuntimeError(f"No valid draws for {dataset} N={n}")
        # Group counts are unequal only if some groups fail construction; expose and verify.
        counts = {str(g): int(v) for g, v in actual_test_draw_counts[n].items()}
        distinct_counts = sorted(set(counts.values()))
        comparisons_vs_full = {
            method: paired_group_inference(collapsed["full_lambda_1"], collapsed[method], seed=9000 + n * 100 + i)
            for i, method in enumerate(METHODS) if method != "full_lambda_1"
        }
        comparisons_vs_sq_shrunk = {
            method: paired_group_inference(collapsed["squared_analytic_shrunk_k300"], collapsed[method], seed=19000 + n * 100 + i)
            for i, method in enumerate(METHODS)
            if method not in ("squared_analytic_shrunk_k300", "full_lambda_1")
        }
        n_records = lambda_records_by_n[n]
        # Per-fold lambda summaries make between-fold variation visible; the 12 draws are
        # dependent and remain descriptive, not inferential.
        by_fold = {}
        for fold_i in fold_indices:
            fr = [r for r in n_records if r["outer_fold"] == fold_i]
            if fr:
                by_fold[str(fold_i)] = summarize_lambdas(fr)
        per_n = {
            "N": int(n),
            "configuration": {
                "outer_folds": FOLDS_DEFAULT, "subset_seeds_per_fold": int(len(seeds)),
                "fit_validation_group_split": f"N={n}: fit={n - max(1, int(np.ceil(n / 3.0)))}, "
                                               f"validation={max(1, int(np.ceil(n / 3.0)))}",
                "validation_anchor_count_for_shrinkage": "raw overlapping anchor count; not independent sample size",
                "kappa": KAPPA, "lambda_grid": LAMBDA_GRID,
            },
            "n_draws_completed": int(len(n_records)),
            "n_test_groups": int(len(collapsed["full_lambda_1"])),
            "test_group_draw_count_range": [min(distinct_counts), max(distinct_counts)] if distinct_counts else [],
            "all_test_groups_have_equal_draw_count": bool(len(distinct_counts) <= 1),
            "mean_test_fde_by_method_group_weighted_m": {
                m: float(np.mean(list(collapsed[m].values()))) for m in METHODS
            },
            "paired_group_comparisons_vs_full_lambda_1": comparisons_vs_full,
            "paired_group_comparisons_vs_squared_analytic_shrunk": comparisons_vs_sq_shrunk,
            "lambda_stability_descriptive_only": summarize_lambdas(n_records),
            "lambda_stability_by_outer_fold_descriptive_only": by_fold,
            "test_group_fde_by_method": collapsed,
            "draw_records": n_records,
            "split_manifests": draw_manifests_by_n[n],
        }
        dataset_result = {
            "dataset": dataset,
            "n_input_groups": int(len(gids)),
            "n_input_segments_or_scenes": int(len(data)),
            "grouping_unit": "route" if dataset == "comma" else "scene",
            "per_N": {str(n): per_n},
        }
        # Incremental write after every N, to leave an auditable checkpoint if interrupted.
        if output_path.exists():
            try:
                old = json.loads(output_path.read_text(encoding="utf-8"))
            except Exception:
                old = {}
        else:
            old = {}
        per_dataset = old.get("per_dataset", {})
        prior_dataset = per_dataset.get(dataset, {})
        prior_n = prior_dataset.get("per_N", {})
        prior_n[str(n)] = per_n
        per_dataset[dataset] = {
            "dataset": dataset, "n_input_groups": int(len(gids)),
            "n_input_segments_or_scenes": int(len(data)),
            "grouping_unit": "route" if dataset == "comma" else "scene",
            "per_N": prior_n,
        }
        output = {
            "title": "Causal grouped calibration target comparison for residual trust coefficient",
            "run_timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "status": "SMOKE_TEST_NOT_FOR_INFERENCE" if smoke else "FULL_RUN",
            "N_list": sorted({int(k) for ds in per_dataset.values() for k in ds.get("per_N", {})}),
            "datasets_completed_or_partial": sorted(per_dataset),
            "provenance": {
                "script": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
                "script_sha256": sha256(Path(__file__).resolve()),
                "python_version": sys.version.split()[0], "numpy": np.__version__,
                "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                "outer_folds": FOLDS_DEFAULT, "subset_seeds": seeds,
                "fold_indices_run": fold_indices, "N_requested": Ns,
                "horizon_s": HORIZON_S, "lambda_grid": LAMBDA_GRID,
                "kappa": KAPPA,
                "comma_protocol": "causal_eval features/physics/targets; route-level groups; dt=0.05 s; stride=15 frames",
                "nuscenes_protocol": "exp_lambda_nusc_causal.collect/gain_causal; CAN speed + backward-difference yaw rate; scene groups; dt=0.02 s; stride=5 frames; all outer test scenes (no TEST_CAP)",
                "selection_data_flow": "fit groups train gain/scaler/MLP; validation groups select lambda only; outer test groups evaluate only",
                "primary_statistical_unit": "comma route; nuScenes scene",
                "repeated_draw_handling": "average each test group's repeated subset-seed FDE before paired bootstrap",
                "uncertainty_rule": "paired percentile bootstrap over independent test groups, 5000 replicates; Wilcoxon one-sided is supplementary",
                "conservative_rule": "minimize mean validation group FDE + 1.96*SE across validation groups; if fewer than 2 validation groups, transparently falls back to group-mean FDE; empirical only, no guarantee",
                "lambda_shrinkage_note": "kappa=300 and n=overlapping validation anchors are empirical, not effective independent sample size",
                "test_oracle": "not used; no test-selected lambda reported",
            },
            "per_dataset": per_dataset,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{dataset} N={n}] completed draws={len(n_records)} groups={len(collapsed['full_lambda_1'])}", flush=True)

    return json.loads(output_path.read_text(encoding="utf-8"))["per_dataset"][dataset]


def create_report(result_path: Path, report_path: Path):
    obj = json.loads(result_path.read_text(encoding="utf-8"))
    lines = [
        "# 因果分组校准目标比较实验",
        "",
        f"- 状态：`{obj['status']}`",
        f"- 运行时间：`{obj['run_timestamp']}`",
        f"- 脚本 SHA-256：`{obj['provenance']['script_sha256']}`",
        "- 论文正文未修改。",
        "- 正向差值定义：参照方法 FDE − 候选方法 FDE；正值表示候选方法更好。",
        "- 区间按独立路线/场景配对 bootstrap；同一测试组跨 subset seed 的重复预测先平均。",
        "",
    ]
    for dataset, ds in obj.get("per_dataset", {}).items():
        unit = ds.get("grouping_unit", "group")
        lines += [f"## {dataset}（独立统计单位：{unit}；输入组数 {ds.get('n_input_groups')}）", ""]
        for n, row in sorted(ds.get("per_N", {}).items(), key=lambda kv: int(kv[0])):
            lines += [f"### N={n}", "",
                      f"完成 {row['n_draws_completed']} 个 fit/validation/test draw；测试 {row['n_test_groups']} 个{unit}；"
                      f"测试组重复 draw 数范围 {row['test_group_draw_count_range']}。", "",
                      "| 策略 | 测试组等权平均 FDE (m) | 相对 λ=1 的改善 (m) | 95% 组级配对 CI (m) | 改善组比例 |",
                      "|---|---:|---:|---:|---:|"]
            ref_fde = row["mean_test_fde_by_method_group_weighted_m"]["full_lambda_1"]
            for method in METHODS:
                fde = row["mean_test_fde_by_method_group_weighted_m"][method]
                if method == "full_lambda_1":
                    lines.append(f"| {method} | {fde:.4f} | — | — | — |")
                else:
                    stats = row["paired_group_comparisons_vs_full_lambda_1"][method]
                    ci = stats["paired_group_bootstrap_CI95_m"]
                    lines.append(
                        f"| {method} | {fde:.4f} | {stats['mean_gain_ref_minus_candidate_m']:.4f} | "
                        f"[{ci[0]:.4f}, {ci[1]:.4f}] | {stats['fraction_groups_candidate_better']:.3f} |"
                    )
            lines += ["", "**λ 稳定性（描述性，draw 之间不独立，不作显著性推断）**", "",
                      "| λ 估计器 | 均值 | 中位数 | SD | IQR |"]
            lines.append("|---|---:|---:|---:|---:|")
            for field, label in [
                ("lambda_sq_raw", "Squared raw"),
                ("lambda_sq_clipped", "Squared clipped"),
                ("lambda_sq_shrunk", "Squared shrunk κ=300"),
                ("lambda_validation_group_fde", "Validation group-FDE"),
                ("lambda_validation_group_fde_upper95", "Validation group-FDE + 1.96SE"),
            ]:
                s = row["lambda_stability_descriptive_only"][field]
                lines.append(f"| {label} | {s['mean']:.4f} | {s['median']:.4f} | {s['sd']:.4f} | [{s['q25']:.4f}, {s['q75']:.4f}] |")
            lines += ["", "**校准策略对比（候选 − 平方误差收缩策略，改善为正）**", "",
                      "| 候选策略 | FDE 改善 (m) | 95% 组级配对 CI (m) | 改善组比例 | 单侧 Wilcoxon p（补充） |",
                      "|---|---:|---:|---:|---:|"]
            for method, stats in row["paired_group_comparisons_vs_squared_analytic_shrunk"].items():
                ci = stats["paired_group_bootstrap_CI95_m"]
                lines.append(
                    f"| {method} | {stats['mean_gain_ref_minus_candidate_m']:.4f} | "
                    f"[{ci[0]:.4f}, {ci[1]:.4f}] | {stats['fraction_groups_candidate_better']:.3f} | "
                    f"{stats['wilcoxon_one_sided_p_candidate_better']:.4g} |"
                )
            lines += ["", "**解释边界**：这是嵌套组级验证下对校准目标的比较；验证组 λ 分布为描述性结果，不能把重复 split 当独立样本。上界惩罚策略是经验选择规则，未提供安全或最优性保证。nuScenes 使用时间因果 CAN/后向差分输入，但定位姿态仍作为锚点状态，不等同于仅传感器输入的真机验证。", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="comma,nuscenes", help="comma-separated: comma,nuscenes")
    parser.add_argument("--Ns", default="3,5,10,20")
    parser.add_argument("--fold-indices", default="0,1,2", help="zero-based outer fold indices")
    parser.add_argument("--seeds", default="0,1,2,3", help="subset seeds per selected fold")
    parser.add_argument("--smoke", action="store_true", help="one small smoke run; never canonical")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    Ns = [int(x) for x in args.Ns.split(",") if x.strip()]
    fold_indices = [int(x) for x in args.fold_indices.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if args.smoke:
        datasets, Ns, fold_indices, seeds = datasets[:], [3], [0], [0]
        output_path = Path(args.output) if args.output else RESULTS / "exp_lambda_group_calibration_smoke.json"
    else:
        output_path = Path(args.output) if args.output else DEFAULT_OUT
    if any(x not in {"comma", "nuscenes"} for x in datasets):
        parser.error("--datasets accepts comma and/or nuscenes")
    if not Ns or any(n < 2 for n in Ns):
        parser.error("N must contain values >=2")
    if not fold_indices or any(i not in range(FOLDS_DEFAULT) for i in fold_indices):
        parser.error("fold indices must be 0, 1, or 2")
    if not seeds:
        parser.error("at least one subset seed is required")

    for dataset in datasets:
        run_dataset(dataset, Ns, fold_indices, seeds, args.smoke, output_path)
    create_report(output_path, REPORT if not args.smoke else RESULTS / "EXP_LAMBDA_GROUP_CALIBRATION_SMOKE_zh.md")
    print(f"Result: {output_path}", flush=True)
    print(f"Report: {REPORT if not args.smoke else RESULTS / 'EXP_LAMBDA_GROUP_CALIBRATION_SMOKE_zh.md'}", flush=True)


if __name__ == "__main__":
    main()
