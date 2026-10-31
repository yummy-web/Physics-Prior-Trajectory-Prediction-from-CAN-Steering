"""MLP convergence sensitivity audit for the canonical grouped lambda experiment.

This script deliberately does not overwrite the canonical result. It replays the
exact split manifests from exp_lambda_group_calibration_results.json, fits the
same causal physics+MLP pipeline under the canonical and higher max_iter
conditions, and compares the resulting group-level FDE and shrinkage gain.

The only intended optimization change is max_iter. Model architecture, alpha,
random seed, early stopping, tol, n_iter_no_change, data, splits, and scoring
remain fixed. Results are diagnostic/sensitivity evidence, not a replacement
for the canonical result unless stability is established separately.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import wilcoxon
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
CANONICAL = RESULTS / "exp_lambda_group_calibration_results.json"
DEFAULT_OUT = RESULTS / "exp_lambda_group_calibration_convergence_results.json"
DEFAULT_REPORT = RESULTS / "EXP_LAMBDA_GROUP_CALIBRATION_CONVERGENCE_zh.md"

sys.path.insert(0, str(HERE))
import causal_eval as E
import causal_protocol as P
import comma_loader as C
import exp_lambda_group_calibration as G
import exp_lambda_nusc_causal as NC
import exp_causal_nuscenes as NU

BOOTSTRAPS = 5000
METHODS = list(G.METHODS)
SHRUNK = "squared_analytic_shrunk_k300"
FULL = "full_lambda_1"
DEFAULT_HIGH_MAX_ITER = 1000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def group_fde_values(pred: np.ndarray, target: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    err = np.linalg.norm(pred - target, axis=1)
    return {str(g): float(np.mean(err[groups == g])) for g in np.unique(groups)}


def score_group_fde(pred: np.ndarray, residual: np.ndarray, target: np.ndarray,
                    groups: np.ndarray, lam: float) -> dict[str, float]:
    return group_fde_values(pred + lam * residual, target, groups)


def paired_group_inference(ref: dict[str, float], candidate: dict[str, float], seed: int) -> dict:
    ids = sorted(set(ref) & set(candidate))
    d = np.asarray([ref[g] - candidate[g] for g in ids], dtype=float)
    if not len(d):
        return {"n_groups": 0}
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
        "paired_group_bootstrap_CI95_m": [
            float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        ],
        "fraction_groups_candidate_better": float(np.mean(d > 0)),
        "fraction_groups_tied": float(np.mean(d == 0)),
        "wilcoxon_one_sided_p_candidate_better": p,
        "statistical_unit": "route" if len(ids) == 42 else "scene/group",
    }


def aggregate_groups(draw_records: list[dict], condition: str, method: str) -> dict[str, float]:
    per_group: dict[str, list[float]] = defaultdict(list)
    for r in draw_records:
        for g, v in r["conditions"][condition]["test_group_fde_by_method"][method].items():
            per_group[str(g)].append(float(v))
    return {g: float(np.mean(v)) for g, v in per_group.items()}


def convergence_summary(records: list[dict], condition: str) -> dict:
    rows = [r["conditions"][condition]["fit_diagnostics"] for r in records]
    max_iter = int(rows[0]["max_iter"]) if rows else None
    n = len(rows)
    n_iter = np.asarray([r["n_iter"] for r in rows], dtype=float) if rows else np.array([])
    warning_counts = np.asarray([r["warning_count"] for r in rows], dtype=int) if rows else np.array([], dtype=int)
    hit = np.asarray([r["hit_max_iter"] for r in rows], dtype=bool) if rows else np.array([], dtype=bool)
    return {
        "n_fits": n,
        "max_iter": max_iter,
        "warning_fit_count": int(np.sum(warning_counts > 0)),
        "warning_fit_rate": float(np.mean(warning_counts > 0)) if n else None,
        "total_warning_count": int(np.sum(warning_counts)),
        "hit_max_iter_count": int(np.sum(hit)),
        "hit_max_iter_rate": float(np.mean(hit)) if n else None,
        "n_iter_mean": float(np.mean(n_iter)) if n else None,
        "n_iter_median": float(np.median(n_iter)) if n else None,
        "n_iter_p95": float(np.percentile(n_iter, 95)) if n else None,
        "n_iter_min": int(np.min(n_iter)) if n else None,
        "n_iter_max": int(np.max(n_iter)) if n else None,
    }


def load_canonical() -> dict:
    if not CANONICAL.exists():
        raise FileNotFoundError(f"Canonical result not found: {CANONICAL}")
    return json.loads(CANONICAL.read_text(encoding="utf-8"))


def load_dataset(name: str):
    if name == "comma":
        data = C.load_all_cached(verbose=False)
        group_map = P.group_keys(data, "comma")
        return data, group_map
    if name == "nuscenes":
        raw = NU.load()
        data = {}
        for k, df in raw.items():
            key = NU.key_of(df, k)
            if key in data:
                raise RuntimeError(f"Duplicate nuScenes group key: {key}")
            data[key] = df
        return data, {k: [k] for k in data}
    raise ValueError(name)


def keys_for_groups(group_map: dict[str, list[str]], groups: list[str]) -> list[str]:
    return [key for g in groups for key in group_map[str(g)]]


def build_arrays(dataset: str, data: dict, keys: list[str], gain: float):
    if dataset == "comma":
        return G.build_comma(data, keys, gain)
    return NC.collect([(key, data[key]) for key in keys], gain)


def fit_gain(dataset: str, data: dict, keys: list[str]) -> float:
    if dataset == "comma":
        return float(E.gain_curv([data[k] for k in keys], is_comma=True))
    return float(NC.gain_causal([(k, data[k]) for k in keys]))


def fit_and_score(dataset: str, arrays: tuple[np.ndarray, ...], max_iter: int) -> dict:
    Ff, PHf, GTf, _ = arrays[0]
    Fv, PHv, GTv, GRPv = arrays[1]
    Ft, PHt, GTt, GRPt = arrays[2]
    scaler = StandardScaler().fit(Ff)
    model = MLPRegressor(
        hidden_layer_sizes=(48, 24),
        max_iter=int(max_iter),
        alpha=1e-3,
        random_state=0,
        early_stopping=True,
        tol=1e-4,
        n_iter_no_change=10,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(scaler.transform(Ff), GTf - PHf)
    warning_messages = [str(w.message) for w in caught]
    convergence_messages = [m for m in warning_messages if "maximum iterations" in m.lower() or "converg" in m.lower()]
    Rv = model.predict(scaler.transform(Fv))
    Rt = model.predict(scaler.transform(Ft))
    numerator = float(np.sum(Rv * (GTv - PHv)))
    denominator = float(np.sum(Rv * Rv)) + 1e-9
    lambda_raw = numerator / denominator
    lambda_clipped = float(np.clip(lambda_raw, 0.0, 1.5))
    n_val = int(len(Fv))
    lambda_shrunk = float(n_val / (n_val + G.KAPPA) * lambda_clipped)
    lambda_fde, _ = G.choose_group_fde_lambda(PHv, Rv, GTv, GRPv, conservative=False)
    lambda_upper, _ = G.choose_group_fde_lambda(PHv, Rv, GTv, GRPv, conservative=True)
    lambdas = {
        "prior_lambda_0": 0.0,
        "full_lambda_1": 1.0,
        "squared_analytic_clipped": lambda_clipped,
        "squared_analytic_shrunk_k300": lambda_shrunk,
        "validation_group_fde_grid": lambda_fde,
        "validation_group_fde_upper95": lambda_upper,
    }
    test_by_method = {
        method: score_group_fde(PHt, Rt, GTt, GRPt, lam)
        for method, lam in lambdas.items()
    }
    return {
        "lambdas": {k: float(v) for k, v in lambdas.items()},
        "test_group_fde_by_method": test_by_method,
        "fit_diagnostics": {
            "max_iter": int(max_iter),
            "n_iter": int(getattr(model, "n_iter_", -1)),
            "loss": float(getattr(model, "loss_", np.nan)),
            "warning_count": int(len(caught)),
            "convergence_warning_count": int(len(convergence_messages)),
            "warning_messages": warning_messages,
            "hit_max_iter": bool(getattr(model, "n_iter_", -1) >= int(max_iter)),
            "early_stopping": True,
            "tol": 1e-4,
            "n_iter_no_change": 10,
        },
    }


def run_dataset(dataset: str, canonical: dict, high_max_iter: int) -> dict:
    data, group_map = load_dataset(dataset)
    canonical_ds = canonical["per_dataset"][dataset]
    out_per_n = {}
    print(f"[{dataset}] groups={len(group_map)} segments/scenes={len(data)}", flush=True)
    for n_str, canonical_n in sorted(canonical_ds["per_N"].items(), key=lambda kv: int(kv[0])):
        n = int(n_str)
        manifests = canonical_n["split_manifests"]
        records = []
        for idx, manifest in enumerate(manifests):
            fit_groups = [str(x) for x in manifest["fit_groups"]]
            val_groups = [str(x) for x in manifest["validation_groups"]]
            test_groups = [str(x) for x in manifest["test_groups"]]
            fit_keys = keys_for_groups(group_map, fit_groups)
            val_keys = keys_for_groups(group_map, val_groups)
            test_keys = keys_for_groups(group_map, test_groups)
            gain = fit_gain(dataset, data, fit_keys)
            arrays = (
                build_arrays(dataset, data, fit_keys, gain),
                build_arrays(dataset, data, val_keys, gain),
                build_arrays(dataset, data, test_keys, gain),
            )
            if len(arrays[0][0]) < (50 if dataset == "nuscenes" else 20) or len(arrays[1][0]) < 5 or len(arrays[2][0]) < 20:
                raise RuntimeError(f"Canonical manifest became invalid: {dataset} N={n} draw={idx}")
            low_iter = 300 if dataset == "nuscenes" else 250
            low = fit_and_score(dataset, arrays, low_iter)
            high = fit_and_score(dataset, arrays, high_max_iter)
            records.append({
                "outer_fold": int(manifest["outer_fold"]),
                "subset_seed": int(manifest["subset_seed"]),
                "training_group_budget_N": n,
                "fit_groups": fit_groups,
                "validation_groups": val_groups,
                "test_groups": test_groups,
                "fit_anchor_count": int(len(arrays[0][0])),
                "validation_anchor_count": int(len(arrays[1][0])),
                "test_anchor_count": int(len(arrays[2][0])),
                "physics_gain_fit_only": float(gain),
                "conditions": {"canonical_max_iter": low, "higher_max_iter": high},
            })
            print(f"[{dataset} N={n} draw={idx+1}/{len(manifests)}] "
                  f"low_iter={low['fit_diagnostics']['n_iter']} high_iter={high['fit_diagnostics']['n_iter']} "
                  f"warn={low['fit_diagnostics']['warning_count']}/{high['fit_diagnostics']['warning_count']}", flush=True)

        condition_summary = {}
        for condition in ("canonical_max_iter", "higher_max_iter"):
            by_method = {method: aggregate_groups(records, condition, method) for method in METHODS}
            mean_by_method = {method: float(np.mean(list(v.values()))) for method, v in by_method.items()}
            condition_summary[condition] = {
                "fit_diagnostics": convergence_summary(records, condition),
                "mean_test_fde_by_method_group_weighted_m": mean_by_method,
                "test_group_fde_by_method": by_method,
                "paired_group_comparisons_shrunk_vs_full": paired_group_inference(
                    by_method[FULL], by_method[SHRUNK], seed=70000 + n
                ),
            }

        method_sensitivity = {}
        low_by = condition_summary["canonical_max_iter"]["test_group_fde_by_method"]
        high_by = condition_summary["higher_max_iter"]["test_group_fde_by_method"]
        for method in METHODS:
            method_sensitivity[method] = paired_group_inference(
                low_by[method], high_by[method], seed=71000 + n + METHODS.index(method)
            )
        low_gain = condition_summary["canonical_max_iter"]["paired_group_comparisons_shrunk_vs_full"]
        high_gain = condition_summary["higher_max_iter"]["paired_group_comparisons_shrunk_vs_full"]
        gain_delta = float(high_gain["mean_gain_ref_minus_candidate_m"] - low_gain["mean_gain_ref_minus_candidate_m"])
        gain_direction_preserved = bool(
            np.sign(low_gain["mean_gain_ref_minus_candidate_m"]) == np.sign(high_gain["mean_gain_ref_minus_candidate_m"])
            or (low_gain["mean_gain_ref_minus_candidate_m"] == 0 and high_gain["mean_gain_ref_minus_candidate_m"] == 0)
        )
        out_per_n[str(n)] = {
            "N": n,
            "n_replayed_draws": len(records),
            "configuration": {
                "split_source": str(CANONICAL.relative_to(ROOT)).replace("\\", "/"),
                "model_hidden_layer_sizes": [48, 24],
                "alpha": 1e-3,
                "random_state": 0,
                "early_stopping": True,
                "tol": 1e-4,
                "n_iter_no_change": 10,
                "canonical_max_iter": 300 if dataset == "nuscenes" else 250,
                "higher_max_iter": int(high_max_iter),
            },
            "condition_summary": condition_summary,
            "method_sensitivity_high_minus_canonical": method_sensitivity,
            "shrinkage_gain_sensitivity": {
                "canonical_gain_full_minus_shrunk_m": float(low_gain["mean_gain_ref_minus_candidate_m"]),
                "higher_gain_full_minus_shrunk_m": float(high_gain["mean_gain_ref_minus_candidate_m"]),
                "higher_minus_canonical_gain_m": gain_delta,
                "gain_direction_preserved": gain_direction_preserved,
                "canonical_gain_CI95_m": low_gain["paired_group_bootstrap_CI95_m"],
                "higher_gain_CI95_m": high_gain["paired_group_bootstrap_CI95_m"],
            },
            "draw_records": records,
        }
    return {
        "dataset": dataset,
        "n_input_groups": int(canonical_ds["n_input_groups"]),
        "n_input_segments_or_scenes": int(canonical_ds["n_input_segments_or_scenes"]),
        "grouping_unit": canonical_ds["grouping_unit"],
        "per_N": out_per_n,
    }


def canonical_reproduction_check(out: dict, canonical: dict, atol: float = 1e-8) -> dict:
    checks = []
    for dataset, ds in out["per_dataset"].items():
        for n, item in ds["per_N"].items():
            got = item["condition_summary"]["canonical_max_iter"]["mean_test_fde_by_method_group_weighted_m"]
            expected = canonical["per_dataset"][dataset]["per_N"][n]["mean_test_fde_by_method_group_weighted_m"]
            for method in METHODS:
                diff = abs(float(got[method]) - float(expected[method]))
                checks.append({"dataset": dataset, "N": int(n), "method": method, "abs_diff_m": diff, "pass": diff <= atol})
    return {"all_pass": bool(all(c["pass"] for c in checks)), "tolerance_m": atol, "checks": checks}


def create_report(result_path: Path, report_path: Path) -> None:
    obj = json.loads(result_path.read_text(encoding="utf-8"))
    lines = [
        "# MLP 收敛敏感性检查（因果分组 λ 校准实验）",
        "",
        "## 目的与原则",
        "",
        "本检查只改变 MLP 的 `max_iter`，不改变数据、因果特征、物理先验、模型结构、随机种子、早停参数、分组切分或统计单位。脚本重放 canonical 结果中的 3 个 outer fold × 4 个 subset seed 清单，并将每个 test route/scene 内的重复评估先平均，再做组级比较。",
        "",
        "- canonical 条件：comma `max_iter=250`，nuScenes `max_iter=300`；",
        f"- higher 条件：两个平台均为 `max_iter={obj['configuration']['higher_max_iter']}`；",
        "- 早停：`early_stopping=True`、`tol=1e-4`、`n_iter_no_change=10`；",
        "- 高迭代结果是敏感性证据，不自动替换 canonical 结果。",
        "",
        "## 结果摘要",
        "",
    ]
    for dataset, ds in obj["per_dataset"].items():
        unit = ds["grouping_unit"]
        lines += [f"### {dataset}（统计单位：{unit}）", "", "| N | canonical 收缩增益 full−shrunk (m) | higher 收缩增益 full−shrunk (m) | higher−canonical (m) | canonical 95% CI | higher 95% CI | 方向保持 | |", "|---:|---:|---:|---:|---|---|:---:|",]
        for n, item in ds["per_N"].items():
            s = item["shrinkage_gain_sensitivity"]
            lines.append(
                f"| {n} | {s['canonical_gain_full_minus_shrunk_m']:.4f} | {s['higher_gain_full_minus_shrunk_m']:.4f} | {s['higher_minus_canonical_gain_m']:.4f} | "
                f"[{s['canonical_gain_CI95_m'][0]:.4f}, {s['canonical_gain_CI95_m'][1]:.4f}] | "
                f"[{s['higher_gain_CI95_m'][0]:.4f}, {s['higher_gain_CI95_m'][1]:.4f}] | {'是' if s['gain_direction_preserved'] else '否'} |"
            )
        lines += ["", "### 拟合收敛诊断", "", "| N | 条件 | 拟合次数 | warning拟合数/比例 | 达到max_iter数/比例 | n_iter均值 | n_iter中位数 | n_iter最大 |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
        for n, item in ds["per_N"].items():
            for condition, label in (("canonical_max_iter", "canonical"), ("higher_max_iter", "higher")):
                d = item["condition_summary"][condition]["fit_diagnostics"]
                lines.append(f"| {n} | {label} | {d['n_fits']} | {d['warning_fit_count']}/{d['warning_fit_rate']:.1%} | {d['hit_max_iter_count']}/{d['hit_max_iter_rate']:.1%} | {d['n_iter_mean']:.1f} | {d['n_iter_median']:.1f} | {d['n_iter_max']} |")
        lines += ["", "### FDE 变化（higher − canonical，单位 m；正值表示高迭代条件误差更大）", "", "| 数据集 | N | full λ=1 | shrunk λ | analytic clipped | validation grid |", "|---|---:|---:|---:|---:|---:|"]
        for dataset, ds in obj["per_dataset"].items():
            for n, item in ds["per_N"].items():
                row = []
                for method in (FULL, SHRUNK, "squared_analytic_clipped", "validation_group_fde_grid"):
                    row.append(item["method_sensitivity_high_minus_canonical"][method]["mean_gain_ref_minus_candidate_m"] * -1.0)
                lines.append(f"| {dataset} | {n} | {row[0]:.5f} | {row[1]:.5f} | {row[2]:.5f} | {row[3]:.5f} |")
        lines += ["", "## 解释边界", "", "1. 如果 higher 条件下收缩增益的方向、量级和组级区间与 canonical 基本一致，可将结论描述为对 MLP 停止条件具有敏感性稳定性；这不等于证明神经网络优化已达到全局最优。", "2. 如果某个 N 的差异明显或方向翻转，应以 higher 结果重新评估该预算的主张，不选择性保留较有利条件。", "3. `warning` 与 `hit_max_iter` 只反映 sklearn 训练停止诊断；早停会在达到 patience 时停止，因此 `n_iter < max_iter` 不必然表示损失已完全收敛。", "4. comma 与 nuScenes 仍按各自平台的 route/scene 统计单位解释，不将窗口数当作独立样本。", ""]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="comma,nuscenes")
    parser.add_argument("--higher-max-iter", type=int, default=DEFAULT_HIGH_MAX_ITER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.higher_max_iter <= 300:
        raise SystemExit("--higher-max-iter must be greater than the canonical nuScenes max_iter=300")
    canonical = load_canonical()
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    per_dataset = {}
    for dataset in datasets:
        per_dataset[dataset] = run_dataset(dataset, canonical, args.higher_max_iter)
    output = {
        "title": "MLP convergence sensitivity audit for causal grouped lambda calibration",
        "run_timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "status": "FULL_RUN",
        "higher_max_iter": int(args.higher_max_iter),
        "configuration": {"higher_max_iter": int(args.higher_max_iter)},
        "canonical_reproduction_check": None,
        "provenance": {
            "script": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
            "script_sha256": sha256(Path(__file__).resolve()),
            "canonical_result": str(CANONICAL.relative_to(ROOT)).replace("\\", "/"),
            "canonical_result_sha256": sha256(CANONICAL),
            "python_version": sys.version.split()[0],
            "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
            "only_changed_parameter": "MLPRegressor.max_iter",
            "canonical_max_iter": {"comma": 250, "nuscenes": 300},
            "higher_max_iter": int(args.higher_max_iter),
            "model": {"hidden_layer_sizes": [48, 24], "alpha": 1e-3, "random_state": 0, "early_stopping": True, "tol": 1e-4, "n_iter_no_change": 10},
            "split_replay": "exact split_manifests from canonical result; no new split or seed",
            "primary_statistical_unit": "comma route; nuScenes scene",
            "bootstrap": "paired percentile bootstrap over independent groups, 5000 replicates",
        },
        "per_dataset": per_dataset,
    }
    output["canonical_reproduction_check"] = canonical_reproduction_check(output, canonical)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    create_report(args.output, args.report)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report}")
    print(f"canonical reproduction all_pass={output['canonical_reproduction_check']['all_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
