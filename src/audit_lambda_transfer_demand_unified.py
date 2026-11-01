from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULT = ROOT / "results" / "supplementary" / "exp_lambda_transfer_demand_unified_results.json"
CANONICAL = ROOT / "results" / "canonical" / "exp_lambda_group_calibration_results.json"
OUT = ROOT / "results" / "supplementary" / "exp_lambda_transfer_demand_unified_audit.json"
REPORT = ROOT / "results" / "reports_zh" / "EXP_LAMBDA_TRANSFER_DEMAND_UNIFIED_AUDIT_zh.md"

sys.path.insert(0, str(HERE))
import exp_lambda_group_calibration as C
from exp_lambda_transfer_demand_unified import N_LIST, source_schedule, scene_values, sha256


def replay_canonical_draw(canonical: dict) -> dict:
    by, _, _ = C.load_dataset("nuscenes")
    gids = list(by.keys())
    np.random.default_rng(0).shuffle(gids)
    folds = [gids[i::3] for i in range(3)]
    test = list(folds[0])
    pool = [g for g in gids if g not in set(test)]
    rng = np.random.default_rng(0)
    subset = list(rng.choice(pool, size=3, replace=False))
    val = list(rng.choice(subset, size=1, replace=False))
    fit = [g for g in subset if g not in set(val)]
    gain = C.fit_gain("nuscenes", by, fit)
    Ff, PHf, GTf, _ = C.build_arrays("nuscenes", by, fit, gain)
    Ft, PHt, GTt, GRPt = C.build_arrays("nuscenes", by, test, gain)
    scaler = StandardScaler().fit(Ff)
    model = MLPRegressor(hidden_layer_sizes=(48, 24), max_iter=300, alpha=1e-3, random_state=0, early_stopping=True)
    model.fit(scaler.transform(Ff), GTf - PHf)
    values = scene_values(PHt + model.predict(scaler.transform(Ft)), GTt, GRPt)
    draw = next(r for r in canonical["per_dataset"]["nuscenes"]["per_N"]["3"]["draw_records"] if r["outer_fold"] == 0 and r["subset_seed"] == 0)
    stored = draw["test_group_fde_by_method"]["full_lambda_1"]
    common = sorted(set(values) & set(stored))
    max_abs = max(abs(values[k] - float(stored[k])) for k in common)
    return {
        "test_group_count": len(test), "n_common_scenes": len(common),
        "scene_0002_replay_m": float(values["scene-0002"]),
        "scene_0002_canonical_m": float(stored["scene-0002"]),
        "replay_test_mean_m": float(np.mean(list(values.values()))),
        "canonical_test_mean_m": float(np.mean([float(stored[k]) for k in common])),
        "max_absolute_scene_difference_m": float(max_abs),
        "pass": bool(len(common) == 189 and max_abs <= 1e-9),
    }


def main() -> None:
    obj = json.loads(RESULT.read_text(encoding="utf-8"))
    canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
    checks = {
        "N_list": obj.get("N_list") == N_LIST,
        "per_N_keys": sorted(int(k) for k in obj.get("per_N", {})) == N_LIST,
        "canonical_source_schedule": obj.get("source_schedule_from_canonical_comma") == {str(k): v for k, v in sorted(source_schedule(canonical).items())},
    }
    expected = {
        3: {"universal", "source", "target_K1", "target_val", "target_oracle"},
        5: {"universal", "source", "target_K1", "target_K2", "target_val", "target_oracle"},
        10: {"universal", "source", "target_K1", "target_K2", "target_K4", "target_val", "target_oracle"},
        20: {"universal", "source", "target_K1", "target_K2", "target_K4", "target_K7", "target_val", "target_oracle"},
    }
    rows = []
    for n in N_LIST:
        row = obj["per_N"][str(n)]
        names = set(row.get("policies", {}))
        ci_names = set(row.get("paired_CI_vs_universal_m", {}))
        item = {"N": n, "n_draws_12": row.get("n_draws") == 12, "n_scenes_566": row.get("n_scenes") == 566, "policy_set": names == expected[n], "CI_for_nonuniversal": ci_names == expected[n] - {"universal"}, "manifest_hash_count": len(row.get("manifest_sha256", [])) == 12}
        item["pass"] = all(v for k, v in item.items() if k != "N")
        rows.append(item)
    checks["per_N"] = rows
    replay = replay_canonical_draw(canonical)
    all_pass = all(checks[k] for k in ("N_list", "per_N_keys", "canonical_source_schedule")) and all(r["pass"] for r in rows) and replay["pass"]
    audit = {"title": "Unified transfer/demand experiment independent audit", "status": "PASS" if all_pass else "FAIL", "result_file_sha256": sha256(RESULT), "script_sha256": sha256(Path(__file__).resolve()), "canonical_file_sha256": sha256(CANONICAL), "original_result_status": obj.get("status"), "checks": checks, "canonical_replay": replay, "note": "This audit checks stored structure and one deterministic canonical replay; it does not replace the full experiment."}
    OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 统一迁移与目标校准需求实验：独立审计", "", "- 审计状态：" + audit["status"], "- 原结果文件状态：" + str(obj.get("status")), "- 审计未修改论文正文。", "", "| N | 12 draws | 566 scenes | policy/CI | manifest | 结果 |", "|---:|:---:|:---:|:---:|:---:|:---:|"]
    for r in rows:
        lines.append("| %s | %s | %s | %s | %s | %s |" % (r["N"], "通过" if r["n_draws_12"] else "失败", "通过" if r["n_scenes_566"] else "失败", "通过" if r["policy_set"] and r["CI_for_nonuniversal"] else "失败", "通过" if r["manifest_hash_count"] else "失败", "通过" if r["pass"] else "失败"))
    lines += ["", "canonical draw 直接重放", "scene-0002 重放 %.12f m；canonical %.12f m" % (replay["scene_0002_replay_m"], replay["scene_0002_canonical_m"]), "测试集平均 FDE 重放 %.12f m；canonical %.12f m" % (replay["replay_test_mean_m"], replay["canonical_test_mean_m"]), "189 个场景最大绝对差 %.3e m" % replay["max_absolute_scene_difference_m"], "重放检查：" + ("通过" if replay["pass"] else "失败"), "", "该审计只检查结果结构和一个确定性 canonical draw，不把需求曲线提升为样本复杂度定律。"]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "replay": replay}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
