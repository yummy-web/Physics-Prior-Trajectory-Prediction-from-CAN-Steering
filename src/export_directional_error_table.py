"""Export a paper-ready directional endpoint-error table.

The script performs no model selection or retraining.  It converts the frozen
group-bootstrap anisotropy result into compact JSON/CSV/Markdown artefacts and
records the interpretation boundary needed by the manuscript.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "results" / "canonical" / "exp_anisotropy_v2_results.json"
DEFAULT_OUT = ROOT / "results" / "supplementary"


def row(platform: str, branch: str, horizon: str, stat: dict) -> dict:
    return {
        "platform": platform,
        "prediction_branch": branch,
        "horizon_s": int(horizon.removesuffix("s")),
        "longitudinal_abs_endpoint_error_mean_m": stat["lon"]["mean"],
        "longitudinal_group_bootstrap_CI95_m": [stat["lon"]["ci_lo"], stat["lon"]["ci_hi"]],
        "lateral_abs_endpoint_error_mean_m": stat["lat"]["mean"],
        "lateral_group_bootstrap_CI95_m": [stat["lat"]["ci_lo"], stat["lat"]["ci_hi"]],
        "lateral_over_longitudinal_ratio": stat["lat_over_lon"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    obj = json.loads(args.source.read_text(encoding="utf-8"))
    rows = []
    for platform in ("comma2k19", "nuscenes"):
        for horizon, stat in obj[platform]["physics_causal"].items():
            rows.append(row(platform, "causal_physics_prior", horizon, stat))
        rows.append(row(platform, "causal_full_residual", "3s", obj[platform]["residual_causal"]["3s"]))
    payload = {
        "title": "Directional absolute endpoint-error decomposition",
        "source_result": str(args.source.relative_to(ROOT)).replace("\\", "/"),
        "coordinate_definition": "local x is longitudinal; local y is lateral",
        "metric": "absolute endpoint component error in metres",
        "statistical_unit": "route for comma2k19; scene for nuScenes",
        "nuscenes_scope": "200-scene causal subset used by exp_anisotropy_v2, not the full 566-scene corpus",
        "interpretation": (
            "Descriptive directional error evidence. It quantifies error allocation between "
            "local longitudinal and lateral axes; it does not establish sensor causality, "
            "observability, or a universal physical mechanism."
        ),
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jpath = args.output_dir / "exp_directional_error_table_results.json"
    cpath = args.output_dir / "exp_directional_error_table.csv"
    mpath = args.output_dir / "EXP_DIRECTIONAL_ERROR_TABLE_zh.md"
    jpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with cpath.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            rr = dict(r)
            rr["longitudinal_group_bootstrap_CI95_m"] = json.dumps(rr["longitudinal_group_bootstrap_CI95_m"])
            rr["lateral_group_bootstrap_CI95_m"] = json.dumps(rr["lateral_group_bootstrap_CI95_m"])
            w.writerow(rr)
    lines = [
        "# 方向性端点误差表", "",
        "局部 x/y 分别定义为纵向/横向。数值为绝对端点分量误差；区间按 comma route、nuScenes scene 聚类自助法计算。nuScenes 使用冻结的 200-scene 因果子集。", "",
        "| 平台 | 分支 | 时域 | 纵向误差 m [95% CI] | 横向误差 m [95% CI] | 横/纵 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lci, yci = r["longitudinal_group_bootstrap_CI95_m"], r["lateral_group_bootstrap_CI95_m"]
        lines.append(
            f"| {r['platform']} | {r['prediction_branch']} | {r['horizon_s']} s | "
            f"{r['longitudinal_abs_endpoint_error_mean_m']:.3f} [{lci[0]:.3f}, {lci[1]:.3f}] | "
            f"{r['lateral_abs_endpoint_error_mean_m']:.3f} [{yci[0]:.3f}, {yci[1]:.3f}] | "
            f"{r['lateral_over_longitudinal_ratio']:.3f} |"
        )
    lines += ["", "该表用于描述误差在局部纵横轴上的分配，不据此推断传感器因果性、系统可观测性或普适机制。", ""]
    mpath.write_text("\n".join(lines), encoding="utf-8")
    print(jpath); print(cpath); print(mpath)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
