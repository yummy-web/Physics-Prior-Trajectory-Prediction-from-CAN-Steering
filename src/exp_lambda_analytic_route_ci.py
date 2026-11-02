"""Compatibility reader for comma2k19 route-level lambda inference.

Route-level point estimates and confidence intervals are generated together by
``exp_lambda_group_calibration.py``.  This module intentionally performs no
independent model fit, preventing a second group-ordering protocol from
producing non-comparable estimates.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "results" / "canonical" / "exp_lambda_group_calibration_results.json"
METHOD = "squared_analytic_shrunk_k300"


def main() -> None:
    obj = json.loads(RESULT.read_text(encoding="utf-8"))
    rows = obj["per_dataset"]["comma"]["per_N"]
    print("N,full_fde_m,shrunk_fde_m,gain_m,ci95_low_m,ci95_high_m,n_routes")
    for n, row in sorted(rows.items(), key=lambda item: int(item[0])):
        means = row["mean_test_fde_by_method_group_weighted_m"]
        inference = row["paired_group_comparisons_vs_full_lambda_1"][METHOD]
        lo, hi = inference["paired_group_bootstrap_CI95_m"]
        print(
            f"{n},{means['full_lambda_1']:.9f},{means[METHOD]:.9f},"
            f"{inference['mean_gain_ref_minus_candidate_m']:.9f},"
            f"{lo:.9f},{hi:.9f},{inference['n_groups']}"
        )


if __name__ == "__main__":
    main()
