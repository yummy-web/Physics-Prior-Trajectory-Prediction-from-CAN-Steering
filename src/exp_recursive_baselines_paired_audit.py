"""Paired group-level audit for the completed RLS and steering-input EKF baselines."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SRC = RES / "exp_recursive_baselines_causal_results.json"
OUT_JSON = RES / "exp_recursive_baselines_paired_audit_results.json"
OUT_MD = RES / "exp_recursive_baselines_paired_audit_results.md"


def paired(a, b, seed, group_unit):
    keys = sorted(set(a) & set(b))
    va = np.array([a[k] for k in keys], float)
    vb = np.array([b[k] for k in keys], float)
    d = va - vb
    rng = np.random.default_rng(seed)
    boot = np.array([np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(4000)])
    # Keep this audit runnable in the repository's minimal runtime. The
    # paired permutation p-value tests the null that the signs of the
    # within-group differences are exchangeable; it is not used as a
    # superiority claim for the main method.
    rng_p = np.random.default_rng(seed + 100000)
    n_perm = 20000
    obs = abs(float(np.mean(d)))
    signs = rng_p.choice(np.array([-1.0, 1.0]), size=(n_perm, len(d)))
    perm_stats = np.abs(np.mean(signs * d[None, :], axis=1))
    p = float((1 + np.sum(perm_stats >= obs)) / (n_perm + 1))
    return {
        "group_unit": group_unit,
        "n_groups": len(keys),
        "mean_diff_RLS_minus_EKF": float(np.mean(d)),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "fraction_RLS_better": float(np.mean(d < 0)),
        "wilcoxon_p_two_sided": p,
    }


def main():
    src = json.loads(SRC.read_text(encoding="utf-8"))
    out = {"experiment": "recursive_baselines_paired_group_audit", "source": SRC.name, "datasets": {}}
    for ds, data in src["datasets"].items():
        folds = []
        for row in data["folds_detail"]:
            folds.append({"fold": row["fold"], "RLS_vs_EKF": paired(row["RLS_fit_only"]["group_means"], row["Steering_EKF"]["group_means"], 9000 + row["fold"], data["group_unit"])})
        out["datasets"][ds] = {"group_unit": data["group_unit"], "n_groups_total": data["n_groups"], "folds": folds, "note": "paired audit compares only the two completed supplementary baselines; it is not a superiority claim against all filters."}
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# RLS and steering-input EKF paired audit", "", "The audit uses the same route/scene groups as the baseline experiment.", "", "| Dataset | Group unit | Fold | Groups | RLS-EKF mean difference | 95% CI | RLS better fraction | permutation p |", "|---|---|---:|---:|---:|---|---:|---:|"]
    for ds, data in out["datasets"].items():
        for row in data["folds"]:
            x = row["RLS_vs_EKF"]
            lines.append(f"| {ds} | {x['group_unit']} | {row['fold']+1} | {x['n_groups']} | {x['mean_diff_RLS_minus_EKF']:.3f} | [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}] | {x['fraction_RLS_better']:.3f} | {x['wilcoxon_p_two_sided']:.4g} |")
    lines += ["", "Interpretation: negative RLS-EKF means RLS has lower FDE. This is only a paired comparison between the two supplementary baselines."]
    OUT_MD.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(OUT_JSON)


if __name__ == "__main__":
    main()
