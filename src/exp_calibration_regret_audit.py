"""Descriptive calibration-regret audit for the frozen lambda results.

No model is retrained here. The script reads the canonical analytic-lambda and
validation-selection JSON files, computes regret to the test-set grid oracle,
and records whether the calculation is point-estimate or clustered inference.
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT_JSON = RES / "exp_calibration_regret_audit_results.json"
OUT_MD = RES / "exp_calibration_regret_audit_results.md"


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def summarize(label, obj):
    rows = []
    for n, d in obj["per_N"].items():
        full = d.get("fde_full", d.get("fde_full_lam1"))
        shr = d.get("fde_analytic_shrunk")
        oracle = d.get("fde_oracle")
        val = d.get("fde_val_selected")
        if full is None or shr is None or oracle is None:
            continue
        denom = full - oracle
        rows.append({
            "N": int(n),
            "full_fde": float(full),
            "analytic_shrunk_fde": float(shr),
            "oracle_fde": float(oracle),
            "full_regret_vs_oracle": float(full - oracle),
            "shrunk_regret_vs_oracle": float(shr - oracle),
            "shrunk_gain_vs_full": float(full - shr),
            "fraction_of_full_to_oracle_gap_closed": float((full - shr) / denom) if abs(denom) > 1e-12 else None,
            "validation_selected_fde": float(val) if val is not None else None,
            "validation_selected_regret_vs_oracle": float(val - oracle) if val is not None else None,
            "statistical_scope": "scene-level CI exists for nuScenes shrink-vs-full; all regret quantities are descriptive point estimates",
        })
    return {"dataset": label, "rows": rows}


def main():
    comma = load("exp_lambda_analytic_results.json")
    nusc = load("exp_lambda_analytic_nusc_results.json")
    # nusc usability JSON supplies validation-selected grid values; analytic JSON
    # supplies the causal scene-clustered shrinkage values.
    nusc_val = load("exp_lambda_usability_results.json")
    nusc_copy = {"per_N": {}}
    for n, d in nusc["per_N"].items():
        x = dict(d)
        if n in nusc_val.get("per_N", {}):
            x["fde_val_selected"] = nusc_val["per_N"][n].get("fde_val_selected")
        nusc_copy["per_N"][n] = x
    out = {
        "experiment": "calibration_regret_audit",
        "description": "Descriptive regret to the test-set grid oracle; not used for model selection.",
        "sources": ["exp_lambda_analytic_results.json", "exp_lambda_analytic_nusc_results.json", "exp_lambda_usability_results.json"],
        "comma2k19": summarize("comma2k19", comma),
        "nuScenes": summarize("nuScenes", nusc_copy),
        "interpretation": [
            "The grid oracle is a descriptive upper bound and is never available at deployment.",
            "The regret and fraction-closed quantities are point estimates; only the nuScenes shrink-vs-full comparison has scene-clustered CIs in the source JSON.",
            "Do not interpret a high fraction closed as a theorem or as a guarantee for unseen platforms.",
        ],
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Lambda calibration regret audit", "", "This is a descriptive audit of frozen results; it does not retrain or select models.", "", "| Dataset | N | Full FDE | Shrunk FDE | Oracle FDE | Shrunk gain vs full | Full-to-oracle gap closed |", "|---|---:|---:|---:|---:|---:|---:|"]
    for ds in ("comma2k19", "nuScenes"):
        for row in out[ds]["rows"]:
            frac = "NA" if row["fraction_of_full_to_oracle_gap_closed"] is None else f"{row['fraction_of_full_to_oracle_gap_closed']:.3f}"
            lines.append(f"| {ds} | {row['N']} | {row['full_fde']:.3f} | {row['analytic_shrunk_fde']:.3f} | {row['oracle_fde']:.3f} | {row['shrunk_gain_vs_full']:.3f} | {frac} |")
    lines += ["", "Interpretation:", "- The oracle uses test-set information and is only a descriptive upper bound.", "- Regret and fraction-closed values are point estimates; they are not significance claims.", "- The source nuScenes JSON contains scene-clustered CIs for shrink-vs-full, but not for every regret quantity."]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_JSON)


if __name__ == "__main__":
    main()
