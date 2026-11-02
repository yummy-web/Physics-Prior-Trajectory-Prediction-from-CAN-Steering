"""Export the frozen representative CRIStAL/LOCSP endpoint rollout.

This exporter needs the locally cached grouped samples but writes no raw data.
It reproduces the published fit/validation/test acquisition split, model seed
and anchor, then stores only four sets of 1/2/3 s endpoint coordinates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import exp_crisal_external_pilot as pilot
from exp_crisal_external_pilot import make_physics_predictions
from exp_crisal_grouped_external import (discover_bags, fit_gain, fit_residual_mlp,
                                         load_or_build_samples, predict_residual)

TEST_ID = "08_04_2024_anneau_cite_scientifique_1_2024-04-08-12-03-32"
VAL_IDS = [
    "08_04_2024_anneau_cite_scientifique_2_2024-04-08-12-14-51",
    "08_04_2024_bleu_2024-04-08-12-47-23",
]
FIT_IDS = [
    "08_04_2024_jaune_2024-04-08-13-16-18",
    "08_04_2024_rouge_2024-04-08-12-57-04",
    "08_04_2024_vert_2024-04-08-12-35-18",
    "08_04_2024_violet_2024-04-08-12-20-41",
]
SEED = 29
ANCHOR = 196
KAPPA_ACQ = 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=ROOT / "results" / "canonical" /
                    "crisal_external_validation" / "representative_endpoint_rollout.json")
    args = ap.parse_args()
    manifest_dir = ROOT / "results" / "canonical" / "crisal_external_validation"
    rows = discover_bags(args.data_root, manifest_dir)
    by_id = {}
    cache = args.data_root / "cache" / "grouped_samples"
    # Public releases intentionally omit raw bags.  An author-side bundle may
    # also contain a partial raw extract; add every validated cache not already
    # represented so figure export does not trigger unnecessary re-decoding.
    known = {item["acquisition_id"] for item in rows}
    rows += [{"acquisition_id": p.stem, "bag_path": p}
             for p in sorted(cache.glob("*.npz")) if p.stem not in known]
    for item in rows:
        cache_path = cache / f"{item['acquisition_id']}.npz"
        if cache_path.exists() and Path(item["bag_path"]).suffix == ".npz":
            with np.load(cache_path, allow_pickle=False) as z:
                by_id[item["acquisition_id"]] = {
                    "X": z["X"], "Y_body": z["Y_body"],
                    "meta": json.loads(str(z["sample_meta"].item())),
                }
        else:
            samples, _ = load_or_build_samples(item, cache, pilot)
            by_id[item["acquisition_id"]] = samples
    missing = sorted(set([TEST_ID, *VAL_IDS, *FIT_IDS]) - set(by_id))
    if missing:
        raise RuntimeError(f"Missing acquisitions: {missing}")
    gain = fit_gain(FIT_IDS, by_id)
    physics = {}
    for aid in [*FIT_IDS, *VAL_IDS, TEST_ID]:
        s = by_id[aid]
        physics[aid] = make_physics_predictions(s, np.arange(len(s["X"])), gain)["CTRV_fit_CAN_steering_gain"]
    model, scaler, _ = fit_residual_mlp(FIT_IDS, by_id, physics, SEED)
    numerator = denominator = 0.0
    for aid in VAL_IDS:
        r = predict_residual(model, scaler, by_id[aid]["X"])
        y, p = by_id[aid]["Y_body"], physics[aid]
        numerator += float(np.mean(np.sum(r * (y - p), axis=(1, 2))))
        denominator += float(np.mean(np.sum(r * r, axis=(1, 2))))
    raw_lambda = numerator / denominator if denominator else 0.0
    validation_lambda = float(np.clip(raw_lambda, 0.0, 1.5))
    deployment_lambda = validation_lambda * len(VAL_IDS) / (len(VAL_IDS) + KAPPA_ACQ)
    test = by_id[TEST_ID]
    if not 0 <= ANCHOR < len(test["X"]):
        raise IndexError(f"anchor {ANCHOR} outside 0..{len(test['X'])-1}")
    residual = predict_residual(model, scaler, test["X"])[ANCHOR]
    p = physics[TEST_ID][ANCHOR]
    payload = {
        "dataset": "CRIStAL/LOCSP",
        "test_acquisition": TEST_ID,
        "fit_acquisitions": FIT_IDS,
        "validation_acquisitions": VAL_IDS,
        "model_seed": SEED,
        "anchor_index": ANCHOR,
        "validation_lambda": validation_lambda,
        "kappa_acq": KAPPA_ACQ,
        "deployment_lambda": deployment_lambda,
        "reference": test["Y_body"][ANCHOR].tolist(),
        "ctrv": p.tolist(),
        "full": (p + residual).tolist(),
        "shrunk": (p + deployment_lambda * residual).tolist(),
        "note": "1/2/3 s endpoints connected for visualization; not a continuous trajectory reconstruction.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
