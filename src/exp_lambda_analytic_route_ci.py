"""
Route-clustered audit for the causal comma2k19 analytic-lambda experiment.

This is a paper-audit companion to exp_lambda_analytic.py. It repeats the
same nested grouped-CV protocol without reading the old result JSON, then
collapses all anchors within route and all repeated subset draws within route
before inference. The independent statistical unit is route (42 routes).

Primary comparison: analytic lambda with kappa=300 vs full residual lambda=1.
The JSON stores route manifests so every reported number is auditable.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import causal_eval as E
import causal_protocol as P
import comma_loader as C

HERE = os.path.dirname(__file__)
RES = os.path.abspath(os.path.join(HERE, "..", "results", "exp_lambda_analytic_route_ci_results.json"))
HORIZON_S = 3.0
DT_S = 0.05
HS = int(round(HORIZON_S / DT_S))
STRIDE = 15
N_LIST = [3, 5, 10, 20]
FOLDS = 3
SUBSET_SEEDS = 4
KAPPA = 300
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
BOOTSTRAPS = 10000


def build(data, keys, gain):
    features, physics, target, groups = [], [], [], []
    for key in keys:
        frame = data[key]
        route = E.route_of(frame["seg"].iloc[0])
        for i0 in E.anchors_for(frame, HS, STRIDE):
            features.append(E.features_causal(frame, i0, HS, DT_S))
            physics.append(E.physics_endpoint_local(frame, i0, HS, DT_S, gain))
            target.append(E.target_local(frame, i0, HS))
            groups.append(route)
    return np.asarray(features), np.asarray(physics), np.asarray(target), np.asarray(groups)


def analytic_lambda(residual_pred, prior_error):
    numerator = float(np.sum(residual_pred * prior_error))
    denominator = float(np.sum(residual_pred * residual_pred)) + 1e-12
    return numerator / denominator


def route_means(fde, groups):
    groups = np.asarray(groups)
    fde = np.asarray(fde, dtype=float)
    return {str(g): float(np.mean(fde[groups == g])) for g in np.unique(groups)}


def bootstrap_ci(values, seed):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    means = values[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def paired_summary(route_full, route_shrunk, seed):
    common = sorted(set(route_full) & set(route_shrunk))
    full = np.asarray([route_full[g] for g in common], dtype=float)
    shrunk = np.asarray([route_shrunk[g] for g in common], dtype=float)
    diff = full - shrunk  # positive = shrinkage improves FDE
    try:
        p = float(stats.wilcoxon(diff, alternative="greater", method="auto").pvalue)
    except Exception:
        p = float("nan")
    return {
        "n_routes": len(common),
        "mean_full_fde": float(np.mean(full)),
        "mean_shrunk_fde": float(np.mean(shrunk)),
        "shrink_gain_full_minus_shrunk": float(np.mean(diff)),
        "shrink_gain_CI95": bootstrap_ci(diff, seed),
        "fraction_routes_shrunk_better": float(np.mean(diff > 0)),
        "wilcoxon_p_one_sided_full_gt_shrunk": p,
        "route_ids": common,
        "route_full_fde": {g: float(route_full[g]) for g in common},
        "route_shrunk_fde": {g: float(route_shrunk[g]) for g in common},
        "route_gain_full_minus_shrunk": {g: float(route_full[g] - route_shrunk[g]) for g in common},
    }


def main():
    data = C.load_all_cached(verbose=False)
    groups = P.group_keys(data, "comma")
    route_ids = sorted(groups)
    rng = np.random.default_rng(0)
    shuffled = list(route_ids)
    rng.shuffle(shuffled)
    folds = [shuffled[i::FOLDS] for i in range(FOLDS)]

    out = {
        "experiment": "comma_route_clustered_analytic_lambda",
        "run_date": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset": "comma2k19",
        "protocol": {
            "causal": True,
            "outer_group": "route",
            "statistical_unit": "route",
            "n_routes": len(route_ids),
            "horizon_s": HORIZON_S,
            "sample_dt_s": DT_S,
            "anchor_stride_samples": STRIDE,
            "outer_folds": FOLDS,
            "subset_seeds": SUBSET_SEEDS,
            "N_list": N_LIST,
            "validation_routes": "ceil(N/3), minimum 1",
            "kappa": KAPPA,
            "kappa_units": "raw validation anchor count; not independent sample count",
            "primary_comparison": "analytic_shrunk_lambda_vs_full_residual_lambda_1",
            "old_draw_level_ci_not_used": True,
        },
        "route_manifest": {"all_routes": route_ids, "outer_test_routes_by_fold": {str(i): sorted(f) for i, f in enumerate(folds)}},
        "per_N": {},
    }

    for N in N_LIST:
        # route -> list of repeated-draw route results. Results are collapsed only
        # after all anchors in a route are averaged.
        full_by_route = defaultdict(list)
        shrunk_by_route = defaultdict(list)
        prior_by_route = defaultdict(list)
        ana_by_route = defaultdict(list)
        draw_records = []

        for fo in range(FOLDS):
            test_routes = set(folds[fo])
            pool = [g for g in route_ids if g not in test_routes]
            test_keys = sum((groups[g] for g in sorted(test_routes)), [])
            for ss in range(SUBSET_SEEDS):
                seed = 100 * fo + ss
                local_rng = np.random.default_rng(seed)
                if len(pool) < N:
                    continue
                subset = list(local_rng.choice(pool, size=N, replace=False))
                n_val_routes = max(1, int(np.ceil(N / 3.0)))
                val_routes = list(local_rng.choice(subset, size=n_val_routes, replace=False))
                fit_routes = [g for g in subset if g not in val_routes]
                if not fit_routes:
                    continue
                fit_keys = sum((groups[g] for g in fit_routes), [])
                val_keys = sum((groups[g] for g in val_routes), [])
                gain = E.gain_curv([data[k] for k in fit_keys], is_comma=True)
                Ff, PHf, GTf, _ = build(data, fit_keys, gain)
                Fv, PHv, GTv, _ = build(data, val_keys, gain)
                Ft, PHt, GTt, test_anchor_routes = build(data, test_keys, gain)
                if min(len(Ff), len(Fv), len(Ft)) < 5:
                    continue
                scaler = StandardScaler().fit(Ff)
                net = MLPRegressor(
                    hidden_layer_sizes=(48, 24), max_iter=250, alpha=1e-3,
                    random_state=0, early_stopping=True,
                ).fit(scaler.transform(Ff), GTf - PHf)
                rv = net.predict(scaler.transform(Fv))
                rt = net.predict(scaler.transform(Ft))
                lhat = float(np.clip(analytic_lambda(rv, GTv - PHv), 0.0, 1.5))
                lshr = float(len(Fv) / (len(Fv) + KAPPA) * lhat)

                fde_prior = np.hypot(PHt[:, 0] - GTt[:, 0], PHt[:, 1] - GTt[:, 1])
                fde_full = np.hypot((PHt + rt)[:, 0] - GTt[:, 0], (PHt + rt)[:, 1] - GTt[:, 1])
                fde_ana = np.hypot((PHt + lhat * rt)[:, 0] - GTt[:, 0], (PHt + lhat * rt)[:, 1] - GTt[:, 1])
                fde_shr = np.hypot((PHt + lshr * rt)[:, 0] - GTt[:, 0], (PHt + lshr * rt)[:, 1] - GTt[:, 1])
                rm_full = route_means(fde_full, test_anchor_routes)
                rm_shr = route_means(fde_shr, test_anchor_routes)
                rm_prior = route_means(fde_prior, test_anchor_routes)
                rm_ana = route_means(fde_ana, test_anchor_routes)
                for route in rm_full:
                    full_by_route[route].append(rm_full[route])
                    shrunk_by_route[route].append(rm_shr[route])
                    prior_by_route[route].append(rm_prior[route])
                    ana_by_route[route].append(rm_ana[route])
                draw_records.append({
                    "outer_fold": fo, "subset_seed": ss,
                    "fit_routes": sorted(fit_routes), "validation_routes": sorted(val_routes),
                    "test_routes": sorted(test_routes), "lambda_analytic": lhat,
                    "lambda_shrunk": lshr, "n_validation_anchors": int(len(Fv)),
                })

        # Collapse repeated seeds/draws within each route before inference.
        route_full = {g: float(np.mean(v)) for g, v in full_by_route.items() if v}
        route_shr = {g: float(np.mean(shrunk_by_route[g])) for g in route_full}
        route_prior = {g: float(np.mean(prior_by_route[g])) for g in route_full}
        route_ana = {g: float(np.mean(ana_by_route[g])) for g in route_full}
        summary = paired_summary(route_full, route_shr, seed=1000 + N)
        summary.update({
            "N": N,
            "n_draws": len(draw_records),
            "fde_prior_route_mean": float(np.mean(list(route_prior.values()))) if route_prior else None,
            "fde_analytic_route_mean": float(np.mean(list(route_ana.values()))) if route_ana else None,
            "analytic_minus_full_route_mean": float(np.mean([route_ana[g] - route_full[g] for g in route_full])) if route_full else None,
            "draw_records": draw_records,
            "route_prior_fde": route_prior,
            "route_analytic_fde": route_ana,
            "routes_with_equal_seed_counts": len({len(v) for v in full_by_route.values()}) <= 1,
            "route_draw_count_min": min((len(v) for v in full_by_route.values()), default=0),
            "route_draw_count_max": max((len(v) for v in full_by_route.values()), default=0),
        })
        out["per_N"][str(N)] = summary
        print(
            "N=%2d routes=%d draws=%d full=%.4f shr=%.4f gain=%.4f CI[%.4f,%.4f] frac=%.3f p=%.3g"
            % (N, summary["n_routes"], summary["n_draws"], summary["mean_full_fde"],
               summary["mean_shrunk_fde"], summary["shrink_gain_full_minus_shrunk"],
               summary["shrink_gain_CI95"][0], summary["shrink_gain_CI95"][1],
               summary["fraction_routes_shrunk_better"], summary["wilcoxon_p_one_sided_full_gt_shrunk"])
        )

    os.makedirs(os.path.dirname(RES), exist_ok=True)
    with open(RES, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("saved ->", RES)


if __name__ == "__main__":
    main()
