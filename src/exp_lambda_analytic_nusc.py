"""
exp_lambda_analytic_nusc.py  (method upgrade / external validity, 2026-10-08)

nuScenes counterpart of exp_lambda_analytic.py. Same closed-form estimator
    lambda_hat = sum_i <r_i, y_i - p_i> / sum_i <r_i, r_i>
estimated on held-in VALIDATION scenes, with small-sample shrinkage
    lambda_shrunk = n/(n+kappa) * clip(lambda_hat, 0, 1.5).

CAUSAL (P0 fix, 2026-10-08): imports collect / gain_causal / route_mean from
exp_lambda_nusc_causal, which uses BACKWARD-difference yaw rate + CAN speed and
NEVER touches central-diff obs_v / df['yaw_rate'] (the leak in the old
exp_lambda_shrinkage_nusc.collect). Nested grouped CV: outer test scenes
untouched, inner split of the N training scenes into fit/val. kappa is a FIXED
prior declared before test evaluation (NOT tuned on test). Its unit is the RAW
validation-anchor count n (overlapping stride-5 windows) -- this is NOT an
effective independent sample size, so kappa is an empirical hyperparameter, not
a principled Bayesian prior. A kappa sweep {100,200,300,500,800} with
SCENE-LEVEL CIs is reported for defensibility.

STAT UNIT (P0 fix, 2026-10-08): the independent unit is the test SCENE, not the
training draw. The 4 subset seeds within a fold reuse the SAME test scenes, so
draw-level bootstrap CIs are too narrow. Primary CIs collapse each scene's
repeated draw-evaluations to one value, then bootstrap over scenes.
"""
import os, json, datetime
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import exp_causal_nuscenes as NU
from exp_lambda_nusc_causal import collect, gain_causal, route_mean

RES = os.path.join(os.path.dirname(__file__), "..", "results", "exp_lambda_analytic_nusc_results.json")
DT = 0.02; H = 3.0
N_LIST = [3, 5, 10, 20]; FOLDS = 3; SUBSET_SEEDS = 4
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
KAPPA = 300
KAPPA_SWEEP = [100, 200, 300, 500, 800]


def route_fde_by_scene(P_, G_, grp):
    """Return (per_scene_mean_fde, scene_ids) so CI can cluster on scenes."""
    fde = np.hypot(P_[:, 0] - G_[:, 0], P_[:, 1] - G_[:, 1])
    rm, gs = route_mean(fde, grp)
    return rm, gs


def route_mean_fde(P_, G_, grp):
    rm, _ = route_fde_by_scene(P_, G_, grp)
    return float(np.mean(rm))


def scene_level_ci(scene_vals_a, scene_vals_b, n_boot=3000, seed=12345):
    """Paired diff (a-b) bootstrapped over SCENES (the independent unit).
    scene_vals_* : dict scene_id -> mean metric over that scene's repeated draws."""
    sids = sorted(set(scene_vals_a) & set(scene_vals_b))
    if not sids:
        return None
    d = np.array([scene_vals_a[s] - scene_vals_b[s] for s in sids])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    bo = d[idx].mean(axis=1)
    return {"n_scenes": int(len(d)), "mean_diff": float(d.mean()),
            "CI95": [float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))],
            "frac_scenes_positive": float(np.mean(d > 0))}


def accum_scene(store, key, rm, gs):
    """Accumulate per-scene FDE; store[key][scene] = list of per-draw means."""
    dd = store.setdefault(key, {})
    for v, g in zip(rm, gs):
        dd.setdefault(g, []).append(float(v))


def collapse_scene(store, key):
    """Average repeated draw-evaluations of each scene -> one value per scene."""
    return {g: float(np.mean(vs)) for g, vs in store.get(key, {}).items()}


def analytic_lambda(r, e):
    num = float(np.sum(r * e)); den = float(np.sum(r * r)) + 1e-9
    return num / den


def main():
    import sys
    global N_LIST
    if len(sys.argv) > 1:
        N_LIST = [int(x) for x in sys.argv[1].split(",")]
    data = NU.load()
    items_all = [(NU.key_of(df, k), df) for k, df in data.items()]
    gids = [k for k, _ in items_all]; byg = {k: df for k, df in items_all}
    rng = np.random.default_rng(0); rng.shuffle(gids)
    folds = [gids[i::FOLDS] for i in range(FOLDS)]
    out = {"horizon_s": H, "dt": DT, "N_list": N_LIST, "kappa": KAPPA, "kappa_sweep": KAPPA_SWEEP,
           "lambdas": LAMBDAS,
           "provenance": {"run_date": datetime.datetime.now().isoformat(timespec="seconds"),
                          "folds": FOLDS, "subset_seeds": SUBSET_SEEDS, "dataset": "nuscenes",
                          "causal": True, "speed_src": "can_speed", "yawrate_src": "backward_diff",
                          "collect_from": "exp_lambda_nusc_causal", "gain_from": "gain_causal",
                          "kappa_unit": "raw_validation_anchor_count (overlapping stride-5 windows; "
                                        "NOT an effective independent sample size)",
                          "kappa_tuned_on_test": False,
                          "kappa_selection": "FIXED prior, declared before test eval; defensibility via "
                                             "kappa_sweep with scene-level CIs (not post-hoc picked)",
                          "stat_unit": "scene (clustered bootstrap over test scenes); draw-level CI is "
                                       "non-independent (4 subset seeds reuse same test scenes per fold)",
                          "scope": "temporally-causal: uses localisation pose + CAN signals available at anchor; "
                                   "NOT a real-vehicle sensor-only causal validation",
                          "note": "CAUSAL closed-form lambda=<r,y-p>/<r,r> + shrinkage; nested grouped CV; "
                                  "replaces leaky exp_lambda_shrinkage_nusc.collect (central-diff obs_v/yaw_rate)"},
           "per_N": {}}
    if os.path.exists(RES):
        try: out["per_N"] = json.load(open(RES)).get("per_N", {})
        except Exception: pass
    for N in N_LIST:
        if N < 2: continue
        R = {"prior": [], "full": [], "grid": [], "ana": [], "anash": [], "oracle": [],
             "lhat_grid": [], "lhat_ana": [], "lhat_anash": []}
        # per-scene accumulators (scene_id -> list of per-draw mean FDE) for clustered CIs
        S = {}
        # kappa sweep: draw-level means + per-scene store for scene-level CI
        Ksweep = {k: {"shrunk": [], "lhat": []} for k in KAPPA_SWEEP}
        Ksweep_scene = {k: {} for k in KAPPA_SWEEP}
        for fo in range(FOLDS):
            test_g = set(folds[fo]); pool = [g for g in gids if g not in test_g]
            if len(pool) < N: continue
            test_items = [(g, byg[g]) for g in test_g]
            for ss in range(SUBSET_SEEDS):
                r2 = np.random.default_rng(100 * fo + ss)
                sub = list(r2.choice(pool, size=N, replace=False))
                n_val = max(1, int(np.ceil(N / 3.0)))
                val_g = list(r2.choice(sub, size=n_val, replace=False))
                fit_g = [g for g in sub if g not in val_g]
                if not fit_g: continue
                fit_items = [(g, byg[g]) for g in fit_g]
                val_items = [(g, byg[g]) for g in val_g]
                a = gain_causal(fit_items)
                Ff, PHf, GTf, _ = collect(fit_items, a)
                Fv, PHv, GTv, GRPv = collect(val_items, a)
                Ft, PHt, GTt, GRPt = collect(test_items, a)
                if len(Ff) < 50 or len(Fv) < 10 or len(Ft) < 20: continue
                sc = StandardScaler().fit(Ff)
                net = MLPRegressor(hidden_layer_sizes=(48, 24), max_iter=300, alpha=1e-3,
                                   random_state=0, early_stopping=True).fit(sc.transform(Ff), GTf - PHf)
                rv = net.predict(sc.transform(Fv)); rt = net.predict(sc.transform(Ft))
                lh = analytic_lambda(rv, GTv - PHv); lh_c = float(np.clip(lh, 0.0, 1.5))
                nval = len(Fv); lh_sh = nval / (nval + KAPPA) * lh_c
                vcurve = {l: route_mean_fde(PHv + l * rv, GTv, GRPv) for l in LAMBDAS}
                lhat_grid = min(vcurve, key=vcurve.get)
                tcurve = {l: route_mean_fde(PHt + l * rt, GTt, GRPt) for l in LAMBDAS}
                R["prior"].append(tcurve[0.0]); R["full"].append(tcurve[1.0])
                R["grid"].append(tcurve[lhat_grid])
                R["ana"].append(route_mean_fde(PHt + lh_c * rt, GTt, GRPt))
                R["anash"].append(route_mean_fde(PHt + lh_sh * rt, GTt, GRPt))
                R["oracle"].append(min(tcurve.values()))
                R["lhat_grid"].append(lhat_grid); R["lhat_ana"].append(lh_c); R["lhat_anash"].append(lh_sh)
                # per-scene paired FDE for scene-clustered CIs
                rm_full, gs_t = route_fde_by_scene(PHt + 1.0 * rt, GTt, GRPt)
                accum_scene(S, "full", rm_full, gs_t)
                accum_scene(S, "anash", *route_fde_by_scene(PHt + lh_sh * rt, GTt, GRPt))
                accum_scene(S, "prior", *route_fde_by_scene(PHt + 0.0 * rt, GTt, GRPt))
                # kappa sweep (same draw, same lhat, only shrinkage factor changes)
                for kap in KAPPA_SWEEP:
                    l_k = nval / (nval + kap) * lh_c
                    Ksweep[kap]["shrunk"].append(route_mean_fde(PHt + l_k * rt, GTt, GRPt))
                    Ksweep[kap]["lhat"].append(l_k)
                    accum_scene(Ksweep_scene[kap], "shrunk", *route_fde_by_scene(PHt + l_k * rt, GTt, GRPt))
                    accum_scene(Ksweep_scene[kap], "full", rm_full, gs_t)
        if not R["full"]: continue
        a_full = np.array(R["full"]); a_sh = np.array(R["anash"])
        d = a_full - a_sh
        # draw-level CI (NON-independent: 4 subset seeds share the same test scenes
        # within a fold -> too narrow; kept only for backward comparison, NOT for claims)
        bo = [float(np.mean(d[np.random.default_rng(k).integers(0, len(d), len(d))])) for k in range(3000)]
        ci_draw = [float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))]
        # PRIMARY: scene-level clustered CI. Average each scene's repeated draw-evaluations,
        # then bootstrap over scenes (the independent statistical unit).
        sc_full = collapse_scene(S, "full"); sc_sh = collapse_scene(S, "anash")
        ci_scene = scene_level_ci(sc_full, sc_sh)
        out["per_N"][str(N)] = {
            "N": N, "n_draws": int(len(a_full)),
            "fde_prior": float(np.mean(R["prior"])), "fde_full": float(np.mean(a_full)),
            "fde_grid": float(np.mean(R["grid"])), "fde_analytic": float(np.mean(R["ana"])),
            "fde_analytic_shrunk": float(np.mean(a_sh)), "fde_oracle": float(np.mean(R["oracle"])),
            "shrunk_gain_vs_full": float(np.mean(d)),
            "shrunk_gain_CI95_draw_nonindep": ci_draw,
            "shrunk_gain_CI95_scene": (ci_scene or {}).get("CI95"),
            "shrunk_gain_scene_detail": ci_scene,
            "stat_unit": "scene (clustered); draw-level CI retained but non-independent",
            "lhat_grid_mean": float(np.mean(R["lhat_grid"])),
            "lhat_analytic_mean": float(np.mean(R["lhat_ana"])),
            "lhat_shrunk_mean": float(np.mean(R["lhat_anash"])),
            "kappa_sweep": {str(kap): {
                "fde_shrunk": float(np.mean(Ksweep[kap]["shrunk"])),
                "lhat_shrunk_mean": float(np.mean(Ksweep[kap]["lhat"])),
                "gain_vs_full": float(np.mean(a_full - np.array(Ksweep[kap]["shrunk"]))),
                "gain_vs_full_CI95_scene": (scene_level_ci(
                    collapse_scene(Ksweep_scene[kap], "full"),
                    collapse_scene(Ksweep_scene[kap], "shrunk")) or {}).get("CI95")}
                for kap in KAPPA_SWEEP}}
        json.dump(out, open(RES, "w"), indent=2)
        o = out["per_N"][str(N)]
        cs = o["shrunk_gain_CI95_scene"] or [float("nan"), float("nan")]
        print("N=%2d full=%.3f anaShr=%.3f oracle=%.3f | shr_vs_full=%.3f sceneCI[%.3f,%.3f] drawCI[%.3f,%.3f] nsc=%s | lam grid=%.2f ana=%.2f shr=%.2f"
              % (N, o["fde_full"], o["fde_analytic_shrunk"], o["fde_oracle"],
                 o["shrunk_gain_vs_full"], cs[0], cs[1], ci_draw[0], ci_draw[1],
                 (ci_scene or {}).get("n_scenes"),
                 o["lhat_grid_mean"], o["lhat_analytic_mean"], o["lhat_shrunk_mean"]))
    # canonical header: reflect ALL N actually present, not just this run's CLI arg
    out["N_list"] = sorted((int(k) for k in out["per_N"]), key=int)
    json.dump(out, open(RES, "w"), indent=2); print("saved ->", RES)


if __name__ == "__main__":
    main()