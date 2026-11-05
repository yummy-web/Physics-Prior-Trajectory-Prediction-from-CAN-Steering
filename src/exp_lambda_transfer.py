"""
exp_lambda_transfer.py  (cross-platform schedule transfer, 2026-10-08)

Question: does a lambda-SCHEDULE estimated on the SOURCE platform (comma) work
when applied verbatim to a TARGET platform (nuScenes)? Tests the memory claim
"direction of the shrinkage law is universal, the numeric level is platform-
dependent."

CAUSAL (P0 fix, 2026-10-08): imports collect / gain_causal / route_mean from
exp_lambda_nusc_causal (backward-diff yaw rate + CAN speed; no central-diff
obs_v / df['yaw_rate']). The SOURCE schedule is estimated purely on comma; the
nuScenes TARGET test scenes never influence schedule selection.

On nuScenes nested grouped CV (identical splits to exp_lambda_analytic_nusc),
evaluate the SAME untouched test scenes under four lambda policies:
  1. source   : comma analytic-shrunk schedule loaded from the current
                canonical comma result
  2. universal: fixed lambda = 1 (naive full residual)
  3. target_val: nuScenes own validation analytic-shrunk lambda (re-estimated here)
  4. target_oracle: nuScenes grid oracle (lower bound)
Reports scene-mean FDE per policy per N, plus SCENE-LEVEL clustered bootstrap
CIs of (source - target_val), (universal - target_val) AND (universal - source)
to show transfer sits between naive and self-tuned.

STAT UNIT (P0 fix, 2026-10-08): the independent unit is the test SCENE. The 4
subset seeds within a fold reuse the SAME test scenes, so draw-level CIs are too
narrow. Primary CIs collapse each scene's repeated draw-evaluations to one value
then bootstrap over scenes. The universal-vs-source CI (previously absent) is now
reported so the source>universal claim is backed by an interval, not just means.
"""
import os, json, datetime
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import exp_causal_nuscenes as NU
from exp_lambda_nusc_causal import collect, gain_causal, route_mean

RES = os.path.join(os.path.dirname(__file__), "..", "results", "exp_lambda_transfer_results.json")
DT = 0.02; H = 3.0
N_LIST = [3, 5, 10, 20]; FOLDS = 3; SUBSET_SEEDS = 4
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
KAPPA = 300
SOURCE_RESULT = os.path.join(
    os.path.dirname(__file__), "..", "results", "canonical",
    "exp_lambda_analytic_results.json",
)


def load_source_schedule():
    """Load the comma-only schedule from the current canonical result."""
    with open(SOURCE_RESULT, "r", encoding="utf-8") as f:
        result = json.load(f)
    schedule = {
        int(n): float(row["lhat_shrunk_mean"])
        for n, row in result["per_N"].items()
    }
    missing = set(N_LIST) - set(schedule)
    if missing:
        raise ValueError(f"canonical comma result lacks budgets: {sorted(missing)}")
    return schedule


def rmf_scene(P_, G_, grp):
    fde = np.hypot(P_[:, 0] - G_[:, 0], P_[:, 1] - G_[:, 1])
    rm, gs = route_mean(fde, grp)
    return rm, gs


def rmf(P_, G_, grp):
    rm, _ = rmf_scene(P_, G_, grp)
    return float(np.mean(rm))


def accum_scene(store, key, rm, gs):
    dd = store.setdefault(key, {})
    for v, g in zip(rm, gs):
        dd.setdefault(g, []).append(float(v))


def collapse_scene(store, key):
    return {g: float(np.mean(vs)) for g, vs in store.get(key, {}).items()}


def scene_level_ci(a, b, n_boot=3000, seed=12345):
    """Paired (a-b) bootstrap over SCENES (independent unit)."""
    sids = sorted(set(a) & set(b))
    if not sids:
        return None
    d = np.array([a[s] - b[s] for s in sids])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    bo = d[idx].mean(axis=1)
    return {"n_scenes": int(len(d)), "mean_diff": float(d.mean()),
            "CI95": [float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))],
            "frac_scenes_positive": float(np.mean(d > 0))}


def analytic_lambda(r, e):
    return float(np.sum(r * e)) / (float(np.sum(r * r)) + 1e-9)


def main():
    import sys
    global N_LIST
    if len(sys.argv) > 1:
        N_LIST = [int(x) for x in sys.argv[1].split(",")]
    source_sched = load_source_schedule()
    data = NU.load()
    items_all = [(NU.key_of(df, k), df) for k, df in data.items()]
    gids = [k for k, _ in items_all]; byg = {k: df for k, df in items_all}
    rng = np.random.default_rng(0); rng.shuffle(gids)
    folds = [gids[i::FOLDS] for i in range(FOLDS)]
    out = {"horizon_s": H, "dt": DT, "N_list": N_LIST, "source_schedule": source_sched,
           "provenance": {"run_date": datetime.datetime.now().isoformat(timespec="seconds"),
                          "folds": FOLDS, "subset_seeds": SUBSET_SEEDS, "dataset": "nuscenes(target) <- comma(source)",
                          "causal": True, "speed_src": "can_speed", "yawrate_src": "backward_diff",
                          "collect_from": "exp_lambda_nusc_causal", "gain_from": "gain_causal",
                          "source_schedule_origin": "current canonical comma-only exp_lambda_analytic result; target test never used for selection",
                          "stat_unit": "scene (clustered bootstrap over test scenes); draw-level CI non-independent "
                                       "(4 subset seeds reuse same test scenes per fold)",
                          "scope": "temporally-causal; localisation pose + CAN at anchor; NOT real-vehicle sensor-only",
                          "ci_coverage": "scene-level CIs for source-vs-tval, universal-vs-tval, AND universal-vs-source",
                          "note": "CAUSAL: apply comma analytic-shrunk lambda schedule to nuScenes; "
                                  "compare vs universal/target-val/target-oracle"},
           "per_N": {}}
    if os.path.exists(RES):
        try: out["per_N"] = json.load(open(RES)).get("per_N", {})
        except Exception: pass
    for N in N_LIST:
        if N < 2: continue
        lam_src = source_sched[N]
        R = {"source": [], "universal": [], "tval": [], "oracle": []}
        S = {}  # per-scene accumulators for clustered CIs
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
                fit_items = [(g, byg[g]) for g in fit_g]; val_items = [(g, byg[g]) for g in val_g]
                a = gain_causal(fit_items)
                Ff, PHf, GTf, _ = collect(fit_items, a)
                Fv, PHv, GTv, GRPv = collect(val_items, a)
                Ft, PHt, GTt, GRPt = collect(test_items, a)
                if len(Ff) < 50 or len(Fv) < 10 or len(Ft) < 20: continue
                sc = StandardScaler().fit(Ff)
                net = MLPRegressor(hidden_layer_sizes=(48, 24), max_iter=300, alpha=1e-3,
                                   random_state=0, early_stopping=True).fit(sc.transform(Ff), GTf - PHf)
                rv = net.predict(sc.transform(Fv)); rt = net.predict(sc.transform(Ft))
                lh = float(np.clip(analytic_lambda(rv, GTv - PHv), 0, 1.5))
                lam_tval = len(Fv) / (len(Fv) + KAPPA) * lh
                tcurve = {l: rmf(PHt + l * rt, GTt, GRPt) for l in LAMBDAS}
                R["source"].append(rmf(PHt + lam_src * rt, GTt, GRPt))
                R["universal"].append(tcurve[1.0])
                R["tval"].append(rmf(PHt + lam_tval * rt, GTt, GRPt))
                R["oracle"].append(min(tcurve.values()))
                accum_scene(S, "source", *rmf_scene(PHt + lam_src * rt, GTt, GRPt))
                accum_scene(S, "universal", *rmf_scene(PHt + 1.0 * rt, GTt, GRPt))
                accum_scene(S, "tval", *rmf_scene(PHt + lam_tval * rt, GTt, GRPt))
        if not R["source"]: continue
        src = np.array(R["source"]); uni = np.array(R["universal"]); tv = np.array(R["tval"])

        def ci_draw(d):  # NON-independent draw-level CI (seeds reuse test scenes); kept for comparison only
            bo = [float(np.mean(d[np.random.default_rng(k).integers(0, len(d), len(d))])) for k in range(3000)]
            return [float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))]
        # PRIMARY scene-clustered CIs
        sc_src = collapse_scene(S, "source"); sc_uni = collapse_scene(S, "universal"); sc_tv = collapse_scene(S, "tval")
        ci_src_tv = scene_level_ci(sc_src, sc_tv)
        ci_uni_tv = scene_level_ci(sc_uni, sc_tv)
        ci_uni_src = scene_level_ci(sc_uni, sc_src)   # P0 fix: previously missing source-vs-universal CI
        out["per_N"][str(N)] = {
            "N": N, "n_draws": int(len(src)), "lambda_source": lam_src,
            "stat_unit": "scene (clustered); draw-level CI non-independent",
            "fde_source": float(np.mean(src)), "fde_universal": float(np.mean(uni)),
            "fde_target_val": float(np.mean(tv)), "fde_target_oracle": float(np.mean(R["oracle"])),
            "source_minus_tval": float(np.mean(src - tv)),
            "source_minus_tval_CI95_scene": (ci_src_tv or {}).get("CI95"),
            "source_minus_tval_CI95_draw_nonindep": ci_draw(src - tv),
            "universal_minus_tval": float(np.mean(uni - tv)),
            "universal_minus_tval_CI95_scene": (ci_uni_tv or {}).get("CI95"),
            "universal_minus_tval_CI95_draw_nonindep": ci_draw(uni - tv),
            "source_vs_universal": float(np.mean(uni - src)),
            "universal_minus_source_CI95_scene": (ci_uni_src or {}).get("CI95"),
            "universal_minus_source_detail": ci_uni_src}
        json.dump(out, open(RES, "w"), indent=2)
        o = out["per_N"][str(N)]
        us = o["universal_minus_source_CI95_scene"] or [float("nan"), float("nan")]
        print("N=%2d lam_src=%.2f | source=%.3f universal=%.3f tval=%.3f oracle=%.3f | uni-src=%+.3f sceneCI[%.3f,%.3f] nsc=%s"
              % (N, o["lambda_source"], o["fde_source"], o["fde_universal"], o["fde_target_val"],
                 o["fde_target_oracle"], o["source_vs_universal"], us[0], us[1],
                 (ci_uni_src or {}).get("n_scenes")))
    out["N_list"] = sorted((int(k) for k in out["per_N"]), key=int)
    json.dump(out, open(RES, "w"), indent=2); print("saved ->", RES)


if __name__ == "__main__":
    main()
