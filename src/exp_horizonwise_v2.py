"""
exp_horizonwise_v2.py  (revision v2, 2026-10)  --  R2-8 horizon-wise learner metrics

The main table reports ADE + horizon-wise FDE for CV/CTRV/CTRA/Physics, but the
learners (MLP, residual) only had a 3 s endpoint FDE, because the residual was
endpoint-trained. R2-8 asks for horizon-wise ADE/FDE for ALL methods.

Here we train SEPARATE endpoint learners at horizons {1,2,3} s (direct MLP and
physics-residual), on the strictly-causal unified-anchor protocol, route-grouped
3-fold, multi-seed. This yields:
  * horizon-wise FDE@{1,2,3}s for MLP and Residual;
  * a horizon-sampled ADE = mean of FDE@{1,2,3}s (clearly labelled as sampled,
    not continuous), comparable to the baselines' ADE.
Grouped-bootstrap CI over routes/scenes.
"""
import os, json, datetime
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import causal_protocol as P
import causal_eval as E
import models_core as M

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "exp_horizonwise_v2_results.json")
DT = {"comma": 0.05, "nuscenes": 0.02}
STRIDE = {"comma": 20, "nuscenes": 12}
HSEC = [1.0, 2.0, 3.0]
MODEL_SEEDS = 2


def load(ds):
    import pickle
    c = ".comma_cache.pkl" if ds == "comma" else ".nusc_cache.pkl"
    return pickle.load(open(os.path.join(os.path.dirname(__file__), c), "rb"))


def run(ds, is_comma, max_groups=None):
    data = load(ds)
    if max_groups and len(data) > max_groups:
        ks = list(data.keys()); np.random.default_rng(7).shuffle(ks); data = {k: data[k] for k in ks[:max_groups]}
    dt = DT[ds]; stride = STRIDE[ds]
    groups = P.group_keys(data, "comma" if is_comma else "nuscenes")
    gids = list(groups.keys()); rng = np.random.default_rng(0); rng.shuffle(gids)
    folds = [gids[i::3] for i in range(3)]

    # per method/horizon accumulate per-anchor FDE + group
    acc = {m: {h: {"fde": [], "grp": []} for h in HSEC} for m in ["MLP", "Residual"]}
    for fo in range(3):
        test_g = set(folds[fo]); tr = [k for g in gids if g not in test_g for k in groups[g]]
        te = [k for g in test_g for k in groups[g]]
        a = E.gain_curv([data[k] for k in tr], is_comma=is_comma)
        for H in HSEC:
            hs = int(round(H / dt))
            Ftr = np.array([E.features_causal(data[s], i0, hs, dt) for s in tr for i0 in E.anchors_for(data[s], hs, stride)])
            GTtr = np.array([E.target_local(data[s], i0, hs) for s in tr for i0 in E.anchors_for(data[s], hs, stride)])
            PHtr = np.array([E.physics_endpoint_local(data[s], i0, hs, dt, a) for s in tr for i0 in E.anchors_for(data[s], hs, stride)])
            anchors = [(s, i0) for s in te for i0 in E.anchors_for(data[s], hs, stride)]
            grp = np.array([E.group_of(data[s], s, is_comma) for s, _ in anchors])
            Fte = np.array([E.features_causal(data[s], i0, hs, dt) for s, i0 in anchors])
            PHte = np.array([E.physics_endpoint_local(data[s], i0, hs, dt, a) for s, i0 in anchors])
            sc = StandardScaler().fit(Ftr)
            mlp_fd = np.zeros(len(anchors)); res_fd = np.zeros(len(anchors))
            for ms in range(MODEL_SEEDS):
                nm = MLPRegressor(hidden_layer_sizes=(48, 24), max_iter=350, alpha=1e-3, random_state=ms, early_stopping=True).fit(sc.transform(Ftr), GTtr)
                nr = MLPRegressor(hidden_layer_sizes=(48, 24), max_iter=350, alpha=1e-3, random_state=ms, early_stopping=True).fit(sc.transform(Ftr), GTtr - PHtr)
                pm = nm.predict(sc.transform(Fte)); pr = PHte + nr.predict(sc.transform(Fte))
                mlp_fd += np.array([E.local_to_fde(pm[j], data[anchors[j][0]], anchors[j][1], hs) for j in range(len(anchors))])
                res_fd += np.array([E.local_to_fde(pr[j], data[anchors[j][0]], anchors[j][1], hs) for j in range(len(anchors))])
            mlp_fd /= MODEL_SEEDS; res_fd /= MODEL_SEEDS
            acc["MLP"][H]["fde"].extend(list(mlp_fd)); acc["MLP"][H]["grp"].extend(list(grp))
            acc["Residual"][H]["fde"].extend(list(res_fd)); acc["Residual"][H]["grp"].extend(list(grp))

    tbl = {}
    for m in ["MLP", "Residual"]:
        tbl[m] = {}
        fde_by_h = []
        for H in HSEC:
            fde = np.array(acc[m][H]["fde"]); grp = np.array(acc[m][H]["grp"])
            mean, lo, hi = M.grouped_bootstrap_ci(fde, grp, n_boot=800)
            tbl[m]["fde@%gs" % H] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
            fde_by_h.append(mean)
        tbl[m]["ade_sampled"] = float(np.mean(fde_by_h))  # mean of FDE@{1,2,3}s
    # PER-HORIZON paired Residual-vs-MLP route-level diff (identical anchors/order),
    # so each horizon's advantage comes with its own CI + p, not just observed means.
    paired = {}
    for H in HSEC:
        rf = np.array(acc["Residual"][H]["fde"]); mf = np.array(acc["MLP"][H]["fde"])
        rg = np.array(acc["Residual"][H]["grp"]); mg = np.array(acc["MLP"][H]["grp"])
        paired["fde@%gs" % H] = E.paired_route_diff(rf, mf, rg) if np.array_equal(rg, mg) \
            else {"note": "anchor order mismatch"}
    return {"dt": dt, "stride": stride, "model_seeds": MODEL_SEEDS, "horizons_s": HSEC,
            "table": tbl, "paired_residual_vs_mlp_by_horizon": paired}


def main():
    out = {"note": "horizon-wise FDE for learners; ade_sampled = mean FDE@{1,2,3}s (sampled, not continuous)",
           "provenance": {"run_date": datetime.datetime.now().isoformat(timespec="seconds"), "causal_heading": True}}
    if os.path.exists(RESULTS):
        try:
            prev = json.load(open(RESULTS))
            for ds in ["comma", "nuscenes"]:
                if ds in prev and prev.get("provenance", {}).get("causal_heading") \
                        and "paired_residual_vs_mlp_by_horizon" in prev[ds]:
                    out[ds] = prev[ds]
        except Exception: pass
    for ds, isc, mg in [("comma", True, None), ("nuscenes", False, 200)]:
        if ds in out: print("[skip]", ds); continue
        out[ds] = run(ds, isc, mg)
        json.dump(out, open(RESULTS, "w"), indent=2)
        t = out[ds]["table"]
        for m in ["MLP", "Residual"]:
            cells = " ".join("%s=%.3f" % (h, t[m][h]["mean"]) for h in ["fde@1s", "fde@2s", "fde@3s"])
            print("  [%s] %-9s %s  ADE_sampled=%.3f" % (ds, m, cells, t[m]["ade_sampled"]))
        for h, v in out[ds]["paired_residual_vs_mlp_by_horizon"].items():
            if "wilcoxon_p" in v:
                print("    paired Res-MLP %-7s diff=%+.3f[%.3f,%.3f] p=%.2g frac=%.2f"
                      % (h, v["mean_diff_a_minus_b"], v["ci_lo"], v["ci_hi"], v["wilcoxon_p"], v["frac_routes_a_better"]))
    json.dump(out, open(RESULTS, "w"), indent=2); print("saved ->", RESULTS)


if __name__ == "__main__":
    main()
