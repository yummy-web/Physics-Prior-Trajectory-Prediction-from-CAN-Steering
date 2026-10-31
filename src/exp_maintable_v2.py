"""
exp_maintable_v2.py  (revision v2, 2026-10)  --  RIGOROUS causal main table

Fixes audit F2 for the main comparison (comma + nuScenes):

  * UNIFIED anchors: every method (CV, CTRV, CTRA, smoothed-CTRV, Physics,
    MLP, Residual) is scored on the SAME anchor list per segment
    (causal_eval.anchors_for), identical history requirement.
  * STRICTLY CAUSAL state everywhere (one-sided backward differences; CAN speed).
  * Multi-seed learners (MLP/residual over model seeds) inside route-grouped
    3-fold CV; gain fit per train fold only.
  * Metrics: ADE@3s and FDE@3s, horizon-wise FDE at {1,2,3}s, 50/90 quantiles,
    grouped-bootstrap CI over routes/scenes.
  * Paired route-level significance of Residual vs the BEST non-hybrid baseline.
  * "EKF" is reported as "smoothed-CTRV" (naming fixed).
  * Physics also reported in ORACLE mode (future steer x pose speed) as an
    explicit upper bound, never the headline.

This supersedes exp_causal_main.py / exp_causal_nuscenes.py as the main table.
"""
import os, json
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import causal_protocol as P
import causal_eval as E

DT = {"comma": 0.05, "nuscenes": 0.02}
STRIDE = {"comma": 20, "nuscenes": 12}   # match the reproducible in-window run
H = 3.0
MODEL_SEEDS = 2   # matches saved JSON; raise for final-camera-ready on a faster host
KIN = ["CV", "CTRV", "CTRA", "EKF"]
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "exp_maintable_v2_results.json")


def load(dataset):
    import pickle
    cache = ".comma_cache.pkl" if dataset == "comma" else ".nusc_cache.pkl"
    return pickle.load(open(os.path.join(os.path.dirname(__file__), cache), "rb"))


def design(data, keys, hs, dt, a, is_comma):
    """Shared design on unified anchors: features, physics path, meta, groups."""
    F, META, GRP = [], [], []
    for s in keys:
        df = data[s]; g = E.group_of(df, s, is_comma)
        for i0 in E.anchors_for(df, hs, STRIDE["comma" if is_comma else "nuscenes"]):
            F.append(E.features_causal(df, i0, hs, dt)); META.append((s, i0)); GRP.append(g)
    return np.array(F), META, np.array(GRP)


def ci(vals, grp):
    import models_core as M
    m, lo, hi = M.grouped_bootstrap_ci(vals, grp, n_boot=1000)
    return {"mean": m, "ci_lo": lo, "ci_hi": hi}


def run(dataset, max_groups=None):
    data = load(dataset); is_comma = (dataset == "comma")
    total_available = len(data)
    sampled = False
    if max_groups and len(data) > max_groups:
        import numpy as _np
        ks = list(data.keys()); _np.random.default_rng(7).shuffle(ks)
        data = {k: data[k] for k in ks[:max_groups]}
        sampled = True
    dt = DT[dataset]; hs = int(round(H / dt))
    keys = list(data.keys())
    groups = P.group_keys(data, "comma" if is_comma else "nuscenes")
    gids = list(groups.keys()); rng = np.random.default_rng(0); rng.shuffle(gids)
    folds = [gids[i::3] for i in range(3)]
    print("[%s] groups=%d anchors-stride=%d hs=%d" % (dataset, len(gids), STRIDE[dataset], hs))

    # accumulate per-anchor errors over the 3 outer folds
    acc = {k: {"ade": [], "fde": [], "fde1": [], "fde2": [], "grp": []} for k in
           KIN + ["Physics", "Physics_oracle", "MLP", "Residual"]}
    h1 = int(round(1.0 / dt)); h2 = int(round(2.0 / dt))

    for fo in range(3):
        test_g = set(folds[fo]); tr_keys = [k for g in gids if g not in test_g for k in groups[g]]
        te_keys = [k for g in test_g for k in groups[g]]
        a = E.gain_curv([data[k] for k in tr_keys], is_comma=is_comma)
        # unified test anchors
        test_anchors = [(s, i0) for s in te_keys for i0 in E.anchors_for(data[s], hs, STRIDE[dataset])]
        tgrp = np.array([E.group_of(data[s], s, is_comma) for s, _ in test_anchors])

        # kinematic + physics on unified anchors
        for s, i0 in test_anchors:
            df = data[s]
            for mdl in KIN:
                p = E.kin_path_local(df, i0, hs, dt, mdl)
                ad, fd, dv = E.ade_fde(p, df, i0, hs)
                acc[mdl]["ade"].append(ad); acc[mdl]["fde"].append(fd)
                acc[mdl]["fde1"].append(dv[h1 - 1]); acc[mdl]["fde2"].append(dv[h2 - 1])
            pp = E.physics_path_local(df, i0, hs, dt, a)
            ad, fd, dv = E.ade_fde(pp, df, i0, hs)
            acc["Physics"]["ade"].append(ad); acc["Physics"]["fde"].append(fd)
            acc["Physics"]["fde1"].append(dv[h1 - 1]); acc["Physics"]["fde2"].append(dv[h2 - 1])
        for k in KIN + ["Physics"]:
            acc[k]["grp"].extend(list(tgrp))

        # oracle physics upper bound on the SAME anchors
        for s, i0 in test_anchors:
            df = data[s]
            po = oracle_path_local(df, i0, hs, dt, a)
            ad, fd, dv = E.ade_fde(po, df, i0, hs)
            acc["Physics_oracle"]["ade"].append(ad); acc["Physics_oracle"]["fde"].append(fd)
            acc["Physics_oracle"]["fde1"].append(dv[h1 - 1]); acc["Physics_oracle"]["fde2"].append(dv[h2 - 1])
        acc["Physics_oracle"]["grp"].extend(list(tgrp))

        # learners (multi-seed), scored on identical anchors
        Ftr, _, _ = design(data, tr_keys, hs, dt, a, is_comma)
        GTtr = np.array([E.target_local(data[s], i0, hs) for s in tr_keys for i0 in E.anchors_for(data[s], hs, STRIDE[dataset])])
        PHtr = np.array([E.physics_endpoint_local(data[s], i0, hs, dt, a) for s in tr_keys for i0 in E.anchors_for(data[s], hs, STRIDE[dataset])])
        Fte = np.array([E.features_causal(data[s], i0, hs, dt) for s, i0 in test_anchors])
        PHte = np.array([E.physics_endpoint_local(data[s], i0, hs, dt, a) for s, i0 in test_anchors])
        sc = StandardScaler().fit(Ftr)
        mlp_fd = np.zeros(len(test_anchors)); res_fd = np.zeros(len(test_anchors))
        for ms in range(MODEL_SEEDS):
            nm = MLPRegressor(hidden_layer_sizes=(48, 24), max_iter=400, alpha=1e-3, random_state=ms, early_stopping=True).fit(sc.transform(Ftr), GTtr)
            nr = MLPRegressor(hidden_layer_sizes=(48, 24), max_iter=400, alpha=1e-3, random_state=ms, early_stopping=True).fit(sc.transform(Ftr), GTtr - PHtr)
            pm = nm.predict(sc.transform(Fte)); pr = PHte + nr.predict(sc.transform(Fte))
            mlp_fd += np.array([E.local_to_fde(pm[j], data[test_anchors[j][0]], test_anchors[j][1], hs) for j in range(len(test_anchors))])
            res_fd += np.array([E.local_to_fde(pr[j], data[test_anchors[j][0]], test_anchors[j][1], hs) for j in range(len(test_anchors))])
        mlp_fd /= MODEL_SEEDS; res_fd /= MODEL_SEEDS
        acc["MLP"]["fde"].extend(list(mlp_fd)); acc["MLP"]["grp"].extend(list(tgrp))
        acc["Residual"]["fde"].extend(list(res_fd)); acc["Residual"]["grp"].extend(list(tgrp))

    # summarize
    tbl = {}
    for k, d in acc.items():
        if not d["fde"]:
            continue
        g = np.array(d["grp"]); fde = np.array(d["fde"])
        row = {"fde": ci(fde, g), "fde_p50": float(np.percentile(fde, 50)), "fde_p90": float(np.percentile(fde, 90)),
               "n": int(len(fde))}
        if d["ade"]:
            row["ade"] = ci(np.array(d["ade"]), g)
            row["fde@1s"] = ci(np.array(d["fde1"]), g); row["fde@2s"] = ci(np.array(d["fde2"]), g)
        tbl[k] = row
    # CTRA = reference kinematic baseline. We CANNOT prove it was pre-registered
    # before seeing results, so this comparison is reported as EXPLORATORY, not a
    # confirmatory pre-specified test (audit round-4). Residual is compared against
    # EVERY non-hybrid baseline on identical anchors, and the family of p-values is
    # Holm-corrected so no single baseline can be cherry-picked post hoc.
    rg = np.array(acc["Residual"]["grp"]); rf = np.array(acc["Residual"]["fde"])
    REFERENCE = "CTRA"
    paired_all = {}
    for b in [k for k in KIN + ["Physics", "MLP"] if k in tbl]:
        bf = np.array(acc[b]["fde"]); bg = np.array(acc[b]["grp"])
        paired_all[b] = E.paired_route_diff(rf, bf, rg) if np.array_equal(rg, bg) else {"note": "anchor order mismatch"}
    # Holm-Bonferroni correction across the baseline family.
    fam = [(b, v["wilcoxon_p"]) for b, v in paired_all.items() if "wilcoxon_p" in v]
    m = len(fam)
    for rank, (b, p) in enumerate(sorted(fam, key=lambda x: x[1])):
        paired_all[b]["p_holm"] = float(min(1.0, p * (m - rank)))
        paired_all[b]["significant_holm_0.05"] = bool(paired_all[b]["p_holm"] < 0.05)
    # descriptive only: strongest baseline on test (reported, never gates the claim).
    cands = {k: tbl[k]["fde"]["mean"] for k in KIN + ["Physics", "MLP"] if k in tbl}
    observed_strongest = min(cands, key=cands.get)
    return {"dt": dt, "hs": hs, "stride": STRIDE[dataset], "model_seeds": MODEL_SEEDS,
            "n_groups_used": len(groups), "n_segments_used": len(data),
            "comparison_status": "EXPLORATORY: reference baseline CTRA not provably "
            "pre-registered; p-values are post-hoc, Holm-corrected across the baseline family",
            "sampling": {"segments_available": total_available, "segments_used": len(data),
                         "subsampled": sampled, "sample_seed": 7,
                         "note": "nuScenes subsampled for in-window compute; raise max_groups on a faster host for the full set"},
            "reference_baseline": REFERENCE,
            "paired_residual_vs_reference": paired_all.get(REFERENCE),
            "paired_residual_vs_all_baselines": paired_all,
            "observed_strongest_baseline_descriptive": observed_strongest,
            "table": tbl}


def oracle_path_local(df, i0, hs, dt, a):
    """ORACLE upper bound path: future steer x future pose-diff speed (leaky)."""
    x = df["x"].values; y = df["y"].values; yaw = np.radians(df["yaw_unwrap"].values)
    st = df["steer"].values; v = df["obs_v"].values; n = len(df)
    px, py, th = x[i0], y[i0], yaw[i0]; path = []
    for k in range(hs):
        j = min(i0 + k, n - 1); th += a * st[j] * v[j] * dt
        px += v[j] * np.cos(th) * dt; py += v[j] * np.sin(th) * dt
        cc, ss = np.cos(-yaw[i0]), np.sin(-yaw[i0])
        path.append([cc * (px - x[i0]) - ss * (py - y[i0]), ss * (px - x[i0]) + cc * (py - y[i0])])
    return np.array(path)


def main():
    import datetime
    out = {"horizon_s": H, "provenance": {"run_date": datetime.datetime.now().isoformat(timespec="seconds"),
                                          "causal_heading": True, "model_seeds": MODEL_SEEDS,
                                          "stride": STRIDE}}
    if os.path.exists(RESULTS):
        try:
            prev = json.load(open(RESULTS))
            for ds in ["comma", "nuscenes"]:
                # resume only cells with the NEW pre-specified-baseline schema
                if ds in prev and prev.get("provenance", {}).get("causal_heading") \
                        and "comparison_status" in prev[ds]:
                    out[ds] = prev[ds]
        except Exception: pass
    for ds, mg in [("comma", None), ("nuscenes", 200)]:
        if ds in out:
            print("[skip]", ds); continue
        out[ds] = run(ds, max_groups=mg)
        json.dump(out, open(RESULTS, "w"), indent=2)
        t = out[ds]["table"]
        for k in KIN + ["Physics", "Physics_oracle", "MLP", "Residual"]:
            if k in t:
                print("  %-15s FDE=%.3f [%.3f,%.3f]  ADE=%s  p50=%.2f p90=%.2f"
                      % (k, t[k]["fde"]["mean"], t[k]["fde"]["ci_lo"], t[k]["fde"]["ci_hi"],
                         ("%.3f" % t[k]["ade"]["mean"]) if "ade" in t[k] else "-",
                         t[k]["fde_p50"], t[k]["fde_p90"]))
        sm = out[ds]["sampling"]
        print("  scenes/segments: used %d of %d available (subsampled=%s)" % (sm["segments_used"], sm["segments_available"], sm["subsampled"]))
        print("  [EXPLORATORY] Res vs all baselines (Holm-corrected):")
        for b, v in out[ds]["paired_residual_vs_all_baselines"].items():
            if "wilcoxon_p" in v:
                print("    vs %-8s diff=%+.3f[%.3f,%.3f] p=%.2g p_holm=%.2g sig=%s frac=%.2f"
                      % (b, v["mean_diff_a_minus_b"], v["ci_lo"], v["ci_hi"],
                         v["wilcoxon_p"], v["p_holm"], v["significant_holm_0.05"], v["frac_routes_a_better"]))
        print("  (descriptive) strongest-on-test baseline =", out[ds]["observed_strongest_baseline_descriptive"])
    json.dump(out, open(RESULTS, "w"), indent=2); print("saved ->", RESULTS)


if __name__ == "__main__":
    main()
