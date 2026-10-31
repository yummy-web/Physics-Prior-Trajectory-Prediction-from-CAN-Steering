"""
exp_anisotropy_v2.py  (revision v2, 2026-10)  --  R2-9 / R1-11 anisotropy, rigorous

Fixes the earlier anisotropy script (which analysed ONLY the physics prior and did
no stratification). Now, under the strictly causal protocol (causal_eval):
  * both the causal PHYSICS prior AND the causal RESIDUAL hybrid are decomposed
    into longitudinal / lateral error at the final step;
  * reported for CAUSAL and ORACLE modes (to show the anisotropy FLIP);
  * stratified by horizon {1,2,3 s} and by speed tercile;
  * grouped-bootstrap CI over routes/scenes.

Conclusion is deliberately shrunk: the dominant error axis is regime-dependent,
not a universal "steering makes lateral tighter" law.
"""
import os, json
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import causal_protocol as P
import causal_eval as E
import models_core as M

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "exp_anisotropy_v2_results.json")


def latlon_of_path(path_local, df, i0, hs_eval):
    """lon/lat error at step hs_eval (1..hs) in anchor-local frame.
    local x axis = heading (longitudinal), y = lateral."""
    gt = E.gt_path_local(df, i0, hs_eval)
    ex = path_local[hs_eval - 1, 0] - gt[hs_eval - 1, 0]
    ey = path_local[hs_eval - 1, 1] - gt[hs_eval - 1, 1]
    return abs(ex), abs(ey)   # |lon|, |lat|


def run(dataset, is_comma, dt, stride, max_groups=None):
    import pickle
    cache = ".comma_cache.pkl" if is_comma else ".nusc_cache.pkl"
    data = pickle.load(open(os.path.join(os.path.dirname(__file__), cache), "rb"))
    if max_groups and len(data) > max_groups:
        ks = list(data.keys()); np.random.default_rng(7).shuffle(ks); data = {k: data[k] for k in ks[:max_groups]}
    hs = int(round(3.0 / dt))
    groups = P.group_keys(data, "comma" if is_comma else "nuscenes")
    gids = list(groups.keys()); rng = np.random.default_rng(0); rng.shuffle(gids)
    folds = [gids[i::3] for i in range(3)]
    horizons = {"1s": int(round(1.0 / dt)), "2s": int(round(2.0 / dt)), "3s": hs}

    # speed terciles (anchor speed)
    rec = {m: {hz: {"lon": [], "lat": [], "grp": [], "spd": []} for hz in horizons}
           for m in ["physics_causal", "physics_oracle", "residual_causal"]}

    for fo in range(3):
        test_g = set(folds[fo]); tr = [k for g in gids if g not in test_g for k in groups[g]]
        te = [k for g in test_g for k in groups[g]]
        a = E.gain_curv([data[k] for k in tr], is_comma=is_comma)
        # train residual
        Ftr = np.array([E.features_causal(data[s], i0, hs, dt) for s in tr for i0 in E.anchors_for(data[s], hs, stride)])
        GTtr = np.array([E.target_local(data[s], i0, hs) for s in tr for i0 in E.anchors_for(data[s], hs, stride)])
        PHtr = np.array([E.physics_endpoint_local(data[s], i0, hs, dt, a) for s in tr for i0 in E.anchors_for(data[s], hs, stride)])
        sc = StandardScaler().fit(Ftr)
        net = MLPRegressor(hidden_layer_sizes=(48, 24), max_iter=400, alpha=1e-3, random_state=0, early_stopping=True).fit(sc.transform(Ftr), GTtr - PHtr)
        for s in te:
            df = data[s]; g = E.group_of(df, s, is_comma)
            sstate = E.causal_state(df); v_can = sstate["v_can"]
            for i0 in E.anchors_for(df, hs, stride):
                pc = E.physics_path_local(df, i0, hs, dt, a)
                po = E.oracle_path_local(df, i0, hs, dt, a) if hasattr(E, "oracle_path_local") else None
                feat = np.array([E.features_causal(df, i0, hs, dt)])
                r = net.predict(sc.transform(feat))[0]
                # residual hybrid full path = physics path + residual applied at endpoint only;
                # for lon/lat we evaluate at endpoint (hs); for 1s/2s use physics path (residual is endpoint-trained)
                for hz, hv in horizons.items():
                    lo, la = latlon_of_path(pc, df, i0, hv)
                    rec["physics_causal"][hz]["lon"].append(lo); rec["physics_causal"][hz]["lat"].append(la)
                    rec["physics_causal"][hz]["grp"].append(g); rec["physics_causal"][hz]["spd"].append(v_can[i0])
                    if po is not None:
                        lo2, la2 = latlon_of_path(po, df, i0, hv)
                        rec["physics_oracle"][hz]["lon"].append(lo2); rec["physics_oracle"][hz]["lat"].append(la2)
                        rec["physics_oracle"][hz]["grp"].append(g); rec["physics_oracle"][hz]["spd"].append(v_can[i0])
                # residual hybrid only at endpoint (3s)
                pc_end = pc.copy(); pc_end[hs - 1] = pc[hs - 1] + r
                lo, la = latlon_of_path(pc_end, df, i0, hs)
                rec["residual_causal"]["3s"]["lon"].append(lo); rec["residual_causal"]["3s"]["lat"].append(la)
                rec["residual_causal"]["3s"]["grp"].append(g); rec["residual_causal"]["3s"]["spd"].append(v_can[i0])

    def ci(vals, grp):
        m, lo, hi = M.grouped_bootstrap_ci(np.array(vals), np.array(grp), n_boot=600)
        return {"mean": m, "ci_lo": lo, "ci_hi": hi}

    out = {"dt": dt, "hs": hs, "stride": stride}
    for mdl, hzs in rec.items():
        out[mdl] = {}
        for hz, d in hzs.items():
            if not d["lon"]:
                continue
            lon = np.array(d["lon"]); lat = np.array(d["lat"]); grp = np.array(d["grp"]); spd = np.array(d["spd"])
            row = {"lon": ci(lon, grp), "lat": ci(lat, grp), "lat_over_lon": float(np.mean(lat) / (np.mean(lon) + 1e-9))}
            # speed-tercile stratification (at 3s only to keep it compact)
            if hz == "3s" and len(spd) > 30:
                q = np.quantile(spd, [1/3, 2/3]); strat = {}
                for name, m in [("low", spd < q[0]), ("mid", (spd >= q[0]) & (spd < q[1])), ("high", spd >= q[1])]:
                    if m.sum() > 10:
                        strat[name] = {"lat_over_lon": float(np.mean(lat[m]) / (np.mean(lon[m]) + 1e-9)),
                                       "lon": float(np.mean(lon[m])), "lat": float(np.mean(lat[m])), "n": int(m.sum())}
                row["speed_strata"] = strat
            out[mdl][hz] = row
    return out


def main():
    out = {"note": "lat/lon>1 => lateral error dominates; causal uses one-sided state"}
    if os.path.exists(RESULTS):
        try: out.update(json.load(open(RESULTS)))
        except Exception: pass
    if "comma2k19" not in out:
        out["comma2k19"] = run("comma", True, 0.05, 20)
        json.dump(out, open(RESULTS, "w"), indent=2)
        print("[comma] done")
    if "nuscenes" not in out:
        out["nuscenes"] = run("nuscenes", False, 0.02, 12, max_groups=200)
        json.dump(out, open(RESULTS, "w"), indent=2)
        print("[nuscenes] done")
    # print summary
    for ds in ["comma2k19", "nuscenes"]:
        if ds not in out: continue
        print("\n[%s] lat/lon ratio (3s):" % ds)
        for mdl in ["physics_causal", "physics_oracle", "residual_causal"]:
            if mdl in out[ds] and "3s" in out[ds][mdl]:
                r = out[ds][mdl]["3s"]
                print("  %-18s lon=%.3f lat=%.3f  lat/lon=%.2f" % (mdl, r["lon"]["mean"], r["lat"]["mean"], r["lat_over_lon"]))
                if "speed_strata" in r:
                    for k, v in r["speed_strata"].items():
                        print("      speed-%s lat/lon=%.2f (lon=%.2f lat=%.2f)" % (k, v["lat_over_lon"], v["lon"], v["lat"]))
    json.dump(out, open(RESULTS, "w"), indent=2); print("\nsaved ->", RESULTS)


if __name__ == "__main__":
    main()
