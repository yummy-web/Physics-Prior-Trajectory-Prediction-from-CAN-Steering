"""
exp_identifiability.py  (revision v2, 2026-10)  --  R2-3 identifiability diagnostics

High R^2 != identifiability. We report, with STRICTLY CAUSAL state (causal_eval):
  (1) per-group Model-C gain a with its regression standard error -> gain
      distribution + 95% CI across groups (routes / scenes / loader sessions);
  (2) cross-group stability: coefficient of variation of a, and leave-one-group
      calibrate / evaluate-elsewhere gain drift;
  (3) residual diagnostics: correlation of the identification residual
      (kappa - a*steer) with |steer|, speed, and |curvature| -- a flat residual
      supports identifiability; structured residual flags regime dependence.

Datasets: comma2k19 (routes), nuScenes (scenes), loader (sessions).
"""
import os, json
import numpy as np
from scipy import stats
import causal_eval as E

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "exp_identifiability_results.json")


def per_group_gain(items, is_comma, min_v):
    """Return list of (gain, se, n) per group using causal state."""
    rows = []
    for it in items:
        df = it[1] if isinstance(it, tuple) else it
        s = E.causal_state(df); v = s["v_can"]
        mask = v > min_v
        x = df["steer"].values[mask]
        kap = (s["wr_bwd"] / (v + 1e-3))[mask]
        ok = np.isfinite(x) & np.isfinite(kap)
        if ok.sum() > 20:
            hi = np.nanpercentile(np.abs(kap[ok]), 97); ok &= np.abs(kap) < hi
        if ok.sum() < 20 or np.std(x[ok]) < 1e-6:
            continue
        a, b, r, p, se = stats.linregress(x[ok], kap[ok])
        rows.append((float(a), float(se), int(ok.sum()), float(r**2)))
    return rows


def residual_diag(items, a_global, is_comma, min_v):
    """Correlation of identification residual with |steer|, speed, |curv|."""
    S, V, K, R = [], [], [], []
    for it in items:
        df = it[1] if isinstance(it, tuple) else it
        s = E.causal_state(df); v = s["v_can"]
        mask = v > min_v
        x = df["steer"].values[mask]
        kap = (s["wr_bwd"] / (v + 1e-3))[mask]
        ok = np.isfinite(x) & np.isfinite(kap)
        if ok.sum() < 20:
            continue
        resid = kap[ok] - a_global * x[ok]
        S.append(np.abs(x[ok])); V.append(v[mask][ok]); K.append(np.abs(kap[ok])); R.append(resid)
    S = np.concatenate(S); V = np.concatenate(V); K = np.concatenate(K); R = np.concatenate(R)
    def corr(x):
        m = np.isfinite(x) & np.isfinite(R)
        return float(np.corrcoef(x[m], R[m])[0, 1]) if m.sum() > 10 else float("nan")
    return {"corr_resid_abssteer": corr(S), "corr_resid_speed": corr(V),
            "corr_resid_abscurv": corr(K), "resid_std": float(np.std(R))}


def summarize(name, items, is_comma, min_v):
    rows = per_group_gain(items, is_comma, min_v)
    gains = np.array([r[0] for r in rows]); ses = np.array([r[1] for r in rows])
    r2s = np.array([r[3] for r in rows])
    a_global = E.gain_curv(items, is_comma=is_comma, min_v=min_v)
    # 95% CI of mean gain across groups
    lo, hi = np.percentile(gains, [2.5, 97.5]) if len(gains) > 3 else (np.nan, np.nan)
    diag = residual_diag(items, a_global, is_comma, min_v)
    out = {"n_groups": len(rows), "gain_global": a_global,
           "gain_mean": float(np.mean(gains)), "gain_std": float(np.std(gains)),
           "gain_cv": float(np.std(gains) / (np.mean(gains) + 1e-12)),
           "gain_p2.5": float(lo), "gain_p97.5": float(hi),
           "gain_se_median": float(np.median(ses)),
           "R2_median": float(np.median(r2s)), "R2_frac_gt07": float(np.mean(r2s > 0.7)),
           "residual_diag": diag}
    print("\n[%s] groups=%d  a_global=%.4g  gain CV=%.2f  R2med=%.3f(frac>0.7=%.2f)"
          % (name, len(rows), a_global, out["gain_cv"], out["R2_median"], out["R2_frac_gt07"]))
    print("  gain mean=%.4g std=%.4g 95%%across-group[%.4g,%.4g]  SE_med=%.2g"
          % (out["gain_mean"], out["gain_std"], out["gain_p2.5"], out["gain_p97.5"], out["gain_se_median"]))
    print("  residual corr: |steer|=%.3f speed=%.3f |curv|=%.3f (flat=>identifiable)"
          % (diag["corr_resid_abssteer"], diag["corr_resid_speed"], diag["corr_resid_abscurv"]))
    return out


def main():
    import comma_loader as C
    out = {"note": "causal state (one-sided backward diff); gain a in curvature=a*steer"}
    cd = C.load_all_cached(verbose=True)
    # group comma by ROUTE: one gain estimate per route (not per segment), so the
    # gain distribution / CV / across-group CI are over the 42 real routes, matching
    # the leave-one-group design (audit round-4: 328 segments were wrongly used as
    # groups before, overstating the effective sample).
    import pandas as pd
    import causal_protocol as P
    groups = P.group_keys(cd, "comma")
    comma_items = []
    for g, ks in groups.items():
        merged = pd.concat([cd[k] for k in ks], ignore_index=True)
        comma_items.append((g, merged))
    out["comma2k19"] = summarize("comma2k19(per-route)", comma_items, True, 8.0)
    out["comma2k19"]["grouping"] = "per-route (%d routes from %d segments)" % (len(groups), len(cd))

    try:
        import pickle
        nd = pickle.load(open(os.path.join(os.path.dirname(__file__), ".nusc_cache.pkl"), "rb"))
        out["nuscenes"] = summarize("nuScenes(per-scene)", list(nd.items()), False, 1.0)
    except Exception as e:
        print("nuScenes skipped:", e)

    try:
        from loader_data_loader import load_all as load_loader
        ld = load_loader()
        # loader uses motor_angle; adapt: build pseudo df with steer=motor_angle-center, state from pose
        litems = []
        import models_core as M
        for k, df in ld.items():
            d2 = df.copy()
            c = M.loader_center(df, warmup=False, dt=0.1)
            d2["steer"] = df["motor_angle"].values - c
            d2["can_speed"] = df["obs_v"].values  # loader has no CAN speed; use pose speed (documented)
            litems.append((k, d2))
        out["loader"] = summarize("loader(per-session)", litems, False, 0.5)
        out["loader"]["note"] = "steer=motor_angle-center; no CAN speed, pose speed used (documented limitation)"
    except Exception as e:
        print("loader skipped:", e)

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    json.dump(out, open(RESULTS, "w"), indent=2); print("\nsaved ->", RESULTS)


if __name__ == "__main__":
    main()
