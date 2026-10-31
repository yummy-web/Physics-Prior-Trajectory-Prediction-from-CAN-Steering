"""
exp_lambda_analytic.py  (method upgrade, 2026-10-08)

Promotes grid-search lambda into a CLOSED-FORM estimator. For pred = p + lambda*r,
least-squares over a validation set gives
    lambda_hat = sum_i <r_i, (y_i - p_i)>  /  sum_i <r_i, r_i>
(inner products over the 2-D local endpoint). Small-sample shrinkage:
    lambda_shrunk = n/(n+kappa) * clip(lambda_hat, 0, 1.5)
so scarce validation shrinks toward 0 (prior), abundant data -> full estimate.

Nested grouped CV (same splits as exp_lambda_usability): inner val routes estimate
lambda analytically, untouched outer test evaluates. Compares on SAME test set:
  prior(0) / full(1) / grid-val-selected / ANALYTIC / analytic+shrink / oracle.

comma, causal protocol, 3 folds x 4 seeds, N in {3,5,10,20}.
"""
import os, json, datetime
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import comma_loader as C
import causal_protocol as P
import causal_eval as E

RES = os.path.join(os.path.dirname(__file__), "..", "results", "exp_lambda_analytic_results.json")
DT = 0.05; H = 3.0; HS = int(round(H / DT)); STRIDE = 15
N_LIST = [3, 5, 10, 20]; FOLDS = 3; SUBSET_SEEDS = 4
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
KAPPA = 300   # FIXED prior in RAW validation-anchor-count units (overlapping stride-5
              # windows -> NOT an effective independent sample size). Declared before test
              # eval; defensibility via kappa_sweep, NOT post-hoc picked on the test set.


def build(data, keys, a):
    PH, GT, GRP, F = [], [], [], []
    for s in keys:
        df = data[s]; route = E.route_of(df["seg"].iloc[0])
        for i0 in E.anchors_for(df, HS, STRIDE):
            F.append(E.features_causal(df, i0, HS, DT))
            PH.append(E.physics_endpoint_local(df, i0, HS, DT, a))
            GT.append(E.target_local(df, i0, HS)); GRP.append(route)
    return np.array(F), np.array(PH), np.array(GT), np.array(GRP)


def route_mean_fde(P_, G_, grp):
    fde = np.hypot(P_[:, 0] - G_[:, 0], P_[:, 1] - G_[:, 1])
    return float(np.mean([np.mean(fde[grp == g]) for g in np.unique(grp)]))


def analytic_lambda(r, resid_err):
    """r, resid_err are (n,2). lambda_hat = sum<r,e>/sum<r,r>, e=(y-p)."""
    num = float(np.sum(r * resid_err)); den = float(np.sum(r * r)) + 1e-9
    return num / den


def main():
    import sys
    data = C.load_all_cached(verbose=False)
    groups = P.group_keys(data, "comma"); gids = list(groups.keys())
    rng = np.random.default_rng(0); rng.shuffle(gids)
    folds = [gids[i::FOLDS] for i in range(FOLDS)]
    out = {"horizon_s": H, "N_list": N_LIST, "kappa": KAPPA,
           "provenance": {"run_date": datetime.datetime.now().isoformat(timespec="seconds"),
                          "folds": FOLDS, "subset_seeds": SUBSET_SEEDS,
                          "note": "closed-form lambda = <r,y-p>/<r,r> + small-sample shrinkage; nested grouped CV"},
           "per_N": {}}
    if os.path.exists(RES):
        try: out["per_N"] = json.load(open(RES)).get("per_N", {})
        except Exception: pass
    for N in N_LIST:
        if N < 2: continue
        R = {"prior": [], "full": [], "grid": [], "ana": [], "anash": [], "oracle": [],
             "lhat_grid": [], "lhat_ana": [], "lhat_anash": []}
        for fo in range(FOLDS):
            test_g = set(folds[fo]); pool = [g for g in gids if g not in test_g]
            if len(pool) < N: continue
            test_keys = sum([groups[g] for g in test_g], [])
            for ss in range(SUBSET_SEEDS):
                r2 = np.random.default_rng(100 * fo + ss)
                sub = list(r2.choice(pool, size=N, replace=False))
                n_val = max(1, int(np.ceil(N / 3.0)))
                val_g = list(r2.choice(sub, size=n_val, replace=False))
                fit_g = [g for g in sub if g not in val_g]
                if not fit_g: continue
                fit_keys = sum([groups[g] for g in fit_g], [])
                val_keys = sum([groups[g] for g in val_g], [])
                a = E.gain_curv([data[k] for k in fit_keys], is_comma=True)
                Ff, PHf, GTf, _ = build(data, fit_keys, a)
                Fv, PHv, GTv, GRPv = build(data, val_keys, a)
                Ft, PHt, GTt, GRPt = build(data, test_keys, a)
                if len(Ff) < 20 or len(Fv) < 5 or len(Ft) < 20: continue
                sc = StandardScaler().fit(Ff)
                net = MLPRegressor(hidden_layer_sizes=(48, 24), max_iter=250, alpha=1e-3,
                                   random_state=0, early_stopping=True).fit(sc.transform(Ff), GTf - PHf)
                rv = net.predict(sc.transform(Fv)); rt = net.predict(sc.transform(Ft))
                # analytic lambda on validation
                lh = analytic_lambda(rv, GTv - PHv)
                lh_c = float(np.clip(lh, 0.0, 1.5))
                nval = len(Fv)
                lh_sh = nval / (nval + KAPPA) * lh_c
                # grid val-selected
                vcurve = {l: route_mean_fde(PHv + l * rv, GTv, GRPv) for l in LAMBDAS}
                lhat_grid = min(vcurve, key=vcurve.get)
                # test
                tcurve = {l: route_mean_fde(PHt + l * rt, GTt, GRPt) for l in LAMBDAS}
                R["prior"].append(tcurve[0.0]); R["full"].append(tcurve[1.0])
                R["grid"].append(tcurve[lhat_grid])
                R["ana"].append(route_mean_fde(PHt + lh_c * rt, GTt, GRPt))
                R["anash"].append(route_mean_fde(PHt + lh_sh * rt, GTt, GRPt))
                R["oracle"].append(min(tcurve.values()))
                R["lhat_grid"].append(lhat_grid); R["lhat_ana"].append(lh_c); R["lhat_anash"].append(lh_sh)
        if not R["full"]: continue
        a_full = np.array(R["full"]); a_ana = np.array(R["ana"]); a_sh = np.array(R["anash"])
        def ci(d):
            bo = [float(np.mean(d[np.random.default_rng(k).integers(0, len(d), len(d))])) for k in range(3000)]
            return float(np.mean(d)), [float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))]
        g_sh, ci_sh = ci(a_full - a_sh)   # shrink-analytic vs full lambda=1
        out["per_N"][str(N)] = {
            "N": N, "n_draws": int(len(a_full)),
            "fde_prior": float(np.mean(R["prior"])), "fde_full": float(np.mean(a_full)),
            "fde_grid": float(np.mean(R["grid"])), "fde_analytic": float(np.mean(a_ana)),
            "fde_analytic_shrunk": float(np.mean(a_sh)), "fde_oracle": float(np.mean(R["oracle"])),
            "shrunk_gain_vs_full": g_sh, "shrunk_gain_CI95": ci_sh,
            "lhat_grid_mean": float(np.mean(R["lhat_grid"])),
            "lhat_analytic_mean": float(np.mean(R["lhat_ana"])),
            "lhat_shrunk_mean": float(np.mean(R["lhat_anash"]))}
        json.dump(out, open(RES, "w"), indent=2)
        o = out["per_N"][str(N)]
        print("N=%2d prior=%.3f full=%.3f grid=%.3f ana=%.3f anaShr=%.3f oracle=%.3f | shr_vs_full=%.3f CI[%.3f,%.3f] | lam grid=%.2f ana=%.2f shr=%.2f"
              % (N, o["fde_prior"], o["fde_full"], o["fde_grid"], o["fde_analytic"], o["fde_analytic_shrunk"],
                 o["fde_oracle"], o["shrunk_gain_vs_full"], ci_sh[0], ci_sh[1],
                 o["lhat_grid_mean"], o["lhat_analytic_mean"], o["lhat_shrunk_mean"]))
    json.dump(out, open(RES, "w"), indent=2); print("saved ->", RES)


if __name__ == "__main__":
    main()