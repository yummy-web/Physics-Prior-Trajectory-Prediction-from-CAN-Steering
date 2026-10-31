"""
exp_gain_transfer_v2.py  (P3 fix, 2026-10-08)

Rigorous prior-saturation: FIXED held-out test set; vary only K = #routes used
to fit the single scalar gain a. Reports FDE on the SAME test set with
route-level 95% CI, for each K. Also reports residual-MLP learning curve on the
SAME test set for contrast (prior appetite ~1 route vs residual appetite many).

Replaces exp_gain_transfer.py, whose K=1..20 draws used DIFFERENT test sets.

comma, causal protocol (causal_eval). test set = 1/3 of routes, fixed by seed 0.
"""
import os, json, datetime
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import comma_loader as C
import causal_protocol as P
import causal_eval as E

RES = os.path.join(os.path.dirname(__file__), "..", "results", "exp_gain_transfer_v2_results.json")
DT = 0.05; H = 3.0; HS = int(round(H / DT)); STRIDE = 15
K_LIST = [1, 2, 3, 5, 10, 20]; SEEDS = 8


def build(data, keys, a):
    PH, GT, GRP, F = [], [], [], []
    for s in keys:
        df = data[s]; route = E.route_of(df["seg"].iloc[0])
        for i0 in E.anchors_for(df, HS, STRIDE):
            PH.append(E.physics_endpoint_local(df, i0, HS, DT, a))
            GT.append(E.target_local(df, i0, HS))
            F.append(E.features_causal(df, i0, HS, DT)); GRP.append(route)
    return np.array(F), np.array(PH), np.array(GT), np.array(GRP)


def route_fde(P_, G_, grp):
    fde = np.hypot(P_[:, 0] - G_[:, 0], P_[:, 1] - G_[:, 1])
    gs = np.unique(grp)
    return np.array([np.mean(fde[grp == g]) for g in gs])


def main():
    data = C.load_all_cached(verbose=False)
    groups = P.group_keys(data, "comma"); gids = list(groups.keys())
    rng = np.random.default_rng(0); rng.shuffle(gids)
    n_test = max(len(gids) // 3, 5)
    test_g = gids[:n_test]; train_pool = gids[n_test:]       # FIXED test set
    test_keys = sum([groups[g] for g in test_g], [])
    out = {"horizon_s": H, "K_list": K_LIST,
           "provenance": {"run_date": datetime.datetime.now().isoformat(timespec="seconds"),
                          "seeds": SEEDS, "n_test_routes": len(test_g), "n_train_pool": len(train_pool),
                          "note": "FIXED held-out test set; vary K = #routes to fit scalar gain a"},
           "prior_saturation": {}, "residual_curve": {}}

    # precompute test ground truth once
    a_all = E.gain_curv([data[k] for k in sum([groups[g] for g in train_pool], [])], is_comma=True)
    out["gain_all_routes"] = float(a_all)

    # ---- A) prior saturation on FIXED test set ----
    for K in K_LIST:
        if K > len(train_pool): continue
        per_seed_mean = []; a_vals = []
        test_route_fde_acc = []
        for ss in range(SEEDS):
            r2 = np.random.default_rng(10 * ss + K)
            fit_g = list(r2.choice(train_pool, size=K, replace=False))
            a = E.gain_curv([data[k] for k in sum([groups[g] for g in fit_g], [])], is_comma=True)
            a_vals.append(a)
            _, PH, GT, GRP = build(data, test_keys, a)
            rf = route_fde(PH, GT, GRP)
            test_route_fde_acc.append(rf); per_seed_mean.append(float(np.mean(rf)))
        # route-level CI: average route-fde across seeds, bootstrap over routes
        mean_route = np.mean(np.vstack(test_route_fde_acc), axis=0)
        rng2 = np.random.default_rng(1)
        bo = [float(np.mean(mean_route[rng2.integers(0, len(mean_route), len(mean_route))])) for _ in range(3000)]
        out["prior_saturation"][str(K)] = {
            "K": K, "fde_mean": float(np.mean(per_seed_mean)),
            "fde_route_CI95": [float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))],
            "gain_a_mean": float(np.mean(a_vals)), "gain_a_std": float(np.std(a_vals))}
        json.dump(out, open(RES, "w"), indent=2)
        o = out["prior_saturation"][str(K)]
        print("PRIOR  K=%2d  FDE=%.3f CI[%.3f,%.3f]  a=%.4f+-%.4f"
              % (K, o["fde_mean"], o["fde_route_CI95"][0], o["fde_route_CI95"][1],
                 o["gain_a_mean"], o["gain_a_std"]))

    # ---- B) residual-MLP learning curve on SAME fixed test set (contrast) ----
    # test features/physics/gt computed once (gain fixed at a_all for the contrast)
    Fte, PHte, GTte, GRPte = build(data, test_keys, a_all)
    for K in K_LIST:
        if K > len(train_pool): continue
        per_seed = []
        for ss in range(SEEDS):
            r2 = np.random.default_rng(55 * ss + K)
            fit_g = list(r2.choice(train_pool, size=K, replace=False))
            tr_keys = sum([groups[g] for g in fit_g], [])
            Ftr, PHtr, GTtr, _ = build(data, tr_keys, a_all)
            if len(Ftr) < 40: continue
            sc = StandardScaler().fit(Ftr)
            net = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400, alpha=1e-3,
                               random_state=0, early_stopping=True).fit(sc.transform(Ftr), GTtr - PHtr)
            rp = net.predict(sc.transform(Fte))
            pl = PHte + rp
            per_seed.append(float(np.mean(route_fde(pl, GTte, GRPte))))
        if per_seed:
            out["residual_curve"][str(K)] = {"K": K, "fde_mean": float(np.mean(per_seed))}
            json.dump(out, open(RES, "w"), indent=2)
            print("RESID  K=%2d  FDE=%.3f" % (K, out["residual_curve"][str(K)]["fde_mean"]))
    json.dump(out, open(RES, "w"), indent=2); print("saved ->", RES)


if __name__ == "__main__":
    main()