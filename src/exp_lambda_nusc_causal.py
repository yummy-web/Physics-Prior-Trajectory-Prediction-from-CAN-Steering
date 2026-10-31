"""
exp_lambda_nusc_causal.py  (P0 fix + P1 uncertainty, 2026-10-08)

CAUSAL nuScenes lambda-shrinkage experiment. Replaces the leaky
exp_lambda_shrinkage_nusc.py, which fed CENTRAL-difference obs_v / yaw_rate
(peek at i+1) into both the gain fit and the residual features.

Causal information set at anchor i0 (mirrors causal_eval.py for comma):
  speed      : can_speed[i0]            -- real on-board CAN sensor (not pose-diff)
  yaw rate   : (yaw[i0]-yaw[i0-1])/dt   -- BACKWARD difference of measured heading
  heading    : yaw_unwrap[i0]           -- measured pose orientation (NOT differenced)
  steer      : steer[i0]                -- real sensor
NO use of obs_v (central-diff) or df['yaw_rate'] (central-diff) anywhere.

Gain a is also refit from the causal (backward) state.

Reports, per N: full lambda curve, lambda*, and PAIRED scene-level CIs for
lambda=0 vs full, lambda* vs full, plus neighbour-of-lambda* significance and
the distribution of per-scene argmin lambda (P1).

nuScenes, scene = group. 3 folds x 4 seeds. N in {3,5,10,20}.
"""
import os, json, datetime
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import exp_causal_nuscenes as NU

RES = os.path.join(os.path.dirname(__file__), "..", "results", "exp_lambda_nusc_causal_results.json")
DT = 0.02; H = 3.0; HS = int(round(H / DT)); HW = 10
N_LIST = [3, 5, 10, 20]; FOLDS = 3; SUBSET_SEEDS = 4
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
TEST_CAP = 60   # cap evaluated test scenes per fold (fixed by seed) for tractable runtime


def causal_state(df):
    """Backward-difference causal channels; NO central-diff columns used."""
    t = df["t"].values
    yaw = np.radians(df["yaw_unwrap"].values)        # measured orientation (not differenced)
    dtb = np.diff(t, prepend=t[0] - (t[1] - t[0]))   # backward dt > 0
    wr_bwd = np.diff(yaw, prepend=yaw[0]) / dtb       # rad/s, backward yaw rate (causal)
    v_can = df["can_speed"].values                    # real sensor (causal)
    st = df["steer"].values
    x = df["x"].values; y = df["y"].values
    return x, y, yaw, v_can, wr_bwd, st


def gain_causal(items, min_v=1.0):
    """Fit kappa = a*steer using CAUSAL backward yaw-rate and CAN speed only."""
    from scipy import stats
    xs, ys = [], []
    for _, df in items:
        _, _, _, v_can, wr_bwd, st = causal_state(df)
        m = v_can > min_v
        kappa = wr_bwd / (v_can + 1e-3)
        ok = m & np.isfinite(kappa) & np.isfinite(st)
        if ok.sum() > 5:
            thr = np.nanpercentile(np.abs(kappa[ok]), 97)
            ok &= np.abs(kappa) < thr
        xs.append(st[ok]); ys.append(kappa[ok])
    a, b, r, p, se = stats.linregress(np.concatenate(xs), np.concatenate(ys))
    return float(a)


def collect(items, a):
    """Causal physics endpoint + residual features + target (local frame)."""
    F, PH, GT, GRP = [], [], [], []
    for k, df in items:
        x, y, yaw, v_can, wr_bwd, st = causal_state(df)
        n = len(df)
        for i0 in range(HW, n - HS, 5):
            # physics rollout: freeze CAN speed + steer at i0 (causal)
            px, py, th = x[i0], y[i0], yaw[i0]
            for _ in range(HS):
                th += a * st[i0] * v_can[i0] * DT
                px += v_can[i0] * np.cos(th) * DT; py += v_can[i0] * np.sin(th) * DT
            cc, ss = np.cos(-yaw[i0]), np.sin(-yaw[i0])
            ph = np.array([cc * (px - x[i0]) - ss * (py - y[i0]),
                           ss * (px - x[i0]) + cc * (py - y[i0])])
            dx = x[i0 + HS] - x[i0]; dy = y[i0 + HS] - y[i0]
            gt = np.array([cc * dx - ss * dy, ss * dx + cc * dy])
            h = slice(max(i0 - HW, 0), i0 + 1)
            sslope = (st[i0] - st[h.start]) / (max(i0 - h.start, 1) * DT)
            # features: ALL causal (can speed, backward yaw rate, steer stats)
            F.append([v_can[i0], wr_bwd[i0], st[i0], float(np.mean(st[h])),
                      float(np.std(st[h])), float(np.mean(v_can[h])), sslope])
            PH.append(ph); GT.append(gt); GRP.append(str(k))
    return np.array(F), np.array(PH), np.array(GT), np.array(GRP)


def route_mean(err, grp):
    gs = np.unique(grp)
    return np.array([np.mean(err[grp == g]) for g in gs]), gs


def paired_ci(a_vec, b_vec, nboot=3000, seed=0):
    """Paired bootstrap on draw-level means; returns mean diff (a-b) and CI."""
    d = np.asarray(a_vec) - np.asarray(b_vec)
    if len(d) == 0: return None
    rng = np.random.default_rng(seed)
    bo = [float(np.mean(d[rng.integers(0, len(d), len(d))])) for _ in range(nboot)]
    return {"mean_diff": float(np.mean(d)), "ci95": [float(np.percentile(bo, 2.5)),
            float(np.percentile(bo, 97.5))], "frac_pos": float(np.mean(d > 0)), "n": int(len(d))}


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
    out = {"horizon_s": H, "dt": DT, "lambdas": LAMBDAS, "N_list": N_LIST,
           "provenance": {"run_date": datetime.datetime.now().isoformat(timespec="seconds"),
                          "folds": FOLDS, "subset_seeds": SUBSET_SEEDS, "dataset": "nuscenes",
                          "causal": True, "speed_src": "can_speed", "yawrate_src": "backward_diff",
                          "note": "replaces leaky exp_lambda_shrinkage_nusc.py (central-diff obs_v/yaw_rate)"},
           "per_N": {}}
    if os.path.exists(RES):
        try: out["per_N"] = json.load(open(RES)).get("per_N", {})
        except Exception: pass
    for N in N_LIST:
        curves = {str(l): [] for l in LAMBDAS}       # draw-level route-mean FDE per lambda
        per_scene_argmin = []
        for fo in range(FOLDS):
            test_g_all = folds[fo]; pool = [g for g in gids if g not in set(test_g_all)]
            if len(pool) < N: continue
            # fix the evaluated test subset per fold (deterministic) to bound runtime
            tg = list(test_g_all)
            if len(tg) > TEST_CAP:
                tg = list(np.random.default_rng(777 + fo).choice(tg, size=TEST_CAP, replace=False))
            test_items = [(g, byg[g]) for g in tg]
            for ss in range(SUBSET_SEEDS):
                r2 = np.random.default_rng(100 * fo + ss)
                sub = list(r2.choice(pool, size=N, replace=False))
                tr_items = [(g, byg[g]) for g in sub]
                a = gain_causal(tr_items)
                Ftr, PHtr, GTtr, _ = collect(tr_items, a)
                Fte, PHte, GTte, GRP = collect(test_items, a)
                if len(Ftr) < 50 or len(Fte) < 20: continue
                sc = StandardScaler().fit(Ftr)
                net = MLPRegressor(hidden_layer_sizes=(48, 24), max_iter=300, alpha=1e-3,
                                   random_state=0, early_stopping=True).fit(sc.transform(Ftr), GTtr - PHtr)
                rp = net.predict(sc.transform(Fte))
                scene_fde_by_lam = {}
                for l in LAMBDAS:
                    pl = PHte + l * rp
                    fde = np.hypot(pl[:, 0] - GTte[:, 0], pl[:, 1] - GTte[:, 1])
                    rm, gs = route_mean(fde, GRP)
                    curves[str(l)].append(float(np.mean(rm)))
                    scene_fde_by_lam[l] = (rm, gs)
                # per-scene argmin lambda (P1): for each test scene pick best lambda
                lam_arr = np.array(LAMBDAS)
                stack = np.vstack([scene_fde_by_lam[l][0] for l in LAMBDAS])  # [n_lam, n_scene]
                per_scene_argmin.extend(lam_arr[np.argmin(stack, axis=0)].tolist())
        means = {l: float(np.mean(v)) for l, v in curves.items() if v}
        if not means: continue
        best = min(means, key=means.get)
        bi = LAMBDAS.index(float(best))
        # P1 stats: paired CIs
        full = curves["1.0"]; zero = curves["0.0"]; star = curves[best]
        stats_block = {
            "lambda0_vs_full": paired_ci(full, zero),      # +ve => lambda0 (prior) beats full
            "star_vs_full": paired_ci(full, star),         # +ve => lambda* beats full
        }
        # neighbour significance: lambda* vs its better-adjacent neighbour
        neigh = []
        for j in (bi - 1, bi + 1):
            if 0 <= j < len(LAMBDAS):
                neigh.append((LAMBDAS[j], curves[str(LAMBDAS[j])]))
        if neigh:
            worst_neighbour = max(neigh, key=lambda t: np.mean(t[1]))  # the neighbour with highest FDE
            stats_block["star_vs_neighbour"] = {
                "neighbour_lambda": worst_neighbour[0],
                **paired_ci(worst_neighbour[1], star)}
        argm = np.array(per_scene_argmin)
        out["per_N"][str(N)] = {
            "N": N, "n_draws": len(full), "lambda_curve": means, "lambda_star": float(best),
            "stats": stats_block,
            "per_scene_argmin_lambda": {"mean": float(np.mean(argm)), "median": float(np.median(argm)),
                                        "frac_zero": float(np.mean(argm == 0.0)),
                                        "frac_ge_1": float(np.mean(argm >= 1.0))}}
        json.dump(out, open(RES, "w"), indent=2)
        o = out["per_N"][str(N)]; sv = stats_block["star_vs_full"]; zv = stats_block["lambda0_vs_full"]
        print("N=%2d lam*=%.2f FDE*=%.3f full=%.3f | star_vs_full d=%.3f CI[%.3f,%.3f] | prior_vs_full d=%.3f CI[%.3f,%.3f] | scene-argmin mean=%.2f fz=%.0f%%"
              % (N, float(best), means[best], means["1.0"], sv["mean_diff"], sv["ci95"][0], sv["ci95"][1],
                 zv["mean_diff"], zv["ci95"][0], zv["ci95"][1],
                 o["per_scene_argmin_lambda"]["mean"], 100 * o["per_scene_argmin_lambda"]["frac_zero"]))
        print("     curve:", {k: round(v, 3) for k, v in means.items()})
    out["N_list"] = sorted((int(k) for k in out["per_N"]), key=int)
    json.dump(out, open(RES, "w"), indent=2); print("saved ->", RES)


if __name__ == "__main__":
    main()