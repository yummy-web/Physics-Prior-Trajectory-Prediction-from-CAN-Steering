"""Empirical route-level learning-curve protocol diagnostic.

Fits power, exponential, and logarithmic candidates and records AIC/BIC. N is
the number of grouped training routes. The output is a protocol diagnostic,
not a universal scaling-law estimate.
"""
import os, json
import numpy as np
from scipy.optimize import curve_fit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import comma_loader as C
import causal_protocol as P
import causal_eval as E

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "exp_scalinglaw_v2_results.json")
DT = 0.05; H = 3.0; HS = int(round(H / DT)); STRIDE = 15


def m_power(N, B, Cc, a): return B + Cc * np.power(N, -a)
def m_exp(N, B, Cc, k): return B + Cc * np.exp(-k * N)
def m_log(N, B, Cc): return B + Cc / np.log(N + 1)


def aic_bic(y, yhat, k):
    n = len(y); rss = np.sum((y - yhat) ** 2)
    sigma2 = rss / n
    ll = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
    return 2 * k - 2 * ll, k * np.log(n) - 2 * ll


def mlp_fde_at_N(data, groups, gids, N, test_keys, seeds=2):
    """Mean route-level causal-MLP FDE using N training routes (avg over seeds+draws)."""
    vals = []
    for draw in range(3):
        r = np.random.default_rng(200 + draw)
        if N > len(gids): return None
        sub = list(r.choice(gids, size=N, replace=False))
        tr = [k for g in sub for k in groups[g]]
        a = E.gain_curv([data[k] for k in tr], is_comma=True)
        Ftr = np.array([E.features_causal(data[s], i0, HS, DT) for s in tr for i0 in E.anchors_for(data[s], HS, STRIDE)])
        GTtr = np.array([E.target_local(data[s], i0, HS) for s in tr for i0 in E.anchors_for(data[s], HS, STRIDE)])
        if len(Ftr) < 20: continue
        sc = StandardScaler().fit(Ftr)
        anchors = [(s, i0) for s in test_keys for i0 in E.anchors_for(data[s], HS, STRIDE)]
        grp = np.array([E.route_of(data[s]["seg"].iloc[0]) for s, _ in anchors])
        Fte = np.array([E.features_causal(data[s], i0, HS, DT) for s, i0 in anchors])
        for ms in range(seeds):
            net = MLPRegressor(hidden_layer_sizes=(48, 24), max_iter=300, alpha=1e-3, random_state=ms, early_stopping=True).fit(sc.transform(Ftr), GTtr)
            pm = net.predict(sc.transform(Fte))
            fde = np.array([E.local_to_fde(pm[j], data[anchors[j][0]], anchors[j][1], HS) for j in range(len(anchors))])
            u = np.unique(grp); vals.append(float(np.mean([fde[grp == g].mean() for g in u])))
    return float(np.mean(vals)), float(np.std(vals))


def main():
    data = C.load_all_cached(verbose=True)
    groups = P.group_keys(data, "comma")
    gids = list(groups.keys()); rng = np.random.default_rng(0); rng.shuffle(gids)
    test_g = gids[-14:]; pool = gids[:-14]
    test_keys = [k for g in test_g for k in groups[g]]

    out = {"horizon_s": H, "metric": "route-level causal MLP FDE@3s",
           "note": "N = number of independent training ROUTES (not correlated segments)",
           "points": {}}
    if os.path.exists(RESULTS):
        try: out.update(json.load(open(RESULTS)))
        except Exception: pass

    N_LIST = [2, 3, 5, 8, 12, 18, 24]
    for N in N_LIST:
        if str(N) in out["points"]:
            print("[skip] N=", N); continue
        if N > len(pool): continue
        r = mlp_fde_at_N(data, groups, pool, N, test_keys)
        if r is None: continue
        out["points"][str(N)] = {"N": N, "fde": r[0], "fde_std": r[1]}
        json.dump(out, open(RESULTS, "w"), indent=2)
        print("  N=%-3d routes  MLP FDE=%.3f (+-%.3f)" % (N, r[0], r[1]))

    # fit candidate models
    pts = sorted(out["points"].values(), key=lambda z: z["N"])
    Nv = np.array([p["N"] for p in pts], float); Ev = np.array([p["fde"] for p in pts])
    fits = {}
    for name, fn, k, p0, bnds in [
        ("power", m_power, 3, [0.5, 5, 1.0], ([0, 0, 0.1], [np.inf, np.inf, 3])),
        ("exponential", m_exp, 3, [0.5, 5, 0.2], ([0, 0, 1e-3], [np.inf, np.inf, 5])),
        ("logarithmic", m_log, 2, [0.5, 5], ([0, 0], [np.inf, np.inf]))]:
        try:
            popt, pcov = curve_fit(fn, Nv, Ev, p0=p0, bounds=bnds, maxfev=20000)
            yhat = fn(Nv, *popt); aic, bic = aic_bic(Ev, yhat, k)
            perr = np.sqrt(np.diag(pcov))
            fits[name] = {"params": [float(x) for x in popt], "param_se": [float(x) for x in perr],
                          "aic": float(aic), "bic": float(bic), "rmse": float(np.sqrt(np.mean((Ev - yhat) ** 2)))}
        except Exception as e:
            fits[name] = {"error": str(e)}
    # bootstrap CI for power-law params over N-points
    if "power" in fits and "params" in fits["power"]:
        boots = []
        rng2 = np.random.default_rng(1)
        for _ in range(1000):
            idx = rng2.integers(0, len(Nv), len(Nv))
            try:
                po, _ = curve_fit(m_power, Nv[idx], Ev[idx], p0=fits["power"]["params"], bounds=([0, 0, 0.1], [np.inf, np.inf, 3]), maxfev=10000)
                boots.append(po)
            except Exception:
                pass
        if boots:
            boots = np.array(boots)
            fits["power"]["ci"] = {n: [float(np.percentile(boots[:, i], 2.5)), float(np.percentile(boots[:, i], 97.5))]
                                   for i, n in enumerate(["B", "C", "alpha"])}
    out["fits"] = fits
    best = min([k for k in fits if "aic" in fits[k]], key=lambda k: fits[k]["aic"]) if any("aic" in v for v in fits.values()) else None
    out["best_by_aic"] = best
    json.dump(out, open(RESULTS, "w"), indent=2)
    print("\nfits:")
    for k, v in fits.items():
        if "aic" in v:
            print("  %-12s AIC=%.2f BIC=%.2f RMSE=%.3f params=%s" % (k, v["aic"], v["bic"], v["rmse"], np.round(v["params"], 3)))
    if best == "power" and "ci" in fits["power"]:
        print("  power-law alpha=%.2f CI%s  B=%.2f C=%.2f" % (fits["power"]["params"][2], fits["power"]["ci"]["alpha"], fits["power"]["params"][0], fits["power"]["params"][1]))
    print("best by AIC:", best, "\nsaved ->", RESULTS)


if __name__ == "__main__":
    main()
