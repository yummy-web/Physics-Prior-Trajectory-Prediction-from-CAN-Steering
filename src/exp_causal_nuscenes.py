"""
exp_causal_nuscenes.py  (revision, 2026-10)  --  nuScenes causal main table

Same causal protocol as exp_causal_main.py (comma), applied to nuScenes:
variable low-speed urban regime (mean 5.4 m/s, within-scene speed CV ~0.28) --
the regime where a speed-aware physics prior should matter most.
Scenes are independent, so the outer group == scene (no route-vs-segment issue).

Methods under one shared information set + oracle column:
  CV, CTRV, CTRA        kinematic baselines (causal)
  Physics (Model-C)     causal vs oracle
  MLP (causal)          direct learning, history features
  Residual (ungated)    physics + residual
  Residual (gated)      physics + causal-gate residual

Metric: FDE@3s mean, grouped-bootstrap 95% CI over scenes.
"""
import os, json
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import models_core as M
import causal_protocol as P
import baselines as B

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "exp_causal_nuscenes_results.json")
DT = 0.02; H = 3.0; HS = int(round(H / DT))
SEED = 0


def load():
    import pickle
    return pickle.load(open(os.path.join(os.path.dirname(__file__), ".nusc_cache.pkl"), "rb"))


def key_of(df, k):
    return str(df["seg"].iloc[0]) if "seg" in df else str(k)


def phys_fde_arrays(items, a, steer_mode, speed_mode):
    fde, grp = [], []
    for k, df in items:
        d = M.comma_phys_metrics_curv(df, HS, DT, a, speed_mode=speed_mode, steer_mode=steer_mode)
        fde.append(d["fde"]); grp.append(np.full(len(d["fde"]), str(k)))
    return np.concatenate(fde), np.concatenate(grp)


def kin_fde_arrays(items, model):
    fde, grp = [], []
    for k, df in items:
        x = df["x"].values; y = df["y"].values; yaw = np.radians(df["yaw_unwrap"].values)
        v = df["obs_v"].values; w = np.radians(df["yaw_rate"].values); n = len(df)
        for i0 in range(0, n - HS, 3):
            v0, w0, yaw0 = v[i0], w[i0], yaw[i0]
            j0 = max(i0 - 5, 0); a0 = (v[i0] - v[j0]) / ((i0 - j0) * DT) if i0 > j0 else 0.0
            px, py = B._rollout_paths(x[i0], y[i0], yaw0, v0, w0, a0, HS, DT, model)
            fde.append(np.hypot(px[-1] - x[i0 + HS], py[-1] - y[i0 + HS])); grp.append(str(k))
    return np.array(fde), np.array(grp)


def gain_curv(items, min_v=1.0):
    from scipy import stats
    xs, ys = [], []
    for k, df in items:
        m = df["obs_v"].values > min_v
        x = df["steer"].values[m]
        yv = (np.radians(df["yaw_rate"].values) / (df["obs_v"].values + 1e-3))[m]
        ok = np.isfinite(x) & np.isfinite(yv) & (np.abs(yv) < np.nanpercentile(np.abs(yv[np.isfinite(yv)]), 97))
        xs.append(x[ok]); ys.append(yv[ok])
    a, b, r, p, se = stats.linregress(np.concatenate(xs), np.concatenate(ys))
    return a


def nusc_mlp_dataset(items, causal=True, hist_win=10):
    X, Y, grp = [], [], []
    for k, df in items:
        n = len(df)
        st = df["steer"].values; v = df["obs_v"].values; cv = df["can_speed"].values
        yr = np.radians(df["yaw_rate"].values)
        x = df["x"].values; y = df["y"].values; yaw = np.radians(df["yaw_unwrap"].values)
        for i0 in range(hist_win, n - HS, 3):
            h = slice(max(i0 - hist_win, 0), i0 + 1)
            if causal:
                sslope = (st[i0] - st[h.start]) / (max(i0 - h.start, 1) * DT)
                feat = [v[i0], cv[i0], yr[i0], st[i0], np.mean(st[h]), np.std(st[h]), np.mean(v[h]), sslope]
            else:
                fut = slice(i0, i0 + HS)
                feat = [v[i0], yr[i0], np.mean(st[fut]), np.std(st[fut]), np.mean(v[fut]), np.std(v[fut])]
            dx = x[i0 + HS] - x[i0]; dy = y[i0 + HS] - y[i0]
            cc, ss = np.cos(-yaw[i0]), np.sin(-yaw[i0])
            X.append(feat); Y.append([cc * dx - ss * dy, ss * dx + cc * dy]); grp.append(str(k))
    return np.array(X), np.array(Y), np.array(grp)


def residual_fde(train, test, a, gated, hist_win=10):
    def collect(items):
        F, PH, GT, STD, GEO, YAW, GRP = [], [], [], [], [], [], []
        for k, df in items:
            n = len(df); st = df["steer"].values; v = df["obs_v"].values; cv = df["can_speed"].values
            yr = np.radians(df["yaw_rate"].values)
            x = df["x"].values; y = df["y"].values; yaw = np.radians(df["yaw_unwrap"].values)
            for i0 in range(hist_win, n - HS, 5):
                px, py, th = x[i0], y[i0], yaw[i0]
                j0 = max(i0 - hist_win, 0)
                for kk in range(HS):
                    j = min(i0 + kk, n - 1)
                    vv = cv[i0]  # can_hold (causal)
                    sd = st[i0]  # hold steer (causal)
                    th += a * sd * vv * DT; px += vv * np.cos(th) * DT; py += vv * np.sin(th) * DT
                cc, ss = np.cos(-yaw[i0]), np.sin(-yaw[i0])
                ph = np.array([cc * (px - x[i0]) - ss * (py - y[i0]), ss * (px - x[i0]) + cc * (py - y[i0])])
                dx = x[i0 + HS] - x[i0]; dy = y[i0 + HS] - y[i0]
                gt = np.array([cc * dx - ss * dy, ss * dx + cc * dy])
                h = slice(j0, i0 + 1); sslope = (st[i0] - st[h.start]) / (max(i0 - h.start, 1) * DT)
                F.append([v[i0], cv[i0], yr[i0], st[i0], np.mean(st[h]), np.std(st[h]), np.mean(v[h]), sslope])
                PH.append(ph); GT.append(gt); STD.append(np.std(st[h]))
                GEO.append((x[i0], y[i0], x[i0 + HS], y[i0 + HS])); YAW.append(yaw[i0]); GRP.append(str(k))
        return (np.array(F), np.array(PH), np.array(GT), np.array(STD), GEO, np.array(YAW), np.array(GRP))
    Ftr, PHtr, GTtr, _, _, _, _ = collect(train)
    Fte, PHte, GTte, STDte, GEOte, YAWte, GRPte = collect(test)
    if len(Ftr) < 50 or len(Fte) < 20:
        return None
    sc = StandardScaler().fit(Ftr)
    net = MLPRegressor(hidden_layer_sizes=(48, 24), max_iter=300, alpha=1e-3,
                       random_state=SEED, early_stopping=True).fit(sc.transform(Ftr), GTtr - PHtr)
    r_te = net.predict(sc.transform(Fte))
    loc = PHte + (1.0 / (1.0 + np.exp(-(STDte - 3.0))))[:, None] * r_te if gated else PHte + r_te
    fde = []
    for p, (x0, y0, gx, gy), yaw0 in zip(loc, GEOte, YAWte):
        c, s = np.cos(yaw0), np.sin(yaw0)
        px = x0 + c * p[0] - s * p[1]; py = y0 + s * p[0] + c * p[1]
        fde.append(np.hypot(px - gx, py - gy))
    return np.array(fde), GRPte


def ci(fde, grp):
    m, lo, hi = M.grouped_bootstrap_ci(fde, grp, n_boot=800)
    return {"fde": m, "ci_lo": lo, "ci_hi": hi, "n": int(len(fde))}


def main():
    data = load(); items = list(data.items())
    print("nuScenes scenes:", len(items), " dt=", DT, " hs=", HS)
    rng = np.random.default_rng(42); idx = np.arange(len(items)); rng.shuffle(idx)
    folds = [idx[i::3] for i in range(3)]
    out = {"horizon_s": H, "dt": DT, "n_scenes": len(items)}
    if os.path.exists(RESULTS):
        try: out.update(json.load(open(RESULTS)))
        except Exception: pass

    for mdl in ["CV", "CTRV", "CTRA"]:
        if mdl in out: print("[skip]", mdl); continue
        f, g = kin_fde_arrays(items, mdl); out[mdl] = ci(f, g)
        json.dump(out, open(RESULTS, "w"), indent=2)
        print("  %-10s FDE=%.3f CI[%.3f,%.3f]" % (mdl, out[mdl]["fde"], out[mdl]["ci_lo"], out[mdl]["ci_hi"]))

    for tag, sm, vm in [("Physics_causal", "hold", "can_hold"), ("Physics_oracle", "future", "obs")]:
        if tag in out: print("[skip]", tag); continue
        fa, ga = [], []
        for fi in range(3):
            te = [items[i] for i in folds[fi]]; tr = [items[i] for i in idx if i not in set(folds[fi])]
            a = gain_curv(tr); f, g = phys_fde_arrays(te, a, sm, vm); fa.append(f); ga.append(g)
        out[tag] = ci(np.concatenate(fa), np.concatenate(ga))
        json.dump(out, open(RESULTS, "w"), indent=2)
        print("  %-14s FDE=%.3f CI[%.3f,%.3f]" % (tag, out[tag]["fde"], out[tag]["ci_lo"], out[tag]["ci_hi"]))

    if "MLP_causal" not in out:
        fa, ga = [], []
        for fi in range(3):
            te = [items[i] for i in folds[fi]]; tr = [items[i] for i in idx if i not in set(folds[fi])]
            Xtr, Ytr, _ = nusc_mlp_dataset(tr, causal=True); Xte, Yte, gte = nusc_mlp_dataset(te, causal=True)
            sc, mdl = M.train_mlp(Xtr, Ytr); pr = mdl.predict(sc.transform(Xte))
            fa.append(np.hypot(pr[:, 0] - Yte[:, 0], pr[:, 1] - Yte[:, 1])); ga.append(gte)
        out["MLP_causal"] = ci(np.concatenate(fa), np.concatenate(ga))
        json.dump(out, open(RESULTS, "w"), indent=2)
        print("  %-14s FDE=%.3f CI[%.3f,%.3f]" % ("MLP_causal", out["MLP_causal"]["fde"], out["MLP_causal"]["ci_lo"], out["MLP_causal"]["ci_hi"]))

    for tag, gated in [("Residual_causal", False), ("Residual_gated_causal", True)]:
        if tag in out: print("[skip]", tag); continue
        fa, ga = [], []
        for fi in range(3):
            te = [items[i] for i in folds[fi]]; tr = [items[i] for i in idx if i not in set(folds[fi])]
            a = gain_curv(tr); r = residual_fde(tr, te, a, gated)
            if r is None: continue
            fa.append(r[0]); ga.append(r[1])
        out[tag] = ci(np.concatenate(fa), np.concatenate(ga))
        json.dump(out, open(RESULTS, "w"), indent=2)
        print("  %-22s FDE=%.3f CI[%.3f,%.3f]" % (tag, out[tag]["fde"], out[tag]["ci_lo"], out[tag]["ci_hi"]))

    json.dump(out, open(RESULTS, "w"), indent=2); print("saved ->", RESULTS)


if __name__ == "__main__":
    main()
