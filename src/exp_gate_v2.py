"""
exp_gate_v2.py  (revision v2, 2026-10)  --  R2-10 gate diagnostics, causal

Reviewer-2 point 10: the heuristic gate keys on steering variability, but the
physics prior can fail for many other reasons; show whether steering variability
actually tracks the true physics residual, and whether a causal learnable gate
helps.

Under the strictly causal protocol (causal_eval), on unified anchors:
  (1) DIAGNOSTIC: correlation between HISTORY steering std and the true physics
      endpoint error |phys - gt| (does the gate signal track real error?);
  (2) GATE VARIANTS sharing one residual network:
        NoGate            phys + r
        Heuristic         phys + sigmoid(std_hist - thr)*r, thr chosen on VALIDATION
        Learn-Scalar      phys + sigmoid(a*std_hist + b)*r, a,b fit on TRAIN
        Learn-MLP         phys + g_phi(causal features)*r, g_phi -> closed-form optimal gate
      all CAUSAL (history-only features), route-grouped, FDE@3s with grouped CI.

If a causal learnable gate still does not beat NoGate, we report that honestly:
gating adds no value under the causal protocol for this prior/metric.
"""
import os, json
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import comma_loader as C
import causal_protocol as P
import causal_eval as E
import models_core as M

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "exp_gate_v2_results.json")
DT = 0.05; H = 3.0; HS = int(round(H / DT)); STRIDE = 15


def collect(data, keys, a):
    """Also precompute GEO (x0,y0,gx,gy) and YAW0 so FDE is vectorized/cheap."""
    F, PH, GT, STD, GEO, YAW, GRP = [], [], [], [], [], [], []
    for s in keys:
        df = data[s]; g = E.route_of(df["seg"].iloc[0])
        st = df["steer"].values
        state = E.causal_state(df); x = state["x"]; y = state["y"]; yaw = state["yaw"]
        for i0 in E.anchors_for(df, HS, STRIDE):
            F.append(E.features_causal(df, i0, HS, DT))
            PH.append(E.physics_endpoint_local(df, i0, HS, DT, a))
            GT.append(E.target_local(df, i0, HS))
            j0 = max(i0 - E.HIST, 0)
            STD.append(float(np.std(st[j0:i0 + 1])))
            GEO.append((x[i0], y[i0], x[i0 + HS], y[i0 + HS])); YAW.append(yaw[i0]); GRP.append(g)
    return (np.array(F), np.array(PH), np.array(GT), np.array(STD),
            np.array(GEO), np.array(YAW), np.array(GRP))


def fde_vec(loc, GEO, YAW):
    """Vectorized global FDE from local predictions + precomputed geometry."""
    c = np.cos(YAW); s = np.sin(YAW)
    px = GEO[:, 0] + c * loc[:, 0] - s * loc[:, 1]
    py = GEO[:, 1] + s * loc[:, 0] + c * loc[:, 1]
    return np.hypot(px - GEO[:, 2], py - GEO[:, 3])


def route_mean(fde, grp):
    u = np.unique(grp); return float(np.mean([fde[grp == g].mean() for g in u]))


def main():
    data = C.load_all_cached(verbose=True)
    groups = P.group_keys(data, "comma")
    gids = list(groups.keys()); rng = np.random.default_rng(0); rng.shuffle(gids)
    test_g = gids[-14:]; val_g = gids[-20:-14]; tr_g = gids[:-20]
    trk = [k for g in tr_g for k in groups[g]]; vak = [k for g in val_g for k in groups[g]]; tek = [k for g in test_g for k in groups[g]]
    a = E.gain_curv([data[k] for k in trk], is_comma=True)

    Ftr, PHtr, GTtr, STDtr, GEOtr, YAWtr, GRPtr = collect(data, trk, a)
    Fva, PHva, GTva, STDva, GEOva, YAWva, GRPva = collect(data, vak, a)
    Fte, PHte, GTte, STDte, GEOte, YAWte, GRPte = collect(data, tek, a)

    sc = StandardScaler().fit(Ftr)
    net = MLPRegressor(hidden_layer_sizes=(48, 24), max_iter=400, alpha=1e-3, random_state=0, early_stopping=True).fit(sc.transform(Ftr), GTtr - PHtr)
    r_te = net.predict(sc.transform(Fte)); r_va = net.predict(sc.transform(Fva)); r_tr = net.predict(sc.transform(Ftr))

    # (1) DIAGNOSTIC: history steering-std vs TRUE physics endpoint error
    phys_err_te = np.hypot((PHte - GTte)[:, 0], (PHte - GTte)[:, 1])
    corr = float(np.corrcoef(STDte, phys_err_te)[0, 1])

    out = {"horizon_s": H, "gain_a": float(a),
           "diag_corr_histStd_vs_true_physErr": corr,
           "note": "all gates causal (history std / causal features); threshold on validation"}

    res = {}
    res["PurePhysics"] = route_mean(fde_vec(PHte, GEOte, YAWte), GRPte)
    res["NoGate"] = route_mean(fde_vec(PHte + r_te, GEOte, YAWte), GRPte)
    # Heuristic: threshold on VALIDATION
    best_thr, best_v = None, np.inf
    for thr in np.linspace(0.2, 8.0, 40):
        g = 1.0 / (1.0 + np.exp(-(STDva - thr)))
        f = route_mean(fde_vec(PHva + g[:, None] * r_va, GEOva, YAWva), GRPva)
        if f < best_v: best_v, best_thr = f, thr
    gH = 1.0 / (1.0 + np.exp(-(STDte - best_thr)))
    res["Heuristic"] = route_mean(fde_vec(PHte + gH[:, None] * r_te, GEOte, YAWte), GRPte)
    # Learn-Scalar: fit a,b on TRAIN (vectorized)
    bestab, bv = (1.0, -3.0), np.inf
    for aa in np.linspace(0.2, 4.0, 15):
        for bb in np.linspace(-10, 3, 20):
            g = 1.0 / (1.0 + np.exp(-(aa * STDtr + bb)))
            f = float(np.mean(fde_vec(PHtr + g[:, None] * r_tr, GEOtr, YAWtr)))
            if f < bv: bv, bestab = f, (aa, bb)
    gS = 1.0 / (1.0 + np.exp(-(bestab[0] * STDte + bestab[1])))
    res["Learn-Scalar"] = route_mean(fde_vec(PHte + gS[:, None] * r_te, GEOte, YAWte), GRPte)
    # Learn-MLP: closed-form optimal gate target, causal features
    num = np.sum(r_tr * (GTtr - PHtr), axis=1); den = np.sum(r_tr * r_tr, axis=1) + 1e-9
    gate_target = np.clip(num / den, 0.0, 1.0)
    gsc = StandardScaler().fit(Ftr)
    gnet = MLPRegressor(hidden_layer_sizes=(24, 12), max_iter=400, alpha=1e-2, random_state=0, early_stopping=True).fit(gsc.transform(Ftr), gate_target)
    gM = np.clip(gnet.predict(gsc.transform(Fte)), 0, 1)
    res["Learn-MLP"] = route_mean(fde_vec(PHte + gM[:, None] * r_te, GEOte, YAWte), GRPte)

    out["val_threshold"] = float(best_thr); out["learn_scalar_ab"] = [float(bestab[0]), float(bestab[1])]
    out["gate_mean_opening"] = {"heuristic": float(gH.mean()), "scalar": float(gS.mean()), "mlp": float(gM.mean())}
    out["fde_by_gate"] = res
    json.dump(out, open(RESULTS, "w"), indent=2)
    print("\nR2-10 gate diagnostics (causal):")
    print("  diag: corr(history steering-std, TRUE physics error) = %.3f" % corr)
    print("  val-selected heuristic threshold = %.2f" % best_thr)
    for k in ["PurePhysics", "NoGate", "Heuristic", "Learn-Scalar", "Learn-MLP"]:
        print("    %-13s FDE=%.3f" % (k, res[k]))
    print("  gate mean opening:", {k: round(v, 3) for k, v in out["gate_mean_opening"].items()})
    print("saved ->", RESULTS)


def _meta_tr(data, keys):
    return [(s, i0) for s in keys for i0 in E.anchors_for(data[s], HS, STRIDE)]


if __name__ == "__main__":
    main()
