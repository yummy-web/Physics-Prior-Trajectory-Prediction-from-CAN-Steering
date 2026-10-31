"""
exp_recursive_baselines_causal.py

Information-matched causal baselines requested by the TVT review.

RLS is a fit-only recursive least-squares predictor for the 3 s local endpoint.
Steering-input EKF filters causal history with position and CAN-speed
observations, then rolls out with anchor CAN speed and steering held constant.
The script is separate from the historical EKF label in baselines.py, which is
only a smoothed CTRV diagnostic.
"""
from __future__ import annotations

import datetime as dtm
import json
import os
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import causal_eval as CE
import causal_protocol as CP
import comma_loader as COMMA
import nuscenes_loader as NUSC
import nuscenes_loader as NUSC

DT_DEFAULT = {"comma": 0.05, "nuscenes": 0.02}
HORIZON_S = 3.0
STRIDE = 5
FOLDS = 3
NUSC_CAP = 200
NUSC_CAP_SEED = 7
RLS_FORGET = 1.0
RLS_DELTA = 100.0
EKF_Q = np.diag([0.03 ** 2, 0.03 ** 2, np.deg2rad(0.7) ** 2, 0.15 ** 2])
EKF_R = np.diag([0.12 ** 2, 0.12 ** 2, 0.25 ** 2])


def _json_path(name: str) -> str:
    return os.path.abspath(os.path.join(HERE, "..", "results", name))


def _group_id(key: str, dataset: str, df) -> str:
    return CE.route_of(df["seg"].iloc[0]) if dataset == "comma" else str(key)


def _dt(df, fallback: float) -> float:
    t = np.asarray(df["t"].values, float)
    d = np.diff(t)
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if len(d) else fallback


def _anchors(df, hs: int, stride: int) -> Iterable[int]:
    return range(CE.HIST, len(df) - hs, stride)


def build_examples(data: Dict[str, object], keys: Sequence[str], dataset: str,
                   hs: int, stride: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, Y, G = [], [], []
    for key in keys:
        df = data[key]
        gid = _group_id(key, dataset, df)
        for i0 in _anchors(df, hs, stride):
            X.append(CE.features_causal(df, i0, hs, _dt(df, DT_DEFAULT[dataset])))
            Y.append(CE.target_local(df, i0, hs))
            G.append(gid)
    return np.asarray(X, float), np.asarray(Y, float), np.asarray(G, object)


def fit_rls(X: np.ndarray, Y: np.ndarray, forget: float = RLS_FORGET,
            delta: float = RLS_DELTA) -> np.ndarray:
    if len(X) == 0:
        raise ValueError("empty RLS training set")
    Xa = np.column_stack([X, np.ones(len(X))])
    p = Xa.shape[1]
    theta = np.zeros((p, Y.shape[1]), float)
    P = np.eye(p, dtype=float) * float(delta)
    for x, y in zip(Xa, Y):
        Px = P @ x
        denom = float(forget + x @ Px)
        gain = Px / max(denom, 1e-12)
        err = y - x @ theta
        theta += np.outer(gain, err)
        P = (P - np.outer(gain, x @ P)) / forget
    return theta


def predict_rls(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.column_stack([X, np.ones(len(X))]) @ theta


def _wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _ekf_predict(x: np.ndarray, u_steer: float, gain_a: float,
                 dti: float) -> Tuple[np.ndarray, np.ndarray]:
    px, py, th, v = x
    c, s = np.cos(th), np.sin(th)
    out = np.array([px + v * c * dti, py + v * s * dti,
                    th + gain_a * u_steer * v * dti, v], float)
    F = np.eye(4)
    F[0, 2] = -v * s * dti; F[0, 3] = c * dti
    F[1, 2] = v * c * dti; F[1, 3] = s * dti
    F[2, 3] = gain_a * u_steer * dti
    return out, F


def steering_input_ekf_path(df, i0: int, hs: int, gain_a: float,
                            fallback_dt: float) -> np.ndarray:
    s = CE.causal_state(df)
    xobs = np.asarray(df["x"].values, float)
    yobs = np.asarray(df["y"].values, float)
    steer = np.asarray(df["steer"].values, float)
    vcan = np.asarray(s["v_can"], float)
    yaw = np.asarray(s["yaw"], float)
    tt = np.asarray(df["t"].values, float)
    j0 = max(0, i0 - CE.HIST)
    state = np.array([xobs[j0], yobs[j0], yaw[j0], max(vcan[j0], 0.0)], float)
    cov = np.eye(4) * 0.2
    H = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]])
    for k in range(j0 + 1, i0 + 1):
        dti = float(tt[k] - tt[k - 1])
        if not np.isfinite(dti) or dti <= 0:
            dti = fallback_dt
        state, F = _ekf_predict(state, float(steer[k - 1]), gain_a, dti)
        cov = F @ cov @ F.T + EKF_Q
        z = np.array([xobs[k], yobs[k], max(vcan[k], 0.0)], float)
        innov = z - H @ state
        S = H @ cov @ H.T + EKF_R
        K = cov @ H.T @ np.linalg.inv(S)
        state = state + K @ innov
        state[2] = _wrap_angle(state[2])
        cov = (np.eye(4) - K @ H) @ cov

    px, py, th, v = state
    v = max(float(v), 0.0)
    path = []
    for _ in range(hs):
        th += gain_a * float(steer[i0]) * v * fallback_dt
        px += v * np.cos(th) * fallback_dt
        py += v * np.sin(th) * fallback_dt
        path.append([px, py])
    path = np.asarray(path)
    c, ss = np.cos(-yaw[i0]), np.sin(-yaw[i0])
    dx = path[:, 0] - xobs[i0]; dy = path[:, 1] - yobs[i0]
    return np.column_stack([c * dx - ss * dy, ss * dx + c * dy])


def grouped_summary(errors: np.ndarray, groups: np.ndarray, nboot: int = 3000,
                    seed: int = 0) -> dict:
    uniq = np.unique(groups)
    gm = np.array([float(np.mean(errors[groups == g])) for g in uniq])
    rng = np.random.default_rng(seed)
    boot = np.array([np.mean(gm[rng.integers(0, len(gm), len(gm))]) for _ in range(nboot)])
    return {"mean_fde": float(np.mean(gm)),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "n_groups": int(len(uniq)), "n_anchors": int(len(errors)),
            "group_means": {str(g): float(m) for g, m in zip(uniq, gm)}}


def evaluate_rls(data, train_keys, test_keys, dataset, hs, stride) -> dict:
    Xtr, Ytr, _ = build_examples(data, train_keys, dataset, hs, stride)
    Xte, Yte, Gte = build_examples(data, test_keys, dataset, hs, stride)
    theta = fit_rls(Xtr, Ytr)
    err = np.linalg.norm(predict_rls(theta, Xte) - Yte, axis=1)
    return grouped_summary(err, Gte)


def evaluate_ekf(data, train_keys, test_keys, dataset, hs, stride) -> dict:
    a = CE.gain_curv([(k, data[k]) for k in train_keys], is_comma=(dataset == "comma"))
    errs, groups = [], []
    for key in test_keys:
        df = data[key]
        gid = _group_id(key, dataset, df)
        dti = _dt(df, DT_DEFAULT[dataset])
        for i0 in _anchors(df, hs, stride):
            path = steering_input_ekf_path(df, i0, hs, a, dti)
            _, fde, _ = CE.ade_fde(path, df, i0, hs)
            errs.append(fde); groups.append(gid)
    return {"gain_a_fit": float(a),
            **grouped_summary(np.asarray(errs), np.asarray(groups, object))}


def make_folds(data: Dict[str, object], dataset: str, nfolds: int) -> List[List[str]]:
    groups = CP.group_keys(data, dataset)
    gids = sorted(groups)
    fold_gids = [gids[i::nfolds] for i in range(nfolds)]
    return [[k for g in fg for k in groups[g]] for fg in fold_gids]


def run_dataset(dataset: str) -> dict:
    if dataset == "comma":
        data = COMMA.load_all_cached(verbose=True)
    elif dataset == "nuscenes":
        data = NUSC.load_all_cached(verbose=True, max_scenes=None)
        keys = sorted(data)
        if len(keys) > NUSC_CAP:
            keys = list(np.random.default_rng(NUSC_CAP_SEED).choice(keys, NUSC_CAP, replace=False))
        data = {k: data[k] for k in keys}
    else:
        raise ValueError(dataset)
    keys = sorted(data)
    hs = int(round(HORIZON_S / DT_DEFAULT[dataset]))
    folds = make_folds(data, dataset, FOLDS)
    per_fold = []
    for fi, test_keys in enumerate(folds):
        train_keys = [k for k in keys if k not in set(test_keys)]
        print(f"[{dataset}] fold {fi+1}/{FOLDS}: train={len(train_keys)} test={len(test_keys)}")
        rls = evaluate_rls(data, train_keys, test_keys, dataset, hs, STRIDE)
        ekf = evaluate_ekf(data, train_keys, test_keys, dataset, hs, STRIDE)
        per_fold.append({"fold": fi, "RLS_fit_only": rls, "Steering_EKF": ekf})
        print("  RLS %.3f [%.3f, %.3f] | EKF %.3f [%.3f, %.3f]" %
              (rls["mean_fde"], *rls["ci95"], ekf["mean_fde"], *ekf["ci95"]))
    def pool(method):
        vals = [f[method]["mean_fde"] for f in per_fold]
        return {"fold_mean_fde": vals, "mean_of_fold_means": float(np.mean(vals))}
    return {
        "dataset": dataset, "horizon_s": HORIZON_S, "dt": DT_DEFAULT[dataset],
        "stride": STRIDE, "folds": FOLDS, "n_segments": len(keys),
        "n_groups": len(CP.group_keys(data, dataset)),
        "group_unit": "route" if dataset == "comma" else "scene",
        "nusc_subset": {"used": len(keys), "available": 566, "seed": NUSC_CAP_SEED}
        if dataset == "nuscenes" else None,
        "causal_protocol": {
            "speed": "CAN speed at anchor; history-only for EKF",
            "steer": "steering at anchor; held during forecast",
            "heading": "backward-difference causal heading from pose",
            "future_control": False, "test_update": False,
        },
        "rls": {"forget": RLS_FORGET, "delta": RLS_DELTA},
        "ekf": {"process_noise": "fixed EKF_Q", "measurement_noise": "fixed EKF_R",
                "gain_fit": "train groups only"},
        "folds_detail": per_fold,
        "summary": {"RLS_fit_only": pool("RLS_fit_only"),
                     "Steering_EKF": pool("Steering_EKF")},
        "run_time": dtm.datetime.now().isoformat(timespec="seconds"),
    }


def main():
    out = {"script": os.path.basename(__file__), "protocol_version": "causal_supplement_v1", "datasets": {}}
    for ds in ("comma", "nuscenes"):
        out["datasets"][ds] = run_dataset(ds)
    path = _json_path("exp_recursive_baselines_causal_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    md = os.path.splitext(path)[0] + ".md"
    with open(md, "w", encoding="utf-8") as f:
        nl = chr(10)
        f.write("# 因果递推基线补充实验" + nl + nl)
        f.write("本实验只用于信息集公平性审计；RLS 为 fit-only 递推最小二乘，EKF 为历史观测更新后用锚点 CAN 速度和转向零阶保持外推。" + nl + nl)
        for ds, res in out["datasets"].items():
            f.write("## " + ds + nl + nl)
            f.write("统计单位：" + res["group_unit"] + "，分组数：" + str(res["n_groups"]) + "，3 s，stride=" + str(res["stride"]) + "。" + nl + nl)
            for name, s in res["summary"].items():
                f.write("- " + name + ": fold mean FDE = " + str(s["fold_mean_fde"]) + "，均值=" + format(s["mean_of_fold_means"], ".3f") + " m。" + nl)
            f.write(nl)
    print("saved", path)


if __name__ == "__main__":
    main()
