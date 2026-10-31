"""
exp_latency_params.py  (revision, 2026-10)

Deployability cost under the causal protocol (R2-11). NO embedded hardware is
available, so we report WORKSTATION latency + parameter/memory only, and the
manuscript must drop any energy / "embedded real-time verified" claim
(revision note Q4). Reports per-prediction wall-clock latency (single-thread,
after warm-up) and parameter counts for each causal method.
"""
import os, json, time, platform
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import models_core as M
import comma_loader as C

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "exp_latency_params_results.json")
DT = 0.05; HS = int(round(3.0 / DT))


def phys_once(df, i0, a):
    st = df["steer"].values; cv = df["can_speed"].values if "can_speed" in df else df["obs_v"].values
    x = df["x"].values; y = df["y"].values; yaw = np.radians(df["yaw_unwrap"].values)
    px, py, th = x[i0], y[i0], yaw[i0]
    for k in range(HS):
        th += a * st[i0] * cv[i0] * DT; px += cv[i0] * np.cos(th) * DT; py += cv[i0] * np.sin(th) * DT
    return px, py


def time_it(fn, n=2000):
    for _ in range(50):
        fn()  # warm-up
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e3  # ms per call


def main():
    data = C.load_all_cached(verbose=True)
    segs = list(data.values())
    a = M.comma_gain_curv(segs[:-30])
    df = segs[0]

    # physics prior latency
    lat_phys = time_it(lambda: phys_once(df, 30, a))

    # MLP latency + params
    X = np.random.randn(400, 8); Y = np.random.randn(400, 2)
    sc = StandardScaler().fit(X)
    net = MLPRegressor(hidden_layer_sizes=(48, 24), max_iter=50, random_state=0).fit(sc.transform(X), Y)
    xone = sc.transform(X[:1])
    lat_mlp = time_it(lambda: net.predict(xone))
    n_params = sum(c.size for c in net.coefs_) + sum(b.size for b in net.intercepts_)

    # residual = physics + MLP
    lat_res = lat_phys + lat_mlp

    out = {
        "platform": platform.platform(), "python": platform.python_version(),
        "cpu": platform.processor() or "unknown", "note": "single-thread, after 50-iter warm-up; NO embedded hw; energy claims dropped",
        "horizon_s": 3.0, "rollout_steps": HS,
        "latency_ms_per_prediction": {
            "physics_causal": round(lat_phys, 4),
            "mlp_causal": round(lat_mlp, 4),
            "residual_causal": round(lat_res, 4),
        },
        "parameters": {"physics_gain_a": 1, "mlp": int(n_params), "residual": int(n_params) + 1},
        "mlp_hidden": [48, 24],
    }
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    json.dump(out, open(RESULTS, "w"), indent=2)
    print(json.dumps(out["latency_ms_per_prediction"], indent=2))
    print("mlp params:", n_params)
    print("saved ->", RESULTS)


if __name__ == "__main__":
    main()
