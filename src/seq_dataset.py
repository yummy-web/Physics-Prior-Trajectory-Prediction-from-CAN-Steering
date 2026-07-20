"""
seq_dataset.py
Build history-window sequence datasets for the GRU/LSTM baseline, and a matching
direct-MLP dataset, so both see identical information for a fair comparison.

For each anchor i0:
  history features over [i0-L+1 .. i0]: per-step [v, yaw_rate(rad/s), steer, dv]
  target: displacement to i0+hs in the anchor-local frame (forward, lateral)
"""
import numpy as np


def build_sequences(segs, hs, dt, steer_col, L=10, stride=3, speed_col="obs_v"):
    """Return X_seq (N,L,F), X_flat (N, L*F) for MLP, Y (N,2), meta list."""
    Xs, Y = [], []
    for df in segs:
        x = df["x"].values; y = df["y"].values
        yaw = np.radians(df["yaw_unwrap"].values)
        v = df[speed_col].values
        w = np.radians(df["yaw_rate"].values)
        steer = df[steer_col].values
        n = len(df)
        for i0 in range(L, n - hs, stride):
            hist = []
            for k in range(i0 - L + 1, i0 + 1):
                dv = v[k] - v[k - 1] if k > 0 else 0.0
                hist.append([v[k], w[k], steer[k], dv])
            dx = x[i0 + hs] - x[i0]; dy = y[i0 + hs] - y[i0]
            c, s = np.cos(-yaw[i0]), np.sin(-yaw[i0])
            Xs.append(hist); Y.append([c * dx - s * dy, s * dx + c * dy])
    Xs = np.array(Xs, dtype=np.float32); Y = np.array(Y, dtype=np.float32)
    return Xs, Y


def standardize_fit(X):
    # X: (N,L,F) -> per-feature mean/std over N,L
    mu = X.reshape(-1, X.shape[-1]).mean(0)
    sd = X.reshape(-1, X.shape[-1]).std(0) + 1e-6
    return mu, sd


def standardize_apply(X, mu, sd):
    return (X - mu) / sd
