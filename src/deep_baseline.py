"""
deep_baseline.py
Small GRU/LSTM sequence predictor as a deep-learning baseline (CPU-friendly).

Purpose in the paper: directly answer "why not a deep temporal model?".
We train a compact GRU to predict the future trajectory and show it needs far
more data than the physics prior to reach comparable accuracy (data-scarcity).

Design:
  Input at each history step: [v, yaw_rate, steer_signal, (aux controls...)]
  The GRU encodes a short history window, then an MLP head predicts the future
  displacement in the anchor-local frame at the requested horizon (direct, no
  compounding), matching the data-driven MLP protocol for fairness.

Requires: torch (CPU). If torch is missing, `available()` returns False and
callers should skip the deep baseline gracefully.
"""
import numpy as np

try:
    import torch
    import torch.nn as nn
    torch.set_num_threads(max(1, __import__("os").cpu_count() // 2))
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def available():
    return _HAS_TORCH


if _HAS_TORCH:
    class GRUPredictor(nn.Module):
        def __init__(self, in_dim, hidden=32, kind="gru"):
            super().__init__()
            rnn = nn.GRU if kind == "gru" else nn.LSTM
            self.rnn = rnn(in_dim, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 2))

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(out[:, -1, :])

    def train_gru(X, Y, in_dim, kind="gru", epochs=60, lr=1e-2, seed=0, verbose=False):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = GRUPredictor(in_dim, hidden=32, kind=kind)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        lossf = nn.MSELoss()
        Xt = torch.tensor(X, dtype=torch.float32)
        Yt = torch.tensor(Y, dtype=torch.float32)
        nval = max(1, int(0.2 * len(Xt)))
        idx = np.random.permutation(len(Xt))
        vi, ti = idx[:nval], idx[nval:]
        best = 1e9; best_state = None; patience = 0
        for ep in range(epochs):
            model.train()
            perm = np.random.permutation(len(ti))
            bs = 128
            for b in range(0, len(perm), bs):
                sel = ti[perm[b:b + bs]]
                opt.zero_grad()
                pred = model(Xt[sel])
                loss = lossf(pred, Yt[sel])
                loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                vloss = lossf(model(Xt[vi]), Yt[vi]).item()
            if vloss < best - 1e-5:
                best = vloss; best_state = {k: v.clone() for k, v in model.state_dict().items()}; patience = 0
            else:
                patience += 1
                if patience > 10:
                    break
        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        return model

    def predict_gru(model, X):
        with torch.no_grad():
            return model(torch.tensor(X, dtype=torch.float32)).numpy()
