"""
causal_protocol.py  (revision, 2026-10)  --  SINGLE SOURCE OF TRUTH

Defines the frozen "causal main protocol" that every revision experiment must
obey, so the manuscript cannot drift back to oracle numbers.

Addresses:
  R2-1  future inputs not available at prediction time
  R2-6  route-level grouping (no leakage across train/test)
  R2-10 causal gating features only

PROTOCOL
--------
Information available at prediction time t0 (CAUSAL):
  - CAN steering & CAN speed up to t0   (history window)
  - current control at t0
  - state estimate at t0 (position, heading, obs_v at t0 only)
NOT available (would be ORACLE / leakage), reported separately as upper bound:
  - future steer[>t0], future obs_v[>t0] (pose-differenced), future CAN speed

SPLITTING
---------
  comma2k19: outer group = ROUTE (the yyyy-mm-dd--hh-mm-ss recording id),
             NOT the 60 s segment. 328 segs come from only 42 routes.
  nuScenes : outer group = scene token.
  loader   : outer group = recording (LOSO), as before.
Gains, scalers, thresholds, early-stopping validation: TRAIN groups only.
"""
import re
import numpy as np

HORIZON_S = 3.0
DT = {"comma": 0.05, "nuscenes": 0.02, "loader": 0.1}

# causal rollout defaults for the physics prior (see models_core.comma_phys_metrics_curv)
CAUSAL_STEER = "hold"       # freeze steering at t0 (deployable). 'hist' = extrapolate
CAUSAL_SPEED = "can_hold"   # freeze CAN speed at t0 (deployable)
ORACLE_STEER = "future"
ORACLE_SPEED = "obs"


def route_of(seg_key):
    """comma route id = the recording timestamp embedded in the seg key."""
    m = re.search(r"(\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2})", str(seg_key))
    return m.group(1) if m else str(seg_key)


def group_keys(data, dataset):
    """Return {group_id: [seg_key,...]} for grouped splitting."""
    groups = {}
    for k in data:
        if dataset == "comma":
            g = route_of(data[k]["seg"].iloc[0] if "seg" in data[k] else k)
        else:
            g = str(k)  # nuScenes scene token / loader recording == key
        groups.setdefault(g, []).append(k)
    return groups


def grouped_kfold(groups, n_folds=5, seed=42):
    """Yield (train_keys, test_keys) splitting at the GROUP level."""
    rng = np.random.default_rng(seed)
    gids = list(groups.keys())
    rng.shuffle(gids)
    folds = [gids[i::n_folds] for i in range(n_folds)]
    for f in range(n_folds):
        test_g = set(folds[f])
        train_keys, test_keys = [], []
        for g, ks in groups.items():
            (test_keys if g in test_g else train_keys).extend(ks)
        yield train_keys, test_keys


def summary(data, dataset):
    g = group_keys(data, dataset)
    sizes = [len(v) for v in g.values()]
    return {"n_segments": len(data), "n_groups": len(g),
            "segs_per_group_mean": float(np.mean(sizes)),
            "segs_per_group_max": int(np.max(sizes))}


if __name__ == "__main__":
    import pickle, os
    here = os.path.dirname(__file__)
    for name, cache, ds in [("comma", ".comma_cache.pkl", "comma"),
                            ("nuscenes", ".nusc_cache.pkl", "nuscenes")]:
        p = os.path.join(here, cache)
        if os.path.exists(p):
            d = pickle.load(open(p, "rb"))
            print(name, summary(d, ds))
