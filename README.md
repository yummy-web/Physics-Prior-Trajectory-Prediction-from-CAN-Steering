# Physics-Prior Trajectory Prediction from CAN Steering

Reproducibility repository for the paper:

> **Physics-Prior Trajectory Prediction from CAN Steering under Data Scarcity:
> Cross-Vehicle Identification, a Bias–Variance Crossover, and a Gated Hybrid Model**

This repository contains all code, figures, and numerical results to reproduce the
experiments. It studies **control-conditioned vehicle trajectory prediction** driven by
the CAN steering signal, and shows — across **three independent vehicle classes** — that a
lightweight physics prior is highly data-efficient, that prediction error is anisotropic
(governed by which control signal is observable), and that a **gated physics–data hybrid**
outperforms both pure physics and pure data-driven models once a modest amount of data is
available.

---

## 1. Datasets

| Dataset | Role | Access |
|---------|------|--------|
| **Articulated wheel loader (private)** | Real engineering case study (222 s, CAN + RTK-GPS) | **Not redistributed** (confidential, provided by an industrial partner). Extracted, anonymized feature CSVs may be shared on request. |
| **comma2k19** | Sedan cross-vehicle validation | Public. Download from the official release: https://github.com/commaai/comma2k19 |
| **nuScenes CAN bus expansion** | Renault Zoe cross-vehicle validation | Public. Download `can_bus.zip` from https://www.nuscenes.org/nuscenes (Downloads → CAN bus expansion) |

> The raw loader bag files are **intentionally excluded** from this repository for
> confidentiality. The public datasets are **not** re-hosted here (they have their own
> licenses); use the official links above and the loader scripts in `src/`.

### Expected local layout
```
<project_root>/
├── extracted_data/                 # loader feature CSVs (private)
├── 公开数据集/  (or any name)
│   ├── Chunk_1/ Chunk_2/           # comma2k19
│   └── can_bus/can_bus/            # nuScenes CAN expansion (scene-*.json)
└── src/                            # this repo's code
```
The loaders auto-discover the dataset folder by scanning for `Chunk_*` (comma) and
`*_vehicle_monitor.json` (nuScenes).

For local reruns outside this repository layout, the loaders also accept:
`LOADER_EXTRACTED_DIR`, `COMMA_CACHE_FILE`, and `NUSC_CACHE_FILE`.

---

## 2. Environment

```bash
conda create -n loader_paper python=3.10 -y
conda activate loader_paper
pip install -r requirements.txt
# optional, for the deep GRU baselines (exp7, exp12):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
CPU-only; the full suite runs in ~5–8 min on a laptop-class CPU. No GPU required.

---

## 3. Run

```bash
cd src
python comma_loader.py         # build comma2k19 cache (first run only)
python nuscenes_loader.py      # build nuScenes cache  (first run only)
python run_all_experiments.py  # run exp1..exp13, writes figures/ and results/
```
Individual experiments (`exp1_*.py` … `exp16_*.py`) can also be run standalone.

---

## 4. Experiments and figures

| Exp | Content | Figures |
|-----|---------|---------|
| exp1 | CAN steering → curvature identification (loader + comma) | fig1, fig1b |
| exp2 | Prediction + data-scarcity crossover + error anisotropy | fig2–4 |
| exp3 | Full baselines (CV/CA/CTRV/CTRA/EKF) + bootstrap CI + significance | fig5, fig6 |
| exp4 | Bias–variance theory of the crossover | fig7 |
| exp5 | Physics-model ablation | fig8 |
| exp6 | CAN degradation robustness (noise/dropout/latency) | fig9 |
| exp7 | Deep GRU baseline vs physics | fig10 |
| exp8 | Grouped (control-conditioned vs extrapolation) + ADE + lat/lon | fig11, fig12 |
| exp9 | Stratified validation (speed × turn) | fig13 |
| exp10 | Gated physics–data hybrid | fig14 |
| exp11 | Cross-dataset validation (loader + comma2k19 + nuScenes) | fig15, fig16 |
| exp12 | Comparison vs physics-informed deep baselines (PERL-style/gated) | fig17 |
| exp13 | Gate ablation: learnable-scalar / learnable-MLP / heuristic / no-gate | fig_gate_ablation |
| exp14 | Crossover sensitivity across public vehicle platforms and random splits | fig18 |
| exp15 | Repeated-split stability of the smallest-budget gate ablation | fig19 |
| exp16 | Loader steering-identification diagnostics for the lower R2 case | fig20 |

All figures are 300 DPI without in-figure titles (captions are in the paper).

---

## 5. Citation

If you use this code, please cite the paper (see `CITATION` once the DOI is assigned).

## 6. License

Code released under the MIT License (see `LICENSE`). Dataset licenses are governed by
their respective providers (comma2k19, nuScenes).
