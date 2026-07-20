# 实验复现说明书（完整版）

> 覆盖 12 组实验、18 张论文图。全部 CPU 可跑，本机（Core Ultra 5 125H）约 5–8 分钟。
> 所有图 300 DPI、无图内标题（期刊用图注）。

---

## 0. 一句话总览

论文验证"数据受限下、基于 CAN 转向信号的车辆轨迹预测"，在**三个独立车型数据集**
（铰接装载机 / comma2k19 轿车 / nuScenes 雷诺 Zoe）上，证明物理先验方法的辨识能力、
误差各向异性、数据稀缺优势，并给出偏差-方差理论与门控混合方法贡献。

---

## 1. 环境搭建

```bash
conda create -n loader_paper python=3.10 -y
conda activate loader_paper
pip install -r requirements.txt
# 可选（exp7/exp12 深度基线需要）：
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

依赖：numpy, pandas, scipy, scikit-learn, matplotlib（必需）；torch CPU（可选）。

---

## 2. 数据准备

```
路径规划/
├── 数据分析/
│   ├── extracted_data/           装载机CSV（已存在；缺失则先跑 01_extract_bag_data.py）
│   └── paper_experiments/        本目录
└── 公开数据集/
    ├── Chunk_1/ Chunk_2/         comma2k19（已下载）
    └── can_bus/can_bus/          nuScenes CAN expansion（已下载，979场景json）
```

**首次运行需建两个缓存**（之后秒读）：
```bash
python comma_loader.py       # 生成 .comma_cache.pkl（328 段可用）
python nuscenes_loader.py    # 生成 .nusc_cache.pkl（566 场景可用）
```
> 本机一次跑完即可；仅受限沙盒才需分批。

---

## 3. 一键运行全部

```bash
python run_all_experiments.py
```
按顺序跑 exp1–exp12，生成 figures/ 下 18 张图与 results/ 下 12 个 JSON。

---

## 4. 实验清单与数据流

| 实验 | 脚本 | 依赖模块 | 数据集 | 产出图 |
|------|------|----------|--------|--------|
| exp1 | exp1_steering_identification.py | loader_data_loader, comma_loader | 装载机+comma | fig1,1b |
| exp2 | exp2_prediction_and_crossover.py | 同上 | 装载机+comma | fig2,3,4 |
| exp3 | exp3_baselines_ci.py | baselines, models_core | 装载机+comma | fig5,6 |
| exp4 | exp4_bias_variance.py | models_core | comma | fig7 |
| exp5 | exp5_ablation.py | models_core | 装载机 | fig8 |
| exp6 | exp6_robustness.py | models_core | comma | fig9 |
| exp7 | exp7_deep_baseline.py | deep_baseline, seq_dataset | comma | fig10 |
| exp8 | exp8_grouped_comparison.py | baselines, models_core | comma | fig11,12 |
| exp9 | exp9_stratified.py | models_core | comma | fig13 |
| exp10 | exp10_hybrid.py | models_core | comma | fig14 |
| exp11 | exp11_cross_dataset.py | 全部 loader | 三数据集 | fig15,16 |
| exp12 | exp12_pi_deep_comparison.py | deep_baseline, seq_dataset, models_core | comma | fig17 |

**核心模块**
- `loader_data_loader.py` 装载机加载（编码器解环绕、运动学真值）
- `comma_loader.py` / `nuscenes_loader.py` 公开数据集加载（统一 schema + 缓存 + 分层）
- `baselines.py` CV/CA/CTRV/CTRA/EKF 向量化基线
- `models_core.py` 增益标定、物理 rollout、MLP、bootstrap CI、Wilcoxon
- `deep_baseline.py` + `seq_dataset.py` GRU/LSTM 序列基线
- `plot_utils.py` 统一 300 DPI、无标题出图

---

## 5. 预期关键结果（核对复现）

| 指标 | 预期 |
|------|------|
| 转向→曲率 R²（装载机/comma/nuScenes） | 0.77 / 0.92 / 0.95 |
| motor_rpm↔转向速率 相关（装载机） | 0.84–0.98（vs 车速≈0） |
| comma 控制条件预测 FDE@3s（B组物理） | ≈0.5–0.6m，显著优于 A组 CTRA1.29/CTRV1.83/CV2.75 |
| 装载机全基线 FDE@3s | 2.8–3.3m，95%CI 大幅重叠（4段样本，不显著） |
| 偏差-方差 | α≈1.0，交叉点 N*≈150 段 |
| 物理消融 | Full 最优；去 warmup 标定伤害最大 |
| CAN 噪声鲁棒性 | 物理 +22% vs MLP +44% |
| GRU 深度基线 | 7200s 仍≈1.2m，不敌物理 |
| 门控混合(exp10/12) | 极稀缺 k=8 与物理持平最优；中/大数据胜纯物理与纯数据 |
| nuScenes 控制条件预测 | 物理 0.87m vs CTRV2.57/CV2.94 |

数值随种子/环境 ±5–10% 波动，趋势应一致。

---

## 6. 运行时间参考（本机 CPU）

- 快实验（exp1,5,9,11,3,8,6,2）：各数秒~30 秒
- 慢实验（exp4,7,10,12，训练多个 MLP/GRU）：各 1–3 分钟
- 全套 run_all：约 5–8 分钟

---

## 7. 常见问题

- **找不到数据集**：确认 `公开数据集/` 下有 `Chunk_1/2`（comma）和 `can_bus/can_bus/`（nuScenes）。
- **exp7/exp12 跳过**：未装 torch，装 CPU 版即可。
- **中文字体警告**：图用英文标签，无影响。
- **缓存损坏**：删除 `.comma_cache.pkl` / `.nusc_cache.pkl` 重建。
