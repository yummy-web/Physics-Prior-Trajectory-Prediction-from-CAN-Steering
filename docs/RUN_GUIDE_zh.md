# 论文实验 — 运行说明（Q2 补强完整版）

本目录是论文的**完整可复现实验管线**，覆盖三个创新点，在**真实装载机数据**（222 秒，铰接式）
与 **comma2k19 公开数据集**（328 段、约 5.5 小时，乘用车阿克曼转向）上双重验证。
全部 CPU 计算，本机（Core Ultra 5 125H）约 **3–5 分钟**跑完全套 7 个实验。

---

## 一、Python 环境

推荐用你已有的 `path` 环境，或新建：

```bash
conda create -n loader_paper python=3.10 -y
conda activate loader_paper
pip install -r requirements.txt
```

核心依赖：numpy、pandas、scipy、scikit-learn、matplotlib（纯 CPU）。

**可选**（仅深度基线 exp7 需要）：PyTorch CPU 版
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
不装也能跑完 exp1–exp6，exp7 会自动跳过并提示。

---

## 二、数据准备（确认路径）

```
路径规划/
├── 数据分析/
│   ├── extracted_data/          ← 装载机提取后的CSV（已存在）
│   └── paper_experiments/       ← 本实验目录
└── 公开数据集/
    ├── Chunk_1/                 ← comma2k19（已下载）
    └── Chunk_2/
```

- comma2k19 首次加载会扫描 328 段并**缓存**到 `paper_experiments/.comma_cache.pkl`，之后秒读。
- 若装载机 `extracted_data/` 缺失，先在 `数据分析/` 下运行 `python 01_extract_bag_data.py`。

---

## 三、运行

### 一键全部
```bash
cd 数据分析/paper_experiments
python run_all_experiments.py
```

### 或分别运行
```bash
python exp1_steering_identification.py     # 创新点1：转向→曲率辨识（双车型）
python exp2_prediction_and_crossover.py    # 创新点2/3：预测+交叉点+误差各向异性
python exp3_baselines_ci.py                # 完整基线(CV/CA/CTRV/CTRA/EKF)+bootstrap CI+显著性
python exp4_bias_variance.py               # 交叉点的偏差-方差理论刻画（较慢，约1-2分钟）
python exp5_ablation.py                    # 物理模型组件消融
python exp6_robustness.py                  # CAN噪声/丢包/时延鲁棒性
python exp7_deep_baseline.py               # 深度GRU基线 vs 物理（需torch，否则自动跳过）
python exp8_grouped_comparison.py          # 基线分组(公平性)+ADE+横纵向分解
python exp9_stratified.py                  # comma按速度×转弯3×3分层验证
python exp10_hybrid.py                     # 改进物理-数据混合模型(方法贡献)
python exp11_cross_dataset.py              # 跨数据集验证(装载机+comma2k19+nuScenes三车型)
python exp12_pi_deep_comparison.py         # 与物理信息深度基线(PERL式/门控混合)公平对比[需torch]
```

**首次运行 nuScenes 相关实验前**：需先构建缓存（约2000个JSON，分批）。若 `.nusc_cache.pkl` 不存在，运行：
```bash
python nuscenes_loader.py     # 生成 .nusc_cache.pkl（566个可用场景）
```
本机一次即可跑完（VM因单次时限才需分批）。

---

## 四、输出

**figures/**（共 11 张，论文用图）

| 文件 | 内容 | 支撑 |
|------|------|------|
| fig1_steering_curvature.png | 转向→曲率辨识（装载机+comma双子图） | 创新点1 |
| fig1b_motor_consistency.png | motor_rpm=转向速率的内部一致性证据 | 转向作动器结论 |
| fig2_prediction_fde.png | 装载机 LOSO 预测 FDE | 方法对比 |
| fig3_crossover.png | 数据稀缺交叉点（两车型） | 创新点3 |
| fig4_error_anisotropy.png | 误差各向异性（速度主导 vs 转向主导） | 创新点2 |
| fig5_baselines_loader.png | 装载机全基线 + 95%CI | 严谨性 |
| fig6_baselines_comma.png | comma2k19 全基线 + 95%CI | 定量主张 |
| fig7_bias_variance.png | 偏差-方差理论拟合 + 交叉点 | 创新点3理论 |
| fig8_ablation.png | 物理模型消融 | 严谨性 |
| fig9_robustness.png | CAN 噪声/丢包/时延鲁棒性 | 工程价值 |
| fig10_deep_vs_physics.png | GRU vs 物理（随数据量） | 回答"为何不用深度模型" |
| fig11_grouped_ade_fde.png | 基线分两组(有/无未来控制)+CI | 公平性(核心抗审) |
| fig12_latlon_curves.png | 横向/纵向误差随时长 | 创新点2补充 |
| fig13_stratified_heatmap.png | 速度×转弯3×3分层 | 结论非平均假象 |
| fig14_hybrid.png | 改进混合模型 vs 纯物理/纯数据 | 方法贡献 |
| fig15_cross_dataset_R2.png | 三数据集转向辨识R²箱线图 | 创新点1跨车型 |
| fig16_cross_dataset_pred.png | nuScenes控制条件预测对比 | 创新点跨数据集 |
| fig17_pi_deep_comparison.png | vs物理信息深度基线(PERL式/门控) | T-ITS SOTA对比 |

> 所有图均为 300 DPI、无图内标题（期刊用图注），满足 IEEE 投稿要求。

**results/**：exp1–exp7 的数值 JSON。

---

## 五、预期关键结果（核对复现是否成功）

| 指标 | 预期值 |
|------|--------|
| 装载机 转向→曲率 合并 R² | ≈ 0.81 |
| 装载机 motor_rpm↔转向速率 相关 | 0.84–0.98（vs 车速 ≈ 0） |
| comma2k19 转向→曲率 R² 中位 | ≈ 0.92 |
| **comma2k19 物理先验 FDE@3s** | **≈ 0.5–0.6 m（显著优于 CTRA 1.33 / CTRV 1.83 / CV 2.75）** |
| 装载机各基线 FDE@3s | 2.8–3.3 m，**95%CI 大幅重叠（4 段样本小，不显著）** |
| 偏差-方差拟合 | E(N)=C/N^α，α≈1.0；交叉点 N*≈150 段（~9000s） |
| 物理消融 | Full 最优；去 warmup 标定伤害最大 |
| CAN 噪声鲁棒性 | 物理退化 ~+22%，MLP ~+44%（物理更稳） |
| GRU 深度基线 | 7200s 数据 FDE@3s≈1.2 m，仍**不敌**物理 0.5 m |
| **基线分组(exp8)** | B组(用未来控制)物理0.54/MLP0.61；A组(无控制)最好CTRA1.29。物理主擂台是B组 |
| 分层验证(exp9) | 9 个速度×转弯格子物理全胜(比 CTRA 优 1.7–3.7×) |
| **改进混合(exp10)** | k=8纯物理最优；k≥24 混合(门控/加性)胜两者，门控混合在 k=60 时约比纯物理降 22% → 方法贡献 |

数值因随机种子/环境有 ±5–10% 波动，**趋势与量级应一致**。

---

## 六、重要说明（论文定位）

- **定量主张由 comma2k19（328 段、统计显著）承担**；装载机 222 秒作为**真实工程车辆案例研究** + 定性发现（转向辨识、误差各向异性）。装载机 4 段的置信区间宽是客观事实，论文中如实作为局限说明——这让工作更抗审。
- 核心论点不是"我们的方法误差最低"，而是：① 用 CAN 转向**指令**预测显著优于只外推运动学的经典模型（CTRV/CTRA），且**跨车型成立**；② 误差主导方向由"哪个信号可观测"决定；③ 少数据下物理先验 > 数据驱动/深度模型，并给出偏差-方差理论与交叉点。

---

## 七、注意事项

- 图内用英文标签（避免缺中文字体报错）。要中文可改 `plot_utils.py` 的 `font.family`。
- exp4 训练较多 MLP，本机约 1–2 分钟属正常。
- 可复现性：随机过程固定 `random_state`/`default_rng` 种子。
- 首次跑会建 comma 缓存（几秒），之后更快。

---

跑完把终端输出或 results/ 发我，我核对数值无误后即可撰写论文（英文投稿 + 中文对照）。
