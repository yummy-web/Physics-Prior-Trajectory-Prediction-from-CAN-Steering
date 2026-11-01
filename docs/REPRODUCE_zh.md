# 复现实验说明

## 数据与环境
数据不随仓库发布。请从 README 的官方链接获取 comma2k19、nuScenes CAN bus expansion 和 CRIStAL/LOCSP，并通过脚本参数或路径配置指向本地目录。

pip install -r requirements.txt
pip install -r requirements-gru.txt
pip install -r requirements-crisal.txt

## 主结果
python src/exp_maintable_v2.py
python src/exp_horizonwise_v2.py
python src/exp_lambda_group_calibration.py
python src/exp_lambda_analytic.py
python src/exp_lambda_analytic_nusc.py
python src/exp_lambda_analytic_route_ci.py
python src/exp_lambda_transfer.py
python src/exp_identifiability.py
python src/exp_recursive_baselines_causal.py

## 补充与审计
python src/exp_calibration_regret_audit.py
python src/exp_gru_baseline_causal.py
python src/exp_lambda_transfer_demand_unified.py
python src/audit_lambda_transfer_demand_unified.py
python src/exp_sample_funnel_audit.py
python src/exp_gate_v2.py
python src/exp_latency_params.py
python src/exp_scalinglaw_v2.py
python src/exp_lambda_group_calibration_convergence.py

gate 与 scaling-law 脚本用于协议敏感性诊断。固定配置、统计单位和边界见 results/reports_zh/PROVENANCE.md。

## CRIStAL/LOCSP
python src/exp_crisal_grouped_external.py --data-root <CRIStAL根目录>

窗口数量不替代路线、scene 或 acquisition 数量；测试集 oracle 是描述性上界；CRIStAL/LOCSP 结果属于同车多采集验证。
## 历史与非冻结诊断
`src/exp_target_calibration_demand_curve.py` 保留用于历史目标校准需求诊断；当前论文引用统一迁移脚本 `src/exp_lambda_transfer_demand_unified.py` 的冻结输出，前者不属于当前冻结证据链。
