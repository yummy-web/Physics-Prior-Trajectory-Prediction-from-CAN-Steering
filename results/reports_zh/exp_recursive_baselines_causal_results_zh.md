# 因果递推基线结果

## 协议

- comma2k19：42 条路线、路线级三折；nuScenes：固定 seed=7 的 200/566 场景子集、scene 级三折。
- 目标：3 s FDE。
- 输入：锚点 CAN speed 和 steering；预测期零阶保持；航向由后向差分重建；测试期不更新参数。
- 方法：fit-only RLS 与 steering-input EKF，固定 Q/R。

## 结果

| 数据集 | fit-only RLS FDE (m) | steering-input EKF FDE (m) |
|---|---:|---:|
| comma2k19 | 2.021 | 2.018 |
| nuScenes | 2.358 | 2.594 |

## 解释边界

这两个基线用于信息集公平性审计，不作为新的方法贡献。实验不覆盖装载机专用 articulated-vehicle 模型、完整标定非线性自行车模型或嵌入式硬件基线。
