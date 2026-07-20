# 第二个公开数据集下载指引（跨数据集验证用）

## 背景与结论

我联网检索确认：**不存在公开的"铰接式装载机 CAN + 轨迹"数据集**（文献里的装载机工作全用私有平台）。
因此第二个数据集只能是**另一类乘用车**，用途是把"CAN 转向 → 路径曲率"辨识范式和"控制条件预测"结论
从"comma2k19 单数据集"升级为"跨两个独立数据集成立"，进一步抗审。

我评估了候选，最优选择是 **nuScenes CAN bus expansion**。

---

## 首选：nuScenes CAN bus expansion（强烈推荐）

**为什么最合适**
- 含 `vehicle_monitor` 消息：**方向盘转向角(steering)、车速(vehicle speed)、横摆角速度**等真实 CAN 信号。
- 含 `pose` 消息：位置、朝向、速度（可作轨迹真值）。
- **体量小**：CAN expansion 本体只有几十 MB（不是完整 nuScenes 的 300GB+！），因为它只是 CAN 数值信号，不含图像/点云。
- 官方 devkit 读取方便，1000 个场景各 20 秒。

**下载步骤**
1. 注册并登录 nuScenes 官网：https://www.nuscenes.org/nuscenes （免费，学术用途）
2. 到 Downloads 页面，找到 **"CAN bus expansion"**（在 Full dataset 区域下方的 expansion 列表里）。
3. 下载 `can_bus.zip`（约几十 MB）。
4. 解压到：`D:\论文与项目\路径规划\公开数据集\nuscenes_can\`
   解压后应有形如 `scene-0001_meta.json`、`scene-0001_ms_imu.json`、`scene-0001_vehicle_monitor.json`、`scene-0001_pose.json` 等一批 JSON 文件。
5. （可选）装官方 devkit 方便校验：`pip install nuscenes-devkit`，但我会写独立解析器，不强制依赖。

**关键信号字段（vehicle_monitor.json 每条记录）**
- `steering`：方向盘角度(度)
- `vehicle_speed`：车速(km/h，注意要换算成 m/s)
- `utime`：时间戳(微秒)
`pose.json` 里有 `pos`(位置)、`orientation`(四元数)、`vel`(速度)、`utime`。

> 下好后把 `nuscenes_can` 文件夹告诉我，我写 `nuscenes_loader.py`，跑同一套 exp1(转向辨识) + exp8(控制条件预测) 流程，产出第三车型的对照结果。

---

## 备选：KITTI raw（若 nuScenes 注册不便）

**特点**：有 OXTS 组合惯导（GPS+IMU），提供位置、速度、**横摆角速度(yaw rate)**，但**没有方向盘转向角信号**。
所以它能验证"运动学预测/航位推算"，但**不能**直接验证"从方向盘转向辨识曲率"（缺转向信号）。作为运动学对照可用，作为核心范式验证不如 nuScenes。

**下载**：https://www.cvlibs.net/datasets/kitti/raw_data.php ，选带 "sync+rectified" 和 OXTS 的序列，单序列几百 MB~几 GB，下 2-3 个 "City"/"Residential" 类含转弯的序列即可。解压到 `公开数据集\kitti_raw\`。

---

## 我的建议

优先 **nuScenes CAN bus expansion**（小、信号对口、最能加分）。KITTI 作为退路。
下好任一个，告诉我文件夹路径，我立即写解析器并补跑跨数据集实验，把结果并入论文。
