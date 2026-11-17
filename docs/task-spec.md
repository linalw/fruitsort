# 任务定义（Task Spec）

> 状态：**v0.2 已确定**（2026-11-17）。改动需同步 `docs/decisions.md` 与 `CHANGELOG.md`。

## 1. 任务描述

- 一句话：单 Franka Panda 从**持续运动的传送带**上抓取水果，按类别放入对应料框。
- 场景：单臂、静止料框、刚性水果、无遮挡、无其他机器人。
- 节拍：v0.2 为串行分拣；目标 ≤ 8 s/件（当前苹果 10.9 s、草莓 10.6 s）。

## 2. 物体

| 属性 | 值 |
| --- | --- |
| 品类 | 苹果、草莓、香蕉、梨（四类） |
| 尺寸/质量 | 苹果 66×66×58 mm / 150 g；草莓 37×37×42 mm / 18 g；香蕉 170×43×32 mm / 110 g；梨 60×60×82 mm / 160 g |
| 颜色 | 深红 / 鲜红 / 黄 / 黄绿（见 `docs/sim-assets.md`） |
| 质地 | 刚体（凸包碰撞），无软体/破损建模 |
| 训练/测试 | 同一套资产；测试用不同随机种子与初始摆放 |

## 3. 硬件

| 项 | 型号 | 备注 |
| --- | --- | --- |
| 机械臂 | Franka Panda（7-DoF） | 臂展 855 mm，ManiSkill3 原生资产 |
| 夹爪 | 平行二指，最大开口 ~80 mm | 二值开合 |
| 相机 | 外部 RGB-D ×2（overhead / side） | 640×480，v0.3 接入感知 |
| 计算 | RTX 5090 32GB | 仿真 + 训练同机 |

## 4. 观测与动作

- **观测（v0.2，oracle）**：各水果位姿/速度、TCP 位姿、关节角、带速、语言指令字符串。
- **观测（v0.3 目标）**：overhead RGB-D；类别由颜色+形状分类；位置由深度反投影 + 手眼标定。
- **动作**：`pd_ee_pose`，7 维 = 末端 xyz + euler XYZ + 夹爪（+1 开 / −1 闭），20 Hz。
- **坐标系**：动作在机器人 root 系；世界系 → root 系用基座旋转（绕 z +90°）。

## 5. 成功判据

- **成功**：所有水果在各自料框内、中心高度 z ≤ 0.16 m、线速度 < 0.1 m/s（settled）。
- **失败**：任一水果落到地面（z < 0.10 且在框外）或飞出带外（|y| > 1.2）。
- **部分指标**：抓取成功率（CLOSE 后物体跟随 TCP）、误抓率、掉落率、按类别成功率。
- **失败分类**：`unreachable / grasp_miss / slip / wrong_bin / dropped / timeout`。

## 6. 数据规范

- 每条 episode：≤ 60 s / 1200 控制步，20 Hz。
- 目录（LeRobot 格式优先，v0.3 落地）：

```text
data/datasets/<name>/
├── meta/            # info.json, episodes.jsonl, tasks.jsonl, stats
├── data/chunk-000/  # parquet: observation.state, action, ...
└── videos/
```

- 相机参数与标定：`assets/calib/<date>/`
- 采集工具：脚本专家（`ScriptedSorter`）生成示教；真机阶段用遥操作 + ROS 2 bag → LeRobot 转换。

## 7. 评测协议

- 20 Hz、1200 步、固定 seed 序列（`--seed 0..N`）。
- 场景预设：`per_class`（4 类各 N 次）、`pair`（苹果+草莓）、`all4`（四类同场）。
- 报告：成功率 + Wilson 95% CI、平均耗时、失败分布。
- 入口：`python scripts/benchmark.py`，结果写 `runs/<id>/metrics.json`。

## 8. Non-goals（v0.2）

- 多机器人协同、动态料框、非刚性/易损建模、成熟度分级、真机部署（v0.5 起）。
