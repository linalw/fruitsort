# 基准与结果

> 官方入口：`python scripts/benchmark.py`。所有数字由该脚本产出，结果写入 `runs/<id>/metrics.json`。
> 环境：RTX 5090 工作站，ManiSkill3 3.0.1 / SAPIEN 3.0.3，CPU 仿真后端（单环境）。

## 协议

| 项 | 值 |
| --- | --- |
| 控制频率 | 20 Hz（`pd_ee_pose`，7 维：xyz + euler + 夹爪） |
| Episode 上限 | 1200 步（60 s） |
| 带速 | 名义 0.06 m/s（实测 ≈0.05 m/s）；可配 0.05–0.18 |
| 成功判据 | 所有水果在正确料框内、z ≤ 0.16 m、速度 < 0.1 m/s |
| 失败判据 | 水果落到地面/飞出带外 |
| 置信区间 | Wilson 95% CI |
| 随机化 | 水果初始 y ±0.05 m、朝向 ±π、带速（若给区间）、关节噪声 0.02 |

## 场景预设

```bash
# 单类（衡量抓取/放置能力）
python scripts/benchmark.py --scenario per_class --episodes 6

# 双类（快速回归，默认）
python scripts/benchmark.py --scenario pair --episodes 10

# 四类同场（完整任务）
python scripts/benchmark.py --scenario all4 --episodes 10
```

## 结果 v0.8.0（四类同场，模型选择 + 伺服/几何执行）

| 执行形态 | 成功率 | 节拍 |
| --- | --- | --- |
| 模型选择 + 伺服（teacher） | 19/20 = 95.0% | 2.82 s / P90 3.02 s |
| 模型选择 + 几何执行器（hybrid） | 39/40 = 97.5% | 3.97 s / P90 4.21 s |

复现：

```bash
python scripts/train_vision_policy.py --data data/datasets/instr-apple-4c \
    data/datasets/instr-strawberry-4c data/datasets/instr-banana \
    data/datasets/instr-pear-single --out runs/vpolicy-4c --epochs 40
python scripts/eval_selection.py --ckpt runs/vpolicy-4c/ckpt.pt
python scripts/benchmark_sorting.py --policy teacher --ckpt runs/vpolicy-4c/ckpt.pt \
    --classes apple strawberry banana pear --episodes 5 --belt 0.08 --spawn-x -0.55 --spawn-spacing 0.6
```

## 结果 v0.2.0（脚本专家，oracle 位姿）

### 单类（每类 6 episodes，带速 0.06 m/s）

| 类别 | 成功率 | 95% CI | 平均耗时 |
| --- | --- | --- | --- |
| 苹果 | **6/6 = 100%** | 61.0–100% | 10.9 s |
| 草莓 | **6/6 = 100%** | 61.0–100% | 10.6 s |
| 香蕉 | 2/6 = 33.3% | 9.7–70.0% | 36.8 s |
| 梨 | 3/6 = 50.0% | 18.8–81.2% | 36.4 s |
| **总体** | **17/24 = 70.8%** | 50.8–85.1% | 23.7 s |

### 双类（苹果 + 草莓，10 episodes）

| 指标 | 值 |
| --- | --- |
| 成功率 | **10/10 = 100%**（95% CI 72.2–100%） |
| 平均耗时 | 18.5 s/次（两次抓放 + 等待） |

### 四类同场

待补（v0.3 目标：≥60%）。当前瓶颈是香蕉/梨的抓取滑落与串行节拍。

## 失败模式（按出现频率）

| 模式 | 根因 | 计划 |
| --- | --- | --- |
| `slip`（香蕉/梨） | 曲面 + 质量分布，闭爪接触面积小；搬运加速度导致滑落 | 学习抓取位姿 / 残差纠偏 / 更慢的搬运轨迹 |
| `grasp_miss` | 高速或横向偏移时夹爪闭合窗口错过 | 提高预瞄精度、加入速度卡尔曼滤波 |
| `unreachable` | 果实漂到 |y|>0.12 或 |x|>0.25 | 机械臂可达域外 | 加宽导轨/改布局或加直线滑台 |
| `dropped` | 放置弹跳/被机械臂碰落 | 降低释放高度、加深料框、软质内衬 |
| `timeout` | 串行节拍不足（4 类同场） | 双臂/双工位、并行抓取、提高带速自适应 |

## 如何复现

```bash
source scripts/env.sh maniskill3
python scripts/benchmark.py --scenario per_class --episodes 6 --belt-speed 0.06
```

> 单次评测在 5090 上约 10–20 分钟（24 episodes，CPU 仿真）。GPU 并行将在 v0.4 用 Isaac Lab 或
> ManiSkill3 GPU backend 加速。
