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

## 验收档（v1.0.0，0.25 m/s，平行夹爪）

| 指标 | 值 |
| --- | --- |
| 总体成功率 | **22/24 = 91.7%**（95% CI 74.2–97.7） |
| 单件节拍 | 均值 **3.94 s**，P90 5.82 s |
| 苹果 / 草莓 / 梨 | 100% / 100% / 100% |
| 香蕉 | 67% |
| 指令选择 | 96/96 = 100% |

复现：
```bash
python scripts/benchmark_sorting.py --policy hybrid --ckpt runs/vpolicy-4c-v3/ckpt.pt \
  --classes apple strawberry banana pear --episodes 6 --belt 0.25 \
  --spawn-x -0.55 --spawn-spacing 0.6 --control-freq 30
```

视频：`data/fruitsort/eval/model_demo_v025.mp4`（四类，3/4；梨该局随机失败）。

## 低速纯策略协议（v1.4，0.08 m/s）

目标：**非脚本**分拣（策略同时输出抓取与搬运），低速优先，成功率 >65%、单件 <10s。

```bash
# 1) 采集同分布示教（四类各 40 局，0.08 m/s / 30 Hz / 平行夹爪）
for k in apple strawberry banana pear; do
  python scripts/record_demos.py --kinds apple strawberry banana pear --episodes 40 \
    --seed 20000 --belt-speed 0.08 --spawn-x -0.55 --spawn-spacing 0.35 \
    --control-freq 30 --images --robot panda_wristcam --instruction $k \
    --out data/datasets/d008-$k
done

# 2) 训练 ACT-lite（增量动作 + 相位条件化 + 下探窗过采样）
python scripts/train_act.py --data data/datasets/d008-{apple,strawberry,banana,pear} \
  --out runs/act-008-4c --epochs 60 --batch 64 --stride 2 --lr 1e-4 \
  --phase --delta-actions --descend-boost 3

# 3) 纯策略闭环（无脚本执行器；残差反馈 + 夹爪去抖）
python scripts/eval_act_pure.py --ckpt runs/act-008-4c-v2/ckpt.pt \
  --instruction apple --episodes 10 --belt 0.08 --spawn-spacing 0.35 \
  --control-freq 30 --fb 0.5 --grip-ema 0.6 --carry-latch
```

指标定义：
- **成功率**：目标水果落入对应料框（env 的 `success`）
- **单件处理时间**：目标水果首次进入抓取区 → 入框（不含传送带运送时间），对应「<10s/件」

## 高速结果（v0.10.0，吸盘末端原型，30 Hz；不纳入验收）

| 带速 | 总体 | 苹果 | 草莓 | 香蕉 | 梨 | 单件节拍 |
| --- | --- | --- | --- | --- | --- | --- |
| **0.30 m/s** | **24/24 = 100%** | 100% | 100% | 100% | 100% | **1.55 s**（P90 1.60） |

高速扰动（0.30 m/s，苹果，各 5 episodes）：none / offset ±0.12 / speed ×U(0.8,1.2) / push 1.5N **全部 5/5**。

吸盘原型的实现：`scripts/eval_suction.py`
- 顶部下探 → 吸附（**位姿跟随近似**，SAPIEN 无运行时固定关节 API）→ 搬运 → 释放
- 明确不是物理级真空模型：附着期间水果位姿跟随 TCP、速度清零

对照：同一 0.30 m/s 下「平行夹爪 + 顶抓」最好 20/24 = 83.3%（香蕉 67%、梨 50–67%）。

## 高速结果（v0.9.1，平行夹爪，30 Hz 控制）

| 带速 | 总体 | 苹果 | 草莓 | 香蕉 | 梨 | 节拍 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.08 m/s | 95–97.5% | 100% | 100% | 80–90% | 100% | 2.8–4.0 s |
| 0.25 m/s | **91.7%** (22/24) | 100% | 100% | 67% | 100% | 3.85 s |
| 0.30 m/s | 79–83% | 83–100% | 100% | 67% | 50–67% | 3.8 s |

- 0.30 m/s 的瓶颈是香蕉/梨的抓握与抬升滑移；苹果/草莓已稳定 100%
- 高速关键修复：预测闭爪（`tcp.x − fruit.x ∈ [−0.01,0.08]`）、抓后带前馈抬升、驱动上界 0.05
- 50 Hz 控制实测 0/24（sim_freq=100 下 2 子步），暂不采用

### 0.30 m/s 的失败排查（已试过，避免重复）
| 尝试 | 结果 |
| --- | --- |
| 香蕉抬升 0.005/步 | 更差（1/6） |
| 梨质量 0.16→0.12 kg | 更差（3/6） |
| 香蕉/梨摩擦 1.2 | 更差（33%） |
| 梨摩擦 1.0 | 单类 5/6，四类仍 50–67% |
| 50 Hz + max_step 0.024 | 0/24 |
| delta 策略 + 比例残差（6cm） | 0/5 |
| 纯策略 / 学习残差 | 0/5 |

结论：0.30 m/s 的香蕉/梨不是"参数没调对"，而是顶面抓取 + 高速相对滑移的物理限制；
需要换抓取方式（侧抓/吸盘）或降低该品类带速。

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
