# 模型架构：多视角时序策略（v0.5）

> 目标：把 12 篇论文里被实测有效的机制，落到「单臂 + 动态传送带 + 水果分拣」这一具体任务上。
> 代码：`src/fruitsort/models/vision_policy.py` · 训练：`scripts/train_vision_policy.py`

## 1. 总体结构

```text
                 ┌── overhead RGB (T 帧) ──┐
                 │                         ├─ 共享 CNN（RGB + 帧差 4 通道）→ 128
                 ├── wrist RGB   (T 帧) ───┘
                 │
                 ├── 状态 49 维 (T 帧) ──── MLP → 128
                 │
                 └── 任务/类别 embedding → 128
                                │
                    + 视角类型编码 + 时间位置编码
                                │
                    Transformer ×4 层（因果掩码，8 头）
                                │
                             [CLS]
                          ┌─────┴─────┐
                动作分块头 K=8×4    未来预测头 H=4×24
                (目标位姿+夹爪)      (4 类水果 pos+vel)
```

## 2. 论文 → 模块映射

| 机制 | 来源论文 | 本项目的实现 |
| --- | --- | --- |
| 多视角输入（固定 + 腕部） | StreamPI / FASTER / TEMPO | overhead + `panda_wristcam` 手眼相机，共享 CNN |
| 帧差作为运动线索 | DynamicWAM / Reflex | 图像 4 通道：RGB + \|I_t − I_{t−1}\| |
| 历史 T 帧 + 因果注意力 | StreamPI 原子时间单元 | T=5，时间位置编码，token 级因果掩码 |
| 动作分块（chunk） | ACT / FASTER / DiscreteRTC | K=8，每步重规划取第 0 步 |
| 未来状态预测辅助任务 | AHEAD / DynamicWAM / TrajWAM | 辅助头预测未来 4 步 × 4 类的 pos/vel |
| 任务/语言条件 | NEO / VLA | 类别 embedding（预留文本编码器接口） |
| 预测与抓取解耦 | AHEAD / TrajWAM | 辅助头只做表征学习，抓取仍由动作头输出目标位姿 |
| 延迟感知执行 | LIBERO-Dynamic / FASTER | 训练时随机丢帧（待加），推理每步重规划 |

## 3. 模块细节

### 视觉编码器（共享）

```text
Conv(4→24, k5,s2) → BN → ReLU
Conv(24→48, k3,s2) → BN → ReLU
Conv(48→96, k3,s2) → BN → ReLU
Conv(96→96, k3,s2) → BN → ReLU
AdaptiveAvgPool → 96 维
```

- 输入 **96×96**（overhead 从 640×480 下采样，wrist 原生 128→96）
- 第 4 通道是相邻帧差，零成本运动特征

### 时序 Transformer

| 项 | 值 |
| --- | --- |
| d_model | 96 |
| 层数 / 头数 | **2 / 4** |
| FFN | 384 |
| 历史 | T = 4（20 Hz 下 0.2 s） |
| 序列长度 | 1 + T×(2 视角 + 1 状态) + 1 指令 = 14 |
| 掩码 | 因果；CLS 看全部 |

### 输出头

- **动作头**：`Linear(96 → 32)` → `(K=8, 4)`，目标 = 世界坐标 (x, y, z) + 夹爪
- **未来头**：`Linear(96 → 72)` → `(H=3, 4 类 × 6)`，预测未来 pos/vel（辅助损失权重 0.2）
- **指令**：类别 id ∈ {all, apple, strawberry, banana, pear} → embedding，加到 CLS

## 3.5 三项硬约束的取舍

| 约束 | 设计 |
| --- | --- |
| **分清水果、执行指令** | 指令 id 注入 CLS token；数据按指令分集（12 条"抓苹果" + 12 条"抓草莓"）；评测按指令判定目标水果是否入框 |
| **速度** | 96×96 输入、2 层 Transformer、T=4、共享 CNN；动作分块 K=8（可 4 步只推理 1 次） |
| **参数少** | **0.39M**，单次前向 GPU 1.2 ms / CPU 3.5 ms |

## 3.6 实测结果（v0.5）

| 指标 | 值 | 说明 |
| --- | --- | --- |
| 参数量 | **405,248（0.41M）** | 含 CNN、2 层 Transformer、动作/未来/选择三个头 |
| GPU 延迟 | **1.9 ms**（batch=1，p95 4.9 ms） | RTX 5090，fp32 |
| CPU 延迟 | **6.1 ms**（单线程，p95 10.1 ms） | 20 Hz 预算 50 ms 的 12% |
| 指令选择准确率 | **100%**（64/64 帧） | 同帧切换 apple / strawberry 指令 |
| 分层执行成功率 | **10/10 = 100%** | 模型选水果 + 脚本专家抓放，14.3 s/件 |
| **纯视觉闭环（模型选择 + 几何伺服）** | **苹果 5/5、草莓 5/5** | `scripts/eval_vision_geo.py`，10.6 / 17.0 s 每件 |
| 纯视觉（四类场景，香蕉） | 2/4 | 4 类同场，选择正确但香蕉抓取弱 |
| 梨 | 无数据 | 专家在四类场景 0/15，需要单独数据/更好抓取 |

> v0.6 起，纯视觉链路（模型选水果 + 视觉伺服 + 几何夹爪/搬运）已在苹果/草莓上到达 **5/5**；
> 分层执行（`scripts/eval_hybrid.py`）仍作为对照与兜底。

## 4. 训练

| 项 | 值 |
| --- | --- |
| 损失 | MSE(动作) + 0.2 × MSE(未来状态) |
| 优化器 | AdamW，lr 3e-4，cosine，weight decay 1e-4 |
| 批大小 | 32 |
| 数据 | 20 条专家示教（带双视角图像），约 7000 个窗口 |
| 参数量 | 约 1.1M |

```bash
python scripts/record_demos.py --kinds apple strawberry --episodes 20 \
    --images --robot panda_wristcam
python scripts/train_vision_policy.py --data data/datasets/vla-pair --epochs 30
python scripts/eval_vision_policy.py --ckpt runs/xxx-vpolicy/ckpt.pt \
    --kinds apple strawberry --episodes 10
```

## 5. 与最强论文的差距（诚实评估）

| 能力 | 论文 | 本项目 |
| --- | --- | --- |
| 视觉主干 | CLIP/DINOv3/PaliGemma | 小型 CNN（数据量不够，先用小模型） |
| 时序 | VLA 主干 + KV cache | 4 层 Transformer |
| 世界模型 | 5B 视频 backbone / 3DGS | 仅未来状态辅助头 |
| 异步执行 | RTC / FASTER 流式 | 每步重规划（20 Hz） |
| 数据规模 | 百万级 episode | 20–60 条示教 |
| 语言 | 自然语言指令 | 类别 embedding |

## 5.5 DAgger 纠偏

BC 闭环会漂移（策略访问到的状态不在示教分布内）。DAgger 用 `policies/markov.py`
这个**无记忆几何标注器**给这些状态打标签：

- 手里有水果 → 去对应料框
- 否则 → 去拦截指令指定的水果；对齐后下探、闭爪

```bash
python scripts/dagger_collect.py --ckpt runs/vpolicy-instr3/ckpt.pt \
    --instruction apple --episodes 8 --out data/datasets/dagger-apple
python scripts/train_vision_policy.py \
    --data data/datasets/instr-apple data/datasets/instr-strawberry \
           data/datasets/dagger-apple data/datasets/dagger-strawberry \
    --out runs/vpolicy-dagger --epochs 30
```

## 6. 三项硬约束的达成情况

| 约束 | 结论 |
| --- | --- |
| 分清水果、执行指令 | ✅ 指针选择头 + 槽位 embedding，同帧指令切换 100% 正确 |
| 速度 | ✅ GPU 1.9 ms / CPU 单线程 6.1 ms，远低于 20 Hz 的 50 ms 预算 |
| 参数少 | ✅ 0.41M，单卡可训，CPU 可推理 |

## 6.4 自蒸馏与延迟增强（v0.7）

- **自蒸馏**：用 5/5 的 teacher（模型+伺服）重新采集 24 条示教，访问分布正是纯模型需要的
- **延迟增强**：训练时以 0.2 概率把最后一帧替换为上一帧（模拟 1 帧推理/感知延迟）
- 结果：**纯模型闭环仍然 0/5**，模型+伺服保持 5/5
- 结论：分布漂移不是靠"补数据分布"就能解决，闭环纠偏层是结构性的

**预测驱动伺服的尝试**：把未来预测头接进伺服（AHEAD 思路），但未来头未学会——
预测 0.1 s 后的水果位置平均误差 **0.96 m**（退化为均值），因此不可用。
当前伺服用 `水果位置 + v·0.10`，苹果/草莓 5/5。

## 6.5 已定位的模型缺陷（v0.6 诊断）

用 20 组对照实验（`/tmp/vision_diag/`）定位到两个训练侧问题：

1. **动作头有"悬停不动点"**：z 用绝对值回归 + 示教里悬停帧占多数，闭环目标 z 平均比水果高 7.7 cm，
   进入抓取区后只有 2% 的帧低于 +3 cm。修复方向：z 改为预测 **Δz**（相对当前 TCP），或对下探帧过采样。
2. **夹爪头类别极不平衡**：闭合帧仅占 18%，BCE 无 pos_weight，模型在 600 个专家帧上从未预测闭合。
   修复方向：`pos_weight≈4.5` + `grip_head.bias` 初始化到 logit(0.18)。

当前的工程绕法是：**夹爪与近距伺服改用几何判据**（d<0.05 且水果在抓取区），模型只负责远距目标与水果选择。
这也是 `docs/gripper-timing.md` 的结论：1 bit 的"何时闭爪"不该交给回归头。

## 7. 后续路线

1. **DAgger**：用专家修正策略访问到的状态（当前最大短板）
2. **延迟增强**：训练时随机丢弃最近 d 帧（d~U{0,4}），模拟推理延迟
3. **视觉主干升级**：数据到 500+ episode 后换 DINOv3/CLIP 冻结特征
4. **异步执行**：接入 action queue + RTC，降低 20 Hz 闭环对推理延迟的敏感度
5. **语言条件**：把类别 embedding 换成文本编码器，支持任意分拣指令
