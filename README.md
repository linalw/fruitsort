# Fruitsort

**动态传送带水果分拣**的具身智能研究项目：单 Franka Panda 从运动中的传送带上抓取苹果 / 草莓 / 香蕉 / 梨，
按类别放入料框。仓库包含可复现的 ManiSkill3 仿真、程序化水果资产、预测拦截基线、基准评测与论文笔记。

![demo](assets/videos/demo_pair.gif)

## 当前结果（v0.2.0）

| 场景 | 成功率 | 95% CI | 平均耗时 |
| --- | --- | --- | --- |
| 苹果 + 草莓（10 episodes） | **100%** (10/10) | 72.2–100% | 18.5 s |
| 苹果单类（6） | 100% (6/6) | 61.0–100% | 10.9 s |
| 草莓单类（6） | 100% (6/6) | 61.0–100% | 10.6 s |
| 香蕉单类（6） | 33.3% (2/6) | 9.7–70.0% | 36.8 s |
| 梨单类（6） | 50.0% (3/6) | 18.8–81.2% | 36.4 s |

> 基线是**脚本专家 + 仿真真值位姿**（性能上界与示教源）。香蕉/梨的滑落是 v0.3 的主要改进点。

## 快速开始

```bash
source scripts/env.sh maniskill3    # ROS 2 Humble + conda 环境
make doctor                          # 环境体检 → docs/env-report.md

# 跑一段演示并录像
python scripts/demo.py --kinds apple strawberry --belt-speed 0.06 \
    --video data/eval/demo.mp4

# 基准评测
python scripts/benchmark.py --scenario pair --episodes 10 --belt-speed 0.06
python scripts/benchmark.py --scenario per_class --episodes 6
```

## 系统结构

```
感知（v0.3: RGB-D） → 目标选择（TTC + 可达性） → 拦截点预测（恒速/卡尔曼）
        → 抓取状态机（TRACK → DESCEND → CLOSE → CARRY → LOWER → RELEASE）
        → 20 Hz 绝对位姿控制（Franka Panda）
```

| 路径 | 内容 |
| --- | --- |
| `src/fruitsort/sim/` | 仿真环境、传送带、料框、程序化水果资产 |
| `src/fruitsort/policies/` | 脚本专家（预测拦截 + 抓放状态机） |
| `src/fruitsort/eval.py` | Wilson 置信区间、episode 汇总 |
| `scripts/benchmark.py` | 官方评测入口（per_class / pair / all4） |
| `scripts/demo.py` | 运行 + 录像 |
| `assets/meshes/` | 程序化生成的水果 GLB（可重建） |
| `docs/` | 任务定义、方案、资产、基准、文献、ManiSkill3 API |
| `notes/papers/` | 论文精读笔记（12 篇 + 补充） |

## 多视角指令条件策略（v0.5）

- **0.41M 参数**，2 层 Transformer + 共享 CNN
- 输入：overhead + 腕部两路 RGB（各 96×96，4 帧历史）+ 49 维状态 + 水果指令
- 输出：8 步动作块（世界目标位姿 + 夹爪）、未来 3 步状态、4 类水果选择 logits
- **实测**：GPU 1.9 ms / CPU 单线程 6.1 ms；指令选择 100%（苹果/草莓/香蕉）
- **纯视觉**：苹果 5/5、草莓 5/5（10.6 / 17.0 s 每件）；分层执行 10/10

```bash
python scripts/train_vision_policy.py --data data/datasets/instr-apple \
    data/datasets/instr-strawberry --out runs/vpolicy --epochs 40
python scripts/bench_latency.py --ckpt runs/vpolicy/ckpt.pt
python scripts/eval_hybrid.py --ckpt runs/vpolicy/ckpt.pt --episodes 10
```

详见 [docs/model-architecture.md](docs/model-architecture.md)。

## 模仿学习（v0.4）

```bash
# 采集 40 条专家示教（成功才保存）
python scripts/record_demos.py --kinds apple strawberry --episodes 40
# 训练 BC（5 帧历史 + 动作分块）
python scripts/train_bc.py --data data/datasets/fruitsort-XXXX --out runs/bc --epochs 80
# 闭环评测（seed 100+，与训练不重叠）
python scripts/eval_bc.py --ckpt runs/bc/ckpt.pt --kinds apple strawberry --episodes 10
```

当前结果：专家 100%，BC 10%（1/10）。离线拟合很好但闭环漂移，说明需要 DAgger / 时序策略，
详见 [docs/learning.md](docs/learning.md)。

## 视觉感知（v0.3，实验性）

```bash
python scripts/calibrate_camera.py          # 标定 overhead 相机（单应）
python scripts/benchmark.py --scenario pair --perception vision
```

位置估计：苹果 4 mm、草莓 8 mm、香蕉 21 mm、梨 15 mm。闭环控制尚未跑通（遮挡/分类歧义），
详见 [docs/perception.md](docs/perception.md)；默认评测仍用 oracle 位姿。

## 文档

- [解决方案与架构](docs/solution-architecture.md) — 分层、选型、论文→模块映射
- [任务定义](docs/task-spec.md) — 成功判据、动作空间、评测协议
- [仿真资产](docs/sim-assets.md) — 水果尺寸/质量/摩擦、机器人、传送带
- [基准结果](docs/benchmark.md) — 协议、数字、失败模式
- [研究路线图](docs/roadmap.md) / [决策记录](docs/decisions.md)
- [动态抓取文献综述](docs/literature-dynamic-grasping.md)
- [ManiSkill3 自定义任务 API](docs/maniskill3-api.md)
- [环境与约定](AGENTS.md)

## 版本管理

- 语义化版本 + git tag（`v0.1.0` / `v0.2.0` …），每个版本对应一个 GitHub Release。
- 详细变更见 [CHANGELOG.md](CHANGELOG.md)；Release 描述包含「做了什么 / 怎么验证 / 已知问题 / 下一步」。

## 环境

RTX 5090 32GB（sm_120，需 CUDA 12.8+ 的 torch）· Ubuntu 22.04 · ROS 2 Humble ·
conda 环境 `maniskill3`（ManiSkill3 3.0.1 / SAPIEN 3.0.3）。
