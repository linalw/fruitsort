# Changelog

本项目遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。
每个版本对应一个 git tag 与 GitHub Release；Release 描述按下方格式写清"做了什么 / 怎么验证 / 已知问题"。

## [Unreleased]

### Planned
- DAgger / ACT / Diffusion Policy
- 图像观测（overhead + wrist）
- RGB-D 感知（颜色+形状分类、深度反投影、手眼标定）
- ACT / Diffusion Policy 模仿学习基线（脚本专家生成示教）
- 30–50 Hz 残差纠偏与异步执行（DSDyn / FASTER 思路）
- 四类同场 ≥ 60%

---

## [1.0.0] - 2026-11-18

### Accepted scope（用户确认）
- **0.25 m/s、平行夹爪、四类同场：22/24 = 91.7%，单件 3.94 s** —— 验收档
- 不使用吸盘末端

### Added
- `scripts/record_video.py --executor hybrid`：按验收策略录视频
  `data/fruitsort/eval/model_demo_v025.mp4`（0.25 m/s 四类，3/4）
- `src/fruitsort/models/act_policy.py` + `scripts/train_act.py` + `scripts/eval_act_suction.py`
  （ACT-lite 6.87M，离线 4.8cm；闭环仍 0/5，作为负结果保留）

### Results
| 带速 | 总体 | 苹果 | 草莓 | 香蕉 | 梨 | 节拍 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.08 | 95–97.5% | 100% | 100% | 80–90% | 100% | 2.8–4.0 s |
| **0.25** | **91.7%** | 100% | 100% | 67% | 100% | 3.94 s |
| 0.30 | 83.3% | 100% | 100% | 67% | 50–67% | 3.8 s |

### Known Issues（明确不达标项）
- 0.30 m/s 全类 ≥85% 未达成（香蕉/梨顶抓物理限制）
- 纯学习策略闭环：0.08 最好 40%，0.30 为 0/16；ACT-lite 6.87M 仍 0/5
- 学习残差 1/5；抓取/搬运仍由确定性执行器完成

---

## [0.11.0] - 2026-11-18

### Fixed
- **示教标签 bug**：`ScriptedSorter._action` 把 `last_target` 存成限速后的 6cm 目标，
  导致所有 BC 标签都是"目标≈当前 TCP"，模型学成悬停不动点。
  现在保存 FSM 的真实目标（未限速），限速只用于执行。

### Added
- `scripts/eval_policy_pure.py`：纯策略闭环（无伺服/无脚本执行器），支持 `--z-gate`、`--delta`
- 重录 33 条 raw 标签示教（苹果 10 / 草莓 10 / 香蕉 8 / 梨 5）

### Results
| 策略 | 0.08 m/s 苹果 |
| --- | --- |
| 旧标签（限速） | 0/5 |
| **raw 标签** | **2/5 = 40%** |
| raw + 学习式残差（8cm） | 1/5（噪声内） |
| 0.30 m/s 纯策略 | 0/4（速度 OOD） |

---

## [0.10.0] - 2026-11-18

### Added
- **吸盘末端原型** `scripts/eval_suction.py`
  - 顶部接近 → 吸附（位姿跟随近似）→ 搬运 → 释放
  - 支持扰动测试与视频录制
- 吸盘演示视频：`data/fruitsort/eval/suction_demo_030.mp4`（3.8 MB，四类各一次）

### Results（0.30 m/s，30 Hz，四类同场）
| 指标 | 值 |
| --- | --- |
| 成功率 | **24/24 = 100%** |
| 单件节拍 | **1.55 s**（P90 1.60 s） |
| 高速扰动（苹果，各 5 次） | none/offset/speed/push 全部 **5/5** |

对照：平行夹爪 + 顶抓在同一速度下最好 83.3%（香蕉/梨是瓶颈）。

### Known Issues
- 吸盘附着用「位姿跟随」近似，不是物理级真空模型；真机需要真实吸盘/负压模型
- 纯策略（抓取+搬运由网络输出）仍未跑通；残差控制器 0/5
- 侧向抓取在带面高度不可达（z=0.46 目标实际只能到 0.598，误差 0.14 m）

---

## [0.9.1] - 2026-11-18

### Changed
- **预测闭爪**：高速时用 `tcp.x − fruit.x ∈ [−0.01, 0.08]` 触发，而不是等位姿对齐
- **抓后带前馈抬升**：抬升时水平跟一段带速（`v_belt·dt·1.2`），减少带面拖拽滑移
- 传送带驱动上界 0.08 → 0.05；梨摩擦 0.8 → 1.0（0.3 m/s 下抓握更稳）
- `benchmark_sorting` 支持 `--max-step`

### Results（四类同场，30 Hz）
| 带速 | 总体 | 苹果 | 草莓 | 香蕉 | 梨 | 节拍 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.25 m/s | **91.7%** | 100% | 100% | 67% | 100% | 3.85 s |
| 0.30 m/s | 79–83% | 83–100% | 100% | 67% | 50–67% | 3.8 s |

0.3 m/s 尚未达到 85%（香蕉/梨是瓶颈）；0.25 m/s 已达标。

---

## [0.9.0] - 2026-11-18

### Added
- **测试视频** `scripts/record_video.py` → `data/fruitsort/eval/model_demo_v08.mp4`
  （798 帧 / 20 fps，四类各一次，4/4）
- **扰动/泛化评测** `scripts/eval_disturbance.py`
  （横向偏移 ±0.12 m / 带速抖动 ×U(0.8,1.2) / 1.5 N 侧推，四条件各 5/5）
- **残差控制器**（智能体产出）：`learning/residual.py`（8068 参数，0.26 ms）、
  `scripts/train_residual.py`、`scripts/eval_residual.py`
- **delta 动作**：`--delta-actions` / `--delta`，预测相对 TCP 的增量
- 控制频率参数 `--control-freq`（20/30/50 Hz）与速度自适应下探预瞄

### Fixed
- **传送带驱动上界 bug**：`belt_top+0.30` 会让已抓起/搬运中的水果继续被带面拖动；
  改为 `+0.08`（高速滑落的主因之一）
- **归一化退化维度**：训练集 `pear_active` 恒为 1，std≈0；2 类场景评测时被放大到 −1e6。
  `Normalizer` 现在对 std 设 1e-3 下限

### Results
| 项 | 结果 |
| --- | --- |
| 测试视频 | 4/4，798 帧 |
| 抗扰（0.08 m/s，四类） | none/offset/speed/push 全部 5/5 |
| 0.30 m/s 苹果（scripted，30 Hz） | 6/6 = 100%，节拍 5.34 s |
| 0.30 m/s 草莓 | 1/6（瓶颈） |
| 纯策略（模型直接输出抓取/搬运） | 0/5；策略搬运片段 3/5 |
| 残差控制器 | 0/5（基座不动点问题，已记录分析） |

### Known Issues
- 0.3 m/s 仅苹果稳定；草莓/整体未达 85%
- 纯策略仍卡在"目标≈当前 TCP"的悬停不动点（delta 动作也未能完全消除）
- 残差限幅 ±3 cm 无法改变吸引域；需要 (a) 修基座 z 头 (b) tcp 相对增量 (c) 在环 DAgger
- 香蕉在 teacher 的远距分支会被带面走廊卡住（已定位）

---

## [0.8.0] - 2026-11-18

### Changed
- **梨几何**：改为更圆回转体 h70/r30（消除低摩擦带面上的翻倒/侧滚）
- **梨物理**：摩擦 1.10/1.00 → 0.80/0.70；梨/香蕉角阻尼 1.0 → 3.0（抑制长距离滚动）
- **香蕉/梨触底判定**：0.015 → 0.045（下探会顶到带面，避免卡在 DESCEND）
- **抓后慢抬**：先用 1.5 cm/步垂直抬 5 cm，再搬运，避免甩飞
- 四类场景示教：苹果/草莓各 12 条（4 类同场）+ 梨 12 条（单类）

### Added
- `scripts/benchmark_sorting.py`：成功率 + 单件节拍（进入抓取区 → 放置完成）
- `--single-class` / `--policy {teacher,scripted,hybrid}` 评测开关
- 四类模型 `runs/vpolicy-4c-v3`（本地，不入库）

### Results（四类同场，带速 0.08 m/s）
| 形态 | 成功率 | 节拍 |
| --- | --- | --- |
| 模型选择 + 伺服（20 episodes） | **19/20 = 95.0%** | 2.82 s / P90 3.02 s |
| 模型选择 + 几何执行器（40 episodes） | **39/40 = 97.5%** | 3.97 s / P90 4.21 s |
| 指令选择（四类） | **96/96 = 100%** | — |

按类别（hybrid）：苹果 100%、草莓 100%、香蕉 90%、梨 100%。

### Verified
- 目标「成功率 ≥85%、单件 ≤7 s」在四类同场下由两组独立评测达成
- 复现命令见 `docs/benchmark.md`

---

## [0.7.0] - 2026-11-17

### Added
- **自蒸馏数据采集** `scripts/collect_self_distill.py` + `policies/hybrid_teacher.py`
  - teacher = 模型选择 + 近距伺服 + 几何夹爪/搬运（v0.6 的 5/5 配置）
  - 采集 24 条成功示教（苹果/草莓各 12）
- **延迟增强**：`ImageDemoDataset(delay_prob=...)`，训练时随机把最后一帧替换为上一帧
- **预测驱动伺服实验** `scripts/eval_vision_pred.py`（未来预测头 → 伺服目标）

### Results
| 配置 | 苹果 |
| --- | --- |
| 原始示教 + 自蒸馏 + 延迟增强，模型+伺服 | **5/5** |
| 同上，纯模型（无伺服） | 0/5 |
| 预测驱动伺服（未来头） | 0/5 |

诊断：未来预测头预测 0.1 s 后的水果位置**平均误差 0.96 m**，退化为均值 → 不可用。

### Finding
1. 自蒸馏补数据分布 + 延迟增强，仍无法消除纯模型的闭环漂移；
2. 未来预测头需要更大容量/更高损失权重，当前 0.41M + 0.1 权重学不动；
3. **学习负责意图与远距引导、几何伺服负责最后几厘米**，是被三组实验反复验证的结构性结论。

---

## [0.6.1] - 2026-11-17

### Changed
- 动作头 z 改为**相对 TCP 的残差回归**（`act_z + tcp_z`），缓解悬停不动点
- 夹爪头 `bias` 初始化到 logit(0.18)≈−1.5；BCE 加 `pos_weight=4.5`
- `eval_vision_geo.py` 支持 `--near 0`（禁用伺服）与 `--grip model`（纯模型夹爪）

### Results
| 配置 | 苹果 |
| --- | --- |
| 模型选择 + 近距伺服 + 几何夹爪 | **5/5** |
| 纯模型（无伺服、模型夹爪） | 0/5 |
| 专家帧上的 z 偏差 | 9 mm（修复前 2 mm） |
| 抓取区内闭爪预测率 | 79% |

### Finding
两个训练缺陷修复后，**纯模型闭环仍然 0/5**。说明剩余瓶颈不是这两个头本身，而是闭环分布漂移：
模型在专家帧上很准（z 偏差 9 mm、闭爪率 79%），但自己走两步就偏离。
几何伺服仍然是必要的纠偏层，这与 AHEAD 的 "predict-then-servo" 结论一致。

---

## [0.6.0] - 2026-11-17

### Added
- **纯视觉闭环跑通**：`scripts/eval_vision_geo.py`
  - 模型选择水果 + 近距视觉伺服（水果位置 + v·0.10）+ 几何夹爪判据
  - 苹果 **5/5**、草莓 **5/5**；四类场景下香蕉 2/4
- **夹爪时序状态机** `policies/gripper_fsm.py` + `tests/test_gripper_fsm.py`（6 个测试）
  - APPROACH→ALIGN→DESCEND→CLOSE→VERIFY→LIFT→CARRY→PLACE→RELEASE→DONE
  - 超时/重试/掉件检测；LIFT 边抬边向料框移动
- **时序设计文档** `docs/gripper-timing.md`：5 类闭爪触发 × 4 类动作块执行的对比与选型
- **四类指令选择评测** `scripts/eval_selection.py`
- 四类数据采集（苹果/草莓/香蕉/梨指令），香蕉 6 条成功；梨在四类场景 0/15

### Results
| 指标 | 值 |
| --- | --- |
| 纯视觉（苹果） | 5/5，10.6 s/件 |
| 纯视觉（草莓） | 5/5，17.0 s/件 |
| 指令选择（苹果/草莓/香蕉） | 100% |
| 指令选择（梨） | 0%（无示教数据） |
| 参数量 / 延迟 | 0.41M / GPU 1.9 ms、CPU 6.1 ms |

### Known Issues
- 动作头存在 z 悬停不动点（目标平均高于水果 7.7 cm），需要改成 Δz 回归或下探帧过采样
- 夹爪 logit 类别不平衡，离线从不预测闭合；当前用几何判据绕过
- 梨在四类场景抓取失败（专家 0/15），需要单独数据或抓取策略
- 四类同场仍只有香蕉 2/4

---

## [0.5.0] - 2026-11-17

### Added
- **多视角指令条件策略** `src/fruitsort/models/vision_policy.py`
  - 双视角（overhead + 腕部）共享 CNN，RGB + 帧差 4 通道，96×96
  - 4 帧历史 + 2 层 Transformer（d=96, 4 头）
  - 动作分块头（K=8）、未来状态辅助头（H=3）、**指针式水果选择头**
  - 参数 **405,248（0.41M）**
- **指令跟随任务**：环境支持 `instruction_kind`，专家支持 `only_kind`
- **示教采集**：`--instruction` 分指令录制（12 条抓苹果 + 12 条抓草莓）
- **DAgger**：`scripts/dagger_collect.py` + Markov 几何标注器 `policies/markov.py`
- **训练/评测/基准**：`train_vision_policy.py`、`eval_vision_policy.py`、`eval_hybrid.py`、`bench_latency.py`
- 文档 `docs/model-architecture.md`

### Results
| 指标 | 值 |
| --- | --- |
| 参数量 | 0.41M |
| GPU 延迟（batch=1） | 1.9 ms（p95 4.9 ms） |
| CPU 延迟（单线程） | 6.1 ms（p95 10.1 ms） |
| 指令选择准确率 | 100%（64/64） |
| 分层执行成功率 | 10/10（14.3 s/件） |
| 纯视觉闭环 | 0/5（抓取时机不稳） |

### Known Issues
- 纯视觉端到端仍未跑通：夹爪闭合时机与目标位姿误差叠加
- 仅 2 类、24 条指令示教；香蕉/梨未覆盖
- 分层执行依赖脚本专家，不是端到端学习

---

## [0.4.0] - 2026-11-17

### Added
- **示教采集** `scripts/record_demos.py`
  - 49 维状态观测（TCP / 关节 / 带速 / 四类水果位姿速度）
  - 4 维回归目标（世界坐标目标位姿 + 夹爪），只保存成功 episode
  - `--noise` 支持鲁棒化数据采集
- **BC 基线** `src/fruitsort/learning/bc.py`
  - 5 帧历史 + 3 层 MLP + 动作分块（K=8）
  - 目标位姿回归（比增量动作更平滑）
- **训练 / 评测脚本** `scripts/train_bc.py`、`scripts/eval_bc.py`
  - 多数据集合并、episode 级划分、EMA 平滑、闭环保留视频
- 文档 `docs/learning.md`

### Results
| 策略 | 场景 | 成功率 |
| --- | --- | --- |
| 脚本专家 | 苹果+草莓，10 episodes | 100% |
| BC（目标位姿 + 历史） | 苹果+草莓，10 episodes | 10%（1/10） |

### Known Issues
- BC 闭环成功率远低于专家：复合误差 + 阶段不可观测 + 动作限速导致的敏感性
- 观测仍是 oracle 位姿，不含图像
- 仅 2 类场景

---

## [0.3.0] - 2026-11-17

### Added
- **RGB-D 感知模块** `src/fruitsort/perception/vision.py`
  - HSV 连通域 + 颜色族分类（红/黄/绿）
  - Hungarian 多目标分配（颜色 + 形状先验 + 历史位置）
  - 平面单应反投影 + 高度视差修正
  - 遮挡时恒速外推（`max_lost=15` 帧）
- **相机标定脚本** `scripts/calibrate_camera.py` → `assets/calib/overhead_homography.npy`
  - 32 个对应点，重投影误差 0.38 px
- `scripts/benchmark.py --perception vision`（默认仍为 oracle）

### Results
- 视觉位置精度：苹果 4 mm、草莓 8 mm、香蕉 21 mm、梨 15 mm（均值）
- 闭环视觉控制：0/5（未达标，原因见 `docs/perception.md`）
- oracle 基线保持：苹果+草莓 5/5、10/10

### Known Issues
- 机械臂遮挡导致下降阶段目标丢失
- 香蕉/梨色相接近，依赖进料顺序先验
- 视觉模式尚未接入残差纠偏

---

## [0.2.0] - 2026-11-17

### Added
- **仿真环境 `FruitSort-v1`**（ManiSkill3 / SAPIEN）
  - 5.0 m 动态传送带（低摩擦带面 + 速度伺服），侧导轨、支腿、条纹
  - 四个类别料框（苹果/草莓/香蕉/梨），彩色边沿
  - Franka Panda，`pd_ee_pose` 20 Hz 控制；1200 步 episode
  - 两路 RGB-D 相机（overhead / side）+ 1280×720 渲染相机
- **四类程序化水果资产**（trimesh → GLB，凸包碰撞）
  - 尺寸/质量/颜色/粗糙度/摩擦按真实水果设定，见 `docs/sim-assets.md`
- **脚本化专家策略** `ScriptedSorter`
  - 预测拦截（t_lead=0.35 s）+ 顶部抓取 + 中转点搬运 + 入框
  - 渐进闭爪、掉件检测、状态看门狗、抓空重试
- **评测体系**
  - `scripts/benchmark.py`：per_class / pair / all4 三种场景
  - Wilson 95% CI、失败分类、`runs/<id>/metrics.json`
  - `scripts/demo.py`：视频录制
- **文档**
  - `docs/solution-architecture.md`：系统分层、选型、论文→模块映射
  - `docs/benchmark.md`：协议与 v0.2 结果
  - `docs/sim-assets.md`：资产规格
  - `docs/maniskill3-api.md`：ManiSkill3 自定义任务 API 笔记
  - `docs/literature-dynamic-grasping.md`：19 篇动态抓取文献综述
  - `notes/papers/`：12 篇 PDF 精读笔记

### Results
| 场景 | 成功率 |
| --- | --- |
| 苹果（6） | 6/6 = 100% |
| 草莓（6） | 6/6 = 100% |
| 香蕉（6） | 2/6 = 33.3% |
| 梨（6） | 3/6 = 50.0% |
| 总体（24） | 17/24 = 70.8% |

### Known Issues
- 香蕉/梨抓取易滑落（曲面 + 质量分布）
- 四类同场串行节拍不足（需要双臂或更快策略）
- 传送带为低摩擦 + 速度伺服近似，缺少自然滚动
- 放置从 0.50 m 释放，未建模碰撞损伤

---

## [0.1.0] - 2026-11-17

### Added
- 研究工程骨架：`AGENTS.md`、`README.md`、`docs/`、`notes/`、`runs/`、`src/fruitsort/`
- 工具：`scripts/env.sh`（ROS 2 + conda）、`scripts/doctor.sh`（环境快照）、
  `scripts/new_run.py`（实验脚手架）、`scripts/arxiv_digest.py`（arXiv 日报）、
  `scripts/fetch_weights.sh`（断点续传）
- `src/fruitsort/eval.py`：Wilson 区间 + episode 汇总
- `src/fruitsort/paths.py`：统一数据/权重/实验路径
- 项目级 skills：`embodied-lit-review`、`robot-stack-map`、`robot-profiling`
- 决策记录 D1：仿真优先，真机后接
