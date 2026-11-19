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

## [1.10.0] - 2026-11-19

### Added
- 控制频率 60 Hz 全链路支持：`SIM_FREQ` 映射（20/30/60 → 100/120/240）
- `ScriptedSorter.rate = control_freq/30`：FSM 时间常量随频率缩放
  （修前 60 Hz 下草莓 0/50，修后 0.20 m/s 6/6）
- `d60-*` 示教（0.12–0.25 m/s、60 Hz、四类各 50 局）+ `act-60hz-4c`（val 0.059）

### Negative result
| 0.20 m/s | 30 Hz | 60 Hz |
| --- | --- | --- |
| 四类合计 | 12/32 = 37.5% | 5/24 = 20.8% |

60 Hz 反而更差（含按频率折算 fb/max-step 的 0/6）。单纯提高控制频率不能解决高速闭爪窗口问题。

---

## [1.9.0] - 2026-11-19

### Added
- `PolicyConfig.pear_extra_depth`：梨抓取深度参数化（默认 0.01，行为不变）

### Results（最终确认跑，推荐配置）
| 带速 | 成功率 |
| --- | --- |
| 0.08 | 18/20 = 90% |
| 0.15 | 19/20 = 95% |

### Negative results（已记录，避免重复尝试）
- 0.20 m/s 苹果：前馈 lead 0/8、fb 0.9 → 3/8、fb 1.2 → 1/8、descend-commit 4/8、chunk 4/8
  → 高速瓶颈是时序决策精度（±3 帧），非增益问题
- 梨：抓取深度 1/3/5cm → 5/8、5/8、0/8；搬运步长 6/4/3cm → 5/8、0/8、0/8
  → 抓到后在搬运中滑脱，属平行夹爪抓锥形薄果的几何极限

---

## [1.8.0] - 2026-11-19

### Added
- 演示视频规范落地：`mp4` + **同名 `.txt`**（脚本自动生成：代码版本 / 权重 / 配置 / 场景 /
  分集整局时长 + 单件处理时间 / 复现命令），并登记 `docs/videos.md`
  - `pure_policy_v008.mp4`、`pure_policy_v015.mp4`、`pure_policy_learned_carry_v015.mp4`
- 高速档示教 0.12–0.25 m/s（四类各 50 局）+ `act-high-4c`

### Results
| 带速 | 成功率 |
| --- | --- |
| 0.08 | 28/32 = 87.5% |
| 0.15 | 31/32 = 96.9% |
| 0.20 | 12/32 = 37.5% |
| 0.25 | 7/32 = 21.9% |

### Fixed
- `record_demos --spawn-x` 默认 −0.75 → −0.55（与评测一致），并警告水果出生超出带尾
  （该 bug 曾让第一版高速示教 0/50）

### Known Issues
- 0.20/0.25 纯策略明显衰减，失败形态是「没抓住」（不是搬运）；已试 fb/commit/chunk 均无效

---

## [1.7.0] - 2026-11-19

### Added
- **演示视频** `data/fruitsort/eval/pure_policy_v015.mp4`（0.15 m/s 四类 4/4，1080p，含相位叠加）
- `record_video.py --executor act`：纯策略录像，失败局自动重试
- `--carry-boost`：料框附近减速窗过采样 → **策略自己学会在框前减速**
  （`act-wide-4c-v2` 在 0.15 四类 32/32 = 100%，无需人工搬运限速）

### Results（同一配置 `act-wide-4c` + `--carry-max-step 0.02`）
| 带速 | 成功率 |
| --- | --- |
| 0.08 | 28/32 = 87.5% |
| 0.15 | 31/32 = 96.9% |
| 0.18 | 苹果 5/5、草莓 4/5 |

### Known Issues
- v2（策略自学减速）在 0.08 退化到 71.9%，高/低速配置尚未统一
- 梨在 0.08 仍是 4–5/8（抓取浅、易滑脱）

---

## [1.6.0] - 2026-11-19

### Added — 宽速域单一策略 + 搬运段限速
- **`act-wide-4c`**（示教 0.06–0.18 m/s，四类各 50 局）：
  **0.15 m/s 31/32 = 96.9%**、**0.08 m/s 28/32 = 87.5%**，单件 4.2–6.7 s
- `--carry-max-step`：搬运段单独限速，修掉在料框处过冲（0.15 草莓 0/10 → 9/10）
- 0.18 m/s：苹果 5/5、草莓 4/5（带宽上限验证）
- 同场景速度专用模型（`act-015-4c`）仅 62.5%，宽速域训练 + 搬运限速是关键

### Known Issues
- 梨 0.08 下 4/8（示教在宽速域仅 28/50，抓取浅易滑脱）
- 搬运限速对梨不利（慢速搬运会滑脱），单一常数无法兼顾快慢，需按类分档或让策略学搬运速度

---

## [1.5.0] - 2026-11-19

### Added — 抗扰 / 提速 / 速度泛化（纯策略）
- **速度域随机化（DR，0.06–0.12 m/s）**：`act-dr-4c` 在 0.08 上 **27/32 = 84.4%**（单件 2.7–3.9 s），
  且在 0.06 / 0.12 上苹果 5/5、草莓 4/5；速度专用模型（`act-008-4c-v2`）在这两档均为 0/5
- **抗扰**：`eval_disturbance.py --policy act`；横向偏移 ±12cm 苹果 3/5、草莓 3/5，
  侧向推力 1.5N 苹果 5/5、草莓 4/5（比专用模型草莓 1/5、3/5 提升）
- **提速 0.15 m/s**：`act-015-4c` 四类 23/40 = 57.5%（苹果 100%、香蕉 70%、梨 60%、草莓 0%）
- `src/fruitsort/policies/act_runner.py`：纯策略闭环执行器（评测/抗扰共用，重构后结果逐位一致）
- `record_demos.py --belt-max`：速度区间采样

### Known Issues
- 0.15 m/s 未达标（草莓 0/10：抓起正常但在料框处过冲后松爪）
- DR 模型梨 4/8：DR 采集时专家抓梨仅 20/40（数据端限制）
- 环境中途改 `belt_speed` 会造成带面突变，故带速抖动扰动项不作为结论

---

## [1.4.0] - 2026-11-19

### Added — 纯策略低速里程碑（0.08 m/s，无脚本执行器）
- **四类 58/80 = 72.5%**（苹果 90%、草莓 60%、香蕉 85%、梨 55%），单件处理时间均值 4.1 s、79/80 <10 s
- 数据：`d008-*` 四类各 40 局（0.08 m/s / 30 Hz / 平行夹爪）
- `train_act.py --phase --delta-actions --descend-boost`：3 类相位头 + 增量动作 + 下探窗过采样
- `eval_act_pure.py`：`--fb`（IBVS 式残差反馈）、`--grip-ema`、`--carry-latch`、
  `--descend-commit`、抓空自动重试、单件处理时间指标
- `scripted.py`：低速下香蕉/梨对齐容差放宽（专家 0.08 抓梨 2/40 → 21/40）

### Fixed（trace 定位）
- 绝对目标位姿 → 轨迹回放、误差无负反馈（改增量=执行位移）
- 悬停 6 cm 被判定为「已抱住」→ 抱空气走向料框
- 搬运中夹爪抖动 → 中途掉果
- 梨示教只有 2 条：场景排队时序超出步数预算

### Known Issues
- 草莓/梨仍有 40–45% 失败（抓取窗口边缘 + 搬运掉果）
- 单件最长 11.5 s（1/80 超出 10 s）
- 抗扰/提速（0.15–0.30 m/s）尚未在纯策略上验证

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
