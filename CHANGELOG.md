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
