# Changelog

本项目遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。
每个版本对应一个 git tag 与 GitHub Release；Release 描述按下方格式写清"做了什么 / 怎么验证 / 已知问题"。

## [Unreleased]

### Planned
- RGB-D 感知（颜色+形状分类、深度反投影、手眼标定）
- ACT / Diffusion Policy 模仿学习基线（脚本专家生成示教）
- 30–50 Hz 残差纠偏与异步执行（DSDyn / FASTER 思路）
- 四类同场 ≥ 60%

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
