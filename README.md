# Fruitsort

机器人水果分拣的具身智能研究：感知 → 抓取 → 操作策略（模仿学习 / VLA）→ 真机评测。

本仓库只做**研究工程化**：统一环境、实验管理、数据落盘、文献沉淀。算法代码放 `src/fruitsort/`。

## 快速开始

```bash
# 1. 环境（非交互 shell 必须，ros2 才会进 PATH）
source scripts/env.sh maniskill3

# 2. 体检，生成 docs/env-report.md
make doctor

# 3. 新建一次实验（自动生成 meta/config/cmd/metrics/notes 并登记）
make run NAME=baseline-dp

# 4. 文献日报（去重，写入 notes/digests/）
make digest
```

## 目录

| 路径 | 内容 |
| --- | --- |
| `AGENTS.md` | 机器环境、坑、工作约定（**先读这个**） |
| `docs/task-spec.md` | 任务定义、成功判据、评测协议（待填） |
| `docs/roadmap.md` | 四阶段研究路线与验收指标 |
| `docs/baselines.md` | 感知/策略/VLA/仿真/数据集清单与复现状态 |
| `docs/env-report.md` | `make doctor` 自动生成 |
| `notes/papers/` | 论文笔记（`_template.md`） |
| `notes/digests/` | arXiv 日报 |
| `notes/ideas.md` | 想法与待验证假设 |
| `runs/` | 实验：`<日期>-<简述>/`，`registry.csv` 总表 |
| `src/fruitsort/` | 代码包：`paths`（路径）、`eval`（Wilson CI / episode 汇总） |
| `data/` | 软链到 9.1T 盘：datasets / checkpoints / raw / rosbags / eval |

## 本机环境

RTX 5090 32GB（sm_120，需 CUDA 12.8+ torch）· 32 核 / 125GB · Ubuntu 22.04 · ROS 2 Humble。
已有环境：`maniskill3`（仿真）、`dynvla`（LeRobot 0.3.3）、`dynvla-sim51`（Isaac Lab）、`rtools`（剖析/可视化）。
详见 `AGENTS.md`。

## 指标约定

所有评测用 `fruitsort.eval.summarize` 产出，写入对应 run 的 `metrics.json`：

```python
from fruitsort.eval import summarize, format_summary
print(format_summary(summarize(episodes)))
# 47/50 = 94.0% (95% CI 83.8–97.9%), 平均 5.82s/次
```
