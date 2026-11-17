# Fruitsort — 具身智能 / 机器人水果分拣研究

本文件是所有 agent 与协作者的入口：机器环境、可用能力、目录约定。**改动环境或约定后请同步更新本文件。**

## 研究目标（待细化）

机器人水果分拣：感知（检测/位姿/成熟度）→ 抓取规划 → 操作策略（模仿学习 / VLA）→ 真机部署与评测。
任务定义、成功判据、评测协议见 [`docs/task-spec.md`](docs/task-spec.md)，路线见 [`docs/roadmap.md`](docs/roadmap.md)。

## 机器环境

| 项目 | 值 |
| --- | --- |
| GPU | 1× RTX 5090 32GB（Blackwell **sm_120**，驱动 580 / CUDA 13.0） |
| CPU / 内存 | 32 核 / 125GB RAM |
| 系统 | Ubuntu 22.04.5，内核 6.8 |
| 磁盘 | `/media/ubuntu/66A27065A2703B9D` 约 9.1T 可用 |
| conda | `/home/ubuntu/linalw/App/minconda3`（base 是 Python 3.14，**勿直接用于深度学习**） |
| ROS | ROS 2 **Humble**（`/opt/ros/humble`），另有 `~/ros2_ws` |
| 权限 | **无 sudo**；只能 conda / pip / uv / npm 安装 |

> ⚠️ 5090 是 sm_120，**必须 CUDA 12.8+ 的 torch**（2.7+）。装旧版 wheel 会报
> `no kernel image is available for execution on the device`。

### 已有 conda 环境（不要擅自改动）

| env | 用途 |
| --- | --- |
| `maniskill3` | ManiSkill3 3.0.1 + SAPIEN 3.0.3，仿真首选 |
| `dynvla` | LeRobot 0.3.3 + transformers，策略训练/推理 |
| `dynvla-sim51` | Isaac Lab 0.54.2 + Isaac Sim 5.1 |
| `physmani` | 物理/操作实验（LeRobot） |
| `sparseworld` / `qwen38` / `bge` / `rdp` / `lingbot` / `light` / `hssd310` | 其他实验环境，用前先确认 |

新建环境统一 **Python 3.11**：

```bash
conda create -y -n fruitsort python=3.11
conda run -n fruitsort pip install ...
```

## 目录约定

```text
Fruitsort/
├── AGENTS.md            # 本文件
├── docs/                # 任务定义、路线图、基线表、环境报告
├── notes/
│   ├── papers/          # 一篇论文一个 md：<arxiv-id>-<slug>.md
│   ├── digests/         # arXiv 日报（scripts/arxiv_digest.py 生成）
│   └── ideas.md         # 随手记的想法/待验证假设
├── scripts/             # 可执行工具（见下）
├── configs/             # 训练/评测配置
├── src/fruitsort/       # 项目代码（可 import 的包）
├── runs/                # 实验产物：runs/<YYYY-MM-DD>-<简述>/
├── assets/              # 小体积模型/标定文件/URDF 等
└── data -> /media/.../data/fruitsort   # 大数据/权重，不进仓库
```

- **大数据与权重只放 `data/`**（软链到 9.1T 盘）：`datasets/ checkpoints/ raw/ rosbags/ eval/`
- **实验**必须用 `scripts/new_run.py` 创建，目录内至少有
  `meta.json`、`config.yaml`、`cmd.txt`、`metrics.json`、`notes.md`，并自动登记到 `runs/registry.csv`。
- **论文笔记**用 `notes/papers/_template.md`，开头带 citation block。

## 工具

| 命令 | 作用 |
| --- | --- |
| `source scripts/env.sh [env]` | 加载 ROS 2 Humble（非交互 shell 也生效），可选激活 conda 环境 |
| `bash scripts/doctor.sh` | 环境体检，结果写入 `docs/env-report.md` |
| `python scripts/new_run.py --name <名> [-c cfg.yaml]` | 新建一次实验目录 + 登记 |
| `python scripts/arxiv_digest.py` | 抓 cs.RO 新论文，生成 `notes/digests/<日期>.md`（去重） |
| `bash scripts/fetch_weights.sh <url> <file>` | 断点续传下载权重到 `data/checkpoints/` |

`Makefile` 里提供了短命令：`make doctor` / `make digest` / `make run NAME=xxx`。

## 可用能力（MCP / Skills）

- MCP：`arxiv`（检索/下载/LaTeX 精读/引文图/订阅）、`huggingface`（Hub 模型/数据集）、
  `context7`（库文档）、`deepwiki`（GitHub 仓库问答）、`llm-wiki`（本地知识库）
- Skills：`embodied-lit-review`、`robot-stack-map`、`robot-profiling`（本项目 `.opencode/skills/`，
  从 `dynamicVLA` 复制；那边更新后可再同步）

## 关键坑

1. **非交互 shell 不加载 `~/.bashrc`**：`ros2`、conda 都不在 PATH，先 `source scripts/env.sh`。
2. **arXiv API 常 429**（共享出口 IP）：`arxiv_digest.py` 已内置退避；MCP 失败时改用
   `webfetch https://arxiv.org/abs/<id>` 或 browser。
3. **Isaac Lab 3.0 四元数是 (x,y,z,w)**，LeRobot / DynamicVLA 是 (w,x,y,z)，跨边界必须转换。
4. 没有 `htop`/`nvtop`/`can-utils`：用 `nvitop`、`glances`、`python-can`（`rtools` 环境已装）。
5. HF token 在 `~/.config/opencode/opencode.jsonc`（勿写入仓库/对话）。下载大模型优先
   `hf download <repo> --local-dir data/checkpoints/<name>`。
6. `runs/` 里的视频/权重放 `data/`，仓库只留元数据与曲线。

## 工作流（建议）

1. 读 `docs/task-spec.md` 明确任务与指标 → `docs/roadmap.md` 选阶段。
2. 每次实验：`new_run.py` 建目录 → 训练/评测 → 写 `notes.md` 结论 → 更新 `metrics.json`。
3. 新读的论文收敛成笔记，并在 `docs/baselines.md` 登记"是否复现/是否可用"。
4. 每周用 `arxiv_digest.py` + `embodied-lit-review` 做一次文献扫描。
