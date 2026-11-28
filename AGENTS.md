# Fruitsort — 具身智能 / 机器人水果分拣研究

本文件是所有 agent 与协作者的入口：机器环境、可用能力、目录约定。**改动环境或约定后请同步更新本文件。**

## 研究目标（待细化）

机器人水果分拣：感知（检测/位姿/成熟度）→ 抓取规划 → 操作策略（模仿学习 / VLA）→ 真机部署与评测。

**当前策略：仿真优先，真机后接**（见 [`docs/decisions.md`](docs/decisions.md) D1）。
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
| `python scripts/demo.py` | 跑仿真 + 录制视频（`--kinds` 选水果） |
| `python scripts/benchmark.py` | 官方评测（per_class / pair / all4） |
| `python scripts/dexterous_sort5.py --robot <uid>` | 灵巧手任务级分拣（`panda_allegro` / `panda_dclaw` / `panda_wristcam` 同台对照） |
| `python scripts/calibrate_gripper.py` | 夹爪标定：指令 → 开口/接触力（`assets/calib/gripper_width.json`） |
| `python scripts/fit_hand_eye.py` | 腕部相机手眼标定 → `assets/calib/camera_hand.json`（真机 `cv2.calibrateHandEye` 同格式） |
| `python scripts/fit_camera_model.py` | 拟合相机 K/T → `assets/calib/camera_<name>.json`（真机可手眼标定同格式） |
| `python scripts/bench_perception.py` | 感知后端对比（召回/分类/3D 误差），支持倾斜机位 |
| `python scripts/train_segmenter.py` | 训练学习式分割（U-Net，RGB+高度→掩码），产出 `runs/seg-unet/seg.pt` |
| `python scripts/train_recognizer.py` | 训练识别头（掩码特征→类别），产出 `runs/seg-head/recognizer.pt` |
| `FRUITSORT_AUDIT_GT=1 python scripts/stream_eval.py ...` | 真值审计：打印模型侧是否读了真值 |
| `python scripts/camera_bench.py [--poses lower]` | 机位对比：px/m、覆盖率、抓取瞬间可见率（结论见 [`docs/camera-placement.md`](docs/camera-placement.md)） |
| `python scripts/stream_eval.py --trace-close \| tee log` + `python scripts/analyze_close.py log` | 闭爪落点诊断（拦截时序）：看每次闭爪离果心多远，结论见 [`docs/interception-timing.md`](docs/interception-timing.md) |
| `python scripts/stream_eval.py [--policy scripted]` | **连续作业**评测（`FruitSortStream-v1`：随机间隔/尺寸/多件在飞，按件统计吞吐）；加 `--video out.mp4` 录演示（画面叠加时间/带速/目标/相位/计数） |

`Makefile` 里提供了短命令：`make doctor` / `make digest` / `make run NAME=xxx`。

## 仿真与评测（v0.2）

```bash
source scripts/env.sh maniskill3
python scripts/benchmark.py --scenario pair --episodes 10 --belt-speed 0.06
```

- 环境：`FruitSort-v1`（`src/fruitsort/sim/tasks/fruit_sort.py`），控制模式 `pd_ee_pose`（20 Hz）。
- 资产：`assets/meshes/*.glb` 由 `python -m fruitsort.sim.assets.fruit_meshes` 生成。
- 结果：`runs/<id>/metrics.json`，协议见 `docs/benchmark.md`。
- 相机机位可覆盖：`FRUITSORT_HEAD_Z` / `FRUITSORT_HEAD_POS="x,y,z"` / `FRUITSORT_HEAD_LOOK="x,y"`；
  **换机位必须重标定**（单应与高度绑定，`FRUITSORT_CALIB` 指定输出名）；机位取舍见 `docs/camera-placement.md`。
- 视觉模式（实验性）：先 `python scripts/calibrate_camera.py`，再 `--perception vision`；
  当前闭环未跑通，默认仍用 oracle 位姿，见 `docs/perception.md`。
- 修改网格后必须 `build_all(force=True)`；否则沿用旧 GLB。

## 连续作业（v0.7，结论见 docs/model-architecture.md 6.26）

```bash
python scripts/stream_eval.py --ckpt runs/act-wide-4c/ckpt.pt --belt 0.08 --seconds 120 \
    --gap-min 0.6 --gap-max 1.2          # 策略在连续流里的吞吐/成功率
python scripts/stream_eval.py --policy scripted --seconds 120   # 规则专家（场景上限）
python scripts/stream_eval.py --policy scripted --record data/fruitsort/datasets/stream-x  # 录连续流示教
```

- 环境 `FruitSortStream-v1`：每类 3 个 slot、单一下料流、间隔 `U(gap)` 米、尺寸 ±15%，
  **只会循环回收**（不回收就永远不再上料）。
- **放料门**（v1.89）：模型会在离料框 15–27 cm 就开爪（果子掉地上）。
  加 `--release-gate 0.10` 后开爪全部落在框内，掉落 4/15 → 1/14，见 [`docs/drop-reduction.md`](docs/drop-reduction.md)。
- **下探段分解**（v1.112）：`stream_eval` 每局打印 `张开追件/闭爪等待/闭爪次数/抓空复位/入口高度差`。
  结论：闭爪确认只占 0.1–1.2 s，下探慢在**相位头/夹爪头抖动**（模型侧），runner 的门已到极限，
  三种修法（`--descend-min-step`/`--zg-hold-chunks`/`--grip-hyst`）实测均未采纳；
  chunk 级去抖（`--debounce-chunks`）16/20→11/20、预测式闭爪（`--close-timing`）16/20→13/20，
  均更差；竖直伺服 `--descend-servo` 16/21、几何进入 `--phase1-geom` 14/23 —— **runner 侧 8 种修法全部无收益**。
  瓶颈是模型每 chunk 重决策的固有节奏（一局进出 phase 1 7–46 次、闭爪 2–34 次），加门只会更乱。
  汇总表见 [`docs/throughput.md`](docs/throughput.md) §8.9。
  另：**同 seed 同配置的仿真是确定性的**（60016 连跑 3 次均 2/4），对照差异可归因配置；
  但 `runner.reset()` 会重建 `stats`，跨目标计数要放在 `ct_total` 这类属性里。
- **搬运保相位辅助**（v1.110，默认开）：`phase==2 且已夹住` 时只把目标**方向**转向料框，
  步长与相位切换仍归模型；稀疏口径单件 7.9→6.8 s、送达统计打平，
  密集口径 10 seeds 配对 58.0%→73.6%（单件 8.2→6.6 s，p=0.0097）。`--no-carry-assist` 关。
  详见 [`docs/throughput.md`](docs/throughput.md) §6。
- **DAgger 下降头**（v1.123 起**默认开启**，权重 `assets/descend_head.pt`）：在**策略自跑状态**上
  标注 descend/close 两分类（val_acc 0.986），`descend` 强制张开并把竖直压到果心 +4 mm。
  **当前主口径：稀疏 67/73 = 91.8%（CI 83–96），0 掉落，7.4 s/件**；密集 37/53 = 69.8%（CI 56–80），0 掉落。
  `--no-descend-head` 回到旧行为（88.9%）。`--descend-z-off` 试过 10 mm 更差，保持 4 mm。
  v1.124 已在新默认下正式重跑 30 个 seed（与 flag 逐位一致）。`--dump-states` 现在连触觉一起转储
  （容器已修成跨局累积）。v1.125 试过 58 维（+触觉）头：dense 11/22 vs 11/21、单件 8.11 vs 7.96 s → 不采纳。
  见 [`docs/throughput.md`](docs/throughput.md) §11。
- **学习式闭爪时机**（v1.121，默认关）：`scripts/train_close_head.py` 用专家闭爪时刻训练（离线 AUC 0.998），
  `--close-head runs/close-head/close.pt` 接入；**闭环 9/20 vs 默认 10/20 更差** ——
  原因是策略悬停的状态专家没去过（分布外），头在错的状态上给高置信度闭爪。见 [`docs/throughput.md`](docs/throughput.md) §10。
- **掉落分类**（v1.117）：`metrics.json` 每局有 `drop_grasped/drop_ungrasped/drop_off_belt`。
  已查明非夹持掉落 = **果子被侧向挤下带边**（y 从 −0.05 漂到 −0.31，臂在带边外继续追），
  不是生成/台账缺陷。**加高挡边试过并否决**：12 cm 挡边让侧掉归零，但送达 16/20→2/19，
  因为料框在挡边外侧、机械臂必须横跨挡边搬运（带面 0.45+0.12 > 搬运高度 0.55）。
  `FRUITSORT_BELT_RAIL` 可配（默认 0.06）。再加搬运高度下限（`--carry-z-floor 0.62`）也只回到
  38.9%，且侧掉仍在（果子是在**接近/下探段**被挤下去的）→ 该类别是场景几何的固有结果，
  只能以 `drop_ungrasped/drop_off_belt` 单独记账。接近段 y 限幅（`--approach-y-limit 0.16`）
  也只做到送达持平、带外掉落仍出现（手指位置由手臂+偏航决定，指令 y 管不到）。
  见 [`docs/drop-reduction.md`](docs/drop-reduction.md) §8–9。
- **腕部相机兜底**（v1.101，默认开）：头部丢检时用 `hand_camera` 补 xy（z 沿用历史），
  配对 5 seeds：apple 87% → 94%、掉落 1 → 0；`--no-hand-fallback` 可关。
- **深度修复**（v1.93）：`PerceptionPipeline` 默认开（邻域最小值填洞 + 掩码补洞），
  深度丢 40% 时召回仍 100%（无修复会掉到 2.5%）。
- **学习式分割**（v1.93）：`--seg-backend learned --seg-thresh 0.75`（U-Net，需 GPU，
  bench 100% 召回/分类、3D 0.21 cm；过曝 1.5× 时召回 75%）。
- **感知 v2**（v1.91）：`--perception rgbdv2` 走"分割掩码→3D 反投影→识别→跟踪"，
  不用单应/不用 `cam_z-depth`，**倾斜机位可用**；模型侧真值读取经审计为 0。
  详见 [`docs/perception-v2.md`](docs/perception-v2.md)。
- **闭爪时机**（v1.88）：模型会在果心上方 3 cm 就闭爪（水平误差只有 1–2 mm），
  加 `--close-z-gate-map apple=0.012 --close-z-off-map apple=0.0` 后 85% 落在果心 ±1.5 cm；
  详见 [`docs/interception-timing.md`](docs/interception-timing.md)。
- 现状：`act-wide-4c` 连续作业 14–17%（单发是 87.5%），规则专家 69–86%；
  模型是单发策略，换队列表现在"接近"相位会冻结（见 6.26.3）。

## 灵巧手（v0.6 实验性，结论见 docs/model-architecture.md 6.25）

```bash
python scripts/make_panda_allegro.py --pitch 0     # 生成 assets/urdf/panda_allegro.urdf
# 必须复制到 ManiSkill 资产目录，否则会触发"下载资产"交互提示
cp assets/urdf/panda_allegro.urdf $CONDA_PREFIX/../envs/maniskill3/lib/python3.10/site-packages/mani_skill/assets/robots/panda/
python scripts/dexterous_sort5.py --robot panda_allegro --kinds apple --episodes 8 --belt 0.08
python scripts/dexterous_sort5.py --robot panda_wristcam  # 同脚本/同种子/同场景的平行夹爪对照
```

- 准入标准：**保持位姿 100 步无 NaN**（`panda_allegro` ✓ / `panda_dclaw` ✓ / `xarm7_ability` ✗）。
- 合并 URDF 必须加虚拟 `panda_tcp`（与 `panda_hand_tcp` 同位），否则 `pd_ee_pose` 映射失配。
- 去掉腕/前臂 link 的 `collision`（长手指会与 Panda 前臂自碰撞，实测 1.7 kN）。
- `pd_ee_pose` 的位姿是**基座系**：抬高 `robot_base` 后动作 z 要减 `base_z`。

## 模仿学习（v0.4）

```bash
python scripts/record_demos.py --kinds apple strawberry --episodes 40   # 示教（只存成功）
python scripts/train_bc.py --data data/datasets/<name> --out runs/bc --epochs 80
python scripts/eval_bc.py --ckpt runs/bc/ckpt.pt --kinds apple strawberry --episodes 10
```

- 观测 49 维、回归目标 4 维（世界目标位姿 + 夹爪），详见 `docs/learning.md`。
- 数据放 `data/datasets/`（不入库）；`runs/**/*.pt` 被 gitignore，checkpoint 只留本地。
- 当前 BC 闭环 10%，专家 100%；瓶颈是分布漂移，优先做 DAgger / ACT。

## 多视角策略（v0.5）

```bash
python scripts/record_demos.py --kinds apple strawberry --episodes 12 \
    --images --robot panda_wristcam --instruction apple
python scripts/train_vision_policy.py --data <dirs...> --out runs/vpolicy --epochs 40
python scripts/bench_latency.py --ckpt runs/vpolicy/ckpt.pt
python scripts/eval_hybrid.py --ckpt runs/vpolicy/ckpt.pt --episodes 10
```

- 模型 0.41M：双视角 96×96 + 4 帧历史 + 2 层 Transformer + 选择头
- GPU 1.9 ms / CPU 单线程 6.1 ms；指令选择 100%；分层执行 10/10
- 纯视觉闭环：苹果/草莓 5/5（`scripts/eval_vision_geo.py`）；夹爪时序见 `docs/gripper-timing.md`

## 演示视频约定（强制）

**每个达到验收标准或有突破的版本，必须录一段演示视频 + 同名 `.txt` 说明**，否则别人无法判断视频背景。

```bash
python scripts/record_video.py --executor act --ckpt runs/<run>/ckpt.pt \
  --classes apple strawberry banana pear --belt 0.15 --spawn-spacing 0.45 \
  --control-freq 30 --fb 0.5 --grip-ema 0.6 --carry-latch --carry-max-step 0.02 \
  --episodes 4 --retries 3 --out data/fruitsort/eval/<name>.mp4
```

- 脚本自动生成 `<name>.txt`：代码版本、权重与离线指标、部署配置、场景、分集结果
  （整局时长 + 单件处理时间）、复现命令
- 视频画面叠加 kind / 带速 / 已用时间 / 策略相位（0 接近，1 下探闭爪，2 搬运）
- 失败局自动重试，只写成功片段
- 录完在 [`docs/videos.md`](docs/videos.md) 登记一行

## 版本与发布约定

- 语义化版本：`vMAJOR.MINOR.PATCH`，tag 与 `CHANGELOG.md` 同步。
- 每个版本写 GitHub Release，内容包含：做了什么 / 怎么验证 / 结果数字 / 已知问题 / 下一步。
- 破坏性改动（场景布局、动作空间、成功判据）必须升 MINOR 并在 `docs/decisions.md` 记录。

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
