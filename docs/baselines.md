# 基线 / 开源资源清单

> 原则：**能用现成的就不自己写**。每条都记录"在本机哪个环境跑过、卡在哪"。
> 状态用 `⬜ 未试 / 🟡 跑通 / 🟢 可用 / 🔴 放弃`。

## 感知

| 资源 | 用途 | 仓库 | 本机状态 |
| --- | --- | --- | --- |
| Ultralytics YOLO | 水果检测/分类 | `ultralytics/ultralytics` | ⬜ |
| RT-DETR | 检测基线 | `lyuwenyu/RT-DETR` | ⬜ |
| SAM 2 | 分割/跟踪 | `facebookresearch/sam2` | ⬜ |
| Grounding DINO | 开放词表检测 | `IDEA-Research/GroundingDINO` | ⬜ |
| FoundationPose | 6D 位姿估计 | `NVlabs/FoundationPose` | ⬜ |
| AnyGrasp | 抓取位姿 | `graspnet/graspnetAPI`（SDK） | ⬜ |
| Contact-GraspNet | 点云抓取 | `NVlabs/contact_graspnet` | ⬜ |
| anomalib | 缺陷/异常检测 | `openvinotoolkit/anomalib` | 已有独立项目 `../anomalib` |

## 策略学习（模仿学习）

| 资源 | 说明 | 仓库 | 本机环境 |
| --- | --- | --- | --- |
| ACT | 低成本双臂模仿学习 | `tonyzhaozh/act` | `dynvla` |
| Diffusion Policy | 扩散策略基线 | `real-stanford/diffusion_policy` | `dynvla` |
| 3D Diffusion Policy (DP3) | 点云输入 DP | `YanjieZe/3D-Diffusion-Policy` | ⬜ |
| LeRobot | 数据格式 + 训练/部署全家桶 | `huggingface/lerobot` | `dynvla` (0.3.3) |
| RoboMimic | 模仿学习 benchmark | `ARISE-Initiative/robomimic` | ⬜ |

## VLA / 基础模型

| 资源 | 说明 | 仓库 | 备注 |
| --- | --- | --- | --- |
| OpenVLA | 开源 VLA，7B | `openvla/openvla` | 需 ~16GB 显存推理 |
| π0 (openpi) | Physical Intelligence 开源 | `Physical-Intelligence/openpi` | JAX 栈 |
| RDT-1B | 双臂扩散 VLA | `thu-ml/RDT` | 1B |
| Octo | 通用操作策略 | `octo-models/octo` | 多 embodiment |
| DynamicVLA | 你自己的项目 | `../dynamicVLA` | 优先复用 |

## 仿真

| 资源 | 说明 | 本机 |
| --- | --- | --- |
| ManiSkill3 | 操作任务、GPU 并行、域随机 | `maniskill3` env，首选 |
| Isaac Lab 3.0 | 高保真 + 大规模 RL | `dynvla-sim51` env |
| Genesis | 高速刚体/软体仿真 | ⬜ |
| MuJoCo / robosuite | 经典操作 benchmark | ⬜ |

> Isaac Lab 3.0 四元数是 `(x, y, z, w)`；LeRobot/ManiSkill 多为 `(w, x, y, z)`，跨边界必须转换。

## 数据集

| 数据集 | 内容 | 获取 |
| --- | --- | --- |
| LeRobot datasets | 大量操作数据，统一格式 | `hf://datasets/lerobot/*` |
| Open X-Embodiment | 多机器人混合 | `huggingface/Open-X-Embodiment` |
| DROID | 76k 真实操作轨迹 | `droid-dataset` |
| BridgeData V2 | 宽泛操作 | `rail-berkeley/bridge_data_v2` |
| AgiBot World | 百万级真实双臂 | `AgiBot-World` |

## 工具链

| 工具 | 用途 | 环境 |
| --- | --- | --- |
| `rosbags` (`rosbags-convert`) | rosbag2 ↔ 其他格式 | `rtools` |
| `rerun-sdk` | 训练/rollout 可视化 | `rtools` |
| `evo` | 轨迹精度评估 | `rtools` |
| `nvitop` / `glances` | GPU / 系统监控 | `rtools` |
| `py-spy` / `scalene` | 性能剖析 | `rtools` |
| `python-can` / `cantools` | CAN 总线调试（夹爪/电机） | `rtools` |
| `wandb` / `tensorboard` | 实验记录 | 按环境 |

## 复现记录

读到值得复现的论文，按下面格式追加：

```markdown
### <论文标题> (<arxiv-id>)
- 日期 / 复现人：
- 代码：<repo>
- 环境：<conda env>
- 结论：🟢 可用 / 🔴 放弃，原因：
- 坑：
```
