# 动态传送带抓取 / 分拣：文献调研与工程参考

> 目标场景：**单臂 + 直线传送带 + 4 类水果（苹果/草莓/香蕉/梨）+ 目标框分拣 + ManiSkill3 仿真**
> 调研时间：2026-11 · 覆盖 **25 篇**（19 篇有详细小节 + `notes/papers/` 中另 6 篇动态操作 VLA 的 PDF：DynamicWAM / FASTER / NEO-VLA / Reflex / StreamPI / TrajWAM）
> 所有链接均已用 `curl -I` / `webfetch` 验证（2026-11-17）：**51 个链接中 50 个返回 200**；唯一例外是 SAGE 的 IJRR DOI 页（`10.1177/02783649211027194`，返 403 是反爬，DOI 本身有效，正文已同时给出 arXiv 链接）。

---

## 0. 结论先行：本项目应采用的 5 条设计原则

| # | 原则 | 一句话 | 依据 |
| --- | --- | --- | --- |
| **P1** | **预测接触时刻的位姿，而不是追当前位姿** | 决策变量必须是"t+T 时刻物体在不在我可达工作空间内、以什么姿态接触"，而不是"它现在在哪" | PAF-Grasp（§1.1.2，传送带 78% vs 反应式 34%）、KARL（§1.1.1）、AHEAD（§1.1.3） |
| **P2** | **把端到端延迟当成一等设计变量，并写进评测** | 实测"拍照→推理→下发→到位"的总延迟 Δt，所有动作都相对 t+Δt 时刻的场景求解；延迟预算按 150–200 ms 设计 | AHEAD 明确给出 ~200 ms 延迟预算（§1.1.3）、LIBERO-Dynamic 用"按模型校准的推理延迟"驱动世界（§1.4.6）、DiscreteRTC 证明 chunk 间停顿是动态任务的致命伤（§1.4.5） |
| **P3** | **"抓哪一颗"用可达性/TTC 排序，不要用最近优先** | 每帧对视野内所有水果算 TTC（剩余可拦截时间）与可达性，选得分最高者；贪心（先到先抓）在传送带上确定性地损失 10–40% 吞吐 | Han/Feng/Yu 的传送带 PnP 最优调度（§1.2.1）、Islam 的 constant-time replanning（§1.2.2）、PAF-Grasp 的 reachability-aware selector（§1.1.2） |
| **P4** | **卡尔曼/EKF 层是"跟带"的必需品，不是可选项** | 用恒定加速度模型外推 6D 位姿，允许物体短暂出视野/被遮挡/检测抖动，跟踪器丢帧时用预测值继续执行而不是停 | KARL 把 EKF 夹在感知与 RL 之间，出视野仍能连续估计（§1.1.1）；Islam 证明"先用早期噪声估计起步、再边跑边改"优于"等好估计"（§1.2.2）；6-DoF 事件跟踪（§1.3.2） |
| **P5** | **策略分层：慢规划 + 快残差，别指望单网络吃掉 300 ms** | 5–10 Hz 的慢策略出 action chunk（含未来偏移补偿），50–100 Hz 的轻量残差/阻抗层做闭环修正；两层分别训练、分别评测 | DSDyn-VLA 的 Flow-Planner + Res-Refiner（§1.4.4）、Residual Policy Learning（§1.4.1）、DiscreteRTC 的异步 chunk 执行（§1.4.5） |

### 0.1 由上述原则推导的量化目标（建议写进 `docs/task-spec.md`）

| 项 | 目标值 | 来源/理由 |
| --- | --- | --- |
| 端到端延迟 Δt | ≤ 150 ms（感知 30 + 策略 50 + 通信/执行 70） | AHEAD 的 200 ms 预算；单件节拍 8 s，但**有效拦截窗口只有 0.3–0.6 s**（臂展后半段） |
| 慢策略频率 | 8–10 Hz（action chunk 长度 ≥ 8 步） | PAF-Grasp 10 Hz 重规划；AHEAD chunk 化 |
| 快残差频率 | ≥ 50 Hz（关节速度/阻抗层） | DSDyn-VLA 的 Res-Refiner 快流；PAF-Grasp 底层 100 Hz |
| 传送带速度（仿真随机化） | 0.05–0.40 m/s，标称 0.15–0.20 m/s | PAF-Grasp 训练采样 0.04–0.40、实机 0.08–0.20；Islam 实机 0.2 |
| 单件可用拦截窗口 | ≥ 1.2 s（= 相机看到 → 到达臂展边界） | 需要 ≥ 1.2 s × 0.2 m/s ≈ 24 cm 的"预警距离"，决定相机安装位置 |
| 抓取点误差容忍 | 位置 ≤ 15 mm、绕竖直轴姿态 ≤ 15° | 水果近似球形，PAF-Grasp/KARL 的实机成功率在 cm 级误差下仍 >75% |

> 一句话总结：**先把"延迟 + 预测 + 调度"这三件非学习的事做对（P1–P4），再谈 VLA（P5）；否则模型再好也追不上带子。**

---

## 1. 分方向文献

### 1.1 移动物体抓取 / 动态抓取 / 拦截（dynamic grasping & interception）

#### 1.1.1 KARL: Kalman-Filter Assisted Reinforcement Learner for Dynamic Object Tracking and Grasping

- **arXiv**：[2506.15945](https://arxiv.org/abs/2506.15945) · 2025 · **IROS 2025**
- **代码**：[github.com/arc-l/karl](https://github.com/arc-l/karl) ✅（官方，Docker 镜像 `kowndidocker/karl:v0`）
- **视频**：[5 分钟 demo（Rutgers Box）](https://rutgers.box.com/s/9yntcg1uw7b0db7jw6nszbu80w3d3jb9) ✅
- **项目页**：[karl-iros25.github.io](https://karl-iros25.github.io/) ✅
- **一句话**：在 eye-on-hand 系统里把 **EKF 层夹在 6D 位姿感知与 RL 控制之间**，物体短暂出视野或急加速时仍保持连续位姿估计，再用 6 阶段课程把机械臂运动范围翻倍。
- **可用点**（对本项目最重要的一篇工程参考）：
  - 硬件配方可直接照抄：**UR5e + Intel RealSense L515 + Contact-GraspNet + R2D2 特征**，腕部相机，ROS1/MoveIt 标定流程写在 README 里；
  - 它把任务拆成"跟踪（保持物体在视野内）→ 接近 → 抓取 → 失败重试"，正是单臂分拣需要的状态机；
  - 失败重试机制（物体被碰飞后可恢复）对易滚的水果（苹果、梨）很关键；
  - 报告 **92.58%** 成功率、比前作 EARL 快 **20%+**。
- **局限**：
  - 物体速度只覆盖 **0–5 cm/s（常规）/ ≤15 cm/s（快档）**，超过 30 cm/s 被视为"破坏性扰动"——**比工业传送带（0.2 m/s = 20 cm/s）还慢**，不能直接照搬其速度假设；
  - 仓库**未声明 LICENSE**（默认版权保留），商用需联系作者；README 的安装流程强依赖 `sudo docker`，本机**无 sudo**，只能手动拆依赖；
  - 依赖 Isaac Gym（已停更），迁到 ManiSkill3 需重写环境。

#### 1.1.2 PAF-Grasp: Physics-Aware Object-Centric SE(3) Forecasting for Real-Time Proactive Dynamic Grasping

- **arXiv**：[2610.17843](https://arxiv.org/abs/2610.17843) · 2026-10
- **项目页（含 demo 视频）**：[am0rarg0.github.io/paf-grasp](https://am0rarg0.github.io/paf-grasp/) ✅
- **代码**：**尚未开源**（项目页标注 "Code — Pending release"）
- **一句话**：不预测整幅未来画面，只预测**接触时刻的刚体 SE(3) 状态**（15 ms 推理），再用"可达性感知的拦截选择器"挑一个运动学与时间都可行的接触点。
- **关键设计**（最值得抄的思想）：
  - 预测器 = 短位姿历史 + **Newton–Euler 结构化的 neural ODE**，只用 Genesis 里随机基元物体轨迹训练，真实物体换 mesh 后**不需要按物体重训**；
  - 选择器 = 把接触帧沿预测轨迹前推，用工作空间/时间约束筛掉不可达解；
  - **10 Hz 在线重规划直到接触**；控制 100 Hz。
- **可用点**：
  - 训练数据完全来自仿真（Genesis），**与"仿真优先"路线一致**；本项目可用 ManiSkill3 代替 Genesis 造同样的数据；
  - 传送带速度采样 **0.04–0.40 m/s**，实机 0.08–0.20 m/s，正好覆盖水果分拣线；
  - 实机只用 **AgileX Piper（轻量低成本 6-DoF 桌面臂）+ D435i**，说明"不必上大臂"也能做动态拦截；
  - 结果：仿真 24.5%→62.0%，实机 15.3%→67.3%；传送带任务 **78% vs π0.5+RTC 34%**。
- **局限**：按"运动族"（抛物/匀速/转台/斜面）各训一个 checkpoint，**跨族泛化未验证**；物体是基元形状，水果的滚动/变形未建模；代码未放出。

#### 1.1.3 AHEAD: Intercepting the Future — Latent-Space Predictive World Model for Dynamic VLA Manipulation

- **arXiv**：[2606.02486](https://arxiv.org/abs/2606.02486) · 2026 · **CoRL 2026**
- **项目页**：[aheadwm.github.io](https://aheadwm.github.io/) ✅ ｜ **代码**：无公开仓库（截至 2026-11）
- **一句话**：给**冻结的 7B OpenVLA** 加一个 4.9M 参数的"潜空间世界模型"wrapper，用光流算出的逐 token 速度/加速度把特征外推到未来时刻，让 VLA "看到"执行时刻的场景。
- **可用点**：
  - 明确指出失败根因是**延迟 > 可抓时间**，并给出 200 ms 量级的延迟预算——这是本项目做延迟测量与优化的直接依据；
  - 实测 **158–171 ms/步**（RTX 4090）；传送带速度做了 **0→25 cm/s** 扫描，全程 >95% 成功率——本项目可把 25 cm/s 当作"这类方法已验证的上限"，超过就需要事件相机或更快的感知；
  - 在 **xArm 7 + 固定第三人称 224×224 RGB** 上做了 5 个动态任务，其中"传送带"类 **29/30–30/30**；
  - 只用 ≈200 条域内轨迹微调世界模型即可跨域，**数据成本极低**，适合本项目自采少量真机数据；
  - 常加速度外推（含零加速度退化）而非纯匀速，能覆盖水果离带后的抛物/滚动。
- **局限**：不改变 VLA 本身，延迟仍然存在（只是被"预演"补偿）；依赖 OpenVLA 7B，5090 上要跑 10 Hz 需量化；代码未开源。

#### 1.1.4 Self-Supervised Learning of Grasping Arbitrary Objects On-the-Move

- **arXiv**：[2411.09904](https://arxiv.org/abs/2411.09904) · 2024
- **代码**：无公开仓库（论文未给出）｜**视频**：无独立项目页
- **一句话**：把"移动抓取"压成 3 个动作基元（静态抓/动态抓/移动），用 3 个 FCN 分别预测抓取基元与**剩余速度误差**，两阶段自监督学习。
- **可用点**：**"残差速度"这个输出头**非常适合传送带——机械臂只需补偿"带速 - 当前末端速度"的残差，而不是学绝对轨迹；仿真里随机化物体形状/环境即可泛化（作者已验证）。
- **局限**：面向移动底盘 + 臂的"边走边抓"，不是固定基座 + 传送带；无代码；物体速度低。

#### 1.1.5 Dual Quaternion-Based Visual Servoing for Grasping Moving Objects

- **arXiv**：[2107.08149](https://arxiv.org/abs/2107.08149) · **IEEE CASE 2021**
- **代码**：无公开仓库
- **一句话**：用**对偶四元数**写 PBVS 误差，把"抓取位姿随物体移动而更新"变成伺服问题，并用零空间避关节限位；配 LoCoMo 抓取排序，物体一动就**重排抓取**。
- **可用点**：这是**最轻量的 baseline**——不需要学习、不需要预测，只要 30–60 Hz 的位姿输入就能"贴着物体飞"。本项目应把它作为 P1 之前的对照实验（脚本化伺服），用来验证标定、延迟、控制链路是否正常。
- **局限**：纯仿真验证（7 轴臂 + 二指夹爪），无真机；依赖视觉标记跟踪；没有前瞻预测，带速一高就跟不上。

---

### 1.2 传送带分拣 / bin picking / 工业分拣

#### 1.2.1 Toward Fast and Optimal Robotic Pick-and-Place on a Moving Conveyor

- **arXiv**：[1912.08009](https://arxiv.org/abs/1912.08009) · 2019
- **代码**：无官方仓库（算法为 DP + 启发式，可自行实现）
- **一句话**：证明"先到先抓"的贪心策略在传送带上**不是时间最优**，用有限时域动态规划求最优抓取序列，在 Delta / SCARA 模型上比贪心提升 **10–40% 吞吐**。
- **可用点**：
  - 直接对应本项目的"**目标框分拣**"：不是所有水果都要抓，只有属于当前目标类别/目标框的才抓——这正是"选择性 PnP 序列优化"；
  - 给出了可实现的代价模型：抓一件的耗时 = 空走 + 下压 + 闭合 + 抬升 + 放置，且与物体在带上的**未来位置**耦合；
  - 规划器只需"物体到达时间 + 各类别目的地"，非常适合作为 VLA 之上的调度器。
- **局限**：抽象成"望远镜式"（telescoping）机器人模型，忽略动力学与避障；假设抓取必成功；未考虑异形/易损物体。

#### 1.2.2 Provably Constant-time Planning and Replanning for Real-time Grasping Objects off a Conveyor Belt

- **arXiv**：[2003.08517](https://arxiv.org/abs/2003.08517) · **RSS 2020**；期刊版 [IJRR 2021](https://doi.org/10.1177/02783649211027194)
- **代码**：无官方仓库（实现基于开源 [SMPL](https://github.com/aurone/smpl) 规划库 ✅）
- **一句话**：提出 **constant-time motion planning (CTMP)** 类算法，保证"每次重规划在固定时间上界内完成"，于是机器人可以**带着早期噪声估计先起步、边跑边按新位姿改轨迹**。
- **核心实验数据**（PR2 7-DoF，带速 0.2 m/s，50 次/组）：

  | 策略 | 成功率 | 感知准确时的成功率 |
  | --- | --- | --- |
  | CTMP（持续重规划） | **69.2%** | 83.3% |
  | 只用第一帧估计 | 16.0% | 66.7% |
  | 等"最好的一帧"再规划 | 34.6% | 55.6% |

- **可用点**：
  - 这是"**必须早动手**"最硬的量化证据：等感知变好反而更差，因为时间窗口没了；
  - 0.2 m/s 的带速与本项目目标一致；
  - 论文明确指出：**带速低到一定程度就不需要重规划**——可以用它来定"什么速度以上必须上闭环"的阈值。
- **局限**：PR2 + 点云配准，感知链路老旧；没有学习成分；只抓单一物体（sugar box）。

#### 1.2.3 Picking a Conveyor Clean by an Autonomously Learning Robot

- **arXiv**：[1511.07608](https://arxiv.org/abs/1511.07608) · 2015（ZenRobotics 工业废料分拣原型）
- **代码**：无公开仓库
- **一句话**：真实工业传送带分拣系统：自标定 + 用夹爪里的反馈传感器做**在线学习**，从随机采样的几何候选里挑最好的抓取，最终把 80 件难处理物体中的 **70 件**投进正确料槽。
- **可用点**：
  - **"自标定"**流程值得抄：相机-传送带-料框的外参在启动时自动恢复，而不是靠人工标定文件；
  - 用**夹爪内的成功/失败信号**做在线学习，是低成本提升成功率的手段（本项目可用真空/力传感器或腕部相机复现）；
  - "把传送带抓干净（pick a conveyor clean）"这个指标 = 本项目的**清空率 / 残留数**，比单件成功率更贴近产线。
- **局限**：2015 年的方法，感知靠简单几何模型；无代码；对象是垃圾，不涉及易损与分类精度。

#### 1.2.4 A Two-Stage Detection–Tracking Framework for Stable Apple Quality Inspection in Dense Conveyor-Belt Environments

- **arXiv**：[2602.19278](https://arxiv.org/abs/2602.19278) · 2026-02
- **代码/数据**：[github.com/luischuquim/Healthy-Defective-Fruits](https://github.com/luischuquim/Healthy-Defective-Fruits) ✅（缺陷分类数据集，论文中引用）
- **一句话**：**YOLOv8 检测 + ByteTrack 保持身份 + ResNet18 缺陷分类 + track-level 多数投票**，把逐帧抖动变成轨迹级稳定判断。
- **可用点**：
  - 这是与本项目**最贴近的水果 + 传送带 + 多目标**组合，可直接作为感知基线；
  - **track-level 聚合**思想可直接迁移到"目标框分拣"：一颗水果的类别/成熟度应在整条轨迹上投票，而不是单帧决定，避免"抓错框"；
  - 提出视频级工业指标（track-level defect ratio、temporal consistency），可写进本项目的 `metrics.json`。
- **局限**：**初步框架，未给出确定性的量化结果**（文中多为"we expect/hypothesize"）；只做检测与分级，**不做抓取**；无速度/延迟指标。

---

### 1.3 高速 / 动态抓取的感知（事件相机、跟踪、速度估计）

#### 1.3.1 EV-Catcher: High-Speed Object Catching Using Low-latency Event-based Neural Networks

- **arXiv**：[2304.07200](https://arxiv.org/abs/2304.07200) · **IEEE RA-L 2022**（7(4)）
- **代码**：无公开仓库
- **一句话**：用 **Binary Event History Image (BEHI)** 这种极轻量事件表征 + 小网络，在 **Jetson NX** 上实时预测高速球的**撞击点**，13 m/s 的乒乓球抓取成功率 81%。
- **可用点**：证明了"**事件相机 + 极轻量表征**"可以在嵌入式设备上做到毫秒级；如果本项目未来要提升带速（>0.5 m/s）或处理香蕉这类细长物体的运动模糊，这是首选硬件路线。
- **局限**：抓的是"抛物线的球"，不是传送带上滚动的水果；需要标定事件相机与臂的外参；无代码、无仿真。

#### 1.3.2 6-DoF Object Tracking with Event-based Optical Flow and Frames

- **arXiv**：[2508.14776](https://arxiv.org/abs/2508.14776) · 2025
- **代码**：无公开仓库（作者为 IIT 团队，代码未随论文发布）
- **一句话**：**低频 RGB 做全局 6D 位姿估计 + 高频事件光流做 6D 速度跟踪**，两者融合，解决高速运动下的运动模糊与帧率不足。
- **可用点**：
  - 这种"**慢而准的绝对位姿 + 快而相对的速度**"异步融合结构，正好是 P4 的工程实现模板；即使不用事件相机，也可以用"低频检测 + 高频光流/UKF"复刻；
  - 给出可用的融合公式（位姿积分 + 速度更新），可直接写进跟踪模块。
- **局限**：事件相机成本与标定门槛；未涉及抓取闭环；无代码。

#### 1.3.3 Visually Guided Model Predictive Robot Control via 6D Object Pose Localization and Tracking

- **arXiv**：[2311.05344](https://arxiv.org/abs/2311.05344) · 2023
- **代码**：无公开仓库（依赖开源 [cosypose](https://github.com/ylabbe/cosypose) 系列组件）
- **一句话**：**学习式 6D 位姿定位器（低频）+ 基于模型的高频 6D 跟踪器**异步组合，输出 **120 Hz** 的位姿流，喂给**力矩级 MPC** 实现"保持相对位姿/动态抓取/人机共操"。
- **可用点**：
  - 120 Hz 位姿 + 力矩 MPC 是"高质量跟带"的参考架构；本项目可先用位置级 MPC/阻抗替代；
  - 明确给出异步（不同频率传感器）融合的工程细节，避免"等最新图像"造成的额外延迟。
- **局限**：需要物体 CAD/mesh；力矩控制对硬件有要求（Panda/UR5e 可，廉价臂不行）；无代码。

#### 1.3.4 Tracking Multiple Moving Objects Using Unscented Kalman Filtering Techniques

- **arXiv**：[1802.01235](https://arxiv.org/abs/1802.01235) · 2018
- **代码**：无
- **一句话**：经典 UKF 多目标跟踪参考实现思路：非线性运动 + 遮挡下仍能维持轨迹，并显式估计速度。
- **可用点**：**当基线**。传送带上的水果运动近似"匀速直线 + 滚动扰动"，KF 足够；只有在水果离带抛物/碰撞时才需要 UKF/EKF。建议本项目：**KF 为主，EKF 处理旋转，UKF 只在离带段启用**。
- **局限**：视频监控场景，非机器人；无代码；不含 6D。

---

### 1.4 学习式动态抓取（RL / 模仿 / 残差策略 / sim2real）

#### 1.4.1 Residual Policy Learning

- **arXiv**：[1812.06298](https://arxiv.org/abs/1812.06298) · 2018
- **代码/视频**：[k-r-allen.github.io/residual-policy-learning](https://k-r-allen.github.io/residual-policy-learning/) ✅（含代码链接）
- **一句话**：在"还不错但不够好"的控制器上学习**残差动作**，比从零 RL 样本效率高得多，能解长时域稀疏奖励任务。
- **可用点**：
  - 本项目的理想形态：**脚本化"追带 + 预抓取"控制器**（P1/P4 实现）+ **RL 残差**修正抓取时机与姿态；
  - 6 个 MuJoCo 任务证明对"模型失配、传感器噪声、控制器标定误差"都有效——正好对应 sim2real 的三大坑。
- **局限**：原始工作是低维状态输入（非视觉）；高维视觉残差需要额外设计（见 DSDyn-VLA）。

#### 1.4.2 PhysMani: Physics-principled 3D World Model for Dynamic Object Manipulation

- **arXiv**：[2607.01938](https://arxiv.org/abs/2607.01938) · 2026 · **ECCV 2026**
- **代码**：[github.com/vLAR-group/PhysMani](https://github.com/vLAR-group/PhysMani) ✅（**CC BY-NC-SA 4.0，非商用**）
- **一句话**：用 **3D Gaussian Splatting + 无散度速度场**（在线优化）学"物理上说得通"的未来动力学，再把预测的 3D 未来通过 cross-attention 注入策略；同时发布 **PhysMani-Bench（16 个动态任务）**。
- **可用点**：
  - 若要做"水果离带后的滚动/碰撞预测"，3D 高斯 + 速度场比视频世界模型更省算力且几何一致；
  - **PhysMani-Bench 的 16 个任务**可作为本项目自定义任务的参考分类（抛/滚/推/接/传送）；
  - 代码 + 数据齐全，是学习式路线里最容易复现的一个（但非商用许可）。
- **局限**：需要多视角/深度与在线优化，训练成本高；水果的软性/摩擦各向异性未覆盖；非商用许可。

#### 1.4.3 TEMPO: Learning Temporal Context for Dynamic Robot Manipulation

- **arXiv**：[2609.16864](https://arxiv.org/abs/2609.16864) · 2026-09 · **CoRL 2026**
- **项目页**：[tempo-robot.github.io](https://tempo-robot.github.io/) ✅
- **代码**：[github.com/tempo-robot/TEMPO](https://github.com/tempo-robot/TEMPO) ✅ ｜ **评测集**：[github.com/tempo-robot/TempoBench](https://github.com/tempo-robot/TempoBench) ✅
- **一句话**：给预训练 VLA 加两个"时间输入"——**冻结视频基础模型抽出的运动摘要**（解决运动歧义 motion ambiguity）+ **紧凑本体历史**（解决状态别名 state aliasing），不改 backbone，算力开销极小。
- **可用点**：
  - 论文的核心论断值得写进设计文档：**运动歧义与状态别名这两类失败"与模型规模、推理延迟无关"**——加参数或加速推理都救不了单帧输入，必须加时序信息；
  - "运动摘要"用冻结的视频基础模型提取，**不需要重训**，是 AHEAD 之外更轻的时序方案；
  - 开源的 **TEMPO-Bench（>50k 标注帧，回归 + 多选两种格式）** 可以直接拿来评测本项目的运动感知/速度估计模块；
  - Bottle Handover 44% → 74%，是"加时序"性价比的直观证据。
- **局限**：四个任务以人机交接为主，不是传送带；未涉及离带后的滚动/抛物；运动摘要是 2D 的，缺 3D 位姿精度。

#### 1.4.4 DSDyn-VLA: Dual-Stream Dynamic Manipulation Framework

- **arXiv**：[2609.39198](https://arxiv.org/abs/2609.39198) · 2026-09
- **代码**：论文声明"将开源全部代码与权重"，**当前未放出**
- **一句话**：**慢速 Flow-Planner**（VLA + 光流时序 + 未来状态补偿延迟）出动作块，**快速 Res-Refiner**（轻量 RL 策略）在 50 Hz 级注入闭环残差——正是 P5 的教科书实现，且明确以**传送带**为场景。
- **可用点**：
  - 把"感知 gap / 延迟 gap / 控制 gap"三个失败模式显式拆开，可作为本项目**失败分类表**的骨架；
  - 慢快双流的接口设计（chunk + residual）可直接借用，不必等其代码；
  - 自带 **DynBench（MuJoCo）**，可作为 ManiSkill3 之外的交叉验证环境。
- **局限**：未开源；MuJoCo 场景保真度低于 ManiSkill3；复现成本高（需 VLA 底座）。

#### 1.4.5 DiscreteRTC: Discrete Diffusion Policies are Natural Asynchronous Executors

- **arXiv**：[2604.25050](https://arxiv.org/abs/2604.25050) · 2026
- **代码**：无公开仓库
- **一句话**：离散扩散策略的"逐步解掩码"天然支持 **inpainting 式异步执行**（冻结已提交动作、续生成剩余），无需额外微调即可消除 chunk 间停顿；真实冰球防守任务成功率比 flow-matching RTC 高 65%。
- **可用点**：
  - 如果本项目采用 Diffusion Policy / flow 策略，**chunk 边界处的停顿就是动态任务的隐形杀手**；本文给出"少改代码"的解法（离散扩散）；
  - 推理成本约为从头生成的 0.7×，对 5090 上跑 10 Hz 很友好。
- **局限**：需要把策略改成离散扩散（token 化）；只在少数任务上验证；无代码。

#### 1.4.6 LIBERO-Dynamic: Benchmarking Robotic Manipulation in Dynamic Environments

- **arXiv**：[2610.24579](https://arxiv.org/abs/2610.24579) · 2026-10
- **项目页**：[renshaojie233.github.io/LIBERO-Dynamic](https://renshaojie233.github.io/LIBERO-Dynamic/) ✅
- **代码**：[github.com/renshaojie233/LIBERO-Dynamic](https://github.com/renshaojie233/LIBERO-Dynamic) ✅；数据集 [HF: libero-dynamic-strict-nonbenchmark-1800](https://huggingface.co/datasets/rsj2003/libero-dynamic-strict-nonbenchmark-1800)
- **一句话**：把全部 **40 个 LIBERO 任务**变成动态版：用接触摩擦驱动物体沿 3 类轨迹族运动，并**按每个模型的实测推理延迟推进物理世界**，从而公平比较"慢模型"。
- **可用点**：
  - **"按模型延迟推进世界"这一评测协议必须抄进 ManiSkill3**：否则快模型与慢模型不可比；
  - 结论"更快的运动会劣化所有策略，但不均匀；执行时序与 horizon 会改变同一策略的行为；异步重规划只有在动作队列不欠载时才受益"——直接影响 P2/P5 的参数选择；
  - 9000 条动态轨迹的生成方式（重放静态示教 + 在线位移补偿）可用于本项目**低成本扩充数据**。
- **局限**：LIBERO 桌面场景，无传送带、无水果；延迟是标定值而非真实系统测量。

---

### 1.5 机械臂选型：文献使用频率与实际约束

#### 1.5.1 本调研样本中的硬件分布

统计口径：25 篇样本（本文件 19 个详细小节 + 6 篇补充 PDF）；下表只统计**论文中明确写出型号**的 12 篇，每篇按其实际做实验（或仿真）用的臂计一次。

| 平台 | 出现次数 | 代表论文 | 典型理由 |
| --- | --- | --- | --- |
| **Franka Emika Panda / Research 3** | **5** | PhysMani、DSDyn-VLA、DynamicWAM、LIBERO-Dynamic、NEO-VLA | 7-DoF 冗余 + 关节力矩传感 + **1 kHz FCI**，最适合做力控/阻抗与高带宽闭环；ManiSkill3/Isaac 原生资产 |
| **UR5e** | **2** | **KARL**（动态抓取）、DiscreteRTC | 850 mm 臂展 + 5 kg 负载 + **500 Hz RTDE** + 内置力传感器；工业现场最常见 |
| **ALOHA / AgileX 双臂** | **3** | FASTER、StreamPI、TrajWAM | 双臂模仿学习生态（LeRobot 数据格式） |
| **UFactory xArm 7** | **1** | **AHEAD**（传送带拦截） | 7-DoF、便宜、ROS 支持好；AHEAD 用它跑完整传送带任务 |
| **AgileX Piper** | **1** | **PAF-Grasp** | 轻量低成本桌面臂，验证"便宜也能做动态拦截" |
| **ABB YuMi (IRB 14000)** | **0** | — | 双臂 0.5 kg/臂、559 mm 臂展，**负载与臂展都不够**（见下） |

> 结论：**动态抓取社区事实上以 Franka 为主、UR5e 为工业备选、xArm 7 为性价比选项；YuMi 在动态抓取论文中几乎不出现。**

#### 1.5.2 官方标称参数（已核对厂商资料）

| 型号 | DoF | 臂展 | 负载 | 重复精度 | 控制接口 | ManiSkill3 原生资产 |
| --- | --- | --- | --- | --- | --- | --- |
| **Franka Research 3** | 7 | **855 mm** | 3 kg | < ±0.1 mm | **FCI 1 kHz**，TCP 速度至 2 m/s | ✅ `panda` |
| **UR5e** | 6 | **850 mm** | 5 kg | ±0.03 mm | **500 Hz**，TCP 至 4 m/s，内置 F/T | ❌（需自建 MJCF/URDF） |
| **xArm 7** | 7 | **700 mm** | 3.5 kg | ±0.1 mm 级 | SDK/以太网（250 Hz 级） | ✅ `xarm7` |
| **ABB YuMi IRB 14000** | 14（7×2） | **559 mm/臂** | **0.5 kg/臂** | ±0.02 mm | 内置 IRC5，TCP 1.5 m/s，0.86 s 抓放循环 | ❌ |

来源：[Franka FR3 数据表](https://franka.de/hubfs/Datasheet%20Franka%20Research%203_R02212_2.5_EN_June%202026.pdf)、[UR5e 技术规格](https://www.universal-robots.com/media/1802457/ur5e-tech-specs-eng.pdf)、[UFactory xArm](https://www.ufactory.cc/xarm-collaborative-robot/)、[ABB YuMi](https://www.abb.com/global/en/areas/robotics/products/robots/collaborative-robots/dual-arm-yumi)。

#### 1.5.3 对本项目的选型建议

**需求反推**：单臂居中、传送带贯穿、两侧各放料框 → 末端需覆盖"带中心 ± 料框深度"，实际需要 **≥ 800 mm 有效臂展**；负载 = 水果（≤ 250 g）+ 夹爪（≤ 1 kg）→ **≥ 2 kg 余量**；带速 0.2 m/s → 末端需 ≥ 0.5 m/s 的瞬时速度与 ≥ 50 Hz 的闭环带宽。

| 结论 | 说明 |
| --- | --- |
| **首选：Franka Panda / FR3** | 855 mm 臂展刚好覆盖"带 + 双侧框"；**7-DoF 冗余**让你在"跟着带子平移"时仍能保持末端竖直抓取姿态（6-DoF 需要靠腕部硬凑，容易到奇异）；1 kHz FCI 让 P5 的快残差层有带宽；**ManiSkill3 原生 `panda` 资产**，仿真与真机同一运动学，sim2real 差异最小。 |
| **备选（工业落地/预算）：UR5e** | 5 kg 负载、850 mm、500 Hz RTDE、内置力传感，工业维护与安全认证齐全；**KARL 已在其上验证动态抓取全栈**（含相机标定与 Contact-GraspNet 集成），工程风险最低。代价：ManiSkill3 无原生资产，需自建 URDF 并校准。 |
| **性价比/教学：xArm 7** | 700 mm 臂展对"带 + 双侧框"偏紧，需要把料框前移或加一条 300 mm 直线滑台；但 **ManiSkill3 原生 `xarm7`**，且 AHEAD 已证明它能做传送带拦截。 |
| **不推荐：ABB YuMi** | 559 mm 臂展 + 0.5 kg/臂负载：抓苹果（~200 g）+ 夹爪已接近上限，且双臂形态对"单臂 + 直线带"没有优势；文献中出现率为 0。 |
| **不建议自研/超低价臂** | 动态抓取吃的是"控制带宽 + 重复精度 + 标定质量"；PAF-Grasp 能用 Piper 成功，是因为它把预测做到 15 ms 并用可达性筛选规避了精度问题——这是很高的方法门槛。 |

> **最终建议**：仿真与真机统一选 **Franka Panda（FR3 真机 / ManiSkill3 `panda` 仿真）**；若采购受限，退到 **xArm 7 + 直线滑台**，并把 UR5e 作为工业客户的兼容目标（代码里把机器人抽象成 `embodiment` 配置，别写死）。

---

## 2. 相关工作对比表

| 方法 | 年份/会议 | 感知 | 规划/控制 | 学习方式 | 硬件 | 开源 |
| --- | --- | --- | --- | --- | --- | --- |
| **KARL** | 2025 IROS | 腕部 RGB-D + FoundationPose + **EKF 6D** | PPO 策略 + 6 阶段课程 + 重试 | RL（Isaac Gym 8192 env） | UR5e + L515 | ✅ 代码 / ⚠️ 无 LICENSE |
| **EARL** | 2023 IROS | 腕部相机主动跟踪 | RL 耦合跟踪+抓取 | RL | 多款商用臂 | ❌（被 KARL 取代） |
| **AHEAD** | 2026 CoRL | 第三人称 RGB + RAFT 光流 → 潜空间世界模型 | 冻结 OpenVLA + 未来 token 替换 | 世界模型 + 少量微调 | xArm 7 | ❌ |
| **PAF-Grasp** | 2026 | RGB-D + mesh 跟踪 → SE(3) 预测（15 ms） | 可达性拦截选择器，10 Hz 重规划 | 神经 ODE（Genesis 仿真训练） | AgileX Piper + D435i | ❌（pending） |
| **DSDyn-VLA** | 2026 | 光流 + 未来状态感知 | Flow-Planner（慢）+ Res-Refiner（快） | VLA + 残差 RL | 仿真 + 实机 | ❌（承诺开源） |
| **PhysMani** | 2026 ECCV | 3DGS + 多视角 | 世界模型预测 + 策略 cross-attention | 世界模型 + 策略 | Franka Panda | ✅ 代码 / 非商用 |
| **TEMPO** | 2026 CoRL | 冻结视频基础模型出运动摘要 + 本体历史 | 单帧 VLA 增强 | 表征增强（不改 backbone） | 未披露（项目页） | ✅ 代码 + TempoBench |
| **DiscreteRTC** | 2026 | — | 离散扩散异步 chunk（inpainting） | 离散扩散策略 | UR5e | ❌ |
| **LIBERO-Dynamic** | 2026 | 仿真观测 | 延迟对齐评测协议 | benchmark（6 策略） | Franka（仿真） | ✅ 代码 + 数据 |
| **Islam CTMP** | 2020 RSS / 2021 IJRR | 点云配准（噪声大） | **constant-time 重规划** | 无（搜索式规划） | PR2 7-DoF | ❌（用 SMPL） |
| **Han/Feng/Yu** | 2019 | 理想化（位置已知） | **DP 最优抓取序列** | 无 | Delta / SCARA 模型 | ❌ |
| **ZenRobotics** | 2015 | 几何模型 + 自标定 | 采样候选 + 在线学习 | 在线学习（夹爪反馈） | 工业臂 + 传送带 | ❌ |
| **Apple 两阶段** | 2026 | YOLOv8 + ByteTrack + ResNet18 | 无（只做分级） | 监督学习 | 无（离线视频） | ✅ 数据集 |
| **EV-Catcher** | 2022 RA-L | **事件相机**（BEHI） | 撞击点回归 + 控制信号 | 监督（轻量网络） | 固定臂 + Jetson NX | ❌ |
| **Event 6-DoF** | 2025 | 事件光流 + RGB 位姿 | 位姿积分跟踪 | 混合（学习 + 模型） | 事件相机 | ❌ |
| **Fourmy MPC** | 2023 | 学习定位 + 模型跟踪（120 Hz） | **力矩 MPC** | 学习感知 + 模型控制 | 未披露 | ❌ |
| **On-the-Move** | 2024 | RGB → 3 个 FCN | 动作基元 + 残差速度 | 自监督 | 移动底盘 + 臂 | ❌ |
| **RPL** | 2018 | 低维状态 | 基础控制器 + 残差 | RL 残差 | MuJoCo | ✅ |
| **对偶四元数伺服** | 2021 CASE | 视觉标记跟踪 + 点云 | PBVS + 零空间 | 无 | 7 轴臂（仿真） | ❌ |

---

## 3. 可直接复用的开源组件清单

| 组件 | 仓库 | License | 能拿来做什么 | 风险/备注 |
| --- | --- | --- | --- | --- |
| **ManiSkill3** | [haosulab/ManiSkill](https://github.com/haosulab/ManiSkill) | Apache-2.0 | 仿真底座；**原生 `panda` / `xarm7` 资产**；GPU 并行域随机 | 本机 `maniskill3` env 已装 3.0.1 |
| **KARL** | [arc-l/karl](https://github.com/arc-l/karl) | **无 LICENSE 文件** | eye-on-hand 跟踪 + EKF + PPO 全栈；UR5e 标定/部署脚本 | 需 `sudo docker` → 本机无 sudo，只能手动拆；Isaac Gym 已停更 |
| **PhysMani** | [vLAR-group/PhysMani](https://github.com/vLAR-group/PhysMani) | **CC BY-NC-SA 4.0（非商用）** | 3DGS 世界模型 + 16 任务 benchmark | 非商用；在线优化成本高 |
| **Residual Policy Learning** | [k-r-allen/residual-policy-learning](https://github.com/k-r-allen/residual-policy-learning) | 未声明（学术代码） | 残差策略的最小实现（MuJoCo） | 低维状态，需扩展到视觉 |
| **Genesis** | [Genesis-Embodied-AI/Genesis](https://github.com/Genesis-Embodied-AI/Genesis) | Apache-2.0 | PAF-Grasp 用的快速刚体仿真，可批量造"预测器"训练数据 | 本机未装；与 ManiSkill 二选一 |
| **TEMPO** | [tempo-robot/TEMPO](https://github.com/tempo-robot/TEMPO) | Apache-2.0 | 给单帧 VLA 加运动摘要 + 本体历史的参考实现（不改 backbone） | 2D 运动摘要，无 3D 位姿 |
| **TempoBench** | [tempo-robot/TempoBench](https://github.com/tempo-robot/TempoBench) | 见仓库（Tempo Robotics 版权） | 运动感知评测集（>50k 帧，回归 + 多选） | 用于评测速度估计模块，非任务策略 |
| **LIBERO-Dynamic** | [renshaojie233/LIBERO-Dynamic](https://github.com/renshaojie233/LIBERO-Dynamic) | MIT | "按模型延迟推进世界"的**评测协议实现**，可直接移植到 ManiSkill3 | LIBERO 场景，无传送带 |
| **ByteTrack** | [ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack) | MIT | 传送带上多水果的**轨迹级身份保持** | 与 YOLO 检测器解耦，易集成 |
| **Ultralytics YOLO** | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | **AGPL-3.0** | 水果检测/分割 + `track` 模式（内置 ByteTrack/BoT-SORT） | **AGPL 传染性**：分发产品需开源或买商业授权；内部研究可用 |
| **LeRobot** | [huggingface/lerobot](https://github.com/huggingface/lerobot) | Apache-2.0 | 数据格式 + ACT/Diffusion 训练部署 | 本机 `dynvla` 已装 0.3.3 |
| **Diffusion Policy** | [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) | MIT | 策略基线（chunk 执行） | 需自行加异步执行（见 DiscreteRTC） |
| **FoundationPose** | [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose) | **NVIDIA 源码许可（非商用）** | 6D 位姿（KARL 用它） | 需 CAD/mesh；商用需授权 |
| **Contact-GraspNet** | [NVlabs/contact_graspnet](https://github.com/NVlabs/contact_graspnet) | NVIDIA 许可（非商用） | 点云抓取位姿（KARL 实机用） | 需深度相机标定 |
| **SMPL** | [aurone/smpl](https://github.com/aurone/smpl) | 未声明 | 搜索式运动规划（Islam CTMP 的实现基础） | 无 license，学术使用 |
| **Healthy-Defective-Fruits** | [luischuquim/Healthy-Defective-Fruits](https://github.com/luischuquim/Healthy-Defective-Fruits) | 见仓库 | 苹果缺陷分类数据集 | 仅苹果，不含草莓/香蕉/梨 |

> **许可红线提醒**：`ultralytics` 是 **AGPL-3.0**、`FoundationPose` / `Contact-GraspNet` / `PhysMani` 是**非商用**。若本项目未来要交付/商业化，感知与位姿模块需提前规划替代（如自训检测器、[FoundationPose-plus-plus](https://github.com/teal024/FoundationPose-plus-plus) 等），否则后期返工成本极高。

---

## 4. 参考文献（BibTeX）

```bibtex
@inproceedings{boyalakuntla2025karl,
  title     = {KARL: Kalman-Filter Assisted Reinforcement Learner for Dynamic Object Tracking and Grasping},
  author    = {Boyalakuntla, Kowndinya and Boularias, Abdeslam and Yu, Jingjin},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2506.15945}
}

@inproceedings{huang2023earl,
  title     = {EARL: Eye-on-Hand Reinforcement Learner for Dynamic Grasping with Active Pose Estimation},
  author    = {Huang, Baichuan and Yu, Jingjin and Jain, Siddarth},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2023},
  url       = {https://arxiv.org/abs/2310.06751}
}

@inproceedings{syed2026ahead,
  title     = {Intercepting the Future: Latent-Space Predictive World Model for Dynamic VLA Manipulation},
  author    = {Syed, Shahram Najam and Jakobsson, Arthur and Hao, Haoran and Ichnowski, Jeffrey},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2606.02486}
}

@article{zhou2026pafgrasp,
  title   = {Physics-Aware Object-Centric SE(3) Forecasting for Real-Time Proactive Dynamic Grasping},
  author  = {Zhou, Yirui and Xu, Hongli and Zheng, Ying and Wang, Yi and Chau, Lap-Pui},
  journal = {arXiv:2610.17843},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.17843}
}

@article{li2026dsdynvla,
  title   = {DSDyn-VLA: A Dual-Stream Dynamic Manipulation Framework with Motion Perception, Future Awareness, and Realtime Correction},
  author  = {Li, Wenhao and Su, Xiu and Han, Yu and Cao, Yichao and You, Shan and Xu, Chang},
  journal = {arXiv:2609.39198},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.39198}
}

@inproceedings{yun2026physmani,
  title     = {PhysMani: Physics-principled 3D World Model for Dynamic Object Manipulation},
  author    = {Yun, Peng and Huang, Shouwang and Li, Hao and Li, Jinxi and Wang, Jianan and Yang, Bo},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2607.01938}
}

@inproceedings{feng2026tempo,
  title     = {TEMPO: Learning Temporal Context for Dynamic Robot Manipulation},
  author    = {Feng, Zhenyang and Heo, Jimin and Sudderth, Erik B. and Jain, Unnat},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2609.16864}
}

@article{wang2026discretertc,
  title   = {DiscreteRTC: Discrete Diffusion Policies are Natural Asynchronous Executors},
  author  = {Wang, Pengcheng and Hong, Kaiwen and Peng, Chensheng and Driggs-Campbell, Katherine and Tomizuka, Masayoshi and Xu, Chenfeng and Tang, Chen},
  journal = {arXiv:2604.25050},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.25050}
}

@article{ren2026liberodynamic,
  title   = {LIBERO-Dynamic: Benchmarking Robotic Manipulation in Dynamic Environments},
  author  = {Ren, Shaojie and Sun, Jingwen and Liu, Zezhi and Chen, Hongjin and Jin, Xin and Zou, Wei and Chen, Zhibo},
  journal = {arXiv:2610.24579},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.24579}
}

@inproceedings{islam2020ctmp,
  title     = {Provably Constant-time Planning and Replanning for Real-time Grasping Objects off a Conveyor Belt},
  author    = {Islam, Fahad and Salzman, Oren and Agarwal, Aditya and Likhachev, Maxim},
  booktitle = {Robotics: Science and Systems (RSS)},
  year      = {2020},
  doi       = {10.15607/RSS.2020.XVI.025},
  url       = {https://arxiv.org/abs/2003.08517}
}

@article{islam2021ijrr,
  title   = {Provably constant-time planning and replanning for real-time grasping objects off a conveyor belt},
  author  = {Islam, Fahad and Salzman, Oren and Agarwal, Aditya and Likhachev, Maxim},
  journal = {The International Journal of Robotics Research},
  year    = {2021},
  doi     = {10.1177/02783649211027194}
}

@article{han2019conveyor,
  title   = {Toward Fast and Optimal Robotic Pick-and-Place on a Moving Conveyor},
  author  = {Han, Shuai D. and Feng, Si Wei and Yu, Jingjin},
  journal = {arXiv:1912.08009},
  year    = {2019},
  url     = {https://arxiv.org/abs/1912.08009}
}

@article{kujala2015conveyor,
  title   = {Picking a Conveyor Clean by an Autonomously Learning Robot},
  author  = {Kujala, Janne V. and Lukka, Tuomas J. and Holopainen, Harri},
  journal = {arXiv:1511.07608},
  year    = {2015},
  url     = {https://arxiv.org/abs/1511.07608}
}

@article{park2026appletracking,
  title   = {A Two-Stage Detection-Tracking Framework for Stable Apple Quality Inspection in Dense Conveyor-Belt Environments},
  author  = {Park, Keonvin and Pal, Aditya and Mok, Jin Hong},
  journal = {arXiv:2602.19278},
  year    = {2026},
  url     = {https://arxiv.org/abs/2602.19278}
}

@article{wang2022evcatcher,
  title   = {EV-Catcher: High-Speed Object Catching Using Low-latency Event-based Neural Networks},
  author  = {Wang, Ziyun and Cladera Ojeda, Fernando and Bisulco, Anthony and Lee, Daewon and Taylor, Camillo J. and Daniilidis, Kostas and Hsieh, M. Ani and Lee, Daniel D. and Isler, Volkan},
  journal = {IEEE Robotics and Automation Letters},
  volume  = {7},
  number  = {4},
  year    = {2022},
  url     = {https://arxiv.org/abs/2304.07200}
}

@article{li2025event6dof,
  title   = {6-DoF Object Tracking with Event-based Optical Flow and Frames},
  author  = {Li, Zhichao and Glover, Arren and Bartolozzi, Chiara and Natale, Lorenzo},
  journal = {arXiv:2508.14776},
  year    = {2025},
  url     = {https://arxiv.org/abs/2508.14776}
}

@article{fourmy2023mpc,
  title   = {Visually Guided Model Predictive Robot Control via 6D Object Pose Localization and Tracking},
  author  = {Fourmy, Mederic and Priban, Vojtech and Behrens, Jan Kristof and Mansard, Nicolas and Sivic, Josef and Petrik, Vladimir},
  journal = {arXiv:2311.05344},
  year    = {2023},
  url     = {https://arxiv.org/abs/2311.05344}
}

@article{chen2018ukf,
  title   = {Tracking Multiple Moving Objects Using Unscented Kalman Filtering Techniques},
  author  = {Chen, Xi and Wang, Xiao and Xuan, Jianhua},
  journal = {arXiv:1802.01235},
  year    = {2018},
  url     = {https://arxiv.org/abs/1802.01235}
}

@article{kiyokawa2024onthemove,
  title   = {Self-Supervised Learning of Grasping Arbitrary Objects On-the-Move},
  author  = {Kiyokawa, Takuya and Nagata, Eiki and Tsurumine, Yoshihisa and Kwon, Yuhwan and Matsubara, Takamitsu},
  journal = {arXiv:2411.09904},
  year    = {2024},
  url     = {https://arxiv.org/abs/2411.09904}
}

@article{silver2018rpl,
  title   = {Residual Policy Learning},
  author  = {Silver, Tom and Allen, Kelsey and Tenenbaum, Josh and Kaelbling, Leslie},
  journal = {arXiv:1812.06298},
  year    = {2018},
  url     = {https://arxiv.org/abs/1812.06298}
}

@inproceedings{defarias2021dualquat,
  title     = {Dual Quaternion-Based Visual Servoing for Grasping Moving Objects},
  author    = {de Farias, Cristiana and Adjigble, Maxime and Tamadazte, Brahim and Stolkin, Rustam and Marturi, Naresh},
  booktitle = {IEEE International Conference on Automation Science and Engineering (CASE)},
  year      = {2021},
  url       = {https://arxiv.org/abs/2107.08149}
}
```

---

## 附：下一步（按优先级）

1. **P2 先量化延迟**：在 ManiSkill3 里加一个 `latency` 配置项，按 LIBERO-Dynamic 的协议"按延迟推进世界"，测出当前基线的成功率-延迟曲线。这决定后面所有设计。
2. **P1+P4 最小闭环**：YOLO+ByteTrack（复用 `ultralytics`）→ KF 外推 → 可达性筛选 → 脚本化拦截（对偶四元数伺服思路）。不做学习，先把传送带抓起来。
3. **P3 调度器**：实现 Han/Feng/Yu 的 DP 序列优化（带类别目标框约束），对比贪心，量化吞吐增益。
4. **P5 再上学习**：脚本拦截器作为基础策略 + Residual Policy Learning 残差；或 ACT/DP + DiscreteRTC 式异步执行。
5. 论文笔记：见 `notes/papers/2606.02486-ahead.md`（AHEAD，已有详版）、`2506.15945-karl-kalman-rl-dynamic-grasping.md`、`2610.17843-paf-grasp-se3-forecasting.md`、`1912.08009-conveyor-pnp-scheduling.md`、`2602.19278-apple-conveyor-detection-tracking.md`。
