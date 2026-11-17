# 决策记录（ADR）

> 只记会长期影响工作方式的决定，一条一个编号。推翻时不要删，加一条新决策并标注取代关系。

## D1 · 仿真优先，真机后接

- 日期：2026-11-17
- 状态：有效
- 背景：本机已有 ManiSkill3（`maniskill3`）与 Isaac Lab 5.1（`dynvla-sim51`），真机链路（ROS2 + Orbbec）也具备，但两者同时推进会拖慢迭代。
- 决定：阶段 0–2 全部在仿真里完成；真机只保留最小链路（能出图、能录数据、能回放），阶段 3 再正式接。
- 影响：
  - 任务定义以仿真可实现为准，物体先做刚体近似。
  - 评测以 `docs/roadmap.md` 阶段 2 的指标为准，真机指标阶段 3 才进入主线。
  - 资产/标定等真机内容先放 `assets/`，不阻塞主线。

## D2 · 机器人选用 Franka Panda

- 日期：2026-11-17
- 状态：有效
- 背景：需要在 ManiSkill3 / Isaac Lab 间选仿真，并在 Panda / UR5e / xArm7 / ALOHA 间选臂。
- 决定：**Franka Panda**（7-DoF + 平行夹爪）。仿真用 ManiSkill3 原生 `panda` 资产。
- 理由：
  - 855 mm 臂展覆盖带面中心 + 两侧料框；7-DoF 可在跟带平移时保持竖直抓姿。
  - 12 篇动态抓取论文中 Franka 系（Panda/FR3）出现频率最高，LIBERO-Dynamic/TEMPO/DSDyn 的 DynBench 都用它。
  - ManiSkill3 原生资产，sim2real 差异最小；后续接 ROS 2 + FCI 也最成熟。
- 备选：UR5e（KARL 全栈验证，但 ManiSkill3 无原生资产）、xArm7（臂展偏紧）。
- 影响：任务与评测以 Panda 工作空间为准；换臂需重做可达区与料框布局。

## D3 · 仿真后端 ManiSkill3

- 日期：2026-11-17
- 状态：有效
- 背景：本机有 ManiSkill3（SAPIEN）与 Isaac Lab 5.1 两套。
- 决定：v0.2 用 **ManiSkill3 3.0.1 + SAPIEN 3.0.3（CPU backend）**；Isaac Lab 留给后续 GPU 大规模并行 / 域随机化。
- 理由：自定义任务 API 简单、启动快、Panda/相机/渲染开箱即用；5090 上单环境 CPU 仿真已能实时。
- 影响：GPU 并行训练（>64 env）时再迁 Isaac Lab；任务逻辑保持与后端解耦（`sim/tasks/`）。

## D4 · 评测协议对齐 LIBERO-Dynamic

- 日期：2026-11-17
- 状态：有效
- 决定：固定 20 Hz、1200 步上限、固定 seed 序列；报告 Wilson 95% CI；失败分类为
  `unreachable / grasp_miss / slip / wrong_bin / dropped / timeout`。
- 附加指标（论文没有、分拣必须有）：分拣正确率、单件节拍、抓取成功率、掉落/破损率。
- 影响：`scripts/benchmark.py` 为唯一官方入口，结果写 `runs/<id>/metrics.json`。

## 模板

```markdown
## D<n> · <一句话决定>
- 日期：
- 状态：有效 / 已被 D<m> 取代
- 背景：
- 决定：
- 影响：
```
