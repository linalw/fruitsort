# 夹爪时序 / 抓取触发 / 动作块执行

> 状态：设计稿（v0.1，2026-11-17）。只描述机制与配置，不含实现代码改动。
> 素材：`notes/papers/*.md`（16 篇精读）+ `src/fruitsort/policies/scripted.py` + `src/fruitsort/models/vision_policy.py`。
> 约定：带论文名的数字来自笔记原文；标「估算」的是本机量级推算，需用 `scripts/` 实测替换。

## 0. 三句话结论

1. **夹爪"何时闭"必须是几何/时间触发器，模型只给"要不要抓"的意图**。AHEAD、PAF-Grasp、KARL 都是这么拆的；把开合交给网络回归，1 bit 信号在 20 条示教里几乎学不出来。
2. **块执行**：我们的延迟量级（模型 1.9–10 ms、感知 1–2 帧、闭爪 80–150 ms）下，`s=1` 逐帧重规划就是最优解；RTC 的收益前提是推理 ≥ 2 个控制步（DiscreteRTC 在 20 Hz 用 d=4），我们还没有这个前提。
3. **瓶颈不在网络前向**。`eval_vision_policy.py` 里 logit 直接过 0 阈值开关爪，是纯视觉 0/5 的直接原因之一（`docs/perception.md` 已定位遮挡与滞后）。

## 1. 现状盘点：我们其实已经在做"几何触发"

`scripted.py` 的 7 状态机里，夹爪是**三段不同口径**拼出来的：

| 阶段 | 预瞄/触发口径 | 问题 |
| --- | --- | --- |
| TRACK | `lead_time=0.35 s`，对齐容差 0.055 m | 0.35 s 是拍的，没有按延迟分解 |
| DESCEND | 预瞄缩短到 `0.12 s`，容差 0.035 m，`tcp[2] <= grasp_z+0.015` 触发 | 用"到达高度"当触发器，忽略物体仍在动 |
| CLOSE | 预瞄再缩到 `0.08 s`，边跟边闭，`grip = -min(1, 0.3+counter/8)` | 渐进闭爪是对的，但没标定 `t_close` |
| 判定 | `is_grasping or (counter>=4 and dist<0.06)` | 只有"抓到了/没抓到"，不区分早/晚/偏 |
| CARRY | `dist>0.10` 判丢 | 阈值单一，梨/香蕉假阴性（代码注释已写） |

`vision_policy.py` 一侧：chunk K=8、夹爪单独头（`grip_head`），这架构没问题；但推理端 `grip = -1.0 if pred4[3] > 0 else 1.0`，**没有平滑、没有门控、没有锁存**，logit 抖一下就开合一次。TEMPO 的失败分析正好对上：基线最大问题是"每帧都重新发起动作"（VLASH 在 Drop Catch 有 **23/50** 次 premature move，TEMPO 降到 **6/50**）。

实测延迟（`docs/model-architecture.md`）：GPU **1.9 ms**（p95 4.9）、CPU 单线程 **6.1 ms**（p95 10.1）。20 Hz 预算 50 ms 的 4–20%，**所以 `d = ceil(τ/50ms) = 1`**。

## 2. 问题一：什么时候闭爪（五类机制）

| 机制 | 代表 | 触发信号 | 关键数字 | 对我们的适用性 |
| --- | --- | --- | --- | --- |
| 几何/到达触发 | 我们的 `scripted.py`；AHEAD `t*−t_close` | TCP 与预测物体位置的距离 + 高度 | AHEAD：在 `t*−t_close` 开始闭合，`t_close` 需自测标定 | ✅ 底座，必须先标定 |
| 预测接触时刻 | **PAF-Grasp** | 沿 SE(3) 轨迹前推 contact frame，解可达性 | 预测器 **15 ms**、**10 Hz** 重规划、100 Hz 控制；带速 0.08–0.20 m/s 下 **78%** vs π0.5+RTC **34%** | ✅ 我们的场景就是"匀速 + 可达性" |
| 状态机 commit（不可回退） | **TEMPO**、**KARL** | 进入 COMMIT 后锁目标 ID 与拦截点 | TEMPO premature move 23/50→6/50；KARL 抓空后可重入跟踪，成功率 **92.58%** | ✅ 成本最低，先做 |
| 学习输出开合 | PhysMani（`BCE(f_open, a_open)`）、DiscreteRTC（10 维动作含二值夹爪） | 网络直接给 0/1 | DiscreteRTC 20 Hz d=4/s=4，同步基线 **0%** | ⚠️ 只能做"意图"，不能做"时刻" |
| 残差/快环 | **DSDyn-VLA** | 慢规划 + 30 Hz 残差纠偏 | refiner **3 ms**，残差限幅 10% 动作范围；48.4 vs 无 refiner 42.7 | ⭕ 二期，先不上 |

**为什么不能纯学习**：一条 10 s 轨迹（20 Hz 下 200 步）里夹爪只翻转 1–2 次，正负样本极度不平衡，回归目标会被"永不闭爪"支配（估算）。DiscreteRTC 能用离散头，是因为它把动作量化成 256–512 bins 并配 8 步 MaskGIT 解掩码，而不是靠一个浅层回归头。TEMPO 的结论更直接：**先补时间上下文（历史桶）比换大模型有效**（Bottle Handover VLASH 38%→TEMPO 74%，只加 2M 参数）。

**我们的规则**：模型输出 `grip_intent ∈ [0,1]`，只在 `APPROACH 且 dist<0.15 m 且 in_zone` 时参与决策，且只用于"是否开始降速/预闭合"；**最终闭合由 `t*` 触发**。

```text
t*  = 最早满足  |p_obj(t) − p_tcp(t)|_xy < ε_xy 且 t ≥ t_now + T_lead  的时刻
T_lead = t_obs + t_infer + t_close + t_servo          # 实测反推，不许拍
close_cmd 发出时刻 = t* − t_close
```

## 3. 问题二：动作块怎么执行

| 方式 | 有效滞后（物体位置误差） | 论文证据 | 什么时候用 |
| --- | --- | --- | --- |
| **每步重规划 s=1** | `d·Δt` = 50 ms | PAF-Grasp 10 Hz 重规划直到接触；DSDyn 5 Hz 规划 + 30 Hz 残差 | ✅ 我们的默认 |
| open-loop 执行 K 步 | `(s/2)·Δt + d·Δt` | TrajWAM `Ta=32` 执行 8；PhysMani 5 Hz + buffer（272.8 ms/keyframe，靠缓冲解耦） | 模型慢（≥200 ms）时 |
| temporal ensemble | 同上，但会**平均掉不同模式** | PhysMani 5 Hz 用；DiscreteRTC 明确批评"会把不同模式平均掉" | 只用于 100 Hz 伺服层，不用于 xyz 重规划 |
| RTC 异步 | `d·Δt`，但需 `s ≥ d` | DiscreteRTC 20 Hz d=4/s=4、5× 插值到 100 Hz；DynamicWAM 重叠 4 步、`γ_max=0.5`，+9.17 点（RealCore 4.58 + RTC 4.59） | 推理 ≥ 100 ms 时 |

两条经验规律必须记住：

- **低频时异步更差**：Reflex 扫出 5 Hz 下 async **8.6%** < sync **16%**（传送带），30 Hz 下 async **31.3%** > sync **22.6%**；最优区是 **chunk 8 / horizon 2 / 30 Hz**。
- **horizon 太小会 underflow**：LIBERO-Dynamic 在 20 Hz 下 K=1 稳态 underflow **828.3/1000 步**、成功率 **0%**；K=6 时 underflow 归零、SR 峰值 **48.4%**（Async 48.4 > Sync 43.1 > RTC 37.8）。我们的模型 2 ms，K=1 不会 underflow——**这条只对慢模型成立**。

## 4. 问题三：三类延迟各自怎么补

| 延迟 | 我们的量级 | 补偿手段 | 依据 |
| --- | --- | --- | --- |
| 推理 `t_infer` | 1.9 ms / p95 4.9（GPU） | 不需要补偿；若换大模型再用 CUDA Graph（Reflex：125.1→65.0 ms）或 FASTER 的逐动作定型（TTFA 80.0→62.1 ms） | — |
| 观测 `t_obs` | **1–2 帧 = 50–100 ms**（`docs/perception.md` 实测） | ① 所有量带时间戳，按 `t_now` 而非"最新帧"外推；② KF 外推（KARL 的 coasting：出视野只预测不更新）；③ 训练时随机间隔采样（StreamPI δ~U[3,7] 帧，T=5 只 +9.2 ms） | PAF：SE(3) 预测比位置-only 高 **37.5 pp** |
| 执行 `t_exec` | 闭爪 80–150 ms + 1 步伺服 50 ms（估算） | 提前 `t_close` 下命令；闭合期间**跟随**物体而不是停在原地；3 步渐变避免冲击 | AHEAD：`d = v_belt·T_lead`，0.2 m/s 时 ≈ 一个果径 |

`T_lead` 的量级（估算，取 `t_obs=50 + t_infer=10 + t_close=120 + t_servo=50 = 230 ms`，设计取 **0.25 s**）：

| 带速 | 0.06 m/s | 0.12 m/s | 0.18 m/s |
| --- | --- | --- | --- |
| 前视位移 | 15 mm | 30 mm | 45 mm |

对比：apple 直径 66 mm、DESCEND 对齐容差 35 mm。**0.18 m/s 时前视位移已超过容差**，所以必须"瞄准拦截点"而不是"追当前位置"——这就是 AHEAD 笔记里"预测误差 = 一个果径就必抓空"的定量版本。

延迟敏感性可当预算表用：PhysMani 额外注入 0/100/200/300 ms → SR **77.5/57.5/37.5/20.0**，约每 100 ms 掉 20 点。我们 20 Hz 下 1 步 = 50 ms，所以"多等一帧"是 8–10 点。

另外：**评测必须让物理不停**（ReflexBench 的核心贡献、LIBERO-Dynamic 用 `D=⌈τ·20⌉` 推进世界）。Reflex 笔记估计"推理时冻结环境"会让传送带类指标乐观 20–40 点。

## 5. 问题四：失败模式与可观测信号

失败占比先看真机统计（DynamicWAM，12 任务 50 次失败）：**时机错误 42%**、横向错位 26%、丢跟踪 18%、接触后滑落 10%。我们的 `docs/benchmark.md` 里香蕉/梨以 `slip` 为主、`grasp_miss` 次之，方向一致。

| 模式 | 物理过程 | 可观测信号 | 判定式（用我们已有的量） |
| --- | --- | --- | --- |
| 过早闭爪 | 闭爪瞬间物体仍在夹爪上游，随后从指间/指前通过 | 闭合时刻的相对 x 为负，且 3 步后仍无接触 | `x_obj − x_tcp < −0.02`（闭爪时刻快照）+ `is_grasping=False` |
| 过晚闭爪 | 闭爪瞬间物体已越过指间平面 | 相对 x 为正且继续增大 | `x_obj − x_tcp > +0.02`（闭爪时刻快照） |
| 横向抓空 | 闭爪时物体在指宽外 | 指间距合拢到最小 + 无接触 | `|Δy| > 0.03` 或（指间距 < 0.015 m 且 `is_grasping=False`） |
| 抓空（高度错） | 抓在物体上方/下方 | TCP 与物体 z 差超过半径 | `|z_obj − z_tcp| > 0.035`（按品类取半径） |
| 滑落 | 搬运中接触力丢失 | 接触力、相对位姿、z 速度 | CARRY 中 `is_grasping: True→False` 或 `|p_obj−tcp| > 0.10`（现有）或 `v_z < −0.3 m/s` |
| 误抓/重复抓 | 状态混叠（刚放开又要抓） | 夹爪开合历史 + 目标 ID | TEMPO 式历史桶（H=90, K=6, L=15）+ COMMIT 锁 |

可用的原始信号（ManiSkill Panda，已确认关节名）：`agent.is_grasping()`（接触力）、`robot.get_qpos()` 里 `panda_finger_joint1/2`（指间距，最大 0.08 m）、`tcp_pose`、物体真值/视觉估计位姿、带速。

**判定窗口**：闭合后 **3 步（150 ms）** 必须出结论；5 步仍无结论 → 保守判失败并重试一次（KARL 的"重试而非终止"）。注意 `is_grasping` 对梨/香蕉有假阴性（`scripted.py` 已注明），所以判据用 **"接触力 OR（距离 + 指间距 plateau）"**。

## 6. 给 Fruitsort 的具体配置

固定前提：20 Hz（物理 100 Hz，`sim dt=0.01`）、0.41M 模型、Panda 平行二指、带速 0.06 名义 / 0.05–0.18 范围、chunk K=8、`hist=4`。

### 6.0 三方案共用的底座（约 1 天）

1. 新增 `GraspTrigger`：输入 `(p_obj, v_obj, p_tcp, v_tcp, t_close, T_lead)`，输出 `(should_close, x_target, phase)`。
2. FSM 增加 `COMMIT`：进入后**锁定 target id 与 t\***，只有 `|p_obj − p_pred| > 0.04 m` 才回退到 TRACK。
3. 记录字段写进 `metrics.json`：`obs_ts / t_infer_ms / close_cmd_ts / contact_ts / is_grasping / finger_gap / rel_pose`。
4. 先做 `scripts/calibrate_gripper.py`：空载测"命令→两指合拢"时间，带载测"命令→接触"，各 30 次取中位数（`t_close`）。

### 6.1 方案 A（推荐）：几何触发 + 模型意图门控 + s=1

| 参数 | 值 | 理由 |
| --- | --- | --- |
| 执行方式 | chunk K=8，**只执行第 0 步**，每步重推 | 模型 2 ms，重推成本可忽略；滞后 50 ms |
| `T_lead` | 0.25 s（标定后替换） | 0.06 m/s → 15 mm；0.18 m/s → 45 mm |
| 远距（>0.15 m） | 全速 `max_step=0.06`，预瞄 `T_lead` | 追赶阶段 |
| 近距（0.05–0.15 m） | 预瞄降到 `0.5·T_lead`，限速 0.03 | 避免"冲过头再回拉" |
| 触发条件 | `|Δxy| < 0.015` 且 `z_tcp − z_grasp ∈ [−5, +10] mm` 且 `v_z ≤ 0` | 把"到达高度"换成"在高度窗内且正在下降"，比现在只看 `+0.015 m` 更稳 |
| 闭爪 | 3 步渐变命令 `0.33/0.66/1.0`（`+1` 开、`−1` 闭） | 现有按 8 步斜率渐变，约 0.3 s 才全闭；3 步 ≈ 150 ms |
| 闭合期间 | x 目标 = `p_obj(t)`，速度匹配 ±0.01 m/s | 跟随而非静止，抵消带速 |
| 判定 | 3 步后按 §5 表分类 | 早/晚/空/滑可区分 |
| 重试 | 同目标 1 次（重算 t\*），出 zone 即记 `missed` | KARL 式重试 |
| 模型门控 | `grip_intent` 只在 `dist<0.15` 生效，其余由 FSM 决定 | 消除 logit 抖动 |

预期收益：吃掉 DynamicWAM 归类为"时机错误"的 42%。工作量 1–2 天，无需训练。

### 6.2 方案 B：RTC-lite（s=2、d=1）

- 触发条件：端到端 p95 **> 100 ms**（换 π0.5 级模型、或 GPU 争用）。NEO-VLA 在 3090 上 282.5 ms，属这一类。
- 做法：ring buffer 存动作块，只覆盖前 `d` 步，剩余步按原 chunk 执行；`s=2` 代价是额外 50 ms 滞后。
- 依据：FASTER 证明同步→异步只省 `0.5·Δt_infer`；DiscreteRTC 证明 chunk 间停顿是动态任务的致命伤，但它的设置是 d=4（≈200 ms）。
- **现在不做**：我们 d=1，s=1 已经比 RTC 更新鲜；引入队列只会增加 underflow 风险（LIBERO-Dynamic K=1 时 0%）。

### 6.3 方案 C：学习式夹爪（离散化 + 延迟随机化）

- 把 `grip_head` 从回归改成 **3 类（开/闭/保持）交叉熵**，或用 DiscreteRTC 的 256 bins。
- 数据要求：每条 demo 记 `close_cmd_ts`、`contact_ts`；训练时随机注入延迟 `d~U{0..3}`（LIBERO-Dynamic 的 `D=⌈τ·20⌉`）。
- 先加 TEMPO 式**夹爪历史桶**（H=90 / K=6 / L=15，3 s 覆盖一个分拣节拍），零初始化 FiLM 到 `z`。这是最便宜的一步，能直接消"刚张开又闭"。
- 风险：现只有 20 条示教（≈7000 窗口），远不够学时机；TEMPO 的收益也主要来自历史而非容量。**排第三**。

### 6.4 排序

| 排序 | 方案 | 工作量 | 预期收益 | 依赖 |
| --- | --- | --- | --- | --- |
| 1 | A：几何触发 + 门控 + s=1 | 1–2 天 | 高（时机类失败 42%） | `t_close` 标定 |
| 2 | C-轻量：夹爪历史桶 + FiLM | 2–3 天 | 中（消重复开合、混叠） | 重录带图像的 20 条 |
| 3 | B：RTC-lite | 3–5 天 | 低（当前延迟不需要） | 换大模型后再评估 |
| 4 | C-完整：离散化夹爪头 | 1–2 周 | 未知（数据不足） | ≥200 条示教 |

### 6.5 落地顺序（消融式，每步单独评测）

1. `calibrate_gripper.py` → 拿到 `t_close`，只替换 `scripted.py` 里的三个预瞄常数（0.35/0.12/0.08 → 统一 `T_lead`）。
2. 加 `COMMIT` 锁 → 测 `premature_close` 次数是否下降。
3. 加 §5 失败分类 → 看 `grasp_miss` 里早/晚/偏的比例。
4. 加重试 1 次 → 看 `missed` 转化率。
5. 最后把模型 `grip_intent` 接进门控（只在 `eval_vision_policy.py` 生效）。

评测口径按 D4：20 Hz、1200 步、带速 0.06/0.12/0.18 各 10 次、Wilson 95% CI；新增必报 **premature_close / late_close / grasp_miss / slip 次数、TTFA（obs→首动作）、close_cmd→contact 时间分布**。

## 7. 不确定与坑

1. **这些笔记全都没有力控细节**：AHEAD 笔记明确写"完全没涉及抓取力与形变"，PhysMani 也没测可变形物体，TrajWAM 只做独立运动刚体。草莓（18 g、易损）只能靠降速 + 浅抓 + 限幅，不能指望论文方案。
2. **仿真的 `t_close` 不可信**：ManiSkill 的夹爪是位置驱动，接触瞬间没有真实延迟建模。仿真与真机必须分别标定，且真机上 `t_close` 会随物体刚度变化（苹果 vs 草莓可能差一倍量级，估算）。
3. **`is_grasping` 假阴性**：梨/香蕉曲面接触面积小，`scripted.py` 已用"距离<0.06 且 counter≥4"兜底；再加指间距 plateau 更稳。
4. **别把 temporal ensemble 用在 xyz 重规划上**：会把不同时刻的拦截点平均掉（DiscreteRTC 的原话是会把不同模式平均掉）。要平滑就平滑到 100 Hz 伺服层。
5. **20 Hz 不能直接套 5 Hz 的结论**：PhysMani 的 5 Hz + buffer、TrajWAM 的 `Ta=32` 执行 8，都是为 200 ms 级推理设计的；我们 2 ms 时照抄只会增加滞后。
6. **AHEAD 的恒加速假设在碰撞后失效**（Plinko 48.6%）：水果离带后滚动、相互碰撞属于这一类，所以"离带后"应切回短视界追踪而不是长外推。
7. **调度与触发是两件事**：锁定单个目标后不要每步换目标，但**排队顺序**仍值得做滚动时域 DP——传送带上贪心（SPT/FIFO）被证明不是最优，吞吐差 **10–40%**（1912.08009）。这属于 `docs/roadmap.md` 阶段 2 之后的事。
