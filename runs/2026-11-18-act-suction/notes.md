# 2026-11-18-act-suction

## 假设 / 目的

0.41M 的 `MultiViewTemporalPolicy` 在 0.30 m/s 苹果上纯策略 0/16，怀疑是**容量/时序建模不足**。
本实验换成 ACT 风格 chunked 策略（6.87M：4 层编码器 + 4 层解码器 + K=16 可学习 query
+ CVAE 隐变量），在同一批吸盘示教上训练，看闭环是否改善。

## 配置

- 数据：`data/datasets/suction-030`（60 条，0.30 m/s）+ `suction-015`（40 条，0.15 m/s）
  = 9282 个窗口（T=8 历史，stride 2）
- 模型：`src/fruitsort/models/act_policy.py::ActPolicy`，6,874,140 参数
  - 视觉：复用 `ConvEncoder`（RGB+帧差，96²，双视角共享权重），BN→GroupNorm（闭环 eval 一致）
  - 编码器 4 层 d=256 / 8 头 / ffn 3d，34 token（1 CLS + 8 帧×(overhead+wrist+state)）
  - 解码器 4 层，K=16 可学习 query，cross-attn 到编码器 memory
  - CVAE：latent 32，训练用 posterior，推理 z=0
- 训练：80 epochs，batch 64，AdamW lr 1e-4 / wd 1e-2，cosine，bf16 autocast，
  dropout 0.15，亮度/对比度抖动 + 10% 帧延迟增强
- 验证：**按类别分层**留出最后 15% episode（apple/strawberry/banana/pear 各有）

## 运行

```bash
python scripts/train_act.py --data data/datasets/suction-030 data/datasets/suction-015 \
    --out runs/2026-11-18-act-suction --epochs 80 --batch 64 --stride 2 --lr 1e-4

python scripts/eval_act_suction.py --ckpt runs/2026-11-18-act-suction/ckpt.pt \
    --instruction apple --episodes 5 --belt 0.30 --chunk-exec 1  --out runs/2026-11-18-act-suction/eval-s1
# 同样跑 --chunk-exec 4 / 8、--gate 0.15、--kinds apple
```

## 观察

### 离线

| 指标 | 值 |
| --- | --- |
| 参数量 | 6.87M |
| best val（归一化总损失，epoch 47） | **0.136**（val_xyz 0.0204，val_grip 0.0578） |
| 训练集对应值（epoch 47） | 0.0447（xyz 0.0108） |
| 验证集目标位姿误差（3D，米） | **4.8 cm 平均 / 3.5 cm p50**；apple 7.7 cm / p95 24 cm |
| 吸附决策准确率 | 97.8%（apple 93.1%） |
| GPU 推理延迟 | **4.4 ms 均值 / 6.6 ms p95**（RTX 5090，2×96²×8 帧） |

对比 v0.5（`runs/vpolicy-suction`，0.41M）：旧 val 0.064 不可比（旧切分 val 100% 是 pear）；
同一切分下 ACT 的 val 更低、离线误差更小 —— **容量/时序建模不是瓶颈**。

### 闭环（0.30 m/s，苹果，吸盘附着，5 episodes）

| 设置 | 成功率 | 附着 | 备注 |
| --- | --- | --- | --- |
| s=1 每步重规划 | **0/5**（CI 0–43.5%） | 0/5 | 全部 700 步跑满，5/5 `no_attach` |
| s=4 open-loop | **0/5** | 1/5 | 附上后第 75 步 fail（bin_err 0.213 m） |
| s=8 open-loop | **0/5** | 2/5 | 326 步 / fail（bin_err 0.765 m） |
| s=1 + 几何门控 0.15 m | **0/5** | 0/5 | 门控挡掉 219–611 次"吸"指令，**轨迹完全不变** |
| 基线 v0.5（0.41M） | 0/5 | — | 同带速、同 instruction |
| s=1 `--kinds apple`（单类，OOD） | 0/5 | 0/5 | 训练数据全是 4 类同场，`active` 位不同属分布外，不作对照 |

### 根因（trace：`tcp / apple / 预测目标 / 吸 logit`）

```
t=  0 tcp=[-0.01 -0.32 0.66] apple=[-0.53 -0.00 0.48] tgt=[-0.67 +0.02 0.60] |tgt-apple|=0.18 logit=-10.5
t= 15 tcp=[-0.26 -0.13 0.64] apple=[-0.38 -0.00 0.48] tgt=[-0.34 -0.01 0.58] |tgt-apple|=0.11 logit=-8.3
t= 20 tcp=[-0.27 -0.06 0.58] apple=[-0.33 -0.00 0.48] tgt=[-0.30 -0.13 0.57] |tgt-apple|=0.16 logit=+0.5  ← 提前翻相
t= 25 tcp=[-0.30 -0.14 0.58] apple=[-0.28 -0.00 0.48] tgt=[-0.48 -0.37 0.64] |tgt-apple|=0.45 logit=+9.7
t= 35 tcp=[-0.46 -0.30 0.63] apple=[-0.18 -0.00 0.48] tgt=[-0.48 -0.34 0.64] |tgt-apple|=0.48 logit=+9.6  ← 悬停在框上
```

1. **追踪本身是好的**：t=0–15 目标位姿贴着苹果（差值恒为 +0.1 m 高度），苹果在以 0.3 m/s 移动也被跟住。
2. **相位提前翻转**：t≈20、TCP 距苹果还有 16 cm 时，模型把"目标"从苹果切到**分拣框**
   并且下达"吸"。示教里这个切换发生在"贴近水果"那一刻，模型学到的是**几何边界附近的
   时间先验**，而闭环里它一直追不上（0.08 m/步限速 + 目标在动），于是"到达自己目标"被
   误判成"已接触"。
3. **单向门**：翻相后不再回头。示教数据里没有"吸了但没吸上"的状态，模型学不到恢复动作。
4. 门控实验证明**这是同一个相变量**：只拦截"吸"指令（219–611 次）后，`bin_err` 与 s=1
   逐位相同（1.051/1.032/1.069）→ 目标位姿的翻转不受夹爪输出控制。
5. s=4/8 能附上，是因为绝对目标过期后 TCP 停在苹果**将要经过**的位置，等于"传送带把水果
   送进吸盘"；但搬运目标偏框心 0.1–0.2 m（apple 离线误差 p95 = 24 cm），仍未释放，
   最终 fail（该 fail 也含"非目标水果被堆挤掉落"的协议噪声，见下）。

## 结论 / 下一步

- **换更强模型（0.41M→6.87M，离线误差降到 cm 级）没有改善闭环**：仍 0/5。瓶颈不是容量，
  也不是图像分辨率，而是**相位状态机**（approach→descend→carry→release）与**缺少纠错/恢复**。
- 下一步（按性价比）：
  1. 加**相位辅助头**（3 分类，标签由几何谓词生成）+ 推理端小 FSM/迟滞：只在
     `dist<8cm` 后允许进入 carry，且进入后不再退出。这直接把"单向门"变成"有状态机兜底"。
  2. **DAgger-lite 补恢复数据**：用当前策略跑、专家给"吸了没吸上"的状态打标（`docs/learning.md`
     已经把这列为第一优先）。
  3. **拦截点标签**：示教标签是"苹果当前位姿"，模型天然带一帧滞后；改成
     `fp + v·Δt_lead` 重打标签，对 0.3 m/s 这种高速目标收益最大（apple 离线误差是 pear 的 2.3 倍）。
  4. **评测协议修正**：`--instruction` 单类场景下 `fail` 会因**非目标水果**堆挤掉带而触发
     （s=4/s=8 那两条 attach 后很快 fail 的可能原因，未坐实）。注意单类场景要用 `record` 端
     同款 `active_kinds`，否则 `active` 特征变成分布外（本实验已验证：`--kinds apple` 直接
     0 附着，不能当对照）。

## 复现清单

- 新文件：`src/fruitsort/models/act_policy.py`、`scripts/train_act.py`、`scripts/eval_act_suction.py`
- 产物：`runs/2026-11-18-act-suction/{ckpt.pt, metrics.json, notes.md, eval-s1,eval-s4,eval-s8,eval-s1-gate15}`
- 训练耗时：80 epochs × 7 s ≈ 10 min（5090，bf16，6 workers）
