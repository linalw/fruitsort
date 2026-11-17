# 模仿学习基线（BC）

> 版本：v0.4.0 · 数据来自脚本专家（oracle 位姿），目标是打通
> **示教采集 → 训练 → 闭环评测** 的完整链路，为 ACT / Diffusion Policy / VLA 提供底座。

## 数据

`scripts/record_demos.py`

| 项 | 值 |
| --- | --- |
| 观测 | 49 维：TCP 位姿(7) + 关节(9) + 带速(1) + 4 类 × [位置(3), 速度(3), active(1), held(1)] |
| 回归目标 | 4 维：世界坐标目标位姿 (x, y, z) + 夹爪（不含欧拉角，抓取全程竖直向下） |
| 控制频率 | 20 Hz |
| 采集策略 | `ScriptedSorter`（预测拦截 + 抓放状态机） |
| 过滤 | 只保留成功 episode |
| 存储 | `data/datasets/<name>/ep_XXXX.npz` + `meta.json` |

```bash
python scripts/record_demos.py --kinds apple strawberry --episodes 40
python scripts/record_demos.py --kinds apple strawberry --episodes 30 --noise 0.02  # 鲁棒化数据
```

**为什么回归"目标位姿"而不是"执行动作"**：执行动作是 `target` 经过 6 cm/步限速后的结果，
在目标较远时是常数、较近时又连续变化，函数不光滑；目标位姿本身是水果位置的平滑函数，
更适合回归。评测时再用同样的限速逻辑转换成动作。

## 模型

`src/fruitsort/learning/bc.py`

```text
历史 5 帧观测 (5×49=245)
  → Linear 512 → GELU → Linear 512 → GELU → Linear 512 → GELU
  → Linear (K × 4) → reshape (K, 4)
```

- 动作分块 K=8（ACT 的 chunk 思想），推理时每步重规划、只执行第 0 步。
- 观测/目标按训练集统计量标准化；损失 MSE；AdamW + cosine。
- 按 episode 划分验证集；评测 seed 100+ 与训练 seed 0–39 不重叠。

```bash
python scripts/train_bc.py \
    --data data/datasets/bc-pair-tgt data/datasets/bc-pair-noisy \
    --out runs/bc-pair-mix --epochs 80
```

## 结果

| 策略 | 场景 | 成功率 |
| --- | --- | --- |
| 脚本专家（oracle） | 苹果+草莓，10 episodes | **100%** |
| BC（目标位姿 + 5 帧历史） | 苹果+草莓，10 episodes | **10%（1/10）** |
| BC（单帧、增量动作目标） | 苹果+草莓，5 episodes | 0% |

离线 val loss 可降到 0.002–0.004（归一化 MSE），专家状态上的动作预测几乎逐帧吻合；
但闭环只有 10%，是典型的 **compounding error / 分布漂移**。这组「低离线误差 + 低闭环成功率」
本身就是本项目 v0.4 最有价值的负结果：**模仿学习的瓶颈不在拟合，而在纠错**。

## 失败分析

1. **复合误差**：每步 1–2 cm 的方向偏差会累积；BC 没有纠正机制，一旦偏离专家分布，预测迅速退化。
2. **阶段不可观测**：脚本专家是有限状态机（TRACK/DESCEND/CARRY/RELEASE），单帧观测无法完全区分
   「接近水果」与「接近料框」等阶段；5 帧历史只能部分缓解。
3. **动作限速**：6 cm/步的限速让动作对目标位置敏感，近距离时方向抖动明显。
4. **数据量小**：40 条示教 ≈ 1.4 万步，对高维连续控制偏少。

## 下一步（按性价比排序）

1. **DAgger**：用专家给 BC 访问到的状态打标签；这是最直接的修复，预计能把 20% 拉到 60%+。
2. **ACT / Diffusion Policy**：时序 Transformer + 动作分块 + 多模态输入，替换 MLP；
   `lerobot` 已在 `dynvla` 环境可用。
3. **图像观测**：接入 overhead + wrist 图像，去掉对 oracle 位姿的依赖（与 `docs/perception.md` 合并）。
4. **更大数据集**：200–500 条示教，覆盖不同带速与初始摆放。
