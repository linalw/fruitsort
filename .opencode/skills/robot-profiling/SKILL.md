---
name: robot-profiling
description: 具身智能训练与推理的性能/数据剖析工具箱。用于"训练太慢""GPU 利用率低""某函数是瓶颈""推理延迟高""轨迹误差怎么评估""怎么可视化 rollout"等场景，覆盖 py-spy/scalene/pyinstrument/torch.profiler、nvitop、rerun、evo、rosbag 分析。
---

# 性能与数据分析

所有命令先 `source scripts/env.sh rtools`（或用 `conda run -n rtools ...`）。

## 0. 先定位瓶颈在哪一层

```bash
nvitop -m full                 # GPU 利用率、显存、功耗、进程树（替代 nvtop）
glances                        # CPU/内存/磁盘/网络（替代 htop）
```

判读：

| 现象 | 结论 | 下一步 |
| --- | --- | --- |
| GPU util 低（<40%）、CPU 高 | dataloader / 预处理瓶颈 | 查 dataloader、图像解码 |
| GPU util 高、step 时间长 | 计算瓶颈 | torch.profiler 看算子 |
| 显存周期性抖动 | batch/序列长度不均 | 采样形状统计 |
| CPU 单核 100% | Python GIL / 单线程 | py-spy 看栈 |

## 1. Python 层

```bash
py-spy top --pid <PID>                                   # 实时看热点函数
py-spy record -o /tmp/opencode/profile.svg --pid <PID>   # 采样火焰图
py-spy dump --pid <PID>                                  # 卡死时抓当前栈

scalene --html --outfile /tmp/opencode/scalene.html train.py   # 逐行 CPU+内存+GPU
pyinstrument -r html -o /tmp/opencode/pyinstr.html train.py    # 调用树，开销低
```

- 训练已在跑就先 `py-spy top --pid`，不用改代码。
- `py-spy` 对已启动进程可以直接 attach；看栈里的 `_wait_for_tstate_lock`、`recv` 往往就是同步点。

## 2. GPU / torch 层

```python
import torch, torch.profiler as prof
with prof.profile(activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.CUDA],
                  record_shapes=True, with_stack=True) as p:
    for _ in range(5):
        step()
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=30))
p.export_chrome_trace("/tmp/opencode/trace.json")   # chrome://tracing 打开
```

```python
torch.cuda.memory_summary()                 # 显存分布
torch.cuda.max_memory_allocated() / 2**30   # GB
```

- 不要用 `torch.cuda.synchronize()` 到处埋点测时间，会把异步流水线打断。
- 多卡/多进程时用 `torch.distributed` 的 `torch.profiler` 并给每个 rank 单独 `export_chrome_trace`。
- 本机没有 `nsys`/`ncu`（需 sudo），torch.profiler + py-spy 足够定位绝大多数问题。

## 3. 可视化 rollout（rerun）

```python
import rerun as rr
rr.init("eval", spawn=True)          # 或 rr.save("run.rrd") 无头保存
for t, frame in enumerate(rollout):
    rr.set_time("step", sequence=t)
    rr.log("cam/front", rr.Image(frame.rgb))
    rr.log("state/qpos", rr.Scalars(float(frame.qpos[0])))
    rr.log("ee_traj", rr.Points3D(frame.ee_xyz.reshape(1, 3)))
```

- 相机 + 关节曲线 + 末端轨迹放同一条时间轴，最容易看出"为什么失败"。
- 无显示环境用 `rr.save(...)`，之后在本地 `rerun file.rrd` 打开。

## 4. 轨迹与成功率评估（evo）

```bash
# 统一格式：t x y z qx qy qz qw，tum / kitti / euroc
evo_traj tum est.txt --ref gt.txt -p --plot_mode xyz --align
evo_ape  tum gt.txt est.txt -va --plot --save_results /tmp/opencode/ape.zip
evo_rpe  tum gt.txt est.txt -va --delta 1 --delta_unit f
evo_res  /tmp/opencode/ape.zip -p
```

- `--align`（Sim3/Umeyama）后再比，否则坐标系差异会淹没误差。
- 操作任务更常用"末端位姿误差 + 成功率"，用 numpy 自己算即可。

## 5. 从 rosbag 里挖数据

```python
from rosbags.highlevel import AnyReader
from pathlib import Path
import numpy as np

rows = []
with AnyReader([Path("bag")]) as r:
    conns = [c for c in r.connections if c.topic == "/joint_states"]
    for conn, ts, raw in r.messages(connections=conns):
        m = r.deserialize(raw, conn.msgtype)
        rows.append((ts / 1e9, *m.position))

a = np.array(rows)
print("时长", a[-1, 0] - a[0, 0], "频率", len(a) / (a[-1, 0] - a[0, 0]))
```

- 先统计**频率/丢帧/时间戳跳变**，再谈算法问题；真机问题一半出在这里。
- 大 bag 不要整体 deserialize 到内存，按 topic 流式读。

## 输出约定

剖析结果、火焰图、trace 放 `/tmp/opencode/`；要留下的结论写进对应 `runs/<日期>-<简述>/notes.md`。
