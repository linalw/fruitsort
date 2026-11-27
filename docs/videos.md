# 演示视频清单

约定（见 `AGENTS.md`）：每个达到验收标准或有突破的版本，都必须有 `mp4` + **同名 `.txt` 说明**。
视频体积大，存放在 `data/fruitsort/eval/`（不进仓库），此处登记背景与结论。

| 视频 | 版本 | 模型 / 配置 | 场景 | 结果 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `pure_policy_v008.mp4` | v1.7 | `act-wide-4c` + `--fb 0.5 --grip-ema 0.6 --carry-latch --carry-max-step 0.02` | 0.08 m/s，四类同场 | **4/4** | 纯策略低速档；单件处理 3.6–7.1 s |
| `pure_policy_v015.mp4` | v1.7 | 同上 | 0.15 m/s，四类同场 | **4/4** | 纯策略中速档；四类各 8 局的整批成功率 31/32 = 96.9% |
| `pure_policy_learned_carry_v015.mp4` | v1.7 | `act-wide-4c-v2`（`--carry-boost 3`）+ 无人工搬运限速 | 0.15 m/s，四类同场 | **4/4** | 搬运减速由策略自己学；整批 32/32 = 100% |

`pure_policy_v015.mp4` 的逐层架构、CVAE 说明、测试结果与逐段视频数据见
[`docs/architecture-act-wide-4c.md`](architecture-act-wide-4c.md)。
| `stream_geom_sparse_3m.mp4` | v1.53 | `act-geom` + `--fb 0.7 --grip-ema 0.7 --carry-latch --carry-max-step 0.02` | 连续作业 0.08 m/s，间距 3–4 m，尺寸 ±15% | **6/6 = 100%** | 完整 120 s 不剪辑（seed 60004）；同配置 5 局 26/30 = 86.7% |
| `stream_geom_dense_1m.mp4` | v1.53 | 同上 | 连续作业 0.08 m/s，间距 0.6–1.2 m（超载） | **12/14 = 92.3%** | 完整 120 s 不剪辑（seed 60002）；同配置 5 局 51/69 = 73.9% |
| `assets/videos/stream_demo_v153.gif` | v1.53 | 同上 | 稀疏档片段 | — | README 首页用（12 s，8 fps） |
| `assets/videos/stream_demo_v153_dense.gif` | v1.53 | 同上 | 密集档片段 | — | 12 s，8 fps |
| `stream_fast_dense_1m.mp4` | v1.54 | 按类提速（apple/straw 0.12、pear 0.10、banana 0.06） | 连续作业 0.08 m/s，间距 0.6–1.2 m | **12/14 = 92.3%**（机会内 12/12，单件 6.1 s，6.0 件/min） | 完整 120 s；5 局 55/68 = 80.9% |
| `assets/videos/stream_fast_v154.gif` | v1.54 | 同上 | 密集档片段 | — | 12 s，8 fps，README 用 |

## 早期（混合架构，非纯策略）

| 视频 | 版本 | 说明 |
| --- | --- | --- |
| `model_demo_v08.mp4` | v0.8 | 模型选果 + 伺服执行器，0.08 m/s 四类 4/4 |
| `model_demo_v025.mp4` | v1.0 | 验收档 0.25 m/s，22/24 = 91.7% |
| `suction_demo_030.mp4` | v0.10 | 吸盘末端原型 0.30 m/s（**已排除，不纳入验收**） |


## 每个视频的 `.txt` 里有什么

代码版本（git commit）、权重路径与离线指标（val / 参数量 / 是否增量动作）、
部署配置（残差反馈、夹爪去抖、搬运限速）、场景参数（带速、间距、控制频率）、
分集结果（整局时长 + **单件处理时间**）、复现命令、以及"视频只保留成功片段"的说明。

## 连续作业视频（v1.53）额外说明

- 场景 `FruitSortStream-v1`：单一下料流、**间隔随机**、尺寸 ±15%、每类 1 个 slot（4 件在飞）。
- 视频是**完整 120 s 连续过程**（不剪辑、不重试），失败件保留在画面里，与同目录 `metrics.json` 一致；
  画面左上角叠加：时间 / 带速 / 间距 / 当前目标类别 / 相位（0 接近、1 下探闭爪、2 搬运）/
  夹爪状态 / delivered-missed-dropped-spawned。
- 同配置 5 局汇总：稀疏 26/30 = **86.7%**、密集 51/69 = **73.9%**
  （对照 `act-gnoisy` 60.0%/53.7%、部署 `act-wide-4c` 33.3%/20.6%）。
- `.txt` 旁注文件在 `data/fruitsort/eval/`，含代码版本、权重、部署配置、场景参数与复现命令。
| `stream_suction_5of5.mp4` | 纯感知 + 吸盘末端，连续作业 **5/5 送达**、机会内 100%、5 件/min、单件 4.6 s | v1.74.0 / `runs/act-suction/ckpt.pt` |
| `head_rgbd_apple_only.mp4` | 头部 RGB-D + 纯感知 + 夹爪，交付口径（只做苹果）：apple 1/1；固定口径 42.3%（CI 26–61%）；**画中画=模型唯一图像输入+检测结果** | v1.86.1 / `runs/act-dagger/ckpt.pt` |
| `head_rgbd_all_kinds.mp4` | 四类都接：apple 1/1、**strawberry 1/3（首次）**、banana/pear 0；加密流 1.5–2.5 m；画中画同上 | v1.86.1 / `runs/act-dagger/ckpt.pt` |
| ~~`head_rgbd_gates_apple.mp4`~~（已被 v190 取代） | | **交付口径（v1.88/1.89 两道门）**：apple 6/6 = 100%，0 掉落；270 s | v1.89.1 / `runs/act-dagger/ckpt.pt` |
| ~~`head_rgbd_gates_all.mp4`~~（已被 v190 取代） | | 四类都接（含失败）：strawberry **3/5**、apple 0/4、3 掉落；180 s | v1.89.1 / `runs/act-dagger/ckpt.pt` |
| `head_rgbd_v190_apple.mp4` | **交付口径（v1.90）**：apple 5/5 = 100%，0 掉落；180 s | v1.90.0 / `runs/act-dagger/ckpt.pt` |
| `head_rgbd_v190_all.mp4` | 四类都接（v1.90 修 bug 后）：apple **5/5**、strawberry 1/3、banana/pear 0，0 掉落；180 s | v1.90.0 / `runs/act-dagger/ckpt.pt` |
| `percv2_masks_apple.mp4` | **感知 v2**（分割掩码可见）：apple 3/3 = 100%，0 掉落；180 s | v1.92.0 / `runs/act-dagger/ckpt.pt` |
| `learned_seg_masks.mp4` | 学习式分割（U-Net）+ ROI 放大窗：apple 1/1；闭环尚未打通 | v1.93.0 / `runs/act-dagger/ckpt.pt` |
| `sort_rate_2026_11_27.mp4` | **当前交付口径**连续作业（2 局 × 90 s）：apple 3/3、0 掉落；统计口径见 `docs/success-rate.md` | v1.96.0 / `runs/act-dagger/ckpt.pt` |
