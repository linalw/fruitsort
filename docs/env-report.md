# 环境报告（自动生成，勿手改）

- 生成时间: 2026-11-17 10:49:26 CST
- 主机: ubun / 用户: ubuntu
- 系统: Ubuntu 22.04.5 LTS 内核 6.8.0-138-generic / x86_64

## GPU

`0, NVIDIA GeForce RTX 5090, 580.105.08, 32607 MiB, 19830 MiB, 48, P1`

- 显存占用:
  - 2628994, 1786 MiB, /home/ubuntu/linalw/App/minconda3/envs/dynvla/bin/python
  - 3508501, 610 MiB, /home/ubuntu/linalw/App/minconda3/envs/dynvla-sim51/bin/python
  - 3511438, 610 MiB, /home/ubuntu/linalw/App/minconda3/envs/dynvla-sim51/bin/python
  - 3577226, 1772 MiB, /home/ubuntu/linalw/App/minconda3/envs/dynvla/bin/python
  - 3687091, 1786 MiB, /home/ubuntu/linalw/App/minconda3/envs/dynvla/bin/python
  - 3712889, 7350 MiB, /home/ubuntu/linalw/App/minconda3/envs/dynvla-sim51/bin/python
  - 716357, 942 MiB, python
  - 717570, 3326 MiB, /home/ubuntu/linalw/App/isaacsim/_build/linux-x86_64/release/kit/python/bin/python3

- nvcc: Build cuda_13.0.r13.0/compiler.36424714_0
- torch 默认架构: sm_120 (Blackwell) → 需要 CUDA 12.8+ 的 wheel

## CPU / 内存 / 磁盘

- CPU: 32 核  Intel(R) Core(TM) i9-14900KF
- 内存: 125 GB total, 66 GB available
- 交换: 1 GB

| 挂载点 | 大小 | 可用 | 用途 |
| --- | --- | --- | --- |
| `/` | 984G | 11G | 系统 |
| `/media/ubuntu/66A27065A2703B9D` | 9.1T | 9.1T | 大数据盘 |

## conda 环境

conda: /home/ubuntu/linalw/App/minconda3

| env | Python | 关键包 |
| --- | --- | --- |
| `bge` | 3.11.17 | torch 2.5.1+cu121, numpy 2.4.6, transformers 4.44.2 |
| `dynvla-sim51` | 3.11.18 | torch 2.7.1+cu128, torchvision 0.22.1+cu128, numpy 1.26.0, transformers 4.57.6, isaaclab 0.54.2, gymnasium 1.2.1, h5py 3.16.0 |
| `dynvla` | 3.10.22 | torch 2.7.1+cu128, torchvision 0.22.1+cu128, numpy 1.26.4, transformers 5.2.0, lerobot 0.3.3, gymnasium 0.29.1, h5py 3.16.0, rerun-sdk 0.22.1, wandb 0.31.0 |
| `hssd310` | 3.10.21 | numpy 2.2.6 |
| `light` | 3.10.22 | torch 2.11.0+cu128, torchvision 0.26.0+cu128, numpy 2.2.6 |
| `lingbot` | 3.10.20 | torch 2.8.0+cu128, torchvision 0.23.0+cu128, numpy 2.2.6, opencv-python 5.0.0.93 |
| `maniskill3` | 3.10.21 | torch 2.15.0.dev20260908+cu130, torchvision 0.30.0.dev20260908+cu130, numpy 2.2.6, mani_skill 3.0.1, sapien 3.0.3, gymnasium 1.3.0, opencv-python 5.0.0.93, h5py 3.16.0 |
| `physmani` | 3.10.22 | torch 2.7.1+cu128, torchvision 0.22.1+cu128, numpy 2.2.6, transformers 5.2.0, lerobot 0.3.3, gymnasium 0.29.1, opencv-python 5.0.0.93, h5py 3.16.0, rerun-sdk 0.22.1, wandb 0.31.0 |
| `qwen38` | 3.12.14 | torch 2.13.0, torchvision 0.28.0, numpy 2.3.5, transformers 5.17.0 |
| `rdp` | 3.11.16 | torch 2.14.0, torchvision 0.29.0, numpy 1.26.4, transformers 5.17.0, opencv-python 4.11.0.86, h5py 3.16.0, wandb 0.30.0 |
| `rtools` | 3.11.18 | numpy 2.4.6, rerun-sdk 0.43.2 |
| `sparseworld` | 3.10.21 | torch 2.8.0, torchvision 0.23.0, numpy 1.26.4, transformers 4.48.3, opencv-python 4.11.0.86 |

## ROS

- 发行版: humble
- 包数量: 325
- 组件: gazebo_dev gazebo_msgs gazebo_plugins gazebo_ros gazebo_ros_pkgs nav2_bringup realsense2_camera realsense2_camera_msgs rviz2 
- 工作空间: OrbbecSDK_ROS2 

## 网络（5s 超时）

- https://export.arxiv.org/api/query?search_query=all:robot&max_results=1 → `200`
- https://huggingface.co → `200`
- https://pypi.org → `200`

## 提示

- 非交互 shell 先 `source scripts/env.sh` 才能用 ros2 / conda env。
- 新建环境用 Python 3.11；安装 torch 需 cu128+（5090 是 sm_120）。
