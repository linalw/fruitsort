---
name: robot-stack-map
description: 本机的机器人软硬件环境地图与操作手册：ROS 2 Humble、conda 环境、GPU、CAN 总线、ManiSkill/Gazebo 仿真。当需要跑仿真、启动 ROS 节点、录制/转换 rosbag、和真机电机通信、或排查"命令找不到/环境不对"时使用。
---

# 机器人环境地图

机器细节见项目根 `AGENTS.md`；本 skill 是可执行的操作手册。

## 0. 先做环境自检

```bash
source scripts/env.sh                 # 入口，必须最先执行
echo $ROS_DISTRO                      # humble
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
conda env list | grep -E 'maniskill3|rtools'
```

非交互 shell 里 `ros2: command not found` 几乎总是因为没 source，而不是没装。

## 1. ROS 2 Humble

```bash
ros2 pkg list | wc -l                 # 本机 325 个包
ros2 node list / topic list -t / topic hz /topic / topic echo --once /topic
ros2 param list /node ; ros2 interface show sensor_msgs/msg/Image
ros2 run rviz2 rviz2                  # 需要 DISPLAY
ros2 launch <pkg> <file.launch.py>
colcon build --symlink-install && source install/setup.bash
```

- 已有：gazebo classic（`gazebo_ros`）、nav2 全家桶、rviz2、rqt、image_transport、tf2、ros2control。
- 缺少：`ros2bag` CLI、`socat`、`can-utils`（装不了）。用下面的 Python 方案替代。
- 跨机/多进程通信异常时先看 `ROS_DOMAIN_ID` 和 `RMW_IMPLEMENTATION`（默认 fastrtps）。

## 2. rosbag 处理（无需 source ROS）

`rtools` 环境里的 `rosbags` 是纯 Python 实现，读写 rosbag1/rosbag2、转 mcap、裁剪话题：

```bash
source scripts/env.sh rtools
rosbags-convert --src record/ --dst out.mcap                 # ros2 db3 -> mcap
rosbags-convert --src a.bag --dst out/ --include-topic /camera/color/image_raw
rosbags-convert --src in.bag --dst out/ --dst-storage sqlite3 --compress zstd
```

Python 里逐条读（做统计/抽帧/画图）：

```python
from rosbags.highlevel import AnyReader
with AnyReader([Path("bag_dir")]) as reader:
    conns = [c for c in reader.connections if c.topic == "/joint_states"]
    for conn, ts, raw in reader.messages(connections=conns):
        msg = reader.deserialize(raw, conn.msgtype)
```

## 3. CAN 总线 / 电机调试

没有 `can-utils`，用 `python-can`：

```bash
source scripts/env.sh rtools
python -m can.viewer -i socketcan -c can0 -b 1000000     # 需要 root 起 can0，通常用现成脚本
```

```python
import can
bus = can.Bus(interface="socketcan", channel="can0", bitrate=1_000_000)
for msg in bus: print(hex(msg.arbitration_id), msg.data.hex())
```

- 解析 DBC 用 `cantools`：`db = cantools.database.load_file("x.dbc")`。
- 常见坑：bitrate 不一致、终端电阻没接、`ip link set can0 up type can bitrate 1000000` 需要 sudo。

## 4. 仿真

| 栈 | 环境 | 备注 |
| --- | --- | --- |
| ManiSkill3（sapien） | `maniskill3` | 最快，适合操作任务与大规模 rollout |
| Gazebo classic | ROS Humble 系统环境 | 和 ros2control 联调 |
| Isaac Lab / Isaac Sim | **未安装** | 需要 20GB+ 下载，先确认再装 |

```bash
conda run -n maniskill3 python -c "import mani_skill, gymnasium as g; print(mani_skill.__version__)"
```

- 无头跑渲染：优先 `render_mode=None`；需要图像时用 `sapien` 的 offscreen renderer，不要依赖 `DISPLAY`。
- RTX 5090 是 sm_120，太老的 torch/cuda 轮子会报 `no kernel image is available`；本机 torch 2.10+cu128 正常。

## 5. 真机/外部设备排障顺序

1. `lsusb` / `ls /dev/tty{USB,ACM}*` / `ls /dev/video*` —— 设备是否枚举。
2. 权限：`groups | grep dialout`，没有就 `sudo usermod`（本机无 sudo，找管理员）。
3. `dmesg | tail` 看 USB 掉线、驱动报错。
4. 相机：`python -c "import cv2; print(cv2.VideoCapture(0).read()[0])"`。
5. 通信类问题先抓包/抓总线，再改代码。
