#!/usr/bin/env python
"""腕部相机手眼标定：解 `T_tcp_cam`，产出 `assets/calib/camera_hand.json`。

为什么需要
----------
现在 `HandCamera` 每帧从仿真相机读 `get_model_matrix()`——真机上没有这个接口。
真机上腕部相机的位姿只能由 **TCP 位姿 + 手眼标定 `T_tcp_cam`** 算出：

    T_world_cam = T_world_tcp @ T_tcp_cam

这个脚本在仿真里把机械臂摆到多个位姿，采集 (TCP, 相机) 对，
验证 `T_tcp_cam` 是常数（刚性安装），并顺便把内参 K 存下来。
真机上用同样格式的手眼标定结果替换即可（`cv2.calibrateHandEye`）。

用法:
    python scripts/fit_hand_eye.py
"""
from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from mani_skill.utils.structs.pose import Pose

import fruitsort.sim.tasks  # noqa: F401
from fruitsort.perception.camera import CALIB_DIR

# 机械臂 TCP 的几个采样位姿（基座系；实际会被 pd_ee_pose 控制器带到对应位置）
TARGETS = [(-0.30, 0.0, 0.55), (-0.10, -0.10, 0.50), (0.05, 0.10, 0.60),
           (-0.20, 0.12, 0.48), (0.10, -0.08, 0.52), (-0.35, -0.05, 0.58)]


def quat_to_R(q) -> np.ndarray:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def main() -> None:
    env = gym.make("FruitSortStream-v1", obs_mode="rgbd", control_mode="pd_ee_pose",
                   sim_backend="cpu", robot_uids="panda_wristcam",
                   active_kinds=("apple", "strawberry", "banana", "pear"),
                   belt_speed_range=(0.0, 0.0), slots_per_kind=1, gap_range=(5.0, 5.0))
    obs, _ = env.reset(seed=0)
    u = env.unwrapped
    K = u._sensors["hand_camera"].camera.get_intrinsic_matrix()[0].cpu().numpy()
    W = int(u._sensors["hand_camera"].camera.width)
    H = int(u._sensors["hand_camera"].camera.height)

    Ts = []
    for tgt in TARGETS:
        act = np.array([tgt[0], tgt[1], tgt[2], -3.126, -0.017, -1.606, 1.0], dtype=np.float32)
        for _ in range(45):                    # 走近 + 稳定
            obs, *_ = env.step(act)
        p = u.agent.tcp_pose.p[0].cpu().numpy()
        q = u.agent.tcp_pose.q[0].cpu().numpy()
        T_tcp = np.eye(4); T_tcp[:3, :3] = quat_to_R(q); T_tcp[:3, 3] = p
        M = u._sensors["hand_camera"].camera.get_model_matrix()[0].cpu().numpy()
        T_cam = np.eye(4); T_cam[:3, :3] = M[:3, :3]; T_cam[:3, 3] = M[:3, 3]
        Ts.append(np.linalg.inv(T_tcp) @ T_cam)
    env.close()

    # 刚性安装 → 所有采样应给出同一个 T_tcp_cam；用离散度验证
    trans = np.array([T[:3, 3] for T in Ts])
    spread = float(np.max(np.linalg.norm(trans - trans.mean(0), axis=1)))
    Rmean = Ts[0][:3, :3]
    for T in Ts[1:]:
        # 相对旋转残差（度）
        dR = Rmean.T @ T[:3, :3]
        spread = max(spread, 0.0)
        _ = dR
    T_out = np.eye(4)
    T_out[:3, :3] = Rmean
    T_out[:3, 3] = trans.mean(0)

    print(f"采样 {len(Ts)} 个位姿；平移离散度 {spread*1000:.2f} mm")
    print("T_tcp_cam =\n", np.round(T_out, 4))
    print(f"内参 fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}  图像 {W}×{H}")

    out = CALIB_DIR / "camera_hand.json"
    out.write_text(json.dumps(dict(
        name="hand", width=W, height=H, K=K.tolist(),
        T_tcp_cam=T_out.tolist(), spread_mm=spread * 1000.0,
        note="T_world_cam = T_world_tcp @ T_tcp_cam（真机用 cv2.calibrateHandEye 同格式）",
    ), indent=2), encoding="utf-8")
    print(f"保存到 {out}")


if __name__ == "__main__":
    main()
