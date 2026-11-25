#!/usr/bin/env python
"""标定 overhead 相机：世界 (x, y) ↔ 像素（单应矩阵）。

做法：把每个水果依次放到带面上一组已知位置（其余水果移出视野），
检测色块质心，拟合单应矩阵，保存到 assets/calib/overhead_homography.npy。

用法:
    python scripts/calibrate_camera.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils.structs.pose import Pose

import fruitsort.sim.tasks  # noqa: F401
from fruitsort.perception.vision import CALIB_DIR, detect_blobs

POINTS = [(-1.75, -0.09), (-1.75, 0.09),
          (-1.30, -0.09), (-1.30, 0.09),
          (-0.90, -0.09), (-0.90, 0.09),
          (-0.55, -0.09), (-0.55, 0.09), (-0.15, -0.09), (-0.15, 0.09),
          (0.20, -0.09), (0.20, 0.09), (0.45, -0.09), (0.45, 0.09)]
FAR = (-3.5, 0.0)


def place(actor, x, y, z):
    pose = Pose.create_from_pq(torch.tensor([[x, y, z]], dtype=torch.float32))
    actor.set_pose(pose)
    actor.set_linear_velocity(torch.zeros(1, 3))
    actor.set_angular_velocity(torch.zeros(1, 3))


def main() -> None:
    env = gym.make(
        "FruitSort-v1", obs_mode="rgbd", control_mode="pd_ee_pose",
        sim_backend="cpu", with_bins=False, belt_speed_range=(0.0, 0.0),
    )
    obs, _ = env.reset(seed=0)
    u = env.unwrapped
    kinds = list(u.active_kinds)
    H = u.belt_top_z

    world_pts, pixel_pts = [], []
    for kind in kinds:
        for (x, y) in POINTS:
            for other in kinds:
                actor = u.fruit_actors[other]
                if other == kind:
                    place(actor, x, y, H + u.specs[other].half_height + 0.002)
                else:
                    place(actor, *FAR, H + 0.06)
            obs, *_ = env.step(np.zeros(7))
            rgb = obs["sensor_data"]["overhead"]["rgb"][0].cpu().numpy()
            blobs = [b for b in detect_blobs(rgb) if b.area > 40]
            if len(blobs) != 1:
                print(f"  ! {kind} @ ({x:+.2f},{y:+.2f}) 检测到 {len(blobs)} 个色块，跳过")
                continue
            pixel_pts.append(blobs[0].uv)
            world_pts.append((x, y))
            print(f"  {kind:11s} ({x:+.2f},{y:+.2f}) -> uv ({blobs[0].uv[0]:6.1f},{blobs[0].uv[1]:6.1f})")

    env.close()
    if len(world_pts) < 6:
        raise SystemExit(f"有效标定点只有 {len(world_pts)} 个，标定失败")

    Hmat, _ = cv2.findHomography(
        np.array(world_pts, dtype=np.float64), np.array(pixel_pts, dtype=np.float64), 0
    )
    proj = cv2.perspectiveTransform(np.array(world_pts, dtype=np.float64)[None], Hmat)[0]
    err = np.linalg.norm(proj - np.array(pixel_pts), axis=1)
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    import os
    _name = os.environ.get("FRUITSORT_CALIB", "overhead_homography.npy")
    np.save(CALIB_DIR / _name, Hmat)
    print(f"\n✓ {len(world_pts)} 点标定完成，重投影误差 mean={err.mean():.2f}px max={err.max():.2f}px")
    print(f"  保存到 {CALIB_DIR / _name}")


if __name__ == "__main__":
    main()
