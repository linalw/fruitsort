#!/usr/bin/env python
"""夹爪标定：指令 `grip` → 指面开口 / 接触力。

为什么需要
----------
`--width-setpoint` 里写的是 `gap ≈ 0.030 + 0.050·g`（空载经验公式），但**夹着苹果时**
指面被果实挡住，实际开口 = 果实直径 − 挤压量，接触力也随 `g` 变。掉落（2/28）都发生在
"夹到上肩"，怀疑是**全闭合时挤压力过大**把果子挤出去。

这个脚本扫一遍 `g`，测：空载开口、夹苹果时的开口、指端法向力。

用法:
    python scripts/calibrate_gripper.py
"""
from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils.structs.pose import Pose

import fruitsort.sim.tasks  # noqa: F401
from fruitsort.perception.camera import CALIB_DIR
from fruitsort.policies.act_runner import finger_normal_force
from fruitsort.policies.scripted import HOME_EULER

GS = (-1.0, -0.7, -0.4, -0.1, 0.2, 0.4, 0.6, 0.7)
FAR = (-9.0, 0.0, 2.0)


def act(x, y, z, grip):
    return np.array([y + 0.5, -x, z, *HOME_EULER, grip], dtype=np.float32)


def main() -> None:
    env = gym.make("FruitSort-v1", obs_mode="rgbd", control_mode="pd_ee_pose",
                   sim_backend="cpu", with_bins=False, belt_speed_range=(0.0, 0.0))
    obs, _ = env.reset(seed=0)
    u = env.unwrapped
    belt = float(u.belt_top_z)
    half = float(u.specs["apple"].half_height)
    X, Y = -0.10, 0.0

    def gap() -> float:
        L = u.agent.robot.links_map["panda_leftfinger"].pose.p[0].cpu().numpy()
        R = u.agent.robot.links_map["panda_rightfinger"].pose.p[0].cpu().numpy()
        return float(np.linalg.norm(np.asarray(L) - np.asarray(R)))

    rows = []
    # ---- 1) 空载：完全张开 → 逐步闭合到 g ----
    for g in GS:
        for _ in range(25):
            obs, *_ = env.step(act(X, Y, belt + 0.25, g))
        rows.append(dict(g=g, mode="empty", gap=gap(), fn=0.0, apple_z=None))
        for _ in range(10):                     # 重新张开
            obs, *_ = env.step(act(X, Y, belt + 0.25, 1.0))

    # ---- 2) 夹着苹果：从上方下探到果心，再闭到 g ----
    for g in GS:
        a = u.fruit_actors["apple"]
        a.set_pose(Pose.create_from_pq(torch.tensor([[X, Y, belt + half + 0.002]],
                                                    dtype=torch.float32)))
        a.set_linear_velocity(torch.zeros(1, 3)); a.set_angular_velocity(torch.zeros(1, 3))
        for _ in range(35):                     # 张开、下探到果心
            obs, *_ = env.step(act(X, Y, belt + half, 1.0))
        for _ in range(30):                     # 闭到 g
            obs, *_ = env.step(act(X, Y, belt + half, g))
        z = float(a.pose.p[0, 2].cpu().numpy())
        rows.append(dict(g=g, mode="apple", gap=gap(), fn=finger_normal_force(u),
                         apple_z=z))
        for _ in range(12):                     # 抬起来，看会不会掉
            obs, *_ = env.step(act(X, Y, belt + 0.25, g))
        z2 = float(a.pose.p[0, 2].cpu().numpy())
        rows[-1]["apple_z_lift"] = z2
        rows[-1]["lifted"] = bool(z2 > belt + 0.06)
    env.close()

    print(f"{'g':>5s} {'空载开口':>9s} {'夹果开口':>9s} {'Fn(N)':>7s} {'抬起':>5s}")
    for r in rows:
        if r["mode"] == "empty":
            print(f"{r['g']:5.2f} {r['gap']*1000:8.1f}mm {'-':>10s} {'-':>7s} {'-':>5s}")
        else:
            print(f"{r['g']:5.2f} {'-':>10s} {r['gap']*1000:8.1f}mm {r['fn']:7.1f} "
                  f"{'是' if r['lifted'] else '否':>5s}")

    out = CALIB_DIR / "gripper_width.json"
    out.write_text(json.dumps(dict(apple_diameter=0.065, rows=rows), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n保存到 {out}")


if __name__ == "__main__":
    main()
