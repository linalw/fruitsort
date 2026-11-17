#!/usr/bin/env python
"""用吸盘控制器采集高速示教（0.15 / 0.30 m/s）。

动作标签 = 世界目标位姿 + 吸盘开关（-1 吸 / +1 放），由策略学习。
只保存成功 episode。

用法:
    python scripts/record_suction_demos.py --speed 0.30 --episodes 15 --out data/datasets/suction-030
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils.structs.pose import Pose

import fruitsort.sim.tasks  # noqa: F401
from fruitsort.policies.scripted import HOME_EULER
from fruitsort.sim.conveyor import BIN_LAYOUT

ROOT = Path(__file__).resolve().parent.parent


def jpg(x):
    return cv2.imencode(".jpg", x[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def obs_vec(u):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from record_demos import obs_vector
    return obs_vector(u)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=["apple", "strawberry", "banana", "pear"])
    ap.add_argument("--speed", type=float, default=0.30)
    ap.add_argument("--episodes", type=int, default=15, help="每类 episode 数")
    ap.add_argument("--spawn-x", type=float, default=-0.55)
    ap.add_argument("--spawn-spacing", type=float, default=0.6)
    ap.add_argument("--control-freq", type=int, default=30)
    ap.add_argument("--seed", type=int, default=11000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out = Path(args.out) if args.out else ROOT / "data" / "datasets" / f"suction-{int(args.speed*100):03d}"
    out.mkdir(parents=True, exist_ok=True)
    saved = 0
    idx = 0

    for kind in args.classes:
        for ep in range(args.episodes):
            env = gym.make(
                "FruitSort-v1", obs_mode="rgbd", control_mode="pd_ee_pose", sim_backend="cpu",
                robot_uids="panda_wristcam", active_kinds=tuple(args.classes),
                belt_speed_range=(args.speed, args.speed),
                spawn_x=args.spawn_x, spawn_spacing=args.spawn_spacing,
                instruction_kind=kind,
                sim_config=dict(control_freq=args.control_freq,
                                sim_freq=120 if args.control_freq == 30 else 100))
            obs, _ = env.reset(seed=args.seed + idx)
            u = env.unwrapped
            actor = u.fruit_actors[kind]
            attached = False
            offset = np.zeros(3)
            phase = "approach"
            info = {}
            obs_list, act_list, oh_list, wr_list = [], [], [], []
            for t in range(700):
                tcp = u.agent.tcp_pose.p[0].cpu().numpy().astype(np.float64)
                fp = actor.pose.p[0].cpu().numpy().astype(np.float64)
                bx, by = BIN_LAYOUT[kind]
                if not attached:
                    if phase == "approach":
                        tgt = np.array([fp[0], fp[1], fp[2] + 0.10])
                        if abs(tcp[0] - fp[0]) < 0.06 and abs(tcp[1] - fp[1]) < 0.06:
                            phase = "descend"
                    else:
                        tgt = np.array([fp[0], fp[1], fp[2] + 0.05])
                        if abs(tcp[0] - fp[0]) < 0.07 and abs(tcp[1] - fp[1]) < 0.07 and tcp[2] < fp[2] + 0.18:
                            attached = True
                            offset = fp - tcp
                            phase = "carry"
                    grip = 1.0
                else:
                    if phase == "carry":
                        tgt = np.array([bx, by, 0.62])
                        if abs(tcp[0] - bx) < 0.04 and abs(tcp[1] - by) < 0.04:
                            phase = "place"
                    elif phase == "place":
                        tgt = np.array([bx, by, 0.52])
                        if tcp[2] < 0.55:
                            phase = "release"
                    else:
                        tgt = np.array([bx, by, 0.52])
                    grip = -1.0  # 吸住

                label = np.array([tgt[0], tgt[1], tgt[2], grip], dtype=np.float32)
                err = tgt - tcp
                d = float(np.linalg.norm(err))
                if d > 0.08:
                    tgt = tcp + err / d * 0.08
                cmd = np.array([tgt[1] + 0.5, -tgt[0], tgt[2], *HOME_EULER,
                                1.0 if not attached else -1.0], dtype=np.float32)
                sd = obs["sensor_data"]
                obs_list.append(obs_vec(u))
                act_list.append(label)
                oh_list.append(jpg(sd["overhead"]["rgb"][0].cpu().numpy()))
                wr_list.append(jpg(sd["hand_camera"]["rgb"][0].cpu().numpy()))
                obs, _, term, trunc, info = env.step(cmd)

                if attached and phase in ("carry", "place"):
                    tcp2 = u.agent.tcp_pose.p[0].cpu().numpy().astype(np.float64)
                    p = np.clip(tcp2 + offset, -2.5, 2.5)
                    actor.set_pose(Pose.create_from_pq(torch.tensor([p], dtype=torch.float32)))
                    actor.set_linear_velocity(torch.zeros(1, 3))
                    actor.set_angular_velocity(torch.zeros(1, 3))
                if attached and phase == "release":
                    p = np.array([bx, by, 0.08])
                    actor.set_pose(Pose.create_from_pq(torch.tensor([p], dtype=torch.float32)))
                    actor.set_linear_velocity(torch.zeros(1, 3))
                    attached = False
                    phase = "done"
                if bool(info.get("success") or info.get("fail")):
                    break

            ok = bool(info.get("success", False))
            if ok and len(obs_list) > 20:
                np.savez_compressed(
                    out / f"ep_{idx:04d}.npz", obs=np.stack(obs_list),
                    action=np.stack(act_list), seed=args.seed + idx,
                    kinds=np.array(args.classes), instruction=np.array(kind),
                    overhead_jpg=np.array(oh_list, dtype=object),
                    wrist_jpg=np.array(wr_list, dtype=object))
                saved += 1
            print(f"[{'✓' if ok else '✗'}] {kind:11s} ep {ep:02d} steps={t + 1}", flush=True)
            idx += 1
            env.close()

    (out / "meta.json").write_text(json.dumps(
        {"speed": args.speed, "episodes": saved, "classes": args.classes,
         "control_freq": args.control_freq, "created": time.strftime("%Y-%m-%dT%H:%M:%S")},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 保存 {saved} 条吸盘示教到 {out}")


if __name__ == "__main__":
    main()
