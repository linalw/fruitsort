#!/usr/bin/env python
"""吸盘末端原型：用"位姿跟随"近似真空吸附（SAPIEN 无运行时固定关节）。

- 接近：顶部下探到水果上方 2cm
- 吸附：水果位姿刚性跟随 TCP（等效于吸住），并清零速度
- 搬运/放置：到框上方 → 下放 → 释放（落到框内）
- 适用：香蕉/梨等顶抓困难的曲面/重物

用法:
    python scripts/eval_suction.py --classes apple strawberry banana pear \
        --episodes 6 --belt 0.30 --control-freq 30
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils.structs.pose import Pose

import fruitsort.sim.tasks  # noqa: F401
from fruitsort.eval import format_summary, summarize
from fruitsort.policies.scripted import HOME_EULER
from fruitsort.sim.conveyor import BIN_LAYOUT

ROOT = Path(__file__).resolve().parent.parent
INSTR_ID = {"apple": 1, "strawberry": 2, "banana": 3, "pear": 4}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=["apple", "strawberry", "banana", "pear"])
    ap.add_argument("--episodes", type=int, default=6, help="每类 episode 数")
    ap.add_argument("--belt", type=float, default=0.30)
    ap.add_argument("--spawn-x", type=float, default=-0.55)
    ap.add_argument("--spawn-spacing", type=float, default=0.6)
    ap.add_argument("--control-freq", type=int, default=30)
    ap.add_argument("--seed", type=int, default=6000)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--disturb", choices=["none", "offset", "speed", "push", "all"], default="none")
    ap.add_argument("--offset", type=float, default=0.12)
    ap.add_argument("--push", type=float, default=1.5)
    ap.add_argument("--video", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import cv2

    writer = None
    records = []
    for ci, kind in enumerate(args.classes):
        for ep in range(args.episodes):
            env = gym.make(
                "FruitSort-v1", obs_mode="state", control_mode="pd_ee_pose", sim_backend="cpu",
                robot_uids="panda_wristcam", active_kinds=tuple(args.classes),
                belt_speed_range=(args.belt, args.belt),
                spawn_x=args.spawn_x, spawn_spacing=args.spawn_spacing,
                instruction_kind=kind,
                render_mode="rgb_array" if args.video else None,
                sim_config=dict(control_freq=args.control_freq,
                                sim_freq=120 if args.control_freq == 30 else 100))
            env.reset(seed=args.seed + ci * 100 + ep)
            u = env.unwrapped
            actor = u.fruit_actors[kind]
            rng = np.random.default_rng(args.seed + ci * 100 + ep)
            if args.disturb in ("offset", "all"):
                p0 = actor.pose.p[0].cpu().numpy()
                p0[1] = float(np.clip(p0[1] + rng.uniform(-args.offset, args.offset), -0.16, 0.16))
                actor.set_pose(Pose.create_from_pq(torch.tensor([p0])))
            pushed = False
            t_enter = None
            attached = False
            offset = np.zeros(3)
            phase = "approach"
            info = {}
            for t in range(args.max_steps):
                if args.disturb in ("speed", "all") and t > 0 and t % 120 == 0:
                    u.belt_speed = u.belt_speed * float(rng.uniform(0.8, 1.2))
                tcp = u.agent.tcp_pose.p[0].cpu().numpy().astype(np.float64)
                fp = actor.pose.p[0].cpu().numpy().astype(np.float64)
                bx, by = BIN_LAYOUT[kind]
                if t_enter is None and u.pick_zone[0] <= fp[0] <= u.pick_zone[1]:
                    t_enter = t
                if args.disturb in ("push", "all") and not pushed and t_enter is not None:
                    actor._bodies[0].add_force_torque(np.array([0.0, args.push, 0.0]), np.zeros(3))
                    pushed = True

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
                    grip = 1.0

                err = tgt - tcp
                d = float(np.linalg.norm(err))
                if d > 0.08:
                    tgt = tcp + err / d * 0.08
                cmd = np.array([tgt[1] + 0.5, -tgt[0], tgt[2], *HOME_EULER, grip], dtype=np.float32)
                _, _, term, trunc, info = env.step(cmd)
                if args.video and t % 2 == 0:
                    img = env.render()
                    arr = img.cpu().numpy() if hasattr(img, "cpu") else np.asarray(img)
                    arr = arr[0] if arr.ndim == 4 else arr
                    if writer is None:
                        h, w = arr.shape[:2]
                        writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
                    writer.write(arr[..., ::-1])

                # 吸附：位姿跟随
                if attached and phase in ("carry", "place"):
                    tcp2 = u.agent.tcp_pose.p[0].cpu().numpy().astype(np.float64)
                    p = np.clip(tcp2 + offset, -2.5, 2.5)
                    actor.set_pose(Pose.create_from_pq(torch.tensor([p], dtype=torch.float32)))
                    actor.set_linear_velocity(torch.zeros(1, 3))
                    actor.set_angular_velocity(torch.zeros(1, 3))
                # 释放
                if attached and phase == "release":
                    p = np.array([bx, by, 0.08])
                    actor.set_pose(Pose.create_from_pq(torch.tensor([p], dtype=torch.float32)))
                    actor.set_linear_velocity(torch.zeros(1, 3))
                    attached = False
                    phase = "done"

                if bool(info.get("success") or info.get("fail")):
                    break
                if phase == "done" and info.get("success"):
                    break

            ok = bool(info.get("success", False))
            cycle = (t + 1 - t_enter) / args.control_freq if (ok and t_enter is not None) else None
            records.append({"kind": kind, "success": ok, "steps": t + 1, "cycle_s": cycle,
                            "disturb": args.disturb})
            print(f"[{'✓' if ok else '✗'}] {kind:11s} ep {ep:02d} disturb={args.disturb} "
                  f"cycle={None if cycle is None else round(cycle, 2)}s steps={t + 1}")
            env.close()

    if writer is not None:
        writer.release()
    overall = summarize([{"success": r["success"], "duration_s": r["steps"] / args.control_freq} for r in records])
    by_kind = {}
    for k in args.classes:
        rs = [r for r in records if r["kind"] == k]
        by_kind[k] = summarize([{"success": r["success"], "duration_s": r["steps"] / args.control_freq} for r in rs])
    cycles = [r["cycle_s"] for r in records if r["cycle_s"] is not None]
    print("\n吸盘原型:", format_summary(overall))
    if cycles:
        print(f"单件节拍: 均值 {np.mean(cycles):.2f}s  P90 {np.percentile(cycles, 90):.2f}s")
    for k, s in by_kind.items():
        print(f"  {k:11s} {s['n_success']}/{s['n_episodes']} = {s['success_rate']:.0%}")

    out = Path(args.out) if args.out else ROOT / "runs" / f"{time.strftime('%Y-%m-%d-%H%M')}-suction"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"classes": args.classes, "belt": args.belt, "control_freq": args.control_freq,
         "disturb": args.disturb,
         "overall": overall, "by_kind": by_kind, "records": records},
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
