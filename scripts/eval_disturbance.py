#!/usr/bin/env python
"""扰动与泛化评测：横向偏移、带速抖动、外部推力。

用法:
    python scripts/eval_disturbance.py --ckpt runs/vpolicy-4c-v3/ckpt.pt \
        --condition offset --episodes 5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

import fruitsort.sim.tasks  # noqa: F401
from fruitsort.eval import format_summary, summarize
from fruitsort.policies.act_runner import ActPureRunner
from fruitsort.policies.hybrid_teacher import HybridTeacher
from fruitsort.policies.scripted import ScriptedSorter

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--condition", choices=["none", "offset", "speed", "push", "all"], default="all")
    ap.add_argument("--classes", nargs="+", default=["apple", "strawberry"])
    ap.add_argument("--instruction", default="apple")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--belt", type=float, default=0.08)
    ap.add_argument("--spawn-x", type=float, default=-0.55)
    ap.add_argument("--offset", type=float, default=0.12)
    ap.add_argument("--push", type=float, default=1.5, help="推力 (N)，在水果进入抓取区时施加一次")
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--policy", choices=["teacher", "scripted", "act"], default="teacher",
                    help="act = 纯 ACT 策略（无脚本执行器）")
    ap.add_argument("--spawn-spacing", type=float, default=0.6)
    ap.add_argument("--control-freq", type=int, default=20)
    ap.add_argument("--fb", type=float, default=0.0)
    ap.add_argument("--fb-z", action="store_true")
    ap.add_argument("--grip-ema", type=float, default=0.0)
    ap.add_argument("--carry-latch", action="store_true")
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = ap.parse_args()

    conds = ["none", "offset", "speed", "push"] if args.condition == "all" else [args.condition]
    rng = np.random.default_rng(args.seed)
    summary = {}

    for cond in conds:
        results = []
        for ep in range(args.episodes):
            env = gym.make(
                "FruitSort-v1", obs_mode="rgbd", control_mode="pd_ee_pose", sim_backend="cpu",
                robot_uids="panda_wristcam", active_kinds=tuple(args.classes),
                belt_speed_range=(args.belt, args.belt),
                spawn_x=args.spawn_x, spawn_spacing=args.spawn_spacing,
                instruction_kind=args.instruction,
                sim_config=dict(control_freq=args.control_freq,
                                sim_freq=120 if args.control_freq == 30 else 100))
            obs, _ = env.reset(seed=args.seed + ep)
            u = env.unwrapped
            kind = args.instruction
            if args.policy == "teacher":
                policy = HybridTeacher(env, args.ckpt, kind, device=args.device)
            elif args.policy == "act":
                policy = ActPureRunner(
                    env, args.ckpt, kind, device=args.device, fb=args.fb, fb_z=args.fb_z,
                    grip_ema=args.grip_ema, carry_latch=args.carry_latch)
            else:
                policy = ScriptedSorter(env, only_kind=kind)

            # 扰动 1：横向偏移（在带面范围内）
            if cond in ("offset", "all"):
                p = u.fruit_actors[kind].pose.p[0].cpu().numpy()
                dy = rng.uniform(-args.offset, args.offset)
                p[1] = float(np.clip(p[1] + dy, -0.16, 0.16))
                from mani_skill.utils.structs.pose import Pose
                import torch
                u.fruit_actors[kind].set_pose(Pose.create_from_pq(torch.tensor([p])))

            pushed = False
            nominal_belt = float(args.belt)   # 带速扰动围绕标称值抖动（不能连乘，否则会漂出量级）
            info, t = {}, 0
            for t in range(args.max_steps):
                # 扰动 2：带速抖动 ±20%
                if cond in ("speed", "all") and t % 120 == 0 and t > 0:
                    u.belt_speed = u.belt_speed * 0.0 + nominal_belt * float(rng.uniform(0.8, 1.2))
                # 扰动 3：进入抓取区时侧向推一次
                fp = u.fruit_actors[kind].pose.p[0].cpu().numpy()
                if (cond in ("push", "all") and not pushed
                        and u.pick_zone[0] <= fp[0] <= u.pick_zone[1]):
                    body = u.fruit_actors[kind]._bodies[0]
                    body.add_force_torque(np.array([0.0, args.push, 0.0]), np.zeros(3))
                    pushed = True

                if args.policy in ("teacher", "act"):
                    cmd = policy.act(obs) if args.policy == "act" else policy.act(obs)[0]
                else:
                    cmd = policy.act()
                obs, _, term, trunc, info = env.step(cmd)
                if bool(info.get("success") or info.get("fail")):
                    break
            ok = bool(info.get("success", False))
            results.append({"success": ok, "duration_s": (t + 1) / args.control_freq})
            print(f"[{'✓' if ok else '✗'}] {cond:7s} ep {ep}")
            env.close()

        s = summarize(results)
        summary[cond] = s
        print(f"  {cond}: {s['n_success']}/{s['n_episodes']} = {s['success_rate']:.0%}")

    print("\n" + "=" * 50)
    print("扰动评测:", format_summary(summary["none"]) if "none" in summary else "")
    out = ROOT / "runs" / f"{time.strftime('%Y-%m-%d-%H%M')}-disturbance"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"condition": args.condition, "classes": args.classes, "instruction": args.instruction,
         "belt": args.belt, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("结果:", out / "metrics.json")


if __name__ == "__main__":
    main()
