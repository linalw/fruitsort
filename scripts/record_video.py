#!/usr/bin/env python
"""录制模型测试视频（当前 v0.8 模型 + 伺服执行器）。

用法:
    python scripts/record_video.py --ckpt runs/vpolicy-4c-v3/ckpt.pt \
        --classes apple strawberry banana pear --belt 0.08 --out data/eval/model_demo.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np

import fruitsort.sim.tasks  # noqa: F401
from fruitsort.policies.act_runner import ActPureRunner
from fruitsort.policies.hybrid_teacher import HybridTeacher

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--classes", nargs="+", default=["apple", "strawberry", "banana", "pear"])
    ap.add_argument("--belt", type=float, default=0.08)
    ap.add_argument("--spawn-x", type=float, default=-0.55)
    ap.add_argument("--spawn-spacing", type=float, default=0.6)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--out", default="data/eval/model_demo.mp4")
    ap.add_argument("--control-freq", type=int, default=20)
    ap.add_argument("--executor", choices=["teacher", "hybrid", "act"], default="hybrid",
                    help="act = 纯 ACT 策略（无脚本执行器，v1.6 配置）")
    ap.add_argument("--fb", type=float, default=0.0)
    ap.add_argument("--grip-ema", type=float, default=0.0)
    ap.add_argument("--carry-latch", action="store_true")
    ap.add_argument("--carry-max-step", type=float, default=0.0)
    ap.add_argument("--retries", type=int, default=3, help="每类最多尝试几次（只为录成功片段）")
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    results = []

    cursor = 0
    for ep in range(args.episodes):
      kind = args.classes[ep % len(args.classes)]
      for attempt in range(args.retries):
        env = gym.make(
            "FruitSort-v1", obs_mode="rgbd", control_mode="pd_ee_pose", sim_backend="cpu",
            robot_uids="panda_wristcam", active_kinds=tuple(args.classes),
            belt_speed_range=(args.belt, args.belt),
            spawn_x=args.spawn_x, spawn_spacing=args.spawn_spacing,
            instruction_kind=kind, render_mode="rgb_array",
            sim_config=dict(control_freq=args.control_freq,
                            sim_freq=120 if args.control_freq == 30 else 100))
        obs, _ = env.reset(seed=args.seed + ep * 10 + attempt)
        info, frames = {}, []
        if args.executor == "act":
            runner = ActPureRunner(
                env, args.ckpt, kind, device=args.device, fb=args.fb,
                grip_ema=args.grip_ema, carry_latch=args.carry_latch,
                carry_max_step=args.carry_max_step)
        else:
            teacher = HybridTeacher(env, args.ckpt, kind, device=args.device)
            from fruitsort.policies.scripted import ScriptedSorter
            selected = teacher._select_kind(obs) if args.executor == "hybrid" else kind
            scripted = ScriptedSorter(env, only_kind=selected) if args.executor == "hybrid" else None
            runner = None
        for t in range(args.max_steps):
            if args.executor == "hybrid":
                cmd = scripted.act()
            elif args.executor == "act":
                cmd = runner.act(obs)
            else:
                cmd, _ = teacher.act(obs)
            obs, _, term, trunc, info = env.step(cmd)
            if t % 2 == 0:
                img = env.render()
                arr = img.cpu().numpy() if hasattr(img, "cpu") else np.asarray(img)
                arr = arr[0] if arr.ndim == 4 else arr
                arr = np.ascontiguousarray(arr[..., ::-1].astype(np.uint8))
                label = f"{kind}  belt={args.belt} m/s  t={t / args.control_freq:.1f}s"
                if runner is not None:
                    label += f"  phase={runner.phase_state}"
                cv2.putText(arr, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA)
                frames.append(arr)
            if bool(info.get("success") or info.get("fail")):
                break
        ok = bool(info.get("success", False))
        env.close()
        print(f"[{'✓' if ok else '✗'}] ep {ep} {kind} steps={t + 1} (attempt {attempt + 1})")
        if not ok:
            continue
        if frames and writer is None:
                h, w = frames[0].shape[:2]
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
        for f in frames:
            writer.write(f)
        results.append(True)
        break

    if writer is not None:
        writer.release()
    print(f"\n视频: {out}")
    print(f"成功率 {sum(results)}/{len(results)}")


if __name__ == "__main__":
    main()
