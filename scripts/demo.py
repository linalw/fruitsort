#!/usr/bin/env python
"""运行 FruitSort 仿真并录制视频。

用法:
    python scripts/demo.py --kinds apple --belt-speed 0 --video data/eval/demo_static.mp4
    python scripts/demo.py --kinds apple strawberry --video data/eval/demo_dynamic.mp4
    python scripts/demo.py --episodes 5 --seed 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np

import fruitsort.sim.tasks  # noqa: F401  (注册 FruitSort-v1)
from fruitsort.policies.scripted import ScriptedSorter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["apple", "strawberry", "banana", "pear"])
    ap.add_argument("--belt-speed", type=float, default=None, help="固定带速；缺省用随机区间")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--video", default="")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--render-every", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    speed = args.belt_speed
    kw = dict(
        obs_mode="state",
        control_mode="pd_ee_pose",
        sim_backend=args.device,
        active_kinds=tuple(args.kinds),
        render_mode="rgb_array",
    )
    if speed is not None:
        kw["belt_speed_range"] = (speed, speed)
    env = gym.make("FruitSort-v1", **kw)

    writer = None
    results = []
    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        policy = ScriptedSorter(env)
        frames, info = [], {}
        for t in range(args.max_steps):
            env.step(policy.act())
            info = env.unwrapped.evaluate()
            if args.video and t % args.render_every == 0:
                img = env.render()
                arr = img.cpu().numpy() if hasattr(img, "cpu") else np.asarray(img)
                frames.append(arr[0] if arr.ndim == 4 else arr)
            if bool(info["success"] or info["fail"]):
                break
        ok = bool(info.get("success", False))
        results.append((ok, t + 1))
        print(f"[ep {ep}] success={ok} steps={t + 1} done={sorted(policy.done)} missed={sorted(policy.missed)}")
        if args.video and frames:
            h, w = frames[0].shape[:2]
            out = Path(args.video) if args.episodes == 1 else Path(args.video).with_name(
                f"{Path(args.video).stem}_ep{ep}.mp4")
            out.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
            for f in frames:
                writer.write(f[..., ::-1])
            writer.release()
            print(f"  视频: {out}")

    env.close()
    n_ok = sum(r[0] for r in results)
    print(f"\n成功率 {n_ok}/{len(results)}，平均步数 {np.mean([r[1] for r in results]):.0f}")


if __name__ == "__main__":
    main()
