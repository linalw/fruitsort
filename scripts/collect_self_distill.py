#!/usr/bin/env python
"""自蒸馏数据采集：用 5/5 的 teacher 策略跑出的轨迹作为示教。

teacher = 模型选择 + 近距伺服 + 几何夹爪/搬运。它访问到的状态分布，
正是纯模型闭环时需要覆盖的分布（比几何标注器的 DAgger 更贴近）。

用法:
    python scripts/collect_self_distill.py --ckpt runs/vpolicy-sel3/ckpt.pt \
        --instruction apple --episodes 12 --out data/datasets/self-apple
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np

import fruitsort.sim.tasks  # noqa: F401
from fruitsort.policies.hybrid_teacher import HybridTeacher

ROOT = Path(__file__).resolve().parent.parent


def jpg(x):
    return cv2.imencode(".jpg", x[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--instruction", default="apple",
                    choices=["all", "apple", "strawberry", "banana", "pear"])
    ap.add_argument("--kinds", nargs="+", default=["apple", "strawberry"])
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--belt-speed", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out = Path(args.out) if args.out else ROOT / "data" / "datasets" / f"self-{args.instruction}"
    out.mkdir(parents=True, exist_ok=True)
    saved = 0

    for ep in range(args.episodes):
        env = gym.make(
            "FruitSort-v1", obs_mode="rgbd", control_mode="pd_ee_pose", sim_backend="cpu",
            robot_uids="panda_wristcam", active_kinds=tuple(args.kinds),
            belt_speed_range=(args.belt_speed, args.belt_speed),
            instruction_kind=None if args.instruction == "all" else args.instruction)
        obs, _ = env.reset(seed=args.seed + ep)
        teacher = HybridTeacher(env, args.ckpt, args.instruction, device=args.device)
        obs_list, act_list, oh_list, wr_list = [], [], [], []
        info, t = {}, 0
        for t in range(args.max_steps):
            cmd, label = teacher.act(obs)
            sd = obs["sensor_data"]
            obs_list.append(teacher.st[-1])
            act_list.append(label)
            oh_list.append(jpg(sd["overhead"]["rgb"][0].cpu().numpy()))
            wr_list.append(jpg(sd["hand_camera"]["rgb"][0].cpu().numpy()))
            obs, _, term, trunc, info = env.step(cmd)
            if bool(info.get("success") or info.get("fail")):
                break

        ok = bool(info.get("success", False))
        if ok and len(obs_list) > 30:
            np.savez_compressed(
                out / f"ep_{ep:04d}.npz",
                obs=np.stack(obs_list), action=np.stack(act_list),
                seed=args.seed + ep, kinds=np.array(args.kinds),
                instruction=np.array(args.instruction),
                overhead_jpg=np.array(oh_list, dtype=object),
                wrist_jpg=np.array(wr_list, dtype=object))
            saved += 1
        print(f"[{'✓' if ok else '✗'}] ep {ep:02d} steps={t + 1}", flush=True)
        env.close()

    (out / "meta.json").write_text(json.dumps(
        {"instruction": args.instruction, "episodes": saved, "teacher": args.ckpt,
         "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 保存 {saved} 条自蒸馏示教到 {out}")


if __name__ == "__main__":
    main()
