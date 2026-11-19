#!/usr/bin/env python
"""采集策略闭环中的**失败/滑脱片段**（带触觉），专门给滑移辅助头提供正样本。

与 `record_demos.py` 的区别
---------------------------
`record_demos` 只存成功示教（滑移帧占比 0.1%），滑移头学不到东西。
这里跑**策略自己的闭环**，把每一帧都存下来（含抓空、搬运滑脱、掉落），
并在 npz 里标记 `aux_only=True`：训练时这些样本只参与滑移辅助损失，
不参与动作/相位的 BC 损失（避免拉低动作精度）。

用法:
    python scripts/collect_slip.py --ckpt runs/act-ta-4c/ckpt.pt \
        --instruction apple --episodes 8 --belt 0.15 --out data/datasets/sdagger-apple-015
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

import fruitsort.sim.tasks  # noqa: F401
from fruitsort.learning.bc import build_tactile, build_obs
from fruitsort.models.act_policy import ActConfig, ActPolicy
from fruitsort.policies.act_runner import ActPureRunner

ROOT = Path(__file__).resolve().parent.parent


def jpg(x):
    return cv2.imencode(".jpg", x[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kinds", nargs="+", default=["apple", "strawberry", "banana", "pear"])
    ap.add_argument("--instruction", default="apple")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--belt", type=float, default=0.15)
    ap.add_argument("--spawn-x", type=float, default=-0.55)
    ap.add_argument("--spawn-spacing", type=float, default=0.45)
    ap.add_argument("--control-freq", type=int, default=30)
    ap.add_argument("--seed", type=int, default=110000)
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    saved, frames_all, slip_frames, fails = 0, 0, 0, 0
    for ep in range(args.episodes):
        env = gym.make(
            "FruitSort-v1", obs_mode="rgbd", control_mode="pd_ee_pose", sim_backend="cpu",
            robot_uids="panda_wristcam", active_kinds=tuple(args.kinds),
            belt_speed_range=(args.belt, args.belt),
            spawn_x=args.spawn_x, spawn_spacing=args.spawn_spacing,
            instruction_kind=args.instruction,
            sim_config=dict(control_freq=args.control_freq,
                            sim_freq=120 if args.control_freq == 30 else 100))
        obs, _ = env.reset(seed=args.seed + ep)
        u = env.unwrapped
        runner = ActPureRunner(env, args.ckpt, args.instruction, device="cpu",
                               fb=0.5, grip_ema=0.6, carry_latch=True, carry_max_step=0.02)
        obs_l, act_l, oh_l, wr_l = [], [], [], []
        info, t = {}, 0
        for t in range(args.max_steps):
            sd = obs["sensor_data"]
            # 与 record_demos 一致：存**原始**触觉（缩放只在训练加载 / 推理时做）
            o = np.concatenate([build_obs(u), build_tactile(u, args.instruction)])
            cmd = runner.act(obs)
            obs_l.append(o)
            act_l.append(np.array([*np.zeros(3), cmd[-1]], dtype=np.float32))
            oh_l.append(jpg(sd["overhead"]["rgb"][0].cpu().numpy()))
            wr_l.append(jpg(sd["hand_camera"]["rgb"][0].cpu().numpy()))
            obs, _, term, trunc, info = env.step(cmd)
            if bool(info.get("success") or info.get("fail")):
                break
        ok = bool(info.get("success", False))
        if len(obs_l) > 20:
            np.savez_compressed(
                out / f"ep_{ep:04d}.npz", obs=np.stack(obs_l), action=np.stack(act_l),
                seed=args.seed + ep, kinds=np.array(args.kinds),
                instruction=np.array(args.instruction), aux_only=np.array(True),
                overhead_jpg=np.array(oh_l, dtype=object),
                wrist_jpg=np.array(wr_l, dtype=object))
            saved += 1
            frames_all += len(obs_l)
            arr = np.stack(obs_l)
            slip_frames += int(((arr[:, 49] > 5.0) & (arr[:, 54] > 0.05)).sum())
        fails += int(not ok)
        print(f"[{'✓' if ok else '✗'}] ep {ep:02d} steps={len(obs_l)}", flush=True)
        env.close()

    (out / "meta.json").write_text(json.dumps(
        {"instruction": args.instruction, "belt": args.belt, "episodes": saved,
         "fail_rate": f"{fails}/{args.episodes}", "frames": frames_all,
         "slip_frames": slip_frames,
         "note": "策略闭环失败片段，仅用于滑移辅助头（aux_only=True）",
         "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"✓ {saved} 局 / {frames_all} 帧，其中滑移帧 {slip_frames}，失败 {fails}/{args.episodes}")


if __name__ == "__main__":
    main()
