#!/usr/bin/env python
"""ACT 策略的纯闭环评测（平行夹爪，无吸盘、无脚本执行器）。

动作（目标位姿 + 夹爪开合 + 何时搬运）全部由模型给出；部署端只叠加
残差反馈与去抖/恢复层，见 `src/fruitsort/policies/act_runner.py` 的说明。

用法:
    python scripts/eval_act_pure.py --ckpt runs/act-008-4c-v2/ckpt.pt \
        --instruction apple --episodes 10 --belt 0.08 --spawn-spacing 0.35 \
        --control-freq 30 --fb 0.5 --grip-ema 0.6 --carry-latch
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import fruitsort.sim.tasks  # noqa: F401
from fruitsort.eval import format_summary, summarize
from fruitsort.policies.act_runner import ActPureRunner

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kinds", nargs="+", default=["apple", "strawberry", "banana", "pear"])
    ap.add_argument("--instruction", default="apple")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--belt", type=float, default=0.25)
    ap.add_argument("--control-freq", type=int, default=30)
    ap.add_argument("--spawn-x", type=float, default=-0.55)
    ap.add_argument("--spawn-spacing", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=15000)
    ap.add_argument("--max-steps", type=int, default=900)
    # ---- 部署配置（v1.4 达标配置）----
    ap.add_argument("--fb", type=float, default=0.0,
                    help="残差反馈增益：target += fb*clip(水果-TCP)。0=纯策略；>0 时策略给前馈、反馈收残差")
    ap.add_argument("--fb-z", action="store_true", help="反馈项也作用于 z（默认只作用 xy）")
    ap.add_argument("--grip-ema", type=float, default=0.0, help="夹爪 logit 平滑系数（0=关闭）")
    ap.add_argument("--carry-latch", action="store_true", help="搬运段连续 3 步预测开爪才松手")
    ap.add_argument("--descend-commit", type=int, default=0, help="下探承诺步数（相位迟滞）")
    ap.add_argument("--hold-latch", action="store_true", help="闭爪期间锁存目标（停住等水果）")
    ap.add_argument("--lead", type=float, default=0.0, help="前馈超前（秒）：目标 x += v*lead")
    ap.add_argument("--z-gate", action="store_true")
    ap.add_argument("--chunk-exec", type=int, default=1, help="每 s 步重规划一次（增量模式下有意义）")
    ap.add_argument("--no-phase-fb", action="store_true", help="相位输入恒为 0（消融）")
    ap.add_argument("--max-step", type=float, default=0.06)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    results = []
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
        kind = args.instruction
        runner = ActPureRunner(
            env, args.ckpt, kind, device=args.device, fb=args.fb, fb_z=args.fb_z,
            grip_ema=args.grip_ema, carry_latch=args.carry_latch,
            descend_commit=args.descend_commit, max_step=args.max_step, lead=args.lead,
            z_gate=args.z_gate, phase_fb=not args.no_phase_fb,
            hold_latch=args.hold_latch, chunk_exec=args.chunk_exec)
        zone_entry = None
        info, t = {}, 0
        for t in range(args.max_steps):
            actor = u.fruit_actors.get(kind)
            if zone_entry is None and actor is not None:
                fx = float(actor.pose.p[0, 0])
                if u.pick_zone[0] <= fx <= u.pick_zone[1]:
                    zone_entry = t
            cmd = runner.act(obs)
            obs, _, term, trunc, info = env.step(cmd)
            if bool(info.get("success") or info.get("fail")):
                break
        ok = bool(info.get("success", False))
        handle_s = ((t + 1 - zone_entry) / args.control_freq) if zone_entry is not None else None
        results.append({"success": ok, "duration_s": (t + 1) / args.control_freq,
                        "handle_s": handle_s})
        st = runner.stats
        print(f"[{'✓' if ok else '✗'}] ep {ep:02d} kind={kind} steps={t + 1} "
              f"handle={'-' if handle_s is None else f'{handle_s:.1f}s'} "
              f"close={st['close'] / max(st['steps'], 1):.2f} "
              f"descend={st['descend']} carry={st['carry']}")
        env.close()

    s = summarize(results)
    print("\nACT 纯策略:", format_summary(s))
    hs = [r["handle_s"] for r in results if r["success"] and r["handle_s"] is not None]
    if hs:
        print(f"单件处理时间（进入抓取区→入框）: 均值 {np.mean(hs):.2f}s  P90 "
              f"{np.percentile(hs, 90):.2f}s  最大 {np.max(hs):.2f}s")
    out = ROOT / "runs" / f"{time.strftime('%Y-%m-%d-%H%M')}-act-pure"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"ckpt": str(args.ckpt), "instruction": args.instruction, "episodes": args.episodes,
         "belt": args.belt, "summary": s, "args": vars(args)}, ensure_ascii=False, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
