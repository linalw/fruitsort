#!/usr/bin/env python
"""FruitSort 基准评测。

场景预设:
    per_class  每个类别单独跑，衡量单类抓取/放置能力
    pair       苹果+草莓（默认难度，用于快速回归）
    all4       四类同时出现（完整任务）

用法:
    python scripts/benchmark.py --scenario per_class --episodes 10
    python scripts/benchmark.py --scenario all4 --episodes 20 --belt-speed 0.06
    python scripts/benchmark.py --scenario pair --episodes 5 --video data/eval/pair.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np

import fruitsort.sim.tasks  # noqa: F401
from fruitsort.eval import format_summary, summarize
from fruitsort.policies.scripted import ScriptedSorter

ROOT = Path(__file__).resolve().parent.parent
ALL = ("apple", "strawberry", "banana", "pear")

SCENARIOS = {
    "per_class": [(k,) for k in ALL],
    "pair": [("apple", "strawberry")],
    "all4": [ALL],
}


def run_episode(kinds, seed, belt_speed, max_steps, video=None):
    kw = dict(
        obs_mode="state",
        control_mode="pd_ee_pose",
        sim_backend="cpu",
        active_kinds=tuple(kinds),
        render_mode="rgb_array" if video else None,
    )
    if belt_speed is not None:
        kw["belt_speed_range"] = (belt_speed, belt_speed)
    env = gym.make("FruitSort-v1", **kw)
    env.reset(seed=seed)
    policy = ScriptedSorter(env)
    frames, info, t = [], {}, 0
    for t in range(max_steps):
        env.step(policy.act())
        info = env.unwrapped.evaluate()
        if video and t % 4 == 0:
            img = env.render()
            arr = img.cpu().numpy() if hasattr(img, "cpu") else np.asarray(img)
            frames.append(arr[0] if arr.ndim == 4 else arr)
        if bool(info["success"] or info["fail"]):
            break

    ok = bool(info.get("success", False))
    rec = {
        "kinds": list(kinds),
        "seed": seed,
        "success": ok,
        "steps": t + 1,
        "done": sorted(policy.done),
        "missed": sorted(policy.missed),
        "dropped": bool(info.get("fail", False)),
        "belt_speed": float(env.unwrapped.belt_speed[0]),
    }
    if video and frames:
        h, w = frames[0].shape[:2]
        out = Path(video)
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
        for f in frames:
            writer.write(f[..., ::-1])
        writer.release()
        rec["video"] = str(out)
    env.close()
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="pair")
    ap.add_argument("--episodes", type=int, default=10, help="每个场景组合的 episode 数")
    ap.add_argument("--belt-speed", type=float, default=0.06)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", default="", help="仅对第一个 episode 录制")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tag = f"{time.strftime('%Y-%m-%d-%H%M')}-bench-{args.scenario}"
    out_dir = Path(args.out) if args.out else ROOT / "runs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    combos = SCENARIOS[args.scenario]
    for combo in combos:
        for ep in range(args.episodes):
            seed = args.seed + ep
            rec = run_episode(
                combo, seed, args.belt_speed, args.max_steps,
                video=args.video if (ep == 0 and combo is combos[0]) else None,
            )
            rec["episode"] = ep
            records.append(rec)
            status = "✓" if rec["success"] else "✗"
            print(f"{status} {args.scenario:9s} {'+'.join(combo):28s} seed={seed:<3d} "
                  f"steps={rec['steps']:4d} missed={rec['missed']}")

    def stats(rs):
        return summarize([{"success": r["success"], "duration_s": r["steps"] / 20.0,
                           "failure": (r["missed"][0] if r["missed"] else
                                       ("dropped" if r["dropped"] else None))} for r in rs])

    overall = stats(records)
    per_combo = {}
    for combo in combos:
        key = "+".join(combo)
        rs = [r for r in records if r["kinds"] == list(combo)]
        if rs:
            per_combo[key] = stats(rs)
    per_class = {}
    for k in ALL:
        rs = [r for r in records if r["kinds"] == [k]]
        if rs:
            per_class[k] = stats(rs)

    metrics = {
        "scenario": args.scenario,
        "belt_speed_nominal": args.belt_speed,
        "episodes": len(records),
        "overall": overall,
        "per_combo": per_combo,
        "per_class": per_class,
        "records": records,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "cmd.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"场景 {args.scenario}  带速 {args.belt_speed} m/s")
    print("总体:", format_summary(overall))
    for key, s in per_combo.items():
        print(f"  {key:28s} {format_summary(s).splitlines()[0]}")
    for key, s in per_class.items():
        print(f"  单类 {key:22s} {format_summary(s).splitlines()[0]}")
    print(f"结果: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
