#!/usr/bin/env python
"""分拣基准：成功率 + 单件节拍（从水果进入抓取区到放置完成）。

用法:
    python scripts/benchmark_sorting.py --ckpt runs/vpolicy-sel3/ckpt.pt \
        --classes apple strawberry --episodes 10 --belt 0.08 --spawn-x -0.55
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
from fruitsort.policies.hybrid_teacher import HybridTeacher

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--classes", nargs="+", default=["apple", "strawberry"])
    ap.add_argument("--episodes", type=int, default=10, help="每个类别的 episode 数")
    ap.add_argument("--belt", type=float, default=0.08)
    ap.add_argument("--spawn-x", type=float, default=-0.55)
    ap.add_argument("--spawn-spacing", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    ap.add_argument("--policy", choices=["teacher", "scripted", "hybrid"], default="teacher")
    ap.add_argument("--single-class", action="store_true", help="每个 episode 只放被测类别")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    records = []
    for ci, kind in enumerate(args.classes):
        for ep in range(args.episodes):
            env = gym.make(
                "FruitSort-v1", obs_mode="rgbd", control_mode="pd_ee_pose", sim_backend="cpu",
                robot_uids="panda_wristcam",
                active_kinds=(kind,) if args.single_class else tuple(args.classes),
                belt_speed_range=(args.belt, args.belt),
                spawn_x=args.spawn_x, spawn_spacing=args.spawn_spacing,
                instruction_kind=kind)
            obs, _ = env.reset(seed=args.seed + ci * 100 + ep)
            teacher = (HybridTeacher(env, args.ckpt, kind, device=args.device)
                       if args.policy in ("teacher", "hybrid") else None)
            from fruitsort.policies.scripted import ScriptedSorter
            selected = kind
            if args.policy == "hybrid":
                selected = teacher._select_kind(obs)
            scripted = ScriptedSorter(env, only_kind=selected) if args.policy in ("scripted", "hybrid") else None
            u = env.unwrapped
            t_enter, t_end, info = None, None, {}
            for t in range(args.max_steps):
                if args.policy == "teacher":
                    cmd, _ = teacher.act(obs)
                else:
                    cmd = scripted.act()
                fp = u.fruit_actors[kind].pose.p[0].cpu().numpy()
                if t_enter is None and u.pick_zone[0] <= fp[0] <= u.pick_zone[1]:
                    t_enter = t
                obs, _, term, trunc, info = env.step(cmd)
                if bool(info.get("success") or info.get("fail")):
                    t_end = t + 1
                    break
            ok = bool(info.get("success", False))
            cycle = (t_end - t_enter) / 20.0 if (ok and t_enter is not None and t_end is not None) else None
            total = (t_end or args.max_steps) / 20.0
            records.append({"kind": kind, "episode": ep, "success": ok, "selected": selected,
                            "cycle_s": cycle, "total_s": total})
            print(f"[{'✓' if ok else '✗'}] {kind:11s} ep {ep:02d} "
                  f"cycle={cycle if cycle is None else round(cycle, 2)}s total={total:.1f}s")
            env.close()

    by_kind = {}
    for k in args.classes:
        rs = [r for r in records if r["kind"] == k]
        s = summarize([{"success": r["success"], "duration_s": r["total_s"]} for r in rs])
        cycles = [r["cycle_s"] for r in rs if r["cycle_s"] is not None]
        by_kind[k] = {"success": s, "mean_cycle_s": float(np.mean(cycles)) if cycles else None,
                      "p90_cycle_s": float(np.percentile(cycles, 90)) if cycles else None}

    overall = summarize([{"success": r["success"], "duration_s": r["total_s"]} for r in records])
    cycles_all = [r["cycle_s"] for r in records if r["cycle_s"] is not None]
    print("\n" + "=" * 60)
    print("总体:", format_summary(overall))
    if cycles_all:
        print(f"单件节拍: 均值 {np.mean(cycles_all):.2f}s  P90 {np.percentile(cycles_all, 90):.2f}s")
    else:
        print("单件节拍: 无")
    for k, v in by_kind.items():
        cyc = f"{v['mean_cycle_s']:.2f}s" if v["mean_cycle_s"] is not None else "n/a"
        print(f"  {k:11s} {v['success']['n_success']}/{v['success']['n_episodes']} = "
              f"{v['success']['success_rate']:.0%}  节拍 {cyc}")

    out = Path(args.out) if args.out else ROOT / "runs" / f"{time.strftime('%Y-%m-%d-%H%M')}-sort-bench"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"classes": args.classes, "policy": args.policy, "single_class": args.single_class,
         "belt": args.belt, "spawn_x": args.spawn_x,
         "overall": overall, "by_kind": by_kind, "records": records},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("结果:", out / "metrics.json")


if __name__ == "__main__":
    main()
