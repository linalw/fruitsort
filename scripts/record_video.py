#!/usr/bin/env python
"""录制模型测试视频（当前 v0.8 模型 + 伺服执行器）。

用法:
    python scripts/record_video.py --ckpt runs/vpolicy-4c-v3/ckpt.pt \
        --classes apple strawberry banana pear --belt 0.08 --out data/eval/model_demo.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
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
    attempt_log = []

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
        zone_entry = None
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
            if zone_entry is None:
                fx = float(env.unwrapped.fruit_actors[kind].pose.p[0, 0])
                if env.unwrapped.pick_zone[0] <= fx <= env.unwrapped.pick_zone[1]:
                    zone_entry = t
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
        attempt_log.append({"kind": kind, "ok": ok, "steps": t + 1})
        if not ok:
            continue
        if frames and writer is None:
                h, w = frames[0].shape[:2]
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
        for f in frames:
            writer.write(f)
        results.append({"kind": kind, "ok": True, "steps": t + 1,
                        "handle_s": ((t + 1 - zone_entry) / args.control_freq)
                        if zone_entry is not None else None})
        break

    if writer is not None:
        writer.release()
    print(f"\n视频: {out}")
    print(f"成功率 {len(results)}/{args.episodes}")

    # ---- 视频说明（每个视频都必须有，否则不知道背景）----
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    ck_meta = {}
    try:
        import torch
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        ck_meta = {"val": ck.get("val"), "params": ck.get("params"),
                   "delta_actions": ck.get("delta_actions"), "cfg": ck.get("cfg")}
    except Exception:  # noqa: BLE001
        pass
    lines = [
        f"演示视频说明 — {out.name}",
        "=" * 60,
        f"生成时间   : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"代码版本   : {commit}",
        "",
        "## 这段视频是什么",
        f"纯策略（非脚本）水果分拣闭环实录：传送带 {args.belt} m/s、平行夹爪、四类同场"
        f"（{', '.join(args.classes)}），每个水果一次分拣。"
        "画面左上角叠加：水果类别 / 带速 / 已用时间 / 策略当前相位"
        "（0=接近，1=下探闭爪，2=搬运）。",
        "",
        "## 模型与部署配置",
        f"权重       : {args.ckpt}",
        f"离线指标   : val={ck_meta.get('val')}, 参数={ck_meta.get('params')}, "
        f"增量动作={ck_meta.get('delta_actions')}",
        f"残差反馈   : fb={args.fb}（target = tcp + Δ_θ + fb·(fruit−TCP)，仅 xy）",
        f"夹爪去抖   : grip_ema={args.grip_ema}, carry_latch={args.carry_latch}",
        f"搬运限速   : carry_max_step={args.carry_max_step}",
        "",
        "## 场景",
        f"带速 {args.belt} m/s | spawn_x={args.spawn_x} | 间距={args.spawn_spacing} m | "
        f"控制频率={args.control_freq} Hz | 机器人=panda_wristcam（7 维位姿 + 夹爪）",
        "",
        f"## 结果：{len(results)}/{args.episodes} 成功",
    ]
    for r in results:
        hs = r.get("handle_s")
        lines.append(f"  - {r['kind']:10s} 成功  整局 {r['steps']} 步 / "
                     f"{r['steps'] / args.control_freq:.1f} s  "
                     f"单件处理时间 {('-' if hs is None else f'{hs:.1f} s')}"
                     "（进入抓取区→入框，不含传送带运送）")
    lines += [
        "",
        "## 复现命令",
        "python scripts/record_video.py --executor act \\",
        f"  --ckpt {args.ckpt} --classes {' '.join(args.classes)} --belt {args.belt} \\",
        f"  --spawn-spacing {args.spawn_spacing} --control-freq {args.control_freq} \\",
        f"  --fb {args.fb} --grip-ema {args.grip_ema} "
        f"{'--carry-latch ' if args.carry_latch else ''}"
        f"{'--carry-max-step ' + str(args.carry_max_step) + ' ' if args.carry_max_step else ''}\\",
        f"  --episodes {args.episodes} --retries {args.retries} --out {args.out}",
        "",
        "## 说明",
        "视频只保留成功片段：每个类最多重试 --retries 次，失败的尝试不会写入视频。",
        "抓取时机、闭爪、搬运方向与释放全部由策略输出；部署端只有残差反馈与去抖/恢复层",
        "（详见 docs/model-architecture.md 6.6）。",
    ]
    txt = out.with_suffix(".txt")
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"说明: {txt}")


if __name__ == "__main__":
    main()
