#!/usr/bin/env python
"""闭环评测 BC 策略。

用法:
    python scripts/eval_bc.py --ckpt runs/xxx-bc/ckpt.pt --kinds apple strawberry --episodes 20
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
from fruitsort.learning.bc import ChunkBC, Normalizer, build_obs
from fruitsort.policies.scripted import HOME_EULER

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kinds", nargs="+", default=["apple", "strawberry"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--belt-speed", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--video", default="")
    ap.add_argument("--smooth", type=float, default=0.6, help="目标位姿 EMA 平滑系数")
    args = ap.parse_args()

    ckpt = torch.load(Path(args.ckpt), map_location="cpu", weights_only=False)
    model = ChunkBC(chunk=ckpt["chunk"], hidden=ckpt["hidden"], hist=ckpt.get("hist", 5))
    model.load_state_dict(ckpt["model"])
    model.eval()
    norm = Normalizer.from_state(ckpt["normalizer"])

    env = gym.make(
        "FruitSort-v1", obs_mode="state", control_mode="pd_ee_pose", sim_backend="cpu",
        active_kinds=tuple(args.kinds), belt_speed_range=(args.belt_speed, args.belt_speed),
        render_mode="rgb_array" if args.video else None,
    )

    results = []
    frames = []
    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        u = env.unwrapped
        info = {}
        history: list[np.ndarray] = []
        prev_tgt = None
        for t in range(args.max_steps):
            o = build_obs(u)
            history.append(o)
            if len(history) > model.hist:
                history.pop(0)
            while len(history) < model.hist:
                history.insert(0, history[0])
            o = np.stack(history).reshape(1, -1)
            with torch.no_grad():
                pred = model(torch.from_numpy(norm.obs(o)).float())[0, 0].numpy()
            pred = norm.unact(pred)
            tgt = np.clip(pred[:3], -2.0, 2.0)
            if prev_tgt is not None:
                tgt = args.smooth * prev_tgt + (1 - args.smooth) * tgt
            prev_tgt = tgt
            grip = float(np.clip(pred[3], -1.0, 1.0))
            tcp = u.agent.tcp_pose.p[0].cpu().numpy()
            err = tgt - tcp
            dist = float(np.linalg.norm(err))
            if dist > 0.06:
                tgt = tcp + err / dist * 0.06
            cmd = np.array([tgt[1] + 0.5, -tgt[0], tgt[2], *HOME_EULER, grip], dtype=np.float32)
            _, _, term, trunc, info = env.step(cmd)
            if args.video and t % 5 == 0:
                img = env.render()
                arr = img.cpu().numpy() if hasattr(img, "cpu") else np.asarray(img)
                frames.append(arr[0] if arr.ndim == 4 else arr)
            if bool(info.get("success") or info.get("fail")):
                break
        ok = bool(info.get("success", False))
        results.append({"success": ok, "duration_s": (t + 1) / 20.0})
        print(f"[{'✓' if ok else '✗'}] ep {ep:02d} steps={t + 1}")

    env.close()
    if args.video and frames:
        import cv2

        out = Path(args.video); out.parent.mkdir(parents=True, exist_ok=True)
        h, w = frames[0].shape[:2]
        wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
        for f in frames:
            wr.write(f[..., ::-1])
        wr.release()
        print("视频:", out)

    s = summarize(results)
    print("\nBC 闭环:", format_summary(s))
    out_dir = ROOT / "runs" / f"{time.strftime('%Y-%m-%d-%H%M')}-bc-eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(
        {"ckpt": str(args.ckpt), "kinds": args.kinds, "episodes": args.episodes,
         "belt_speed": args.belt_speed, "summary": s}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
