#!/usr/bin/env python
"""分层执行评测：模型负责"按指令选水果"，脚本专家负责抓取/放置。

这是把「指令理解」与「低层控制」解耦的最小闭环，用于验证模型的指令选择能力
在真实任务里的端到端贡献。

用法:
    python scripts/eval_hybrid.py --ckpt runs/vpolicy-sel3/ckpt.pt --episodes 10
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
from fruitsort.eval import format_summary, summarize
from fruitsort.learning.bc import Normalizer, build_obs
from fruitsort.models.vision_policy import MultiViewTemporalPolicy, PolicyConfig
from fruitsort.policies.scripted import ScriptedSorter

ROOT = Path(__file__).resolve().parent.parent
INSTR_ID = {"apple": 1, "strawberry": 2, "banana": 3, "pear": 4}
ID_KIND = {0: "apple", 1: "strawberry", 2: "banana", 3: "pear"}


def prep(x, size):
    if x.shape[0] != size:
        x = cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
    return (x.astype(np.float32) / 255.0).transpose(2, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kinds", nargs="+", default=["apple", "strawberry"])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--belt-speed", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=700)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(Path(args.ckpt), map_location="cpu", weights_only=False)
    cfg = PolicyConfig(**ck["cfg"])
    model = MultiViewTemporalPolicy(cfg).to(args.device)
    model.load_state_dict(ck["model"]); model.eval()
    ns = Normalizer.from_state(ck["norm_state"])

    results = []
    correct = 0
    for ep in range(args.episodes):
        instr = args.kinds[ep % len(args.kinds)]
        env = gym.make("FruitSort-v1", obs_mode="rgbd", control_mode="pd_ee_pose",
                       sim_backend="cpu", robot_uids="panda_wristcam",
                       active_kinds=tuple(args.kinds), belt_speed_range=(args.belt_speed, args.belt_speed),
                       instruction_kind=instr)
        obs, _ = env.reset(seed=args.seed + ep)
        u = env.unwrapped
        # 用模型做一次指令选择（取第一帧，构造 hist 帧）
        sd = obs["sensor_data"]
        oh = prep(sd["overhead"]["rgb"][0].cpu().numpy(), cfg.img_size)
        wr = prep(sd["hand_camera"]["rgb"][0].cpu().numpy(), cfg.img_size)
        st = build_obs(u)
        oh_t = torch.from_numpy(np.stack([oh] * cfg.hist)[None]).float().to(args.device)
        wr_t = torch.from_numpy(np.stack([wr] * cfg.hist)[None]).float().to(args.device)
        st_t = torch.from_numpy(ns.obs(np.stack([st] * cfg.hist).astype(np.float32))[None]).float().to(args.device)
        with torch.no_grad():
            _, _, logits = model(oh_t, wr_t, st_t, torch.tensor([INSTR_ID[instr]], device=args.device))
        slot = int(logits.argmax(-1).item())
        chosen = ID_KIND.get(slot, instr)
        correct += int(chosen == instr)

        policy = ScriptedSorter(env, only_kind=chosen)
        info = {}
        for t in range(args.max_steps):
            _, _, term, trunc, info = env.step(policy.act())
            if bool(info.get("success") or info.get("fail")):
                break
        ok = bool(info.get("success", False))
        results.append({"success": ok, "duration_s": (t + 1) / 20.0})
        print(f"[{'✓' if ok else '✗'}] ep {ep:02d} instr={instr} chosen={chosen} steps={t + 1}")

        env.close()
    s = summarize(results)
    print(f"\n指令选择: {correct}/{args.episodes} = {correct/args.episodes:.0%}")
    print("端到端:", format_summary(s))
    out_dir = ROOT / "runs" / f"{time.strftime('%Y-%m-%d-%H%M')}-hybrid-eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(
        {"ckpt": str(args.ckpt), "episodes": args.episodes, "belt_speed": args.belt_speed,
         "selection_acc": correct / args.episodes, "summary": s}, ensure_ascii=False, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
