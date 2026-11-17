#!/usr/bin/env python
"""指令选择准确率评测：同一帧下切换指令，检查选择头是否指向对应水果。

用法:
    python scripts/eval_selection.py --ckpt runs/vpolicy-sel3/ckpt.pt --kinds apple strawberry banana pear
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
from fruitsort.learning.bc import Normalizer, build_obs
from fruitsort.models.vision_policy import MultiViewTemporalPolicy, PolicyConfig

ROOT = Path(__file__).resolve().parent.parent
KIND_ID = {"apple": 1, "strawberry": 2, "banana": 3, "pear": 4}
SLOT = {"apple": 0, "strawberry": 1, "banana": 2, "pear": 3}


def prep(x, size):
    if x.shape[0] != size:
        x = cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
    return (x.astype(np.float32) / 255.0).transpose(2, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kinds", nargs="+", default=["apple", "strawberry", "banana", "pear"])
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--belt-speed", type=float, default=0.06)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(Path(args.ckpt), map_location="cpu", weights_only=False)
    cfg = PolicyConfig(**ck["cfg"])
    model = MultiViewTemporalPolicy(cfg).to(args.device)
    model.load_state_dict(ck["model"]); model.eval()
    ns = Normalizer.from_state(ck["norm_state"])

    env = gym.make("FruitSort-v1", obs_mode="rgbd", control_mode="pd_ee_pose",
                   sim_backend="cpu", robot_uids="panda_wristcam",
                   active_kinds=tuple(args.kinds), belt_speed_range=(args.belt_speed, args.belt_speed))
    obs, _ = env.reset(seed=args.seed)
    u = env.unwrapped

    ohs, wrs, sts = [], [], []
    counts = {k: [0, 0] for k in args.kinds}  # correct, total
    for t in range(args.steps):
        sd = obs["sensor_data"]
        ohs.append(prep(sd["overhead"]["rgb"][0].cpu().numpy(), cfg.img_size))
        wrs.append(prep(sd["hand_camera"]["rgb"][0].cpu().numpy(), cfg.img_size))
        sts.append(build_obs(u))
        for h in (ohs, wrs, sts):
            if len(h) > cfg.hist:
                h.pop(0)
        while len(ohs) < cfg.hist:
            ohs.insert(0, ohs[0]); wrs.insert(0, wrs[0]); sts.insert(0, sts[0])

        if t > 20 and t % 4 == 0:
            oh_t = torch.from_numpy(np.stack(ohs)[None]).float().to(args.device)
            wr_t = torch.from_numpy(np.stack(wrs)[None]).float().to(args.device)
            st_t = torch.from_numpy(ns.obs(np.stack(sts).astype(np.float32))[None]).float().to(args.device)
            for k in args.kinds:
                with torch.no_grad():
                    _, _, logits = model(oh_t, wr_t, st_t,
                                         torch.tensor([KIND_ID[k]], device=args.device))
                pred = int(logits.argmax(-1).item())
                counts[k][0] += int(pred == SLOT[k])
                counts[k][1] += 1
        obs, _, _, _, _ = env.step(np.zeros(7))
    env.close()

    total_c = sum(c for c, _ in counts.values())
    total_n = sum(n for _, n in counts.values())
    print(f"总体指令选择: {total_c}/{total_n} = {total_c / max(total_n, 1):.1%}")
    for k, (c, n) in counts.items():
        print(f"  {k:11s} {c}/{n} = {c / max(n, 1):.1%}")

    out = ROOT / "runs" / f"{time.strftime('%Y-%m-%d-%H%M')}-selection"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"ckpt": str(args.ckpt), "kinds": args.kinds,
         "overall": [total_c, total_n], "per_kind": counts}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
