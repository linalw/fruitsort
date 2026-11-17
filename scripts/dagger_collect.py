#!/usr/bin/env python
"""DAgger 数据采集：跑当前策略，用 Markov 几何标注器给访问到的状态打标签。

输出格式与 record_demos.py 完全一致，可直接加入训练集。

用法:
    python scripts/dagger_collect.py --ckpt runs/xxx/ckpt.pt \
        --instruction apple --episodes 10 --out data/datasets/dagger-apple
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
from fruitsort.policies.markov import markov_label
from fruitsort.policies.scripted import HOME_EULER

ROOT = Path(__file__).resolve().parent.parent
INSTR_ID = {"all": 0, "apple": 1, "strawberry": 2, "banana": 3, "pear": 4}


def prep(x, size):
    if x.shape[0] != size:
        x = cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
    return (x.astype(np.float32) / 255.0).transpose(2, 0, 1)


def jpg(x):
    return cv2.imencode(".jpg", x[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--instruction", default="apple",
                    choices=["all", "apple", "strawberry", "banana", "pear"])
    ap.add_argument("--kinds", nargs="+", default=["apple", "strawberry"])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--belt-speed", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--smooth", type=float, default=0.3)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(Path(args.ckpt), map_location="cpu", weights_only=False)
    cfg = PolicyConfig(**ck["cfg"])
    model = MultiViewTemporalPolicy(cfg).to(args.device)
    model.load_state_dict(ck["model"]); model.eval()
    ns = Normalizer.from_state(ck["norm_state"]); na = Normalizer.from_state(ck["norm_act"])

    out = Path(args.out) if args.out else ROOT / "data" / "datasets" / f"dagger-{args.instruction}"
    out.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        "FruitSort-v1", obs_mode="rgbd", control_mode="pd_ee_pose", sim_backend="cpu",
        robot_uids="panda_wristcam", active_kinds=tuple(args.kinds),
        belt_speed_range=(args.belt_speed, args.belt_speed),
        instruction_kind=None if args.instruction == "all" else args.instruction,
    )
    kid = INSTR_ID[args.instruction]
    saved = 0
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        u = env.unwrapped
        ohs, wrs, sts = [], [], []
        obs_list, act_list, oh_list, wr_list = [], [], [], []
        prev_tgt = None
        for t in range(args.max_steps):
            sd = obs["sensor_data"]
            oh = prep(sd["overhead"]["rgb"][0].cpu().numpy(), cfg.img_size)
            wr = prep(sd["hand_camera"]["rgb"][0].cpu().numpy(), cfg.img_size)
            st = build_obs(u)
            ohs.append(oh); wrs.append(wr); sts.append(st)
            for h in (ohs, wrs, sts):
                if len(h) > cfg.hist:
                    h.pop(0)
            while len(ohs) < cfg.hist:
                ohs.insert(0, ohs[0]); wrs.insert(0, wrs[0]); sts.insert(0, sts[0])

            with torch.no_grad():
                pred, _, _ = model(
                    torch.from_numpy(np.stack(ohs)[None]).float().to(args.device),
                    torch.from_numpy(np.stack(wrs)[None]).float().to(args.device),
                    torch.from_numpy(ns.obs(np.stack(sts).astype(np.float32))[None]).float().to(args.device),
                    torch.tensor([kid], device=args.device))
            p4 = pred[0, 0].cpu().numpy()
            tgt = np.clip(na.unact(p4[:3]), -2, 2)
            grip = -1.0 if p4[3] > 0 else 1.0  # logit>0 = 闭爪
            if prev_tgt is not None:
                tgt = args.smooth * prev_tgt + (1 - args.smooth) * tgt
            prev_tgt = tgt

            # 标签：Markov 几何标注
            lab_t, lab_g = markov_label(u, None if args.instruction == "all" else args.instruction)
            obs_list.append(st)
            act_list.append(np.array([*lab_t, lab_g], dtype=np.float32))
            oh_list.append(jpg(sd["overhead"]["rgb"][0].cpu().numpy()))
            wr_list.append(jpg(sd["hand_camera"]["rgb"][0].cpu().numpy()))

            tcp = u.agent.tcp_pose.p[0].cpu().numpy()
            err = tgt - tcp; d = np.linalg.norm(err)
            if d > 0.06:
                tgt = tcp + err / d * 0.06
            cmd = np.array([tgt[1] + 0.5, -tgt[0], tgt[2], *HOME_EULER, grip], dtype=np.float32)
            obs, _, term, trunc, info = env.step(cmd)
            if bool(info.get("success") or info.get("fail")):
                break

        if len(obs_list) > 30:
            f = out / f"ep_{ep:04d}.npz"
            np.savez_compressed(f, obs=np.stack(obs_list), action=np.stack(act_list),
                                seed=args.seed + ep, kinds=np.array(args.kinds),
                                instruction=np.array(args.instruction),
                                overhead_jpg=np.array(oh_list, dtype=object),
                                wrist_jpg=np.array(wr_list, dtype=object))
            saved += 1
        print(f"[{'✓' if info.get('success') else '✗'}] ep {ep:02d} steps={len(obs_list)}", flush=True)

    env.close()
    meta = {"instruction": args.instruction, "episodes": saved, "source": str(args.ckpt),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 保存 {saved} 条 DAgger 数据到 {out}")


if __name__ == "__main__":
    main()
