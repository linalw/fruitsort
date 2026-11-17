#!/usr/bin/env python
"""纯视觉策略（模型选择 + 几何伺服/夹爪/搬运）——诊断实验的最优配置。

来源：`/tmp/vision_diag/` 的 AF_blend045_early_l 配置（5/5）。
- 远距（>0.45 m）用模型目标
- 近距用 水果位置 + v*0.10
- 夹爪：d<0.05 且水果在抓取区 → 闭
- 搬运/放置：确定性（via → bin → release）

用法:
    python scripts/eval_vision_geo.py --ckpt runs/vpolicy-sel3/ckpt.pt \
        --instruction apple --episodes 5
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
from fruitsort.policies.scripted import HOME_EULER
from fruitsort.sim.conveyor import BIN_LAYOUT

ROOT = Path(__file__).resolve().parent.parent
INSTR_ID = {"all": 0, "apple": 1, "strawberry": 2, "banana": 3, "pear": 4}
ID_KIND = {0: "apple", 1: "strawberry", 2: "banana", 3: "pear"}
MAX_STEP = 0.06


def prep_img(x, size):
    if x.shape[0] != size:
        x = cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
    return (x.astype(np.float32) / 255.0).transpose(2, 0, 1)


def to_cmd(tgt, tcp, grip):
    tgt = np.asarray(tgt, dtype=np.float64).copy()
    err = tgt - tcp
    d = float(np.linalg.norm(err))
    if d > MAX_STEP:
        tgt = tcp + err / d * MAX_STEP
    return np.array([tgt[1] + 0.5, -tgt[0], tgt[2], *HOME_EULER, float(grip)], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kinds", nargs="+", default=["apple", "strawberry"])
    ap.add_argument("--instruction", default="apple",
                    choices=["all", "apple", "strawberry", "banana", "pear"])
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--belt-speed", type=float, default=0.06)
    ap.add_argument("--spawn-x", type=float, default=-0.75)
    ap.add_argument("--spawn-spacing", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--near", type=float, default=0.45, help="<=0 表示禁用近距伺服，全用模型")
    ap.add_argument("--grip", choices=["geo", "model"], default="geo")
    ap.add_argument("--delta", action="store_true", help="模型输出相对 TCP 的增量")
    ap.add_argument("--lead", type=float, default=0.10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(Path(args.ckpt), map_location="cpu", weights_only=False)
    cfg = PolicyConfig(**ck["cfg"])
    model = MultiViewTemporalPolicy(cfg).to(args.device)
    model.load_state_dict(ck["model"]); model.eval()
    ns = Normalizer.from_state(ck["norm_state"]); na = Normalizer.from_state(ck["norm_act"])

    results, selections = [], []
    for ep in range(args.episodes):
        env = gym.make(
            "FruitSort-v1", obs_mode="rgbd", control_mode="pd_ee_pose", sim_backend="cpu",
            robot_uids="panda_wristcam", active_kinds=tuple(args.kinds),
            belt_speed_range=(args.belt_speed, args.belt_speed),
            spawn_x=args.spawn_x, spawn_spacing=args.spawn_spacing,
            instruction_kind=None if args.instruction == "all" else args.instruction)
        obs, _ = env.reset(seed=args.seed + ep)
        u = env.unwrapped
        # 用模型做一次指令选择（决定抓哪颗；选择头离线 100%）
        sd0 = obs["sensor_data"]
        _oh = prep_img(sd0["overhead"]["rgb"][0].cpu().numpy(), cfg.img_size)
        _wr = prep_img(sd0["hand_camera"]["rgb"][0].cpu().numpy(), cfg.img_size)
        _st = build_obs(u)
        with torch.no_grad():
            _, _, _sel = model(
                torch.from_numpy(np.stack([_oh] * cfg.hist)[None]).float().to(args.device),
                torch.from_numpy(np.stack([_wr] * cfg.hist)[None]).float().to(args.device),
                torch.from_numpy(ns.obs(np.stack([_st] * cfg.hist).astype(np.float32))[None]).float().to(args.device),
                torch.tensor([INSTR_ID[args.instruction]], device=args.device))
        selected_kind = ID_KIND.get(int(_sel.argmax(-1).item()), "apple")
        if selected_kind not in u.fruit_actors:
            selected_kind = args.instruction if args.instruction in u.fruit_actors else list(u.fruit_actors)[0]
        oh_h, wr_h, st_h = [], [], []
        carrying, via_done, carry_steps, close_streak = None, False, 0, 0
        prev_tgt = None
        info, t = {}, 0
        for t in range(args.max_steps):
            sd = obs["sensor_data"]
            oh_h.append(prep_img(sd["overhead"]["rgb"][0].cpu().numpy(), cfg.img_size))
            wr_h.append(prep_img(sd["hand_camera"]["rgb"][0].cpu().numpy(), cfg.img_size))
            st_h.append(build_obs(u))
            for h in (oh_h, wr_h, st_h):
                if len(h) > cfg.hist:
                    h.pop(0)
            while len(oh_h) < cfg.hist:
                oh_h.insert(0, oh_h[0]); wr_h.insert(0, wr_h[0]); st_h.insert(0, st_h[0])

            tcp = u.agent.tcp_pose.p[0].cpu().numpy().astype(np.float64)
            kind = selected_kind
            actor = u.fruit_actors[kind]
            fp = actor.pose.p[0].cpu().numpy().astype(np.float64)
            fv = actor.linear_velocity[0].cpu().numpy().astype(np.float64)
            in_zone = bool(u.pick_zone[0] <= fp[0] <= u.pick_zone[1])
            dist = float(np.linalg.norm(fp - tcp))
            grasped = bool(u.agent.is_grasping(actor)[0])

            # 抓取锁存
            if carrying is None and close_streak >= 1 and (grasped or (close_streak >= 4 and dist < 0.05)):
                carrying, via_done, carry_steps = kind, False, 0

            if carrying is not None:
                carry_steps += 1
                bx, by = BIN_LAYOUT[carrying]
                if dist > 0.10 or carry_steps > 200:
                    carrying = None
                elif not via_done and (abs(tcp[0]) > 0.06 or abs(tcp[1] + 0.34) > 0.06):
                    tgt, grip = np.array([0.0, -0.34, 0.60]), -1.0
                else:
                    via_done = True
                    over = abs(tcp[0] - bx) < 0.04 and abs(tcp[1] - by) < 0.04
                    if over and tcp[2] < 0.53:
                        tgt, grip = np.array([bx, by, 0.60]), 1.0
                    else:
                        tgt, grip = np.array([bx, by, 0.60 if not over else 0.48]), -1.0
            elif args.near > 0 and dist <= args.near and fp[2] > 0.3:
                tgt = fp + fv * args.lead
                grip = -1.0 if (dist < 0.05 and in_zone) else 1.0
            else:
                oh_t = torch.from_numpy(np.stack(oh_h)[None]).float().to(args.device)
                wr_t = torch.from_numpy(np.stack(wr_h)[None]).float().to(args.device)
                st_t = torch.from_numpy(ns.obs(np.stack(st_h).astype(np.float32))[None]).float().to(args.device)
                with torch.no_grad():
                    pred, _, sel = model(oh_t, wr_t, st_t,
                                         torch.tensor([INSTR_ID[args.instruction]], device=args.device))
                p4 = pred[0, 0].cpu().numpy()
                tgt = na.unact(p4[:3])
                if args.delta:
                    tgt = tcp + tgt
                tgt = np.clip(tgt, -2, 2)
                grip = (-1.0 if p4[3] > 0 else 1.0) if args.grip == "model" else 1.0
                selections.append(int(sel[0].argmax().item()))

            close_streak = close_streak + 1 if grip < 0 else 0
            obs, _, term, trunc, info = env.step(to_cmd(tgt, tcp, grip))
            if bool(info.get("success") or info.get("fail")):
                break
        ok = bool(info.get("success", False))
        results.append({"success": ok, "duration_s": (t + 1) / 20.0})
        print(f"[{'✓' if ok else '✗'}] ep {ep:02d} instr={args.instruction} steps={t + 1}")
        env.close()

    s = summarize(results)
    print("\n纯视觉（模型+几何）:", format_summary(s))
    out = ROOT / "runs" / f"{time.strftime('%Y-%m-%d-%H%M')}-vgeo-eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"ckpt": str(args.ckpt), "instruction": args.instruction, "episodes": args.episodes,
         "near": args.near, "lead": args.lead, "delta": args.delta, "summary": s}, ensure_ascii=False, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
