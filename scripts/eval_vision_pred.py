#!/usr/bin/env python
"""预测驱动伺服：把模型的未来预测头接进闭环（AHEAD / TrajWAM 思路）。

- 模型每步输出：目标位姿、未来 H 步的 4 类水果 pos/vel、选择 logits
- 远距（>near）：用模型目标
- 近距：用模型预测的 t+0.1s 水果位置作为伺服目标
- 夹爪：几何判据（论文结论：1 bit 闭爪时刻不交给回归头）
- 搬运/放置：确定性

用法:
    python scripts/eval_vision_pred.py --ckpt runs/vpolicy-sel3/ckpt.pt \
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
SLOT = {"apple": 0, "strawberry": 1, "banana": 2, "pear": 3}
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
    ap.add_argument("--seed", type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--near", type=float, default=0.45)
    ap.add_argument("--frame-ahead", type=int, default=1, help="未来头取第几步（1 步=50ms）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(Path(args.ckpt), map_location="cpu", weights_only=False)
    cfg = PolicyConfig(**ck["cfg"])
    model = MultiViewTemporalPolicy(cfg).to(args.device)
    model.load_state_dict(ck["model"]); model.eval()
    ns = Normalizer.from_state(ck["norm_state"]); na = Normalizer.from_state(ck["norm_act"])
    nf = Normalizer.from_state(ck["norm_fut"])

    results = []
    for ep in range(args.episodes):
        env = gym.make(
            "FruitSort-v1", obs_mode="rgbd", control_mode="pd_ee_pose", sim_backend="cpu",
            robot_uids="panda_wristcam", active_kinds=tuple(args.kinds),
            belt_speed_range=(args.belt_speed, args.belt_speed),
            instruction_kind=None if args.instruction == "all" else args.instruction)
        obs, _ = env.reset(seed=args.seed + ep)
        u = env.unwrapped
        oh_h, wr_h, st_h = [], [], []
        carrying, via_done, carry_steps, close_streak = None, False, 0, 0
        kind = None
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

            with torch.no_grad():
                pred, fut, sel = model(
                    torch.from_numpy(np.stack(oh_h)[None]).float().to(args.device),
                    torch.from_numpy(np.stack(wr_h)[None]).float().to(args.device),
                    torch.from_numpy(ns.obs(np.stack(st_h).astype(np.float32))[None]).float().to(args.device),
                    torch.tensor([INSTR_ID[args.instruction]], device=args.device))
            if kind is None:
                kind = ID_KIND.get(int(sel.argmax(-1).item()), "apple")
                if kind not in u.fruit_actors:
                    kind = args.instruction if args.instruction in u.fruit_actors else list(u.fruit_actors)[0]

            fut_world = nf.unact(fut[0, :, :].cpu().numpy())  # (H, 24)
            fi = min(args.frame_ahead, fut_world.shape[0] - 1)
            slot = SLOT.get(kind, 0)
            fut_pos = fut_world[fi, slot * 6:slot * 6 + 3]

            tcp = u.agent.tcp_pose.p[0].cpu().numpy().astype(np.float64)
            actor = u.fruit_actors[kind]
            fp = actor.pose.p[0].cpu().numpy().astype(np.float64)
            in_zone = bool(u.pick_zone[0] <= fp[0] <= u.pick_zone[1])
            dist = float(np.linalg.norm(fp - tcp))
            grasped = bool(u.agent.is_grasping(actor)[0])

            if carrying is None and close_streak >= 1 and (grasped or (close_streak >= 4 and dist < 0.05)):
                carrying, via_done, carry_steps = kind, False, 0

            tgt, grip = None, 1.0
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

            if tgt is None and dist <= args.near and fp[2] > 0.3:
                # 未来预测驱动伺服（论文的 predict-then-act）
                tgt = fut_pos
                grip = -1.0 if (dist < 0.05 and in_zone) else 1.0
            if tgt is None:
                tgt = np.clip(na.unact(pred[0, 0, :3].cpu().numpy()), -2, 2)
                grip = 1.0

            close_streak = close_streak + 1 if grip < 0 else 0
            obs, _, term, trunc, info = env.step(to_cmd(tgt, tcp, grip))
            if bool(info.get("success") or info.get("fail")):
                break
        ok = bool(info.get("success", False))
        results.append({"success": ok, "duration_s": (t + 1) / 20.0})
        print(f"[{'✓' if ok else '✗'}] ep {ep:02d} instr={args.instruction} steps={t + 1}")
        env.close()

    s = summarize(results)
    print("\n预测驱动伺服:", format_summary(s))
    out = ROOT / "runs" / f"{time.strftime('%Y-%m-%d-%H%M')}-vpred-eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"ckpt": str(args.ckpt), "instruction": args.instruction, "episodes": args.episodes,
         "frame_ahead": args.frame_ahead, "summary": s}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
