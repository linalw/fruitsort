#!/usr/bin/env python
"""闭环评测多视角时序策略。

用法:
    python scripts/eval_vision_policy.py --ckpt runs/xxx-vpolicy/ckpt.pt \
        --kinds apple strawberry --episodes 10 --belt-speed 0.06
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


def prep_img(x: np.ndarray) -> np.ndarray:
    """HWC uint8 → (3,H,W) float [0,1]，并缩放到 128。"""
    if x.shape[0] != 128:
        x = cv2.resize(x, (128, 128), interpolation=cv2.INTER_AREA)
    return (x.astype(np.float32) / 255.0).transpose(2, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kinds", nargs="+", default=["apple", "strawberry"])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--belt-speed", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--smooth", type=float, default=0.0)
    ap.add_argument("--video", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--instruction", default="apple",
                    choices=["all", "apple", "strawberry", "banana", "pear"])
    args = ap.parse_args()

    ck = torch.load(Path(args.ckpt), map_location="cpu", weights_only=False)
    cfg = PolicyConfig(**ck["cfg"])
    model = MultiViewTemporalPolicy(cfg).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    ns = Normalizer.from_state(ck["norm_state"])
    na = Normalizer.from_state(ck["norm_act"])

    env = gym.make(
        "FruitSort-v1", obs_mode="rgbd", control_mode="pd_ee_pose", sim_backend="cpu",
        robot_uids="panda_wristcam", active_kinds=tuple(args.kinds),
        belt_speed_range=(args.belt_speed, args.belt_speed),
        instruction_kind=None if args.instruction == "all" else args.instruction,
        render_mode="rgb_array" if args.video else None,
    )

    results, frames = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        u = env.unwrapped
        oh_hist, wr_hist, st_hist = [], [], []
        prev_tgt = None
        carrying = None
        info = {}
        for t in range(args.max_steps):
            sd = obs["sensor_data"]
            oh = prep_img(sd["overhead"]["rgb"][0].cpu().numpy())
            wr = prep_img(sd["hand_camera"]["rgb"][0].cpu().numpy())
            st = build_obs(u)
            oh_hist.append(oh); wr_hist.append(wr); st_hist.append(st)
            for h in (oh_hist, wr_hist, st_hist):
                if len(h) > cfg.hist:
                    h.pop(0)
            while len(oh_hist) < cfg.hist:
                oh_hist.insert(0, oh_hist[0]); wr_hist.insert(0, wr_hist[0]); st_hist.insert(0, st_hist[0])

            oh_t = torch.from_numpy(np.stack(oh_hist)[None]).to(args.device)
            wr_t = torch.from_numpy(np.stack(wr_hist)[None]).to(args.device)
            st_t = torch.from_numpy(ns.obs(np.stack(st_hist).astype(np.float32))[None]).to(args.device)
            instr_id = {"all": 0, "apple": 1, "strawberry": 2, "banana": 3, "pear": 4}[args.instruction]
            kind = torch.full((1,), instr_id, dtype=torch.long, device=args.device)
            with torch.no_grad():
                pred, _, _ = model(oh_t, wr_t, st_t, kind)
            pred4 = pred[0, 0].cpu().numpy()
            tgt = np.clip(na.unact(pred4[:3]), -2.0, 2.0)
            grip = -1.0 if pred4[3] > 0 else 1.0
            if prev_tgt is not None:
                tgt = args.smooth * prev_tgt + (1 - args.smooth) * tgt
            prev_tgt = tgt

            # 近距视觉伺服：目标水果进入 12 cm 后，末端对准水果；夹爪用几何规则
            # （学习模型负责远距决策，伺服/夹爪是确定性模块——AHEAD 的 predict-then-servo 思路）
            servo_kind = args.instruction if args.instruction != "all" else None
            if servo_kind in u.fruit_actors:
                fp = u.fruit_actors[servo_kind].pose.p[0].cpu().numpy()
                tcp_now = u.agent.tcp_pose.p[0].cpu().numpy()
                d = float(np.linalg.norm(fp - tcp_now))
                in_zone = u.pick_zone[0] <= fp[0] <= u.pick_zone[1]
                grasped = bool(u.agent.is_grasping(u.fruit_actors[servo_kind])[0])
                if carrying is None and grasped:
                    carrying = servo_kind  # 锁存，避免搬运中接触力抖动导致重新去够水果
                if carrying is not None:
                    bx, by = BIN_LAYOUT[carrying]
                    over = abs(tcp_now[0] - bx) < 0.03 and abs(tcp_now[1] - by) < 0.03
                    held_z = u.fruit_actors[carrying].pose.p[0, 2].item()
                    if over and tcp_now[2] < 0.57:
                        tgt = np.array([bx, by, 0.50], dtype=np.float32)
                        grip = 1.0
                        if held_z < 0.25:
                            carrying = None
                    else:
                        tgt = np.array([bx, by, 0.58], dtype=np.float32)
                        grip = -1.0
                elif d < 0.45 and fp[2] > 0.3:
                    # 近距视觉伺服：目标 = 水果位置 + 速度前馈；夹爪用几何判据
                    fv = u.fruit_actors[servo_kind].linear_velocity[0].cpu().numpy()
                    tgt = fp + fv * 0.10
                    close = d < 0.05 and abs(tcp_now[2] - fp[2]) < 0.04
                    grip = -1.0 if close else 1.0

            tcp = u.agent.tcp_pose.p[0].cpu().numpy()
            err = tgt - tcp
            dist = float(np.linalg.norm(err))
            if dist > 0.06:
                tgt = tcp + err / dist * 0.06
            cmd = np.array([tgt[1] + 0.5, -tgt[0], tgt[2], *HOME_EULER, grip], dtype=np.float32)
            obs, _, term, trunc, info = env.step(cmd)

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
        out = Path(args.video); out.parent.mkdir(parents=True, exist_ok=True)
        h, w = frames[0].shape[:2]
        wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
        for f in frames:
            wr.write(f[..., ::-1])
        wr.release()
        print("视频:", out)

    s = summarize(results)
    print("\n视觉策略闭环:", format_summary(s))
    out_dir = ROOT / "runs" / f"{time.strftime('%Y-%m-%d-%H%M')}-vpolicy-eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(
        {"ckpt": str(args.ckpt), "kinds": args.kinds, "instruction": args.instruction, "episodes": args.episodes,
         "belt_speed": args.belt_speed, "summary": s}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
