#!/usr/bin/env python
"""纯策略 + 吸盘末端：策略输出接近/吸附触发/搬运/释放，吸盘只做附着物理。

与 eval_policy_pure 的区别：
- 策略的夹爪动作 <0 且 TCP 靠近水果时，吸盘附着（等效真空吸住）
- 策略的夹爪动作 >0 且位于目标框上方时，释放

用法:
    python scripts/eval_policy_suction.py --ckpt runs/xxx/ckpt.pt \
        --instruction apple --episodes 5 --belt 0.30
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
from mani_skill.utils.structs.pose import Pose

import fruitsort.sim.tasks  # noqa: F401
from fruitsort.eval import format_summary, summarize
from fruitsort.learning.bc import Normalizer, build_obs
from fruitsort.models.vision_policy import MultiViewTemporalPolicy, PolicyConfig
from fruitsort.policies.scripted import HOME_EULER
from fruitsort.sim.conveyor import BIN_LAYOUT

ROOT = Path(__file__).resolve().parent.parent
INSTR_ID = {"all": 0, "apple": 1, "strawberry": 2, "banana": 3, "pear": 4}
ID_KIND = {0: "apple", 1: "strawberry", 2: "banana", 3: "pear"}


def prep(x, size):
    if x.shape[0] != size:
        x = cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
    return (x.astype(np.float32) / 255.0).transpose(2, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kinds", nargs="+", default=["apple", "strawberry", "banana", "pear"])
    ap.add_argument("--instruction", default="apple")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--belt", type=float, default=0.30)
    ap.add_argument("--control-freq", type=int, default=30)
    ap.add_argument("--spawn-x", type=float, default=-0.55)
    ap.add_argument("--spawn-spacing", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=13000)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--max-step", type=float, default=0.08)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(Path(args.ckpt), map_location="cpu", weights_only=False)
    cfg = PolicyConfig(**ck["cfg"])
    model = MultiViewTemporalPolicy(cfg).to(args.device)
    model.load_state_dict(ck["model"]); model.eval()
    ns = Normalizer.from_state(ck["norm_state"]); na = Normalizer.from_state(ck["norm_act"])

    results = []
    for ep in range(args.episodes):
        env = gym.make(
            "FruitSort-v1", obs_mode="rgbd", control_mode="pd_ee_pose", sim_backend="cpu",
            robot_uids="panda_wristcam", active_kinds=tuple(args.kinds),
            belt_speed_range=(args.belt, args.belt),
            spawn_x=args.spawn_x, spawn_spacing=args.spawn_spacing,
            instruction_kind=args.instruction,
            sim_config=dict(control_freq=args.control_freq,
                            sim_freq=120 if args.control_freq == 30 else 100))
        obs, _ = env.reset(seed=args.seed + ep)
        u = env.unwrapped
        kind = args.instruction
        actor = u.fruit_actors[kind]
        bx, by = BIN_LAYOUT[kind]
        oh, wr, st = [], [], []
        attached, offset = False, np.zeros(3)
        info, t = {}, 0
        for t in range(args.max_steps):
            sd = obs["sensor_data"]
            oh.append(prep(sd["overhead"]["rgb"][0].cpu().numpy(), cfg.img_size))
            wr.append(prep(sd["hand_camera"]["rgb"][0].cpu().numpy(), cfg.img_size))
            st.append(build_obs(u))
            for h in (oh, wr, st):
                if len(h) > cfg.hist:
                    h.pop(0)
            while len(oh) < cfg.hist:
                oh.insert(0, oh[0]); wr.insert(0, wr[0]); st.insert(0, st[0])

            with torch.no_grad():
                pred, _, sel = model(
                    torch.from_numpy(np.stack(oh)[None]).float().to(args.device),
                    torch.from_numpy(np.stack(wr)[None]).float().to(args.device),
                    torch.from_numpy(ns.obs(np.stack(st).astype(np.float32))[None]).float().to(args.device),
                    torch.tensor([INSTR_ID[args.instruction]], device=args.device))
            if ep == 0 and t == 0:
                kind = ID_KIND.get(int(sel.argmax(-1).item()), args.instruction)
                if kind in u.fruit_actors:
                    actor = u.fruit_actors[kind]
                    bx, by = BIN_LAYOUT[kind]

            p4 = pred[0, 0].cpu().numpy()
            tcp = u.agent.tcp_pose.p[0].cpu().numpy().astype(np.float64)
            tgt = np.clip(na.unact(p4[:3]), -2, 2)
            grip = -1.0 if p4[3] > 0 else 1.0

            fp = actor.pose.p[0].cpu().numpy().astype(np.float64)
            dist = float(np.linalg.norm(fp - tcp))
            over_bin = abs(tcp[0] - bx) < 0.07 and abs(tcp[1] - by) < 0.07
            # 吸盘附着：策略下达"吸"且靠近水果
            if not attached and grip < 0 and dist < 0.10 and fp[2] > 0.3:
                attached = True
                offset = fp - tcp
            # 吸盘释放：策略下达"放"且位于框上方
            if attached and grip > 0 and over_bin and tcp[2] < 0.62:
                p = np.array([bx, by, 0.08])
                actor.set_pose(Pose.create_from_pq(torch.tensor([p], dtype=torch.float32)))
                actor.set_linear_velocity(torch.zeros(1, 3))
                attached = False
                grip = 1.0

            err = tgt - tcp
            d = float(np.linalg.norm(err))
            if d > args.max_step:
                tgt = tcp + err / d * args.max_step
            cmd = np.array([tgt[1] + 0.5, -tgt[0], tgt[2], *HOME_EULER, grip], dtype=np.float32)
            obs, _, term, trunc, info = env.step(cmd)

            if attached:
                tcp2 = u.agent.tcp_pose.p[0].cpu().numpy().astype(np.float64)
                p = np.clip(tcp2 + offset, -2.5, 2.5)
                actor.set_pose(Pose.create_from_pq(torch.tensor([p], dtype=torch.float32)))
                actor.set_linear_velocity(torch.zeros(1, 3))
                actor.set_angular_velocity(torch.zeros(1, 3))

            if bool(info.get("success") or info.get("fail")):
                break
        ok = bool(info.get("success", False))
        results.append({"success": ok, "duration_s": (t + 1) / args.control_freq})
        print(f"[{'✓' if ok else '✗'}] ep {ep:02d} kind={kind} steps={t + 1}")
        env.close()

    s = summarize(results)
    print("\n纯策略+吸盘:", format_summary(s))
    out = ROOT / "runs" / f"{time.strftime('%Y-%m-%d-%H%M')}-pure-suction"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"ckpt": str(args.ckpt), "instruction": args.instruction, "episodes": args.episodes,
         "belt": args.belt, "summary": s}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
