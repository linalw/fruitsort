#!/usr/bin/env python
"""纯策略闭环：模型直接输出目标位姿 + 夹爪，不使用任何几何伺服/脚本执行器。

用法:
    python scripts/eval_policy_pure.py --ckpt runs/xxx/ckpt.pt \
        --instruction apple --episodes 5 --belt 0.08
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
    ap.add_argument("--belt-speed", type=float, default=0.08)
    ap.add_argument("--spawn-x", type=float, default=-0.55)
    ap.add_argument("--spawn-spacing", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--max-step", type=float, default=0.06)
    ap.add_argument("--delta", action="store_true")
    ap.add_argument("--z-gate", action="store_true", help="用水果高度门控目标 z")
    ap.add_argument("--phase-gate", action="store_true", help="几何相位门控：未抓住前不许切搬运")
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
            belt_speed_range=(args.belt_speed, args.belt_speed),
            spawn_x=args.spawn_x, spawn_spacing=args.spawn_spacing,
            instruction_kind=args.instruction)
        obs, _ = env.reset(seed=args.seed + ep)
        u = env.unwrapped
        oh, wr, st = [], [], []
        kind = args.instruction
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
            p4 = pred[0, 0].cpu().numpy()
            tcp = u.agent.tcp_pose.p[0].cpu().numpy().astype(np.float64)
            tgt = na.unact(p4[:3])
            if args.delta:
                tgt = tcp + tgt
            if args.z_gate and kind in u.fruit_actors:
                fz = u.fruit_actors[kind].pose.p[0, 2].item()
                tgt[2] = float(np.clip(tgt[2], fz - 0.02, 0.90))
            tgt = np.clip(tgt, -2.0, 2.0)
            grip = -1.0 if p4[3] > 0 else 1.0

            if args.phase_gate and kind in u.fruit_actors:
                actor2 = u.fruit_actors[kind]
                fp2 = actor2.pose.p[0].cpu().numpy().astype(np.float64)
                dist2 = float(np.linalg.norm(fp2 - tcp))
                holding = bool(u.agent.is_grasping(actor2)[0]) or dist2 < 0.05
                if not holding:
                    # 未抓住：只允许接近/下探，禁止提前切到料框
                    if dist2 > 0.10:
                        tgt = np.array([fp2[0], fp2[1], fp2[2] + 0.08])
                        grip = 1.0
                    else:
                        tgt = np.array([fp2[0], fp2[1], fp2[2] + 0.02])
                        grip = -1.0
                else:
                    tgt[2] = float(np.clip(tgt[2], 0.45, 0.9))

            err = tgt - tcp
            d = float(np.linalg.norm(err))
            if d > args.max_step:
                tgt = tcp + err / d * args.max_step
            cmd = np.array([tgt[1] + 0.5, -tgt[0], tgt[2], *HOME_EULER, grip], dtype=np.float32)
            obs, _, term, trunc, info = env.step(cmd)
            if bool(info.get("success") or info.get("fail")):
                break
        ok = bool(info.get("success", False))
        results.append({"success": ok, "duration_s": (t + 1) / 20.0})
        print(f"[{'✓' if ok else '✗'}] ep {ep:02d} kind={kind} steps={t + 1}")
        env.close()

    s = summarize(results)
    print("\n纯策略:", format_summary(s))
    out = ROOT / "runs" / f"{time.strftime('%Y-%m-%d-%H%M')}-pure-policy"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"ckpt": str(args.ckpt), "instruction": args.instruction, "episodes": args.episodes,
         "delta": args.delta, "z_gate": args.z_gate,
         "phase_gate": args.phase_gate, "summary": s}, ensure_ascii=False, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
