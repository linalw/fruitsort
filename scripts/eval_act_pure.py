#!/usr/bin/env python
"""ACT 策略的纯闭环评测（平行夹爪，无吸盘、无脚本执行器）。

只有策略输出：目标位姿 + 夹爪开合都由模型给出。唯一的额外信号是「相位」——
模型自己预测它（相位辅助头的 k=0 输出），并把自己的预测反馈回下一步的状态，
等价于把 FSM 交给策略而不是脚本。

用法:
    python scripts/eval_act_pure.py --ckpt runs/xxx/ckpt.pt \
        --instruction apple --episodes 5 --belt 0.25
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
from fruitsort.models.act_policy import ActConfig, ActPolicy
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
    ap.add_argument("--belt", type=float, default=0.25)
    ap.add_argument("--control-freq", type=int, default=30)
    ap.add_argument("--spawn-x", type=float, default=-0.55)
    ap.add_argument("--spawn-spacing", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=15000)
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--max-step", type=float, default=0.06)
    ap.add_argument("--z-gate", action="store_true")
    ap.add_argument("--no-phase-fb", action="store_true",
                    help="关闭相位反馈（对照：模型输入恒为 0）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(Path(args.ckpt), map_location="cpu", weights_only=False)
    cfg = ActConfig(**ck["cfg"])
    model = ActPolicy(cfg).to(args.device)
    model.load_state_dict(ck["model"]); model.eval()
    ns = Normalizer.from_state(ck["norm_state"]); na = Normalizer.from_state(ck["norm_act"])
    use_phase = cfg.obs_dim > 49

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
        oh, wr, st = [], [], []
        info, t = {}, 0
        phase_state, phase_cand, same, grasp_lost = 0, 0, 0, 0
        n_close, n_dec, n_carry = 0, 0, 0
        for t in range(args.max_steps):
            actor = u.fruit_actors.get(kind)
            grasping = False
            if actor is not None:
                grasping = bool(u.agent.is_grasping(actor)[0])
                fz = float(actor.pose.p[0, 2])
                if grasping:
                    phase_state, grasp_lost = 2, 0
                elif phase_state == 2:
                    # 抓丢了：回退到接近段，避免"单向门"
                    grasp_lost += 1
                    if fz < 0.52 or grasp_lost > 20:
                        phase_state, same, phase_cand = 0, 0, 0

            sd = obs["sensor_data"]
            oh.append(prep(sd["overhead"]["rgb"][0].cpu().numpy(), cfg.img_size))
            wr.append(prep(sd["hand_camera"]["rgb"][0].cpu().numpy(), cfg.img_size))
            sb = build_obs(u)
            if use_phase:
                onehot = np.zeros(3, dtype=np.float32)
                onehot[phase_state if not args.no_phase_fb else 0] = 1.0
                sb = np.concatenate([sb, onehot])
            st.append(sb)
            for h in (oh, wr, st):
                if len(h) > cfg.hist:
                    h.pop(0)
            while len(oh) < cfg.hist:
                oh.insert(0, oh[0]); wr.insert(0, wr[0]); st.insert(0, st[0])

            with torch.no_grad():
                if cfg.n_phase:
                    act, _, _, logits = model(
                        torch.from_numpy(np.stack(oh)[None]).float().to(args.device),
                        torch.from_numpy(np.stack(wr)[None]).float().to(args.device),
                        torch.from_numpy(ns.obs(np.stack(st).astype(np.float32))[None]).float().to(args.device),
                        torch.tensor([INSTR_ID[args.instruction]], device=args.device),
                        use_prior=True, phase_out=True)
                else:
                    act, _, _ = model(
                        torch.from_numpy(np.stack(oh)[None]).float().to(args.device),
                        torch.from_numpy(np.stack(wr)[None]).float().to(args.device),
                        torch.from_numpy(ns.obs(np.stack(st).astype(np.float32))[None]).float().to(args.device),
                        torch.tensor([INSTR_ID[args.instruction]], device=args.device),
                        use_prior=True)
                    logits = None
            p4 = act[0, 0].cpu().numpy()
            tcp = u.agent.tcp_pose.p[0].cpu().numpy().astype(np.float64)
            tgt = np.clip(na.unact(p4[:3]), -2, 2)
            if args.z_gate and kind in u.fruit_actors:
                fz = u.fruit_actors[kind].pose.p[0, 2].item()
                tgt[2] = float(np.clip(tgt[2], fz - 0.02, 0.90))
            grip = -1.0 if p4[3] > 0 else 1.0
            n_close += int(grip < 0)
            n_carry += int(phase_state == 2)
            n_dec += int(phase_state == 1)

            # 相位反馈：用模型自己的预测（k=0），未真正抓住时不允许自认搬运，
            # 并要求连续 2 帧一致才切换（迟滞）
            if logits is not None and phase_state != 2:
                p_cls = int(logits[0, 0].argmax())
                if p_cls == 2:
                    p_cls = 1
                if p_cls == phase_cand:
                    same += 1
                else:
                    phase_cand, same = p_cls, 0
                if same >= 1:
                    phase_state = phase_cand

            err = tgt - tcp
            d = float(np.linalg.norm(err))
            if d > args.max_step:
                tgt = tcp + err / d * args.max_step
            cmd = np.array([tgt[1] + 0.5, -tgt[0], tgt[2], *HOME_EULER, grip], dtype=np.float32)
            obs, _, term, trunc, info = env.step(cmd)
            if bool(info.get("success") or info.get("fail")):
                break
        ok = bool(info.get("success", False))
        results.append({"success": ok, "duration_s": (t + 1) / args.control_freq})
        print(f"[{'✓' if ok else '✗'}] ep {ep:02d} kind={kind} steps={t + 1} "
              f"close={n_close / max(t + 1, 1):.2f} descend={n_dec} carry={n_carry}")
        env.close()

    s = summarize(results)
    print("\nACT 纯策略:", format_summary(s))
    out = ROOT / "runs" / f"{time.strftime('%Y-%m-%d-%H%M')}-act-pure"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"ckpt": str(args.ckpt), "instruction": args.instruction, "episodes": args.episodes,
         "belt": args.belt, "z_gate": args.z_gate, "phase_fb": not args.no_phase_fb,
         "summary": s}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
