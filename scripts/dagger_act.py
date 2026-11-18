#!/usr/bin/env python
"""ACT 的 DAgger 采集：学生跑闭环，用平行夹爪 Markov 标注器给访问到的状态打标。

标签规则（无记忆，适配任意状态）：
- 已抓住：搬去对应料框（上方向 0.60，到框后下放 0.50 并开爪）
- 未抓住：追拦截点（水果 + v*0.10）；对齐后下探到果心并闭爪

只保存数据，不做筛选（失败状态同样重要）。

用法:
    python scripts/dagger_act.py --ckpt runs/act-d500/ckpt.pt \
        --episodes 6 --belt 0.25 --out data/datasets/dagger-act-r1
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
from fruitsort.models.act_policy import ActConfig, ActPolicy
from fruitsort.policies.scripted import HOME_EULER
from fruitsort.sim.conveyor import BIN_LAYOUT

ROOT = Path(__file__).resolve().parent.parent
INSTR_ID = {"apple": 1, "strawberry": 2, "banana": 3, "pear": 4}


def prep(x, size):
    if x.shape[0] != size:
        x = cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
    return (x.astype(np.float32) / 255.0).transpose(2, 0, 1)


def jpg(x):
    return cv2.imencode(".jpg", x[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def label(kind, u):
    tcp = u.agent.tcp_pose.p[0].cpu().numpy().astype(np.float64)
    actor = u.fruit_actors[kind]
    fp = actor.pose.p[0].cpu().numpy().astype(np.float64)
    fv = actor.linear_velocity[0].cpu().numpy().astype(np.float64)
    bx, by = BIN_LAYOUT[kind]
    holding = bool(u.agent.is_grasping(actor)[0])
    if holding:
        over = abs(tcp[0] - bx) < 0.06 and abs(tcp[1] - by) < 0.06
        if over and tcp[2] < 0.58:
            return np.array([bx, by, 0.50, 1.0], dtype=np.float32)
        return np.array([bx, by, 0.60, -1.0], dtype=np.float32)
    x = float(np.clip(fp[0] + fv[0] * 0.10, u.pick_zone[0] - 0.02, u.pick_zone[1]))
    aligned = abs(tcp[0] - x) < 0.025 and abs(tcp[1] - fp[1]) < 0.025
    if aligned and u.pick_zone[0] <= fp[0] <= u.pick_zone[1]:
        return np.array([x, fp[1], fp[2], -1.0], dtype=np.float32)
    return np.array([x, fp[1], fp[2] + 0.10, 1.0], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kinds", nargs="+", default=["apple", "strawberry", "banana", "pear"])
    ap.add_argument("--instruction", default="apple")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--belt", type=float, default=0.25)
    ap.add_argument("--control-freq", type=int, default=30)
    ap.add_argument("--spawn-x", type=float, default=-0.55)
    ap.add_argument("--spawn-spacing", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=40000)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ck = torch.load(Path(args.ckpt), map_location="cpu", weights_only=False)
    cfg = ActConfig(**ck["cfg"])
    model = ActPolicy(cfg).to("cpu")
    model.load_state_dict(ck["model"]); model.eval()
    ns = Normalizer.from_state(ck["norm_state"]); na = Normalizer.from_state(ck["norm_act"])

    out = Path(args.out) if args.out else ROOT / "data" / "datasets" / f"dagger-act-{args.instruction}"
    out.mkdir(parents=True, exist_ok=True)
    saved = 0

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
        oh, wr, st = [], [], []
        obs_list, act_list, oh_list, wr_list = [], [], [], []
        info, t = {}, 0
        phase_state, phase_cand, same, grasp_lost = 0, 0, 0, 0
        for t in range(args.max_steps):
            actor = u.fruit_actors.get(args.instruction)
            if actor is not None:
                grasping = bool(u.agent.is_grasping(actor)[0])
                fz = float(actor.pose.p[0, 2])
                if grasping:
                    phase_state, grasp_lost = 2, 0
                elif phase_state == 2:
                    grasp_lost += 1
                    if fz < 0.52 or grasp_lost > 20:
                        phase_state, same, phase_cand = 0, 0, 0
            sd = obs["sensor_data"]
            oh.append(prep(sd["overhead"]["rgb"][0].cpu().numpy(), cfg.img_size))
            wr.append(prep(sd["hand_camera"]["rgb"][0].cpu().numpy(), cfg.img_size))
            sb = build_obs(u)
            if cfg.obs_dim > 49:
                onehot = np.zeros(3, dtype=np.float32)
                onehot[phase_state] = 1.0
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
                        torch.from_numpy(np.stack(oh)[None]).float(),
                        torch.from_numpy(np.stack(wr)[None]).float(),
                        torch.from_numpy(ns.obs(np.stack(st).astype(np.float32))[None]).float(),
                        torch.tensor([INSTR_ID[args.instruction]]),
                        use_prior=True, phase_out=True)
                else:
                    act, _, _ = model(
                        torch.from_numpy(np.stack(oh)[None]).float(),
                        torch.from_numpy(np.stack(wr)[None]).float(),
                        torch.from_numpy(ns.obs(np.stack(st).astype(np.float32))[None]).float(),
                        torch.tensor([INSTR_ID[args.instruction]]),
                        use_prior=True)
                    logits = None
            p4 = act[0, 0].numpy()
            tcp = u.agent.tcp_pose.p[0].cpu().numpy().astype(np.float64)
            tgt = np.clip(na.unact(p4[:3]), -2, 2)
            grip = -1.0 if p4[3] > 0 else 1.0

            lab = label(args.instruction, u)
            # 过滤退化状态：只保留水果在抓取区、且距离合理（或已抓住）的帧
            fp_now = u.fruit_actors[args.instruction].pose.p[0].cpu().numpy()
            tcp_now = u.agent.tcp_pose.p[0].cpu().numpy()
            in_zone = u.pick_zone[0] - 0.05 <= fp_now[0] <= u.pick_zone[1] + 0.05
            holding_now = bool(u.agent.is_grasping(u.fruit_actors[args.instruction])[0])
            keep = (in_zone and np.linalg.norm(fp_now - tcp_now) < 0.6) or holding_now
            if keep and len(obs_list) < 200:
                obs_list.append(build_obs(u))
                act_list.append(lab)
                oh_list.append(jpg(sd["overhead"]["rgb"][0].cpu().numpy()))
                wr_list.append(jpg(sd["hand_camera"]["rgb"][0].cpu().numpy()))

            if logits is not None and phase_state != 2:
                p_cls = int(logits[0, 0].argmax())
                p_cls = 1 if p_cls == 2 else p_cls
                if p_cls == phase_cand:
                    same += 1
                else:
                    phase_cand, same = p_cls, 0
                if same >= 1:
                    phase_state = phase_cand

            err = tgt - tcp
            d = float(np.linalg.norm(err))
            if d > 0.06:
                tgt = tcp + err / d * 0.06
            cmd = np.array([tgt[1] + 0.5, -tgt[0], tgt[2], *HOME_EULER, grip], dtype=np.float32)
            obs, _, term, trunc, info = env.step(cmd)
            if bool(info.get("success") or info.get("fail")):
                break

        if len(obs_list) > 20:
            np.savez_compressed(
                out / f"ep_{ep:04d}.npz", obs=np.stack(obs_list), action=np.stack(act_list),
                seed=args.seed + ep, kinds=np.array(args.kinds),
                instruction=np.array(args.instruction),
                overhead_jpg=np.array(oh_list, dtype=object),
                wrist_jpg=np.array(wr_list, dtype=object))
            saved += 1
        print(f"[{'✓' if info.get('success') else '✗'}] ep {ep:02d} steps={len(obs_list)}", flush=True)
        env.close()

    (out / "meta.json").write_text(json.dumps(
        {"instruction": args.instruction, "episodes": saved, "belt": args.belt,
         "source": args.ckpt, "created": time.strftime("%Y-%m-%dT%H:%M:%S")},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ 保存 {saved} 条到 {out}")


if __name__ == "__main__":
    main()
