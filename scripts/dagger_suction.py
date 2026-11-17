#!/usr/bin/env python
"""DAgger（吸盘版）：跑当前策略，用 suction 控制器的状态标签给访问到的状态打标。

只存成功/失败都保留（DAgger 需要失败状态），输出格式与示教一致。

用法:
    python scripts/dagger_suction.py --ckpt runs/vpolicy-mix/ckpt.pt \
        --instruction apple --episodes 8 --belt 0.30 --out data/datasets/dagger-suc-apple
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
from fruitsort.learning.bc import Normalizer, build_obs
from fruitsort.models.vision_policy import MultiViewTemporalPolicy, PolicyConfig
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--instruction", default="apple")
    ap.add_argument("--kinds", nargs="+", default=["apple", "strawberry", "banana", "pear"])
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--belt", type=float, default=0.30)
    ap.add_argument("--control-freq", type=int, default=30)
    ap.add_argument("--spawn-x", type=float, default=-0.55)
    ap.add_argument("--spawn-spacing", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=14000)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ck = torch.load(Path(args.ckpt), map_location="cpu", weights_only=False)
    cfg = PolicyConfig(**ck["cfg"])
    model = MultiViewTemporalPolicy(cfg).to("cpu")
    model.load_state_dict(ck["model"]); model.eval()
    ns = Normalizer.from_state(ck["norm_state"]); na = Normalizer.from_state(ck["norm_act"])

    out = Path(args.out) if args.out else ROOT / "data" / "datasets" / f"dagger-suc-{args.instruction}"
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
        kind = args.instruction
        actor = u.fruit_actors[kind]
        bx, by = BIN_LAYOUT[kind]
        oh, wr, st = [], [], []
        attached, offset = False, np.zeros(3)
        obs_list, act_list, oh_list, wr_list = [], [], [], []
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
                pred, _, _ = model(
                    torch.from_numpy(np.stack(oh)[None]).float(),
                    torch.from_numpy(np.stack(wr)[None]).float(),
                    torch.from_numpy(ns.obs(np.stack(st).astype(np.float32))[None]).float(),
                    torch.tensor([INSTR_ID[args.instruction]]))
            p4 = pred[0, 0].cpu().numpy()
            tcp = u.agent.tcp_pose.p[0].cpu().numpy().astype(np.float64)
            tgt = np.clip(na.unact(p4[:3]), -2, 2)
            grip = -1.0 if p4[3] > 0 else 1.0

            # ---- suction 状态标签 ----
            fp = actor.pose.p[0].cpu().numpy().astype(np.float64)
            dist = float(np.linalg.norm(fp - tcp))
            over_bin = abs(tcp[0] - bx) < 0.07 and abs(tcp[1] - by) < 0.07
            if attached:
                if over_bin and tcp[2] < 0.60:
                    lab = np.array([bx, by, 0.52, 1.0], dtype=np.float32)
                else:
                    lab = np.array([bx, by, 0.62, -1.0], dtype=np.float32)
            else:
                if dist > 0.10:
                    lab = np.array([fp[0], fp[1], fp[2] + 0.10, 1.0], dtype=np.float32)
                else:
                    lab = np.array([fp[0], fp[1], fp[2] + 0.05, -1.0], dtype=np.float32)

            obs_list.append(obs_vec(u) if False else build_obs(u))
            act_list.append(lab)
            oh_list.append(jpg(sd["overhead"]["rgb"][0].cpu().numpy()))
            wr_list.append(jpg(sd["hand_camera"]["rgb"][0].cpu().numpy()))

            # 吸盘附着/释放（执行策略动作）
            if not attached and grip < 0 and dist < 0.10 and fp[2] > 0.3:
                attached = True
                offset = fp - tcp
            if attached and grip > 0 and over_bin and tcp[2] < 0.62:
                p = np.array([bx, by, 0.08])
                actor.set_pose(Pose.create_from_pq(torch.tensor([p], dtype=torch.float32)))
                actor.set_linear_velocity(torch.zeros(1, 3))
                attached = False
                grip = 1.0

            err = tgt - tcp
            d = float(np.linalg.norm(err))
            if d > 0.08:
                tgt = tcp + err / d * 0.08
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

        if len(obs_list) > 20:
            np.savez_compressed(
                out / f"ep_{ep:04d}.npz", obs=np.stack(obs_list),
                action=np.stack(act_list), seed=args.seed + ep,
                kinds=np.array(args.kinds), instruction=np.array(args.instruction),
                overhead_jpg=np.array(oh_list, dtype=object),
                wrist_jpg=np.array(wr_list, dtype=object))
            saved += 1
        print(f"[{'✓' if info.get('success') else '✗'}] ep {ep:02d} steps={len(obs_list)}", flush=True)
        env.close()

    (out / "meta.json").write_text(json.dumps(
        {"instruction": args.instruction, "episodes": saved, "belt": args.belt,
         "source": str(args.ckpt), "created": time.strftime("%Y-%m-%dT%H:%M:%S")},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 保存 {saved} 条 DAgger 数据到 {out}")


if __name__ == "__main__":
    main()
