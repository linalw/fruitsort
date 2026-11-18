#!/usr/bin/env python
"""DAgger（状态反馈教师版）：学生跑闭环，用**伺服教师**给每个访问状态打标。

与 `dagger_act.py` 的差别
------------------------
`dagger_act.py` 的标签是手写几何规则，学生停在远处时规则仍然回「悬停接近」，
于是把「永远悬停」教进策略（实测相位 CE 从 0.093 恶化到 1.31）。

这里改用**状态反馈教师**（hybrid 的伺服规则，Markov 化）：
- 已抓住 → 抬升 / 过中点 / 到框上方 / 释放（4 段几何谓词）
- 目标已脱离可抓区间（越过 12cm、掉下带面、出抓取区）→ **放弃并回 ready 位**
  （这一条是关键：它给出「失败恢复」标签，而不是「继续悬停」）
- 近距（<45cm）→ 伺服到拦截点 `fp + v*lead`，`dist<5cm` 时闭爪（预瞄闭爪）
- 远距 → 飞到拦截点上方 9cm

标签是当前状态的纯函数，因此 BC 学到的是反馈律而不是轨迹回放。

用法:
    python scripts/dagger_teacher.py --ckpt runs/act-phase-delta2/ckpt.pt \
        --episodes 8 --belt 0.25 --out data/datasets/dagger-teacher-r1
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
INSTR_ID = {"all": 0, "apple": 1, "strawberry": 2, "banana": 3, "pear": 4}
READY = np.array([0.0, -0.34, 0.66])
LEAD = 0.10


def prep(x, size):
    if x.shape[0] != size:
        x = cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
    return (x.astype(np.float32) / 255.0).transpose(2, 0, 1)


def jpg(x):
    return cv2.imencode(".jpg", x[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def teacher_label(u, kind: str):
    """状态反馈教师：当前状态的纯函数 → (目标位姿[3], 夹爪[+1 开/-1 闭])。"""
    tcp = u.agent.tcp_pose.p[0].cpu().numpy().astype(np.float64)
    actor = u.fruit_actors[kind]
    fp = actor.pose.p[0].cpu().numpy().astype(np.float64)
    fv = actor.linear_velocity[0].cpu().numpy().astype(np.float64)
    dist = float(np.linalg.norm(fp - tcp))
    z0, z1 = u.pick_zone
    in_zone = z0 <= fp[0] <= z1
    grasped = bool(u.agent.is_grasping(actor)[0]) or (dist < 0.06 and fp[2] > 0.55)
    bx, by = BIN_LAYOUT[kind]

    if grasped:
        # 搬运四段：先垂直抬、再过中点、到框上方、下放释放
        if abs(tcp[0]) > 0.06 or abs(tcp[1] + 0.34) > 0.06:
            if tcp[2] < 0.60:
                return np.array([tcp[0], tcp[1], tcp[2] + 0.02]), -1.0
            return np.array([0.0, -0.34, 0.60]), -1.0
        over = abs(tcp[0] - bx) < 0.04 and abs(tcp[1] - by) < 0.04
        if over and tcp[2] < 0.56:
            return np.array([bx, by, 0.50]), 1.0
        return np.array([bx, by, 0.60 if not over else 0.48]), -1.0

    # 恢复分支：目标已不可抓 → 放弃，回 ready（教「不要悬停」）
    escaped = (fp[0] > z1 + 0.05) or (fp[2] < 0.35) or (tcp[0] - fp[0] > 0.12)
    if escaped:
        return READY.copy(), 1.0

    x_pred = float(np.clip(fp[0] + fv[0] * LEAD, z0 - 0.02, z1))
    if dist <= 0.45 and in_zone:
        grip = -1.0 if dist < 0.05 else 1.0
        return np.array([x_pred, fp[1], fp[2]]), grip
    return np.array([x_pred, fp[1], fp[2] + 0.09]), 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="学生权重；--rollout teacher 时不需要")
    ap.add_argument("--rollout", choices=["student", "teacher"], default="student",
                    help="student：学生闭环+教师打标（DAgger）；teacher：教师自己跑（覆盖抓取/搬运段）")
    ap.add_argument("--kinds", nargs="+", default=["apple", "strawberry", "banana", "pear"])
    ap.add_argument("--instruction", default="apple")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--belt", type=float, default=0.25)
    ap.add_argument("--control-freq", type=int, default=30)
    ap.add_argument("--spawn-x", type=float, default=-0.55)
    ap.add_argument("--spawn-spacing", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=50000)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--max-keep", type=int, default=200, help="每局最多保留帧数")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.rollout == "student":
        ck = torch.load(Path(args.ckpt), map_location="cpu", weights_only=False)
        cfg = ActConfig(**ck["cfg"])
        model = ActPolicy(cfg).to("cpu")
        model.load_state_dict(ck["model"]); model.eval()
        ns = Normalizer.from_state(ck["norm_state"]); na = Normalizer.from_state(ck["norm_act"])
        delta = bool(ck.get("delta_actions", False))
    else:
        # 教师自跑：需要与训练一致的 obs 维度（相位开/关都用 52 维的 cfg 占位）
        cfg = ActConfig(obs_dim=52, n_phase=3)
        model = None; ns = na = None; delta = False

    out = Path(args.out) if args.out else ROOT / "data" / "datasets" / f"dagger-teacher-{args.instruction}"
    out.mkdir(parents=True, exist_ok=True)
    saved, total_kept, ok_cnt = 0, 0, 0

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
        obs_l, act_l, oh_l, wr_l = [], [], [], []
        phase_state, cand, same = 0, 0, 0
        info, t = {}, 0
        for t in range(args.max_steps):
            actor = u.fruit_actors.get(args.instruction)
            grasping = False
            if actor is not None:
                grasping = bool(u.agent.is_grasping(actor)[0])
                if grasping:
                    phase_state, cand, same = 2, 0, 0
            sd = obs["sensor_data"]
            tcp0 = u.agent.tcp_pose.p[0].cpu().numpy().astype(np.float64)
            oh.append(prep(sd["overhead"]["rgb"][0].cpu().numpy(), cfg.img_size))
            wr.append(prep(sd["hand_camera"]["rgb"][0].cpu().numpy(), cfg.img_size))
            sb = build_obs(u)
            if cfg.obs_dim > 49:
                one = np.zeros(3, dtype=np.float32); one[phase_state] = 1.0
                sb = np.concatenate([sb, one])
            st.append(sb)
            for h in (oh, wr, st):
                if len(h) > cfg.hist:
                    h.pop(0)
            while len(oh) < cfg.hist:
                oh.insert(0, oh[0]); wr.insert(0, wr[0]); st.insert(0, st[0])

            # ---- 学生动作（与环境交互）----
            if model is None:
                lab0, grip0 = teacher_label(u, args.instruction)
                tgt = lab0.copy(); grip = grip0
                err = tgt - tcp0; d0 = float(np.linalg.norm(err))
                if d0 > 0.06:
                    tgt = tcp0 + err / d0 * 0.06
                cmd = np.array([tgt[1] + 0.5, -tgt[0], tgt[2], *HOME_EULER, grip],
                               dtype=np.float32)
                lab, lab_grip = lab0, grip0
                if len(obs_l) < args.max_keep:
                    obs_l.append(build_obs(u))
                    act_l.append(np.array([*lab, lab_grip], dtype=np.float32))
                    oh_l.append(jpg(sd["overhead"]["rgb"][0].cpu().numpy()))
                    wr_l.append(jpg(sd["hand_camera"]["rgb"][0].cpu().numpy()))
                obs, _, term, trunc, info = env.step(cmd)
                if bool(info.get("success") or info.get("fail")):
                    break
                continue
            with torch.no_grad():
                if cfg.n_phase:
                    act, _, _, logits = model(
                        torch.from_numpy(np.stack(oh)[None]).float(),
                        torch.from_numpy(np.stack(wr)[None]).float(),
                        torch.from_numpy(ns.obs(np.stack(st).astype(np.float32))[None]).float(),
                        torch.tensor([INSTR_ID[args.instruction]]), use_prior=True, phase_out=True)
                else:
                    act, _, _ = model(
                        torch.from_numpy(np.stack(oh)[None]).float(),
                        torch.from_numpy(np.stack(wr)[None]).float(),
                        torch.from_numpy(ns.obs(np.stack(st).astype(np.float32))[None]).float(),
                        torch.tensor([INSTR_ID[args.instruction]]), use_prior=True)
                    logits = None
            p4 = act[0, 0].numpy()
            tcp = u.agent.tcp_pose.p[0].cpu().numpy().astype(np.float64)
            pred = na.unact(p4[:3])
            tgt = tcp + np.clip(pred, -0.06, 0.06) if delta else np.clip(pred, -2, 2)
            grip = -1.0 if p4[3] > 0 else 1.0
            if logits is not None and phase_state != 2:
                pc = int(logits[0, 0].argmax())
                pc = 1 if pc == 2 else pc
                if pc == cand:
                    same += 1
                else:
                    cand, same = pc, 0
                if same >= 1:
                    phase_state = cand
            err = tgt - tcp
            d = float(np.linalg.norm(err))
            if d > 0.06:
                tgt = tcp + err / d * 0.06
            cmd = np.array([tgt[1] + 0.5, -tgt[0], tgt[2], *HOME_EULER, grip], dtype=np.float32)

            # ---- 教师标签（状态反馈，Markov）----
            lab, lab_grip = teacher_label(u, args.instruction)
            keep = True
            if actor is not None:
                fp = actor.pose.p[0].cpu().numpy()
                # 丢掉「水果长时间不在场景内」的退化帧（等下一个水果）
                keep = float(np.linalg.norm(fp - tcp)) < 0.9 or grasping
            if keep and len(obs_l) < args.max_keep:
                obs_l.append(build_obs(u))
                act_l.append(np.array([*lab, lab_grip], dtype=np.float32))
                oh_l.append(jpg(sd["overhead"]["rgb"][0].cpu().numpy()))
                wr_l.append(jpg(sd["hand_camera"]["rgb"][0].cpu().numpy()))

            obs, _, term, trunc, info = env.step(cmd)
            if bool(info.get("success") or info.get("fail")):
                break

        succ = bool(info.get("success", False))
        ok_cnt += int(succ)
        if len(obs_l) > 20:
            np.savez_compressed(
                out / f"ep_{ep:04d}.npz", obs=np.stack(obs_l), action=np.stack(act_l),
                seed=args.seed + ep, kinds=np.array(args.kinds),
                instruction=np.array(args.instruction),
                overhead_jpg=np.array(oh_l, dtype=object),
                wrist_jpg=np.array(wr_l, dtype=object))
            saved += 1; total_kept += len(obs_l)
        print(f"[{'✓' if succ else '✗'}] ep {ep:02d} steps={len(obs_l)}", flush=True)
        env.close()

    (out / "meta.json").write_text(json.dumps(
        {"instruction": args.instruction, "episodes": saved, "frames": total_kept,
         "student_success": f"{ok_cnt}/{args.episodes}", "belt": args.belt,
         "source": args.ckpt, "label": "state-feedback teacher (Markov)",
         "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"✓ 保存 {saved} 局 / {total_kept} 帧到 {out}（学生成功率 {ok_cnt}/{args.episodes}）")


if __name__ == "__main__":
    main()
