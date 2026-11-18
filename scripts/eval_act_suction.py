#!/usr/bin/env python
"""评测 ACT-lite chunked 策略（吸盘末端）：每步重规划 vs open-loop 执行整段。

策略输出 K 步世界目标位姿 + 夹住 logit；本脚本负责把第 i 步目标转成
`pd_ee_pose` 指令，并沿用 `eval_policy_suction.py` 的吸盘附着/释放逻辑：

- 策略说「吸」(grip logit > 0) 且 TCP 距目标水果 < 10cm → 附着
- 策略说「放」且 TCP 在目标框上方 → 释放并把水果放到框底

`--chunk-exec s`：s=1 每步重规划（receding horizon），s>1 每 s 步推理一次、
中间 s-1 步执行同一 chunk 的后续动作（open-loop）。

用法:
    python scripts/eval_act_suction.py --ckpt runs/xxx/ckpt.pt \
        --instruction apple --episodes 5 --belt 0.30 --chunk-exec 1
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
from fruitsort.models.act_policy import ActConfig, ActPolicy, count_params
from fruitsort.policies.scripted import HOME_EULER
from fruitsort.sim.conveyor import BIN_LAYOUT

ROOT = Path(__file__).resolve().parent.parent
INSTR_ID = {"all": 0, "apple": 1, "strawberry": 2, "banana": 3, "pear": 4}


def prep(x, size):
    if x.shape[0] != size:
        x = cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
    return (x.astype(np.float32) / 255.0).transpose(2, 0, 1)


@torch.no_grad()
def infer(model, dev, oh, wr, st, kind_id, ns):
    """oh/wr: list[(3,H,W)]，st: list[49]，返回 (K,4) 归一化动作。"""
    o = torch.from_numpy(np.stack(oh)[None]).float().to(dev)
    w = torch.from_numpy(np.stack(wr)[None]).float().to(dev)
    s = torch.from_numpy(ns.obs(np.stack(st).astype(np.float32))[None]).float().to(dev)
    k = torch.tensor([kind_id], device=dev)
    return model.act(o, w, s, k)[0].cpu().numpy()


def latency(model, dev, cfg, ns, n=50):
    oh = [prep(np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8), cfg.img_size)] * cfg.hist
    st = [np.zeros(49, dtype=np.float32)] * cfg.hist
    for _ in range(10):
        infer(model, dev, oh, oh, st, 1, ns)
    torch.cuda.synchronize() if dev.startswith("cuda") else None
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        infer(model, dev, oh, oh, st, 1, ns)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return {"mean_ms": round(float(ts.mean()), 3),
            "p95_ms": round(float(np.percentile(ts, 95)), 3),
            "min_ms": round(float(ts.min()), 3)}


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
    ap.add_argument("--max-step", type=float, default=0.08, help="每步目标位姿最大位移（m）")
    ap.add_argument("--chunk-exec", type=int, default=1, help=">1: 每 s 步推理一次并执行整段")
    ap.add_argument("--gate", type=float, default=0.0,
                    help="几何门控半径：目标水果 >R 时拒绝策略的吸附意图（docs/gripper-timing.md 方案A）")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(Path(args.ckpt), map_location="cpu", weights_only=False)
    cfg = ActConfig(**ck["cfg"])
    model = ActPolicy(cfg).to(args.device)
    model.load_state_dict(ck["model"]); model.eval()
    ns = Normalizer.from_state(ck["norm_state"]); na = Normalizer.from_state(ck["norm_act"])
    delta = bool(ck.get("delta_actions", False))
    lat = latency(model, args.device, cfg, ns)
    print(f"参数量 {count_params(model)/1e6:.2f}M；推理 {lat['mean_ms']:.2f} ms "
          f"(p95 {lat['p95_ms']:.2f}) on {args.device}；delta_actions={delta}")

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
        attached, offset, released = False, np.zeros(3), False
        buf: list[np.ndarray] = []
        info, t = {}, 0
        n_infer, n_blocked = 0, 0
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

            if buf:
                p4 = buf.pop(0)
            else:
                p4 = infer(model, args.device, oh, wr, st, INSTR_ID[args.instruction], ns)
                n_infer += 1
                if args.chunk_exec > 1:
                    buf = [p4[i] for i in range(1, min(args.chunk_exec, p4.shape[0]))]
                p4 = p4[0]

            tcp = u.agent.tcp_pose.p[0].cpu().numpy().astype(np.float64)
            tgt = np.clip(na.unact(p4[:3]), -2, 2)
            if delta:
                tgt = np.clip(tcp + tgt, -2, 2)
            # chunk 内后续动作是绝对目标，不重新叠加当前 tcp（delta 模式除外）
            grip = -1.0 if p4[3] > 0 else 1.0

            fp = actor.pose.p[0].cpu().numpy().astype(np.float64)
            dist = float(np.linalg.norm(fp - tcp))
            over_bin = abs(tcp[0] - bx) < 0.07 and abs(tcp[1] - by) < 0.07
            # 几何门控：策略在远离水果时下达"吸"→ 视为过早吸附意图，拒绝
            if args.gate > 0 and grip < 0 and not attached and dist > args.gate:
                grip = 1.0
                n_blocked += 1
            if not attached and grip < 0 and dist < 0.10 and fp[2] > 0.3:
                attached = True
                offset = fp - tcp
            if attached and grip > 0 and over_bin and tcp[2] < 0.62:
                p = np.array([bx, by, 0.08])
                actor.set_pose(Pose.create_from_pq(torch.tensor([p], dtype=torch.float32)))
                actor.set_linear_velocity(torch.zeros(1, 3))
                attached = False
                released = True
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
        fp = actor.pose.p[0].cpu().numpy()
        bin_err = float(np.linalg.norm(fp[:2] - np.array([bx, by])))
        if ok:
            why = None
        elif released:
            why = "wrong_landing"
        elif attached:
            why = "not_released"
        elif dist < 0.10:
            why = "attached_too_late"   # 曾贴近但没吸上
        else:
            why = "no_attach"
        results.append({"success": ok, "duration_s": (t + 1) / args.control_freq,
                        "steps": t + 1, "failure": why, "attached": attached,
                        "released": released, "inferences": n_infer,
                        "gate_blocks": n_blocked,
                        "bin_err_m": round(bin_err, 3)})
        print(f"[{'✓' if ok else '✗'}] ep {ep:02d} steps={t + 1} infer={n_infer} "
              f"attach={attached} release={released} bin_err={bin_err:.3f} "
              f"gate_block={n_blocked} why={why}", flush=True)
        env.close()

    s = summarize(results)
    print(f"\nACT+吸盘 (chunk_exec={args.chunk_exec}, belt={args.belt}): {format_summary(s)}")
    out = Path(args.out) if args.out else ROOT / "runs" / f"{time.strftime('%Y-%m-%d-%H%M')}-act-suction"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({
        "ckpt": str(args.ckpt), "params": count_params(model),
        "instruction": args.instruction, "episodes": args.episodes,
        "belt": args.belt, "chunk_exec": args.chunk_exec,
        "delta_actions": delta, "gate": args.gate, "latency": lat, "summary": s,
        "episodes_detail": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {out/'metrics.json'}")


if __name__ == "__main__":
    main()
