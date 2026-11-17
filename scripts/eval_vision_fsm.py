#!/usr/bin/env python
"""模型 + 夹爪状态机的闭环评测（v0.6）。

- 模型：按指令选择水果（选择头），并在 APPROACH 阶段给出预瞄目标
- 状态机 `GripperFSM`：负责对齐、下探、闭爪时序、验证、搬运、放置
- 执行：6 cm/步限速 + pd_ee_pose

用法:
    python scripts/eval_vision_fsm.py --ckpt runs/vpolicy-sel3/ckpt.pt \
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
from fruitsort.policies.gripper_fsm import GripperFSM, GripperFSMConfig, GripperState
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
    ap.add_argument("--kinds", nargs="+", default=["apple", "strawberry"])
    ap.add_argument("--instruction", default="apple",
                    choices=["all", "apple", "strawberry", "banana", "pear"])
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--belt-speed", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--model-approach", action="store_true",
                    help="APPROACH 阶段用模型目标，其余阶段交给状态机")
    ap.add_argument("--video", default="")
    ap.add_argument("--lead", type=float, default=0.12)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(Path(args.ckpt), map_location="cpu", weights_only=False)
    cfg = PolicyConfig(**ck["cfg"])
    model = MultiViewTemporalPolicy(cfg).to(args.device)
    model.load_state_dict(ck["model"]); model.eval()
    ns = Normalizer.from_state(ck["norm_state"]); na = Normalizer.from_state(ck["norm_act"])

    results, frames = [], []
    for ep in range(args.episodes):
        env = gym.make(
            "FruitSort-v1", obs_mode="rgbd", control_mode="pd_ee_pose", sim_backend="cpu",
            robot_uids="panda_wristcam", active_kinds=tuple(args.kinds),
            belt_speed_range=(args.belt_speed, args.belt_speed),
            instruction_kind=None if args.instruction == "all" else args.instruction,
            render_mode="rgb_array" if args.video else None,
        )
        obs, _ = env.reset(seed=args.seed + ep)
        u = env.unwrapped
        fsm = GripperFSM(GripperFSMConfig(lead=args.lead, max_retries=args.retries))
        ohs, wrs, sts = [], [], []
        prev_tgt = None
        info, t = {}, 0
        for t in range(args.max_steps):
            sd = obs["sensor_data"]
            oh = prep(sd["overhead"]["rgb"][0].cpu().numpy(), cfg.img_size)
            wr = prep(sd["hand_camera"]["rgb"][0].cpu().numpy(), cfg.img_size)
            st = build_obs(u)
            ohs.append(oh); wrs.append(wr); sts.append(st)
            for h in (ohs, wrs, sts):
                if len(h) > cfg.hist:
                    h.pop(0)
            while len(ohs) < cfg.hist:
                ohs.insert(0, ohs[0]); wrs.insert(0, wrs[0]); sts.insert(0, sts[0])

            with torch.no_grad():
                pred, _, sel = model(
                    torch.from_numpy(np.stack(ohs)[None]).float().to(args.device),
                    torch.from_numpy(np.stack(wrs)[None]).float().to(args.device),
                    torch.from_numpy(ns.obs(np.stack(sts).astype(np.float32))[None]).float().to(args.device),
                    torch.tensor([INSTR_ID[args.instruction]], device=args.device))
            slot = int(sel.argmax(-1).item())
            chosen = ID_KIND.get(slot, args.instruction if args.instruction != "all" else "apple")
            if chosen not in u.fruit_actors:
                chosen = args.instruction if args.instruction in u.fruit_actors else list(u.fruit_actors)[0]

            fp = u.fruit_actors[chosen].pose.p[0].cpu().numpy()
            fv = u.fruit_actors[chosen].linear_velocity[0].cpu().numpy()
            tcp = u.agent.tcp_pose.p[0].cpu().numpy()
            gap = float(u.agent.robot.get_qpos()[0, -2:].sum())
            grasped = bool(u.agent.is_grasping(u.fruit_actors[chosen])[0])

            out = fsm.step(GripperState(
                tcp_pos=tcp, fruit_pos=fp, fruit_vel=fv, fingers_gap=gap,
                is_grasping=grasped, bin_xy=np.array(BIN_LAYOUT[chosen]), dt=0.05))
            tgt = out.target.copy()
            grip = out.grip

            if args.model_approach and out.phase in ("APPROACH", "ALIGN"):
                mtgt = np.clip(na.unact(pred[0, 0, :3].cpu().numpy()), -2, 2)
                if prev_tgt is not None:
                    mtgt = 0.3 * prev_tgt + 0.7 * mtgt
                prev_tgt = mtgt
                tgt = mtgt

            err = tgt - tcp
            d = float(np.linalg.norm(err))
            if d > 0.06:
                tgt = tcp + err / d * 0.06
            cmd = np.array([tgt[1] + 0.5, -tgt[0], tgt[2], *HOME_EULER, grip], dtype=np.float32)
            obs, _, term, trunc, info = env.step(cmd)

            if args.video and t % 5 == 0:
                img = env.render()
                arr = img.cpu().numpy() if hasattr(img, "cpu") else np.asarray(img)
                frames.append(arr[0] if arr.ndim == 4 else arr)
            if bool(info.get("success") or info.get("fail")):
                break
            # FSM 完成后不要立刻退出：水果还在下落，需等待 settled 判定

        ok = bool(info.get("success", False))
        results.append({"success": ok, "duration_s": (t + 1) / 20.0})
        print(f"[{'✓' if ok else '✗'}] ep {ep:02d} instr={args.instruction} "
              f"phase={fsm.phase} retries={fsm.retries} steps={t + 1}")
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
    print("\n模型+状态机:", format_summary(s))
    out_dir = ROOT / "runs" / f"{time.strftime('%Y-%m-%d-%H%M')}-vfsm-eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(
        {"ckpt": str(args.ckpt), "instruction": args.instruction, "episodes": args.episodes,
         "model_approach": args.model_approach, "summary": s}, ensure_ascii=False, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
