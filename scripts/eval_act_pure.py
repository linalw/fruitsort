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
    ap.add_argument("--carry-latch", action="store_true",
                    help="搬运段锁住夹爪：只有连续 3 步预测开爪才真的松手（防止中途抖动掉果）")
    ap.add_argument("--grip-ema", type=float, default=0.0,
                    help="夹爪 logit 指数平滑系数（0=关闭）。搬运中 logit 抖动会造成中途松爪掉落，"
                         "平滑后开/闭只跟随持续意图")
    ap.add_argument("--fb-z", action="store_true",
                    help="反馈项也作用于 z（默认只作用于 xy：竖直方向的下探/闭爪深度由策略决定）")
    ap.add_argument("--fb", type=float, default=0.0,
                    help="残差反馈增益：目标 += fb * clip(水果 - TCP)。0=纯策略；"
                         ">0 时策略提供前馈、反馈项负责把残余误差收掉（IBVS 式）")
    ap.add_argument("--descend-commit", type=int, default=0,
                    help="下探承诺步数：相位头一旦预测「下探」就保持 N 步（专家 FSM 进入 DESCEND 后不退回；"
                         "单帧回归无法表达这种承诺，于是闭环里会上下点头）")
    ap.add_argument("--hold-latch", action="store_true",
                    help="闭爪期间锁存目标（停住等水果进夹持区），而不是继续跟着水果走")
    ap.add_argument("--lead", type=float, default=0.0,
                    help="前馈超前补偿（秒）：目标 x 加上 v*lead，抵消策略的固定跟踪滞后（残差项）")
    ap.add_argument("--chunk-exec", type=int, default=1,
                    help="每 s 步重规划一次，其余步执行已预测的动作块（增量模式下有意义）")
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
    delta_actions = bool(ck.get("delta_actions", False))

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
        zone_entry = None   # 水果首次进入抓取区的步数（用于「单件处理时间」）
        queue = []          # 已预测但未执行的动作块（增量）
        closed_steps = 0    # 闭爪但未夹住的连续步数（再现专家的重试行为）
        grip_ema = None     # 夹爪 logit 平滑状态
        prev_grip = 1.0     # 上一步的夹爪指令（用于判定是否真的夹住）
        open_streak = 0     # 搬运段连续预测「开爪」的步数
        commit = 0          # 下探承诺剩余步数
        n_close, n_dec, n_carry = 0, 0, 0
        for t in range(args.max_steps):
            actor = u.fruit_actors.get(kind)
            grasping = False
            tcp = u.agent.tcp_pose.p[0].cpu().numpy().astype(np.float64)
            if actor is not None:
                fpos_now = actor.pose.p[0].cpu().numpy().astype(np.float64)
                # 搬运判据：手指接触，或「贴得很近且上一步确实闭爪」。
                # 不能用 |Δ|<8cm 单独判定——悬停在水果上方 6cm 也会满足，
                # 模型会以为自己抱着东西，一路"抱着空气"走向料框（草莓失败局就是这样）
                fd = float(np.linalg.norm(fpos_now - tcp))
                grasping = bool(u.agent.is_grasping(actor)[0]) or (fd < 0.045 and prev_grip < 0)
                fz = float(actor.pose.p[0, 2])
                if grasping:
                    phase_state, grasp_lost = 2, 0
                elif phase_state == 2:
                    # 抓丢了：回退到接近段，避免"单向门"
                    grasp_lost += 1
                    if fz < 0.52 or grasp_lost > 20:
                        phase_state, same, phase_cand = 0, 0, 0

            if zone_entry is None and actor is not None:
                fx = float(actor.pose.p[0, 0])
                if u.pick_zone[0] <= fx <= u.pick_zone[1]:
                    zone_entry = t
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

            if queue:
                p4 = queue.pop(0)
                logits = queue_logits
            else:
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
              acts_np = act[0].cpu().numpy()
              if args.chunk_exec > 1 and delta_actions:
                  queue = [a for a in acts_np[: args.chunk_exec]]
                  queue_logits = logits
              p4 = acts_np[0]
            tcp = u.agent.tcp_pose.p[0].cpu().numpy().astype(np.float64)
            tgt = na.unact(p4[:3])
            if delta_actions:
                # 训练/推理一致的增量模式：模型输出「相对当前 TCP 的位移」，
                # 滞后由每一步的误差反馈自然纠正（绝对目标会退化成轨迹回放）
                tgt = tcp + np.clip(tgt, -args.max_step, args.max_step)
            else:
                tgt = np.clip(tgt, -2, 2)
            if args.fb and kind in u.fruit_actors and not (actor is not None and grasping):
                fpos = u.fruit_actors[kind].pose.p[0].cpu().numpy().astype(np.float64)
                corr = np.clip(fpos - tcp, -0.15, 0.15)
                if not args.fb_z:
                    corr[2] = 0.0
                tgt = tgt + args.fb * corr
            if args.lead and kind in u.fruit_actors:
                # 残差控制器的前馈项：策略负责形状，前馈负责相位（补偿固定滞后）
                tgt[0] += args.lead * args.belt
            if args.z_gate and kind in u.fruit_actors:
                fz = u.fruit_actors[kind].pose.p[0, 2].item()
                tgt[2] = float(np.clip(tgt[2], fz - 0.02, 0.90))
            glogit = float(p4[3])
            if args.grip_ema > 0:
                a = args.grip_ema
                grip_ema = glogit if grip_ema is None else a * grip_ema + (1 - a) * glogit
                glogit = grip_ema
            grip = -1.0 if glogit > 0 else 1.0
            if args.carry_latch and grasping:
                open_streak = open_streak + 1 if glogit < 0 else 0
                if open_streak < 3:
                    grip = -1.0
            else:
                open_streak = 0
            n_close += int(grip < 0)
            n_carry += int(phase_state == 2)
            n_dec += int(phase_state == 1)

            # 相位反馈：用模型自己的预测（k=0），未真正抓住时不允许自认搬运，
            # 并要求连续 2 帧一致才切换（迟滞）；「下探」额外带承诺步数
            if logits is not None and phase_state != 2:
                p_cls = int(logits[0, 0].argmax())
                if p_cls == 2:
                    p_cls = 1
                if p_cls == 1:
                    phase_cand, same = 1, 1
                    commit = args.descend_commit
                    phase_state = 1
                elif commit > 0:
                    commit -= 1          # 承诺保持下探，不退回接近
                else:
                    if p_cls == phase_cand:
                        same += 1
                    else:
                        phase_cand, same = p_cls, 0
                    if same >= 1:
                        phase_state = phase_cand

            # 抓空复位（对应专家 FSM 的 close_steps=12）：闭爪 12 步没夹住就开爪重试，
            # 否则一次闭空会把整局浪费掉（trace 里模型闭着爪一路走向料框）
            if grip < 0 and not grasping:
                closed_steps += 1
                if closed_steps > 12:
                    grip = 1.0
                    closed_steps = 0
                    if phase_state == 1:
                        phase_state, same, phase_cand, commit = 0, 0, 0, 0
            else:
                closed_steps = 0
            if args.hold_latch and grip < 0 and not grasping:
                # 闭爪窗口内不再移动：停在原地等水果进夹持区
                tgt = tcp.copy()
            err = tgt - tcp
            d = float(np.linalg.norm(err))
            if d > args.max_step:
                tgt = tcp + err / d * args.max_step
            # 安全界限：不允许把目标压到果心以下（否则夹爪顶住带面，PD 饱和锁死手臂），
            # 也不允许在水果已经被越过时继续下探（解除下探段）
            if actor is not None:
                fpos = actor.pose.p[0].cpu().numpy().astype(np.float64)
                if not grasping:
                    tgt[2] = max(tgt[2], float(fpos[2]) - 0.02)
                    if float(tcp[0] - fpos[0]) < -0.08 and phase_state == 1:
                        phase_state, same, phase_cand = 0, 0, 0
            prev_grip = grip
            cmd = np.array([tgt[1] + 0.5, -tgt[0], tgt[2], *HOME_EULER, grip], dtype=np.float32)
            obs, _, term, trunc, info = env.step(cmd)
            if bool(info.get("success") or info.get("fail")):
                break
        ok = bool(info.get("success", False))
        handle_s = ((t + 1 - zone_entry) / args.control_freq) if zone_entry is not None else None
        results.append({"success": ok, "duration_s": (t + 1) / args.control_freq,
                        "handle_s": handle_s})
        print(f"[{'✓' if ok else '✗'}] ep {ep:02d} kind={kind} steps={t + 1} "
              f"handle={'-' if handle_s is None else f'{handle_s:.1f}s'} "
              f"close={n_close / max(t + 1, 1):.2f} descend={n_dec} carry={n_carry}")
        env.close()

    s = summarize(results)
    print("\nACT 纯策略:", format_summary(s))
    hs = [r["handle_s"] for r in results if r["success"] and r["handle_s"] is not None]
    if hs:
        print(f"单件处理时间（进入抓取区→入框）: 均值 {np.mean(hs):.2f}s  P90 "
              f"{np.percentile(hs, 90):.2f}s  最大 {np.max(hs):.2f}s")
    out = ROOT / "runs" / f"{time.strftime('%Y-%m-%d-%H%M')}-act-pure"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"ckpt": str(args.ckpt), "instruction": args.instruction, "episodes": args.episodes,
         "belt": args.belt, "z_gate": args.z_gate, "phase_fb": not args.no_phase_fb,
         "fb": args.fb, "hold_latch": args.hold_latch, "chunk_exec": args.chunk_exec,
         "grasp_retry": True, "descend_commit": args.descend_commit,
         "grip_ema": args.grip_ema, "carry_latch": args.carry_latch,
         "summary": s}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
