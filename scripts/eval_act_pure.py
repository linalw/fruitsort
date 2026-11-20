#!/usr/bin/env python
"""ACT 策略的纯闭环评测（平行夹爪，无吸盘、无脚本执行器）。

动作（目标位姿 + 夹爪开合 + 何时搬运）全部由模型给出；部署端只叠加
残差反馈与去抖/恢复层，见 `src/fruitsort/policies/act_runner.py` 的说明。

用法:
    python scripts/eval_act_pure.py --ckpt runs/act-008-4c-v2/ckpt.pt \
        --instruction apple --episodes 10 --belt 0.08 --spawn-spacing 0.35 \
        --control-freq 30 --fb 0.5 --grip-ema 0.6 --carry-latch
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import fruitsort.sim.tasks  # noqa: F401
from fruitsort.eval import format_summary, summarize
from fruitsort.learning.bc import build_tactile
from fruitsort.policies.act_runner import ActPureRunner

ROOT = Path(__file__).resolve().parent.parent
SIM_FREQ = {20: 100, 30: 120, 60: 240}


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
    # ---- 部署配置（v1.4 达标配置）----
    ap.add_argument("--fb", type=float, default=0.0,
                    help="残差反馈增益：target += fb*clip(水果-TCP)。0=纯策略；>0 时策略给前馈、反馈收残差")
    ap.add_argument("--fb-z", action="store_true", help="反馈项也作用于 z（默认只作用 xy）")
    ap.add_argument("--grip-ema", type=float, default=0.0, help="夹爪 logit 平滑系数（0=关闭）")
    ap.add_argument("--carry-latch", action="store_true", help="搬运段连续 3 步预测开爪才松手")
    ap.add_argument("--regrasp", action="store_true",
                    help="触觉预测驱动：闭爪前若滑移头预测会滑，则开爪→下沉 2cm→再闭")
    ap.add_argument("--regrasp-thr", type=float, default=0.6)
    ap.add_argument("--drop-nn", default="", help="独立掉落预测器权重（runs/dropnet/ckpt.pt）")
    ap.add_argument("--grasp-gate", action="store_true",
                    help="抓取验证门：进搬运段后判 双指力/对称度/相对速度，不达标则带偏移重试")
    ap.add_argument("--gate-min-force", type=float, default=2.0)
    ap.add_argument("--gate-asym", type=float, default=0.5)
    ap.add_argument("--width-setpoint", action="store_true",
                    help="闭合到 w*(类)−δ 而不是合到底（位置等价的轻力抓取）")
    ap.add_argument("--width-delta", type=float, default=0.002)
    ap.add_argument("--width-g", type=float, default=None, help="直接用固定闭合指令值扫标定")
    ap.add_argument("--close-ramp", type=float, default=0.0,
                    help="渐进闭爪步长（如 0.15）：不一步合到底，减少闭爪瞬间弹飞")
    ap.add_argument("--side-servo", action="store_true",
                    help="闭爪时单侧接触 → 朝弱侧平移再闭（论文②）")
    ap.add_argument("--close-force-th", type=float, default=0.0,
                    help="闭合到力阈（N）：超过就停止继续闭合（力→位置近似）")
    ap.add_argument("--slip-guard", action="store_true",
                    help="触觉补救：搬运中检测到滑移就降速（抑制惯性滑脱）")
    ap.add_argument("--slip-thr", type=float, default=0.05)
    ap.add_argument("--descend-commit", type=int, default=0, help="下探承诺步数（相位迟滞）")
    ap.add_argument("--hold-latch", action="store_true", help="闭爪期间锁存目标（停住等水果）")
    ap.add_argument("--lead", type=float, default=0.0, help="前馈超前（秒）：目标 x += v*lead")
    ap.add_argument("--z-gate", action="store_true")
    ap.add_argument("--chunk-exec", type=int, default=1, help="每 s 步重规划一次（增量模式下有意义）")
    ap.add_argument("--no-phase-fb", action="store_true", help="相位输入恒为 0（消融）")
    ap.add_argument("--max-step", type=float, default=0.06)
    ap.add_argument("--carry-max-step", type=float, default=0.0,
                    help="搬运段专用步长上限（0=与 max-step 相同）；抑制在料框处过冲")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    results = []
    for ep in range(args.episodes):
        env = gym.make(
            "FruitSort-v1", obs_mode="rgbd", control_mode="pd_ee_pose", sim_backend="cpu",
            robot_uids="panda_wristcam", active_kinds=tuple(args.kinds),
            belt_speed_range=(args.belt, args.belt),
            spawn_x=args.spawn_x, spawn_spacing=args.spawn_spacing,
            instruction_kind=args.instruction,
            sim_config=dict(control_freq=args.control_freq,
                            sim_freq=SIM_FREQ.get(args.control_freq, 4 * args.control_freq)))
        obs, _ = env.reset(seed=args.seed + ep)
        u = env.unwrapped
        kind = args.instruction
        runner = ActPureRunner(
            env, args.ckpt, kind, device=args.device, fb=args.fb, fb_z=args.fb_z,
            grip_ema=args.grip_ema, carry_latch=args.carry_latch,
            descend_commit=args.descend_commit, max_step=args.max_step, lead=args.lead,
            z_gate=args.z_gate, phase_fb=not args.no_phase_fb,
            hold_latch=args.hold_latch, chunk_exec=args.chunk_exec,
            carry_max_step=args.carry_max_step, slip_guard=args.slip_guard,
            slip_thr=args.slip_thr, regrasp=args.regrasp,
            regrasp_thr=args.regrasp_thr, drop_ckpt=args.drop_nn,
            grasp_gate=args.grasp_gate, gate_min_force=args.gate_min_force,
            gate_asym=args.gate_asym, close_force_th=args.close_force_th,
            side_servo=args.side_servo,
            width_map=({"apple": 0.062, "strawberry": 0.034, "banana": 0.044, "pear": 0.068}
                       if args.width_setpoint else {}),
            width_delta=args.width_delta, width_g=args.width_g,
            close_ramp=args.close_ramp)
        zone_entry = None
        info, t = {}, 0
        # 「无损」指标（触觉）：峰值法向力、最大摩擦利用率、最大挤压比、累计滑移距离
        Fn_peak = rho_max = delta_max = slip_dist = 0.0
        held_any = False
        for t in range(args.max_steps):
            actor = u.fruit_actors.get(kind)
            if zone_entry is None and actor is not None:
                fx = float(actor.pose.p[0, 0])
                if u.pick_zone[0] <= fx <= u.pick_zone[1]:
                    zone_entry = t
            cmd = runner.act(obs)
            tac = build_tactile(u, kind)
            Fn_peak = max(Fn_peak, float(tac[0]))
            if float(tac[0]) > 1.0:   # Fn≈0 时 rho 无意义（自由运动期会爆表）
                rho_max = max(rho_max, float(tac[2]))
            delta_max = max(delta_max, float(tac[4]))
            if runner.phase_state == 2:
                held_any = True
                slip_dist += float(tac[5]) / args.control_freq
            obs, _, term, trunc, info = env.step(cmd)
            if bool(info.get("success") or info.get("fail")):
                break
        ok = bool(info.get("success", False))
        handle_s = ((t + 1 - zone_entry) / args.control_freq) if zone_entry is not None else None
        results.append({"success": ok, "duration_s": (t + 1) / args.control_freq,
                        "handle_s": handle_s, "Fn_peak": Fn_peak, "rho_max": rho_max,
                        "delta_max": delta_max, "slip_m": slip_dist,
                        "dropped": bool(held_any and not ok)})
        st = runner.stats
        print(f"[{'✓' if ok else '✗'}] ep {ep:02d} kind={kind} steps={t + 1} "
              f"handle={'-' if handle_s is None else f'{handle_s:.1f}s'} "
              f"close={st['close'] / max(st['steps'], 1):.2f} "
              f"descend={st['descend']} carry={st['carry']} "
              f"Fn={Fn_peak:.1f}N rho={rho_max:.2f} slip={slip_dist * 100:.1f}cm "
              f"slipguard={st.get('slip', 0)} regrasp={st.get('regrasp', 0)} "
              f"p_slip={st.get('slip_prob_max', 0):.2f} p_drop={st.get('drop_prob_max', 0):.2f} "
              f"gate={st.get('gate_fail', 0)} retry={st.get('retries', 0)} "
              f"side={st.get('side_push', 0)} gw={st.get('grip_w', 1.0):+.2f}")
        env.close()

    s = summarize(results)
    print("\nACT 纯策略:", format_summary(s))
    hs = [r["handle_s"] for r in results if r["success"] and r["handle_s"] is not None]
    if hs:
        print(f"单件处理时间（进入抓取区→入框）: 均值 {np.mean(hs):.2f}s  P90 "
              f"{np.percentile(hs, 90):.2f}s  最大 {np.max(hs):.2f}s")
    print(f"无损指标: 峰值法向力 均值 {np.mean([r['Fn_peak'] for r in results]):.1f} N | "
          f"最大摩擦利用率 均值 {np.mean([r['rho_max'] for r in results]):.2f} | "
          f"累计滑移 均值 {np.mean([r['slip_m'] for r in results]) * 100:.1f} cm | "
          f"抓起后掉落 {sum(r['dropped'] for r in results)}/{len(results)}")
    out = ROOT / "runs" / f"{time.strftime('%Y-%m-%d-%H%M')}-act-pure"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"ckpt": str(args.ckpt), "instruction": args.instruction, "episodes": args.episodes,
         "belt": args.belt, "summary": s, "args": vars(args)}, ensure_ascii=False, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
