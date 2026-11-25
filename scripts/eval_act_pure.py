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


def parse_map(spec: str) -> dict:
    """'apple=0.025,banana=0.02' -> {'apple': 0.025, 'banana': 0.02}"""
    out = {}
    for kv in (spec or "").split(","):
        kv = kv.strip()
        if not kv or "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        out[k.strip()] = float(v)
    return out

ROOT = Path(__file__).resolve().parent.parent
SIM_FREQ = {20: 100, 30: 120, 60: 240}


def queue_level(u) -> float:
    """队列标志：场上 |x|<0.6 且在带面的水果数 /4（与训练特征同源计算）。

    用于"按队列状态切换双策略"：<=阈值视为单件模式（用单发更强的模型），
    否则视为队列模式（用连续更强的模型）。
    """
    n = 0
    for k in ("apple", "strawberry", "banana", "pear"):
        a = u.fruit_actors.get(k)
        if a is None:
            continue
        p = a.pose.p[0].cpu().numpy()
        if abs(float(p[0])) < 0.6 and 0.40 < float(p[2]) < 0.60:
            n += 1
    return n / 4.0



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
    ap.add_argument("--z-bias", type=float, default=0.0,
                    help="抓取高度偏置（米，作用于下探/闭爪段）；梨建议 +0.015")
    ap.add_argument("--yaw-align", action="store_true",
                    help="用仿真真值 yaw 旋转手腕，使闭合方向垂直于水果长轴")
    ap.add_argument("--yaw-align-off", type=float, default=0.0)
    ap.add_argument("--yaw-align-extent", type=float, default=0.0,
                    help="只对世界长度 ≥ 该值的目标做腕部对准（0=全部；香蕉建议 0.12）")
    ap.add_argument("--yaw-hold-frac", type=float, default=1.0,
                    help="非长条目标保持的偏航比例（1=全保持，0.5=半保持）")
    ap.add_argument("--yaw-hold-cap", type=float, default=0.0,
                    help="非长条目标保持的偏航上限（rad，0=不限；建议 0.3–0.5）")
    ap.add_argument("--yaw-hold-fade", type=float, default=0.12,
                    help="距果心该距离内保持量线性淡出到 0（0=不淡出）")
    ap.add_argument("--yaw-align-fast", action="store_true",
                    help="预转用满增益/每步修正（快 3 倍，但抓取瞬间带角速度，易甩掉薄果）")
    ap.add_argument("--yaw-align-rot-frac", type=float, default=0.0,
                    help="预转腕时同时靠近的比例（0=原地只转；0.35=边转边靠近，省单件时间）")
    ap.add_argument("--stall-guard", type=float, default=0.0,
                    help="停滞守卫的最小接近步长（米，0=关闭）")
    ap.add_argument("--single-ckpt", default="",
                    help="双策略：队列标志<=--queue-thr 时改用该模型（单件模式）")
    ap.add_argument("--queue-thr", type=float, default=0.30)
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
    ap.add_argument("--lead-long", type=float, default=0.0,
                    help="长条目标的拦截点前移（秒）。实测有害：长条会被推到尖端去抓（默认关）")
    ap.add_argument("--lead-long-extent", type=float, default=0.11)
    ap.add_argument("--lead-long-taper", type=float, default=0.15,
                    help="前移在距果心该距离内线性衰减到 0（0=不衰减）")
    ap.add_argument("--z-gate", action="store_true")
    ap.add_argument("--chunk-exec", type=int, default=1, help="每 s 步重规划一次（增量模式下有意义）")
    ap.add_argument("--no-phase-fb", action="store_true", help="相位输入恒为 0（消融）")
    ap.add_argument("--max-step", type=float, default=0.06)
    ap.add_argument("--carry-max-step", type=float, default=0.0,
                    help="搬运段专用步长上限（0=与 max-step 相同）；抑制在料框处过冲")
    ap.add_argument("--close-z-off-map", default="",
                    help="闭爪门触发时的下探高度偏移（果心+该值），如 'strawberry=-0.005'")
    ap.add_argument("--close-z-gate-map", default="",
                    help="按类闭爪高度门，如 'apple=0.009,strawberry=0.006,pear=0.011'")
    ap.add_argument("--close-z-gate", type=float, default=0.0,
                    help="闭爪高度门：离果心高于该值时禁止闭爪、继续下探（0=关闭）")
    ap.add_argument("--close-commit", type=float, default=0.0,
                    help="闭爪承诺：离果心 z 低于该值且水平对齐时强制闭爪并冻结水平目标（0=关闭）")
    ap.add_argument("--close-commit-xy", type=float, default=0.030)
    ap.add_argument("--close-commit-map", default="",
                    help="按类闭爪承诺阈值，如 'apple=0.025,banana=0.02'（覆盖 --close-commit）")
    ap.add_argument("--close-commit-steps", type=int, default=8)
    ap.add_argument("--grasp-proxy", default="auto", choices=["auto", "always", "never"],
                    help="抓取近似判据（离得近+闭爪）：auto=仅当查不到真实接触时用")
    ap.add_argument("--suction", action="store_true",
                    help="吸盘/软末端：模型说吸且贴近果心即附着，搬运跟 TCP，到框上方释放")
    ap.add_argument("--suction-radius", type=float, default=0.10)
    ap.add_argument("--suction-carry", default="scripted", choices=["scripted", "policy"],
                    help="吸住后怎么搬到框：scripted=直接奔框（推荐）/policy=用策略搬运段")
    ap.add_argument("--grasp-verify", action="store_true",
                    help="接触但指面未到果身高度/未抬起时不算抓住（避免'带空气搬运'）")
    ap.add_argument("--grasp-axis-offset", type=float, default=0.0,
                    help="抓取点沿长轴偏移（米，只作用于长条目标；感知 yaw 给方向）")
    ap.add_argument("--grasp-axis-extent", type=float, default=0.12)
    ap.add_argument("--near-hold", action=argparse.BooleanOptionalAction, default=True,
                    help="感知：丢检且夹爪已压到目标上方时，位置取 TCP（不按带速外推；默认开）")
    ap.add_argument("--hand-cam", action="store_true",
                    help="fused 时再用腕部相机补最后 5 cm 的盲区（实验项，实测精度 1-3 cm）")
    ap.add_argument("--perception", default="oracle", choices=["oracle", "vision", "rgbd", "fused"],
                    help="oracle=仿真真值（默认）；vision=只用 overhead RGB 感知")
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
        src = None
        if args.perception == "vision":
            from fruitsort.perception.adapter import VisionSource
            src = VisionSource(env.unwrapped, dt=1.0 / args.control_freq)
        elif args.perception == "rgbd":
            from fruitsort.perception.adapter import RGBDSource
            src = RGBDSource(env.unwrapped, dt=1.0 / args.control_freq)
        elif args.perception == "fused":
            from fruitsort.perception.adapter import FusedSource
            src = FusedSource(env.unwrapped, dt=1.0 / args.control_freq,
                              use_hand=args.hand_cam, near_hold=args.near_hold,
                              grasp_axis_offset=args.grasp_axis_offset,
                              grasp_axis_extent=args.grasp_axis_extent)
        u = env.unwrapped
        kind = args.instruction
        # 按队列状态切换的双策略（单发场景 queue≈0.25 → 单件模式）
        if args.single_ckpt:
            q = queue_level(u)
            ckpt = args.single_ckpt if q <= args.queue_thr else args.ckpt
            print(f"[dual] queue={q:.2f} -> {Path(ckpt).parent.name}", flush=True)
        else:
            ckpt = args.ckpt
        runner = ActPureRunner(
            env, ckpt, kind, device=args.device, fb=args.fb, fb_z=args.fb_z,
            grip_ema=args.grip_ema, carry_latch=args.carry_latch,
            descend_commit=args.descend_commit, max_step=args.max_step, lead=args.lead,
            z_gate=args.z_gate, phase_fb=not args.no_phase_fb,
            lead_long=args.lead_long, lead_long_extent=args.lead_long_extent,
            lead_long_taper=args.lead_long_taper,
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
            close_ramp=args.close_ramp, stall_guard=args.stall_guard,
            yaw_align=args.yaw_align, yaw_align_off=args.yaw_align_off,
            yaw_align_extent=args.yaw_align_extent,
            yaw_align_rot_frac=args.yaw_align_rot_frac,
            yaw_align_fast=args.yaw_align_fast,
            yaw_hold_frac=args.yaw_hold_frac, yaw_hold_cap=args.yaw_hold_cap,
            yaw_hold_fade=args.yaw_hold_fade,
            grasp_verify=args.grasp_verify, grasp_proxy=args.grasp_proxy,
            suction=args.suction, suction_radius=args.suction_radius,
            suction_carry=args.suction_carry, close_z_gate=args.close_z_gate,
            close_commit=args.close_commit, close_commit_xy=args.close_commit_xy,
            close_z_gate_map=parse_map(args.close_z_gate_map),
            close_z_off_map=parse_map(args.close_z_off_map),
            close_commit_map=parse_map(args.close_commit_map),
            close_commit_steps=args.close_commit_steps,
            grasp_z_bias=({args.instruction: args.z_bias} if args.z_bias else {}))
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
            if src is not None:
                runner.set_actors(src.update(obs, held_kind=(runner.kind if runner.grasping else None)))
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
