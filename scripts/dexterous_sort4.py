#!/usr/bin/env python
"""Panda + D'Claw 任务级分拣（pin IK 版本）。

> ⚠️ 这是**诊断用**中间版本：D'Claw 腕到指尖 0.28 m、张开占地 24 cm，在本任务的
> 带面高度下无法把手指放到水果两侧（见 docs/model-architecture.md 6.25）。
> 正式的任务级对照请用 `scripts/dexterous_sort5.py`（pd_ee_pose + ScriptedSorter），
> 它支持 `--robot panda_allegro / panda_dclaw / panda_wristcam`。

原说明：Panda + D'Claw 任务级分拣（仿真 FK 查表抓取 + pin IK 搬运）。

为什么这么设计
--------------
1. **抓取精度靠仿真自身 FK**：pin 的 URDF 里 9 个手关节取同一归一化值，而仿真里每个手指
   关节范围不同 → 指尖位置差 5–8 cm。启动时把手指稳定到抓取姿态，然后用
   `robot.set_qpos()` + 读 `dc_link_f*_head` 做位置 IK，生成"y → 关节"查表，
   运行期只按 y 查表（运行期不再 set_qpos，避免打乱 PD）。
2. **搬运用 pin IK**：抬升/到框/放下不需要亚厘米精度，pin 的 8 cm 偏差可以接受。
3. **抓取站位固定、等水果送进来**：手的水平速度 ≲0.06 m/s < 带速 0.084 m/s，
   边追边降永远追不上；在固定站位降到位后等水果滑进张开的指间。
4. **偏航 157°**：三指张开时有一只手指正好挡在来料方向（方位角 −172°），
   转 157° 让 148° 的指间缺口正对来料。
5. 手张开时 x 方向占地 24 cm（会挡水果），下探用 pre-hand=+0.2（跨距 10.8 cm、占地 13 cm），
   闭指到 +0.45（跨距 6.6 cm）夹住 ~8 cm 的水果。

动作布局（pd_joint_pos）= [panda_joint1..7 (rad)] + [9 指归一化]

用法:
    python scripts/dexterous_sort4.py --kinds apple --episodes 1
    python scripts/dexterous_sort4.py --episodes 8 --belt 0.08
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pinocchio as pin
import torch

import fruitsort.sim.tasks  # noqa: F401
import fruitsort.sim.agents.panda_dclaw  # noqa: F401
from fruitsort.sim.conveyor import BIN_LAYOUT

ROOT = Path(__file__).resolve().parent.parent
KINDS = ("apple", "strawberry", "banana", "pear")
URDF = ("/home/ubuntu/linalw/App/minconda3/envs/maniskill3/lib/python3.10/"
        "site-packages/mani_skill/assets/robots/panda/panda_dclaw.urdf")
OPEN = -0.85
R_WB = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # R_z(+90°)
P_WB = np.array([0.0, -0.5, 0.0])                                        # robot_base
HOME_Q = np.array([0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, np.pi * 3 / 4, np.pi / 4])
TIP_LINKS = ("dc_link_f1_head", "dc_link_f2_head", "dc_link_f3_head")


class PinIK:
    """搬运用的粗略 IK（位置-only，误差 ~cm 级可接受）。"""

    def __init__(self, urdf: str):
        self.model = pin.buildModelFromUrdf(urdf)
        self.data = self.model.createData()
        self.tcp = self.model.getFrameId("panda_tcp")
        self.fids = [self.model.getFrameId(f) for f in TIP_LINKS]
        q = pin.neutral(self.model)
        q[:7] = HOME_Q
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        tip0 = np.mean([self.data.oMf[i].translation for i in self.fids], axis=0)
        self.o_local = self.data.oMf[self.tcp].rotation.T @ (
            tip0 - self.data.oMf[self.tcp].translation)
        self.R = self.data.oMf[self.tcp].rotation.copy()

    def tip_world_pin(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        p = np.mean([self.data.oMf[i].translation for i in self.fids], axis=0)
        return R_WB @ p + P_WB

    def track(self, tip_world, q_cmd, rate: float, hand_val: float = 0.0, iters: int = 8):
        pos = (R_WB.T @ (np.asarray(tip_world, float) - P_WB)) - self.R @ self.o_local
        T_des = pin.SE3(self.R.copy(), pos)
        q = np.zeros(self.model.nq)
        q[:7] = q_cmd
        q[7:] = float(hand_val)
        for _ in range(iters):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            e = pin.log6(self.data.oMf[self.tcp].inverse() * T_des).vector
            if np.linalg.norm(e) < 1e-4:
                break
            J = pin.computeFrameJacobian(self.model, self.data, q, self.tcp,
                                         pin.ReferenceFrame.LOCAL)[:, :7]
            q[:7] = np.clip(q[:7] + np.clip(
                J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), e), -0.25, 0.25), -2.9, 2.9)
        sol = q[:7]
        return q_cmd + np.clip(sol - q_cmd, -rate, rate)


class SimIK:
    """仿真 FK 位置 IK（只在启动标定时用）。"""

    def __init__(self, robot, tips, hand_q):
        self.robot, self.tips = robot, tips
        self.hand = np.asarray(hand_q, dtype=np.float64).copy()

    def tip(self, q7):
        self.robot.set_qpos(torch.tensor(
            np.concatenate([q7, self.hand])[None], dtype=torch.float32))
        return np.mean([l.pose.p[0].cpu().numpy() for l in self.tips], axis=0)

    def solve(self, target, q7, iters: int = 20, lam: float = 1e-3, clip: float = 0.12):
        q = np.asarray(q7, dtype=np.float64).copy()
        tgt = np.asarray(target, dtype=np.float64)
        for _ in range(iters):
            p = self.tip(q)
            err = tgt - p
            if np.linalg.norm(err) < 2e-3:
                break
            J = np.zeros((3, 7))
            for k in range(7):
                q2 = q.copy()
                q2[k] += 1e-3
                J[:, k] = (self.tip(q2) - p) / 1e-3
            q += np.clip(J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err), -clip, clip)
        return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=list(KINDS))
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--belt", type=float, default=0.08)
    ap.add_argument("--spawn-x", type=float, default=-0.55)
    ap.add_argument("--spawn-spacing", type=float, default=0.45)
    ap.add_argument("--control-freq", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--rate", type=float, default=0.10)
    ap.add_argument("--grasp-x", type=float, default=-0.14)
    ap.add_argument("--grasp-dz", type=float, default=0.0)
    ap.add_argument("--pre-hand", type=float, default=0.20)
    ap.add_argument("--close-val", type=float, default=0.45)
    ap.add_argument("--lift-dz", type=float, default=0.16)
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--base-z", type=float, default=0.30,
                    help="机器人基座抬高（D'Claw 腕到指尖 0.29 m，标准座高下够不到带面）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    # 抬高基座：D'Claw 腕到指尖 0.29 m，标准安装高度下静息姿态手指埋在传送带下方。
    # 这是"更长的末端需要更高的安装座"的机械选择，任务本身（水果/带速/料框/判据）不变。
    from fruitsort.sim.tasks.fruit_sort import FruitSortEnv
    FruitSortEnv.robot_base = (0.0, -0.50, float(args.base_z))
    global P_WB
    P_WB = np.array([0.0, -0.5, float(args.base_z)])

    env = gym.make(
        "FruitSort-v1", obs_mode="state", control_mode="pd_joint_pos", sim_backend="cpu",
        robot_uids="panda_dclaw", active_kinds=tuple(args.kinds),
        belt_speed_range=(args.belt, args.belt), spawn_x=args.spawn_x,
        spawn_spacing=args.spawn_spacing, instruction_kind=None,
        sim_config=dict(control_freq=args.control_freq, sim_freq=120))
    env.reset(seed=args.seed)
    u = env.unwrapped
    robot = u.agent.robot
    tips = [robot.links_map[n] for n in TIP_LINKS]
    ik = PinIK(URDF)

    # ---------------- 启动：抬到安全高度 + 稳定手指 + 一次性偏差标定 ----------------
    # 静息姿态下 D'Claw 指尖在 z≈−0.02（埋进地面），必须先抬起来
    q_arm0 = robot.get_qpos()[0].cpu().numpy()[:7].copy()
    for _ in range(70):
        q_arm0 = ik.track([args.grasp_x, 0.0, 0.60], q_arm0, 0.10, args.pre_hand)
        env.step(np.concatenate([q_arm0, np.full(9, args.pre_hand)]).astype(np.float32))
    for _ in range(40):
        env.step(np.concatenate([q_arm0, np.full(9, args.pre_hand)]).astype(np.float32))
    # pin 模型与仿真的指尖偏差（来自手指关节映射），在悬停位测一次，之后按阶段校正
    tip_real0 = np.mean([robot.links_map[n].pose.p[0].cpu().numpy() for n in TIP_LINKS], axis=0)
    tip_pin0 = ik.tip_world_pin(np.r_[q_arm0, np.full(9, args.pre_hand)])
    tip_delta = tip_real0 - tip_pin0
    print(f"指尖偏差（仿真−pin）= {np.round(tip_delta, 3).tolist()} m", flush=True)
    env.reset(seed=args.seed)

    results = []
    t0 = time.time()
    for kind in args.kinds:
        for ep in range(args.episodes):
            env.reset(seed=args.seed + ep * 17)
            u = env.unwrapped
            robot = u.agent.robot
            fruit = u.fruit_actors[kind]
            bx, by = BIN_LAYOUT[kind]
            q_arm = robot.get_qpos()[0].cpu().numpy()[:7].copy()
            hand = np.full(9, args.pre_hand, dtype=np.float64)
            state, t, info, n_close, n_desc, max_fn = "WAIT", 0, {}, 0, 0, 0.0
            tip_links = [robot.links_map[n] for n in TIP_LINKS]
            for t in range(args.max_steps):
                fp = fruit.pose.p[0].cpu().numpy()
                tip_real = np.mean([l.pose.p[0].cpu().numpy() for l in tip_links], axis=0)

                if state == "WAIT":
                    if fp[0] > args.grasp_x - 0.20 and fp[2] > 0.3:
                        state = "TRACK"
                elif state == "TRACK":
                    q_arm = ik.track(np.array([args.grasp_x, fp[1], 0.58]) - tip_delta,
                                     q_arm, args.rate, args.pre_hand)
                    hand[:] = args.pre_hand
                    if fp[0] > args.grasp_x - 0.20:
                        state, n_desc = "DESCEND", 0
                elif state == "DESCEND":
                    n_desc += 1
                    if n_desc % 15 == 0:      # 每 15 步做一次一次性偏差校正（不用每步反馈，避免发散）
                        tip_delta = tip_real - ik.tip_world_pin(np.r_[q_arm, hand])
                    q_arm = ik.track(np.array([args.grasp_x, fp[1], fp[2] + args.grasp_dz])
                                     - tip_delta, q_arm, args.rate, args.pre_hand)
                    hand[:] = args.pre_hand
                    if abs(tip_real[0] - fp[0]) < 0.030 and abs(tip_real[1] - fp[1]) < 0.030 \
                            and abs(tip_real[2] - fp[2]) < 0.035:
                        state, n_close = "CLOSE", 0
                elif state == "CLOSE":
                    hand[:] = args.close_val
                    n_close += 1
                    for ln in ("dc_link_f1_3", "dc_link_f2_3", "dc_link_f3_3") + TIP_LINKS:
                        if ln in robot.links_map:
                            max_fn = max(max_fn, float(
                                robot.links_map[ln].get_net_contact_forces()[0].norm()))
                    if n_close >= 15:
                        state = "LIFT"
                elif state == "LIFT":
                    q_arm = ik.track(np.array([fp[0], fp[1], fp[2] + args.lift_dz]) - tip_delta,
                                     q_arm, args.rate, args.close_val)
                    hand[:] = args.close_val
                    if fp[2] > 0.56:
                        state = "MOVE"
                elif state == "MOVE":
                    q_arm = ik.track(np.array([bx, by, 0.66]) - tip_delta, q_arm, args.rate, args.close_val)
                    hand[:] = args.close_val
                    if abs(tip_real[0] - bx) < 0.09 and abs(tip_real[1] - by) < 0.09:
                        state = "RELEASE"
                elif state == "RELEASE":
                    q_arm = ik.track(np.array([bx, by, 0.56]) - tip_delta, q_arm, args.rate, OPEN)
                    hand[:] = OPEN
                    if tip_real[2] < 0.62:
                        state = "DONE"
                elif state == "DONE":
                    hand[:] = OPEN
                    if t % 10 == 0:
                        info = u.evaluate()
                        if bool(info["success"][0]):
                            break

                obs, _, term, trunc, info = env.step(
                    np.concatenate([q_arm, hand]).astype(np.float32))
                if bool(term) or bool(trunc):
                    break
                if t % 60 == 0:
                    qa = robot.get_qpos()[0].cpu().numpy()
                    hits = sorted(((float(l.get_net_contact_forces()[0].norm()), l.name)
                                   for l in robot.links), reverse=True)[:2]
                    print(f"    t={t:4d} {state:8s} 果={np.round(fp, 2).tolist()} "
                          f"真实指尖={np.round(tip_real, 2).tolist()} 臂差最大={np.abs(qa[:7]-q_arm).max():.2f} "
                          f"接触={[(round(v, 0), n) for v, n in hits]}", flush=True)

            ok = bool(info.get("success", [False])[0]) if info else False
            results.append({"kind": kind, "ok": ok, "steps": t + 1, "state": state,
                            "t_s": (t + 1) / args.control_freq, "max_fn": max_fn})
            print(f"[{'✓' if ok else '✗'}] {kind:10s} ep{ep} 步 {t + 1} "
                  f"({(t + 1) / args.control_freq:.1f}s) 终态 {state} 峰值接触 {max_fn:.1f}N",
                  flush=True)

    n, ok = len(results), sum(r["ok"] for r in results)
    succ = [r["t_s"] for r in results if r["ok"]]
    print(f"\nPanda+D'Claw 分拣 @{args.belt} m/s: {ok}/{n} = {ok / max(n, 1) * 100:.1f}%"
          + (f" | 单件均值 {np.mean(succ):.2f}s（n={len(succ)}）" if succ else ""))
    for k in args.kinds:
        sub = [r for r in results if r["kind"] == k]
        if sub:
            print(f"  {k:10s} {sum(r['ok'] for r in sub)}/{len(sub)}"
                  f"  单件 {np.mean([r['t_s'] for r in sub]):.1f}s")
    out = Path(args.out) if args.out else ROOT / "runs" / f"{time.strftime('%Y-%m-%d-%H%M')}-dclaw-sort4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"robot": "panda_dclaw", "belt": args.belt, "results": results,
         "success": ok, "n": n, "args": vars(args)}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"结果写入 {out}/metrics.json | 用时 {(time.time() - t0) / 60:.1f} min")
    env.close()


if __name__ == "__main__":
    main()
