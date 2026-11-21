#!/usr/bin/env python
"""灵巧手（xArm7 + Ability Hand）的**脚本化分拣**：任务级对照用。

为什么需要它
------------
静态抓取强度测试对"水果在掌内的放置位"极其敏感（同一颗梨 1.7N ↔ >40N 随 2cm 变化），
不能作为结论。任务级对照必须让机器人**自己够到、抓起、搬到框里**，因此需要 IK。
本机没有 pinocchio（现已 `pip install pin` 装上），这里用 pinocchio 对 xArm7 做数值 IK。

控制
----
动作 = 17 维关节目标（7 臂 + 10 手）。手指姿态只有"张开/握拳"两档；
臂用 IK 跟踪（palm 目标 = 指尖目标 ∘ 掌内偏移）。

用法:
    python scripts/dexterous_sort.py --kinds apple strawberry banana pear --episodes 8 --belt 0.08
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
from fruitsort.sim.conveyor import BIN_LAYOUT

ROOT = Path(__file__).resolve().parent.parent
KINDS = ("apple", "strawberry", "banana", "pear")
URDF = ("/home/ubuntu/linalw/App/minconda3/envs/maniskill3/lib/python3.10/"
        "site-packages/mani_skill/assets/robots/xarm7/xarm7_ability_right_hand.urdf")


def patch_hand():
    from mani_skill.agents.robots.xarm.xarm7_ability import XArm7Ability

    def _tcp(self):
        lm = self.robot.links_map
        for n in ("palm", "hand_palm", "base_link", "hand_base"):
            if n in lm:
                return lm[n].pose
        return self.robot.links[-1].pose

    def _is_grasping(self, obj=None, min_force: float = 0.5, **kw):
        out = torch.zeros(self.robot.get_qpos().shape[0], dtype=torch.bool)
        try:
            tips = [l for l in self.robot.links if "tip" in l.name]
            f = sum(float(t.get_net_contact_forces().abs().sum()) for t in tips)
            if f > min_force:
                out[:] = True
        except Exception:  # noqa: BLE001
            pass
        return out

    XArm7Ability.tcp_pose = property(_tcp)
    XArm7Ability.is_grasping = _is_grasping


class ArmIK:
    """xArm7 7 关节数值 IK（手部姿态固定）。"""

    def __init__(self, urdf: str, arm_joints: list[str], palm_frame: str, tip_frames: list[str]):
        self.model = pin.buildModelFromUrdf(urdf)
        self.data = self.model.createData()
        self.arm_joints = arm_joints
        self.arm_idx = [self.model.getJointId(j) for j in arm_joints]
        self.palm_id = self.model.getFrameId(palm_frame)
        self.tip_ids = [self.model.getFrameId(f) for f in tip_frames if f in
                        [fr.name for fr in self.model.frames]]
        self.q_nom = pin.neutral(self.model)

    def fk(self, q):
        return pin.forwardKinematics(self.model, self.data, q), pin.updateFramePlacements(self.model, self.data)

    def palm_pose(self, q):
        self.fk(q)
        return self.data.oMf[self.palm_id].copy()

    def tip_center(self, q):
        self.fk(q)
        return np.mean([self.data.oMf[i].translation for i in self.tip_ids], axis=0)

    def tip_offset_local(self, q0):
        """指尖中心相对 palm 的局部偏移（用于把"指尖目标"换成"palm 目标"）。"""
        M = self.palm_pose(q0)
        c = self.tip_center(q0)
        return M.inverse() * c

    def solve(self, q_init, palm_target: pin.SE3, iters: int = 60, damping: float = 1e-2):
        q = q_init.copy()
        for _ in range(iters):
            self.fk(q)
            err = pin.log6(self.data.oMf[self.palm_id].inverse() * palm_target).vector
            if np.linalg.norm(err) < 1e-3:
                break
            J = pin.computeFrameJacobian(self.model, self.data, q, self.palm_id,
                                         pin.ReferenceFrame.LOCAL)
            cols = [self.model.joints[i].idx_q for i in self.arm_idx]
            Jv = J[:, cols]
            dq = -Jv.T @ np.linalg.solve(Jv @ Jv.T + damping * np.eye(6), err)
            for k, i in enumerate(self.arm_idx):
                j = self.model.joints[i]
                q[j.idx_q] = np.clip(q[j.idx_q] + dq[k], -3.0, 3.0)
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
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    patch_hand()
    env = gym.make(
        "FruitSort-v1", obs_mode="state", control_mode="pd_joint_pos", sim_backend="cpu",
        robot_uids="xarm7_ability", active_kinds=tuple(args.kinds),
        belt_speed_range=(args.belt, args.belt), spawn_x=args.spawn_x,
        spawn_spacing=args.spawn_spacing, instruction_kind=None,
        sim_config=dict(control_freq=args.control_freq, sim_freq=120))
    obs, _ = env.reset(seed=args.seed)
    u = env.unwrapped
    robot = u.agent.robot
    names = [j.name for j in robot.get_active_joints()]
    dof = int(robot.get_qpos().shape[-1])
    arm_names = names[:7]
    hand_idx = [i for i, n in enumerate(names) if not n.startswith("joint")]
    lim = robot.get_qlimits().cpu().numpy()[0]
    q_open = robot.get_qpos()[0].cpu().numpy().copy()
    q_close = q_open.copy()
    for i in hand_idx:
        lo, hi = lim[i]
        q_close[i] = hi if abs(hi) < abs(lo) else lo

    ik = ArmIK(URDF, arm_names, "base",
               ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"])
    off_local = ik.tip_offset_local(q_open)

    def cmd_to_tip(tip_target: np.ndarray, q_hand: np.ndarray, q_arm_init: np.ndarray):
        """把"指尖目标"换成 palm 目标，再解 IK。"""
        M_palm_des = pin.SE3(np.eye(3), tip_target) * pin.SE3(np.eye(3), -off_local)
        q = ik.solve(np.concatenate([q_arm_init, q_open[7:]]), M_palm_des)
        out = q_open.copy()
        out[:7] = q[:7]
        out[7:] = q_hand[7:]
        return out

    results = []
    t0 = time.time()
    for kind in args.kinds:
        for ep in range(args.episodes):
            env.reset(seed=args.seed + ep * 17)
            u = env.unwrapped
            fruit = u.fruit_actors[kind]
            bx, by = BIN_LAYOUT[kind]
            state, q_arm, t = "WAIT", q_open[:7].copy(), 0
            info = {}
            n_close = 0
            for t in range(args.max_steps):
                fp = fruit.pose.p[0].cpu().numpy()
                fv = fruit.linear_velocity[0].cpu().numpy()
                tcp_tip = ik.tip_center(np.concatenate([q_arm, q_open[7:]]))
                d = float(np.linalg.norm(tcp_tip - fp))
                held = d < 0.08
                if state == "WAIT":
                    if u.pick_zone[0] <= fp[0] <= u.pick_zone[1] and fp[2] > 0.3:
                        state = "TRACK"
                elif state == "TRACK":
                    x = float(np.clip(fp[0] + fv[0] * 0.15, u.pick_zone[0], u.pick_zone[1]))
                    q_arm = cmd_to_tip(np.array([x, fp[1], fp[2] + 0.10]), q_open, q_arm)[:7]
                    if d < 0.055:
                        state = "DESCEND"
                elif state == "DESCEND":
                    q_arm = cmd_to_tip(np.array([fp[0] + fv[0] * 0.05, fp[1], fp[2] + 0.012]),
                                       q_close, q_arm)[:7]
                    if d < 0.03:
                        state = "CLOSE"
                elif state == "CLOSE":
                    n_close += 1
                    if held:
                        state = "LIFT"
                    elif n_close > 25:
                        state = "WAIT"
                elif state == "LIFT":
                    q_arm = cmd_to_tip(np.array([fp[0], fp[1], fp[2] + 0.16]), q_close, q_arm)[:7]
                    if fp[2] > 0.56 or t % 25 == 0:
                        state = "MOVE"
                elif state == "MOVE":
                    q_arm = cmd_to_tip(np.array([bx, by, 0.62]), q_close, q_arm)[:7]
                    if abs(tcp_tip[0] - bx) < 0.05 and abs(tcp_tip[1] - by) < 0.05:
                        state = "RELEASE"
                elif state == "RELEASE":
                    q_arm = cmd_to_tip(np.array([bx, by, 0.52]), q_open, q_arm)[:7]
                    if tcp_tip[2] < 0.56:
                        state = "DONE"
                act = np.concatenate([q_arm[:7], q_open[7:] if state in ("WAIT", "TRACK", "DESCEND", "RELEASE", "DONE") else q_close[7:]])
                obs, _, term, trunc, info = env.step(act.astype(np.float32))
                if state == "DONE" and t % 10 == 0:
                    info = u.evaluate()
                    if bool(info["success"][0]):
                        break
                if term or trunc:
                    break
            ok = bool(info.get("success", torch.zeros(1, dtype=torch.bool))[0])
            results.append({"kind": kind, "ok": ok, "steps": t + 1, "state": state})
            print(f"[{'✓' if ok else '✗'}] {kind:10s} ep{ep} 步 {t+1} 终态 {state}", flush=True)

    n = len(results)
    ok = sum(r["ok"] for r in results)
    print(f"\n灵巧手脚本化分拣: {ok}/{n} = {ok/max(n,1)*100:.1f}% | 用时 {(time.time()-t0)/60:.1f} min")
    for k in args.kinds:
        sub = [r for r in results if r["kind"] == k]
        if sub:
            print(f"  {k:10s} {sum(r['ok'] for r in sub)}/{len(sub)}")
    out = Path(args.out) if args.out else ROOT / "runs" / f"{time.strftime('%Y-%m-%d-%H%M')}-dexsort"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"robot": "xarm7_ability", "belt": args.belt, "results": results,
         "success": ok, "n": n}, ensure_ascii=False, indent=2), encoding="utf-8")
    env.close()


if __name__ == "__main__":
    main()
