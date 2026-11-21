#!/usr/bin/env python
"""灵巧手（xArm7 + Ability Hand）任务级分拣：**用仿真自身 FK 做数值 IK**。

为什么重写
----------
上一版用 pinocchio 的 URDF frame 做 IK，指尖 frame 与 SAPIEN link 原点不一致
（pin 指尖 0.296,-0.016,0.161 vs sim 0.101,-0.404,0.666），且直接下发关节目标会让
仿真 NaN。实测发现 **`robot.set_qpos(q)` 之后 link 位姿会立即更新**，因此可以
直接用仿真自己的 FK 做数值 IK，不需要任何 frame 标定。

控制：动作 = 17 维关节目标（7 臂 + 10 手），每步相对当前指令的变化 ≤ `--rate` rad，
手指只有"张开/握拳"两档。

用法:
    python scripts/dexterous_sort2.py --kinds apple --episodes 1     # 冒烟
    python scripts/dexterous_sort2.py --episodes 8 --belt 0.08
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
from fruitsort.sim.conveyor import BIN_LAYOUT

ROOT = Path(__file__).resolve().parent.parent
KINDS = ("apple", "strawberry", "banana", "pear")


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


class SimIK:
    """用仿真自身 FK 的位置数值 IK（只解 7 个臂关节，手指姿态固定）。"""

    def __init__(self, robot, tips, q_ref: np.ndarray, lo=None, hi=None):
        self.robot = robot
        self.tips = tips
        self.q_ref = q_ref.copy()
        self.lo = lo
        self.hi = hi

    def tip_center(self) -> np.ndarray:
        return np.mean([t.pose.p[0].cpu().numpy() for t in self.tips], axis=0)

    def fk(self, q: np.ndarray) -> np.ndarray:
        qq = q.copy()
        if self.lo is not None:
            qq[:7] = np.clip(qq[:7], self.lo[:7], self.hi[:7])
        self.robot.set_qpos(torch.tensor(qq[None], dtype=torch.float32))
        p = self.tip_center()
        return p if np.all(np.isfinite(p)) else None

    def solve(self, target, q_init, iters: int = 12, damping: float = 1e-3,
              step_clip: float = 0.12, tol: float = 3e-3) -> np.ndarray:
        q = q_init.copy()
        for _ in range(iters):
            p = self.fk(q)
            if p is None:
                break
            err = np.asarray(target) - p
            if np.linalg.norm(err) < tol:
                break
            J = np.zeros((3, 7))
            for k in range(7):
                q2 = q.copy()
                q2[k] += 1e-3
                p2 = self.fk(q2)
                J[:, k] = ((p2 - p) / 1e-3) if p2 is not None else 0.0
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), err)
            q[:7] += np.clip(dq, -step_clip, step_clip)
            if self.lo is not None:
                q[:7] = np.clip(q[:7], self.lo[:7], self.hi[:7])
        self.robot.set_qpos(torch.tensor(self.q_ref[None], dtype=torch.float32))
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
    ap.add_argument("--rate", type=float, default=0.15, help="每步关节指令最大变化 (rad)")
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
    hand_idx = [i for i, n in enumerate(names) if not n.startswith("joint")]
    lim = robot.get_qlimits().cpu().numpy()[0]
    q_now = robot.get_qpos()[0].cpu().numpy().copy()
    q_open = q_now.copy()
    q_close = q_now.copy()
    for i in hand_idx:
        lo, hi = lim[i]
        q_close[i] = hi if abs(hi) < abs(lo) else lo
    tips = [l for l in robot.links if "tip" in l.name]
    lo_, hi_ = lim[:, 0], lim[:, 1]
    ik = SimIK(robot, tips, q_now, lo_, hi_)

    # ---------- 启动时预计算 IK 查表（此阶段扰动无害，运行期不再写 sim 状态）----------
    print('预计算 IK 查表…', flush=True)
    grid = {}
    q_warm = robot.get_qpos()[0].cpu().numpy().copy()
    ik.q_ref = q_warm.copy()
    xs = np.linspace(-0.30, 0.26, 12)
    for zname, dz in (("hover", 0.10), ("grasp", 0.015)):
        for x in xs:
            tgt = np.array([x, 0.0, 0.48 + dz])
            q_warm = ik.solve(tgt, q_warm)
            grid[(zname, round(float(x), 3))] = q_warm[:7].copy()
    for kind in args.kinds:
        bx, by = BIN_LAYOUT[kind]
        for zname, z in (("bin_hi", 0.62), ("bin_lo", 0.52)):
            tgt = np.array([bx, by, z])
            q_warm = ik.solve(tgt, q_warm)
            grid[(zname + kind, 0.0)] = q_warm[:7].copy()
    print(f'  查表 {len(grid)} 项完成', flush=True)
    env.reset(seed=args.seed)

    def lookup(zname, x):
        keys = [k for k in grid if k[0] == zname]
        if not keys:
            return None
        k = min(keys, key=lambda kk: abs(kk[1] - float(x)))
        return grid[k]

    results = []
    t0 = time.time()
    for kind in args.kinds:
        for ep in range(args.episodes):
            env.reset(seed=args.seed + ep * 17)
            u = env.unwrapped
            robot = u.agent.robot
            ik.robot = robot
            ik.tips = [l for l in robot.links if "tip" in l.name]
            fruit = u.fruit_actors[kind]
            bx, by = BIN_LAYOUT[kind]
            q_cmd = robot.get_qpos()[0].cpu().numpy().copy()
            q_target = q_cmd.copy()
            last_tip = ik.tip_center()
            state, t = "WAIT", 0
            info = {}
            n_close = 0
            for t in range(args.max_steps):
                fp = fruit.pose.p[0].cpu().numpy()
                fv = fruit.linear_velocity[0].cpu().numpy()
                tip = ik.tip_center()
                if not np.all(np.isfinite(tip)):
                    tip = last_tip
                else:
                    last_tip = tip
                if state == "WAIT":
                    if u.pick_zone[0] <= fp[0] <= u.pick_zone[1] and fp[2] > 0.3:
                        state = "TRACK"
                elif state == "TRACK":
                    x = float(np.clip(fp[0] + fv[0] * 0.15, u.pick_zone[0], u.pick_zone[1]))
                    q_target = q_cmd.copy()
                    qg = lookup("hover", x)
                    if qg is not None:
                        q_target[:7] = qg
                    if abs(tip[0] - x) < 0.04 and abs(tip[1] - fp[1]) < 0.04:
                        state = "DESCEND"
                elif state == "DESCEND":
                    q_target = q_cmd.copy()
                    qg = lookup("grasp", fp[0] + fv[0] * 0.05)
                    if qg is not None:
                        q_target[:7] = qg
                    if abs(tip[0] - fp[0]) < 0.035 and abs(tip[1] - fp[1]) < 0.035 \
                            and tip[2] < fp[2] + 0.06:
                        state = "CLOSE"
                elif state == "CLOSE":
                    q_target = q_cmd.copy()
                    q_target[7:] = q_close[7:]
                    n_close += 1
                    if float(np.linalg.norm(tip - fp)) < 0.08 or n_close > 30:
                        state = "LIFT"
                elif state == "LIFT":
                    q_target = q_cmd.copy()
                    qg = lookup("hover", fp[0])
                    if qg is not None:
                        q_target[:7] = qg
                    q_target[7:] = q_close[7:]
                    if fp[2] > 0.55:
                        state = "MOVE"
                elif state == "MOVE":
                    q_target = q_cmd.copy()
                    qg = grid.get(("bin_hi" + kind, 0.0))
                    if qg is not None:
                        q_target[:7] = qg
                    q_target[7:] = q_close[7:]
                    if abs(tip[0] - bx) < 0.06 and abs(tip[1] - by) < 0.06:
                        state = "RELEASE"
                elif state == "RELEASE":
                    q_target = q_cmd.copy()
                    qg = grid.get(("bin_lo" + kind, 0.0))
                    if qg is not None:
                        q_target[:7] = qg
                    q_target[7:] = q_open[7:]
                    if tip[2] < 0.56:
                        state = "DONE"
                elif state == "DONE":
                    q_target = q_cmd.copy()
                    q_target[7:] = q_open[7:]
                    if t % 10 == 0:
                        info = u.evaluate()
                        if bool(info["success"][0]):
                            break
                # 关节指令限速
                dq = np.clip(q_target - q_cmd, -args.rate, args.rate)
                q_cmd = q_cmd + dq
                obs, _, term, trunc, info = env.step(q_cmd.astype(np.float32))
                if term or trunc:
                    print(f"   [终止] t={t} term={term} trunc={trunc} "
                          f"success={bool(info.get('success',[False])[0]) if torch.is_tensor(info.get('success', None)) else info.get('success')} "
                          f"fail={bool(info.get('fail',[False])[0]) if torch.is_tensor(info.get('fail', None)) else info.get('fail')}")
                    break
            ok = bool(info.get("success", torch.zeros(1, dtype=torch.bool))[0])
            results.append({"kind": kind, "ok": ok, "steps": t + 1, "state": state})
            print(f"[{'✓' if ok else '✗'}] {kind:10s} ep{ep} 步 {t+1} 终态 {state}", flush=True)

    n = len(results)
    ok = sum(r["ok"] for r in results)
    print(f"\n灵巧手脚本化分拣: {ok}/{n} = {ok / max(n,1) * 100:.1f}% | 用时 {(time.time()-t0)/60:.1f} min")
    for k in args.kinds:
        sub = [r for r in results if r["kind"] == k]
        if sub:
            print(f"  {k:10s} {sum(r['ok'] for r in sub)}/{len(sub)}")
    out = Path(args.out) if args.out else ROOT / "runs" / f"{time.strftime('%Y-%m-%d-%H%M')}-dexsort2"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"robot": "xarm7_ability", "belt": args.belt, "results": results,
         "success": ok, "n": n, "rate": args.rate}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    env.close()


if __name__ == "__main__":
    main()
