#!/usr/bin/env python
"""灵巧手 vs 平行夹爪：**任务级**抓放对照（抓取 → 抬升 → 转向料框 → 释放 → 判入框）。

为什么不用 IK
-------------
本机没有 pinocchio、ManiSkill 的 robot 也没暴露 compute_ik；17 维臂+手的 IK
代价过大。这里用"预置位 + 关节空间插值"完成抓放：
  1. 把目标水果放到"指尖中心"（按类标定的深度偏置，见 --offset）；
  2. 闭手指（朝限位收拢的一侧）；
  3. 抬升：肩关节 −0.3 rad（手整体上移）；
  4. 转向料框：base 关节按目标料框方位 ±1.2 rad（大致对准左/右两列料框）；
  5. 松手指；等 20 步后读 env 的 success（水果落入对应料框）。

用法:
    python scripts/dexterous_pickplace.py --robot xarm7_ability --episodes 8
    python scripts/dexterous_pickplace.py --robot panda_wristcam --episodes 8
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


def _make_is_grasping():
    def _is_grasping(self, obj=None, min_force: float = 0.5, **kw):
        out = torch.zeros(self.robot.get_qpos().shape[0], dtype=torch.bool)
        try:
            tips = [l for l in self.robot.links if "tip" in l.name or "_L2" in l.name]
            f = sum(float(t.get_net_contact_forces().abs().sum()) for t in tips)
            if f > min_force:
                out[:] = True
        except Exception:  # noqa: BLE001
            pass
        return out
    return _is_grasping


def patch(uid: str):
    if uid.startswith("xarm7"):
        from mani_skill.agents.robots.xarm.xarm7_ability import XArm7Ability

        def _tcp(self):
            lm = self.robot.links_map
            for n in ("palm", "hand_palm", "base_link", "hand_base"):
                if n in lm:
                    return lm[n].pose
            return self.robot.links[-1].pose

        XArm7Ability.tcp_pose = property(_tcp)
        XArm7Ability.is_grasping = _make_is_grasping()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="xarm7_ability")
    ap.add_argument("--kinds", nargs="+", default=list(KINDS))
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--offset", type=float, default=None,
                    help="水果放在指尖中心外的深度（默认：灵巧手 0.06，夹爪 0.0）")
    ap.add_argument("--close-steps", type=int, default=25)
    ap.add_argument("--lift-steps", type=int, default=25)
    ap.add_argument("--move-steps", type=int, default=35)
    ap.add_argument("--yaw", type=float, default=1.2)
    ap.add_argument("--mode", choices=["lift", "place"], default="lift",
                    help="lift=只测抓取+抬升（本机无 IK，搬运不可达）；place=继续转向料框")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    patch(args.robot)
    hand = not args.robot.startswith("panda")
    offset = args.offset if args.offset is not None else (0.06 if hand else 0.0)
    results = []
    t0 = time.time()
    for kind in args.kinds:
        for ep in range(args.episodes):
            env = gym.make(
                "FruitSort-v1", obs_mode="state", control_mode="pd_joint_pos", sim_backend="cpu",
                robot_uids=args.robot, active_kinds=tuple(args.kinds),
                belt_speed_range=(0.0, 0.0), spawn_x=-0.55, spawn_spacing=0.35,
                instruction_kind=kind,
            )
            obs, _ = env.reset(seed=args.seed + ep)
            u = env.unwrapped
            r = u.agent.robot
            na = int(env.action_space.shape[0])
            dof = int(r.get_qpos().shape[-1])
            names = [j.name for j in r.get_active_joints()]
            lim = r.get_qlimits().cpu().numpy()[0]
            q = r.get_qpos()[0].cpu().numpy().copy()

            tips = [l for l in r.links if "tip" in l.name or "_L2" in l.name]
            if not tips:
                tips = [l for l in r.links if "finger" in l.name]
            tip_p = torch.stack([t.pose.p[0] for t in tips]).mean(0).cpu().numpy()
            palm_p = u.agent.tcp_pose.p[0].cpu().numpy()
            nrm = palm_p - tip_p
            nrm = nrm / max(np.linalg.norm(nrm), 1e-6) if np.linalg.norm(nrm) > 1e-6 else np.array([0, 0, -1.0])

            fruit = u.fruit_actors[kind]
            if hand:
                target = tip_p + nrm * offset
            else:
                target = palm_p.copy()          # 平行夹爪：两指中点就是抓取点
                target[2] += offset
            from mani_skill.utils.structs.pose import Pose
            fruit.set_pose(Pose.create_from_pq(torch.tensor([target], dtype=torch.float32)))
            fruit.set_linear_velocity(torch.zeros(1, 3))
            fruit.set_angular_velocity(torch.zeros(1, 3))

            # 闭合姿态
            hand_idx = [i for i, n in enumerate(names) if not n.startswith("joint")]
            q_close = q.copy()
            for i in hand_idx:
                lo, hi = lim[i]
                q_close[i] = hi if abs(hi) < abs(lo) else lo

            def act_of(qq):
                if na == dof:
                    return qq.astype(np.float32)
                if na == dof - 1:                       # 平行夹爪：7 臂 + 1 夹爪
                    return np.concatenate([qq[:7], np.array([-1.0])]).astype(np.float32)
                return qq[:na].astype(np.float32)

            # 1) 闭指
            for _ in range(args.close_steps):
                env.step(act_of(q_close))
            fn = 0.0
            try:
                fn = sum(float(t.get_net_contact_forces().abs().sum()) for t in tips)
            except Exception:  # noqa: BLE001
                pass
            held_close = float(np.linalg.norm(fruit.pose.p[0].cpu().numpy() - tip_p))

            # 2) 抬升（肩关节）
            q_lift = q_close.copy()
            q_lift[1] -= 0.30
            for i in range(args.lift_steps):
                a = act_of(q_lift)
                env.step(a)
            z_after_lift = float(fruit.pose.p[0].cpu().numpy()[2])

            ok_lift = float(np.linalg.norm(fruit.pose.p[0].cpu().numpy() - tip_p)) < 0.10
            if args.mode == "lift":
                results.append({"kind": kind, "ok": ok_lift, "fn": fn,
                                "close_gap": held_close, "z_lift": z_after_lift})
                print(f"[{'✓' if ok_lift else '✗'}] {kind:10s} ep{ep} 接触 {fn:6.1f}N "
                      f"抬升后 z {z_after_lift:.2f} 果-指尖 {np.linalg.norm(fruit.pose.p[0].cpu().numpy()-tip_p)*100:.1f}cm",
                      flush=True)
                env.close()
                continue

            # 3) 转向料框
            q_move = q_lift.copy()
            bx, by = BIN_LAYOUT[kind]
            q_move[0] += args.yaw * np.sign(bx) if abs(bx) > 1e-6 else 0.0
            for _ in range(args.move_steps):
                env.step(act_of(q_move))

            # 4) 松指 + 沉降
            q_open = q_move.copy()
            for i in hand_idx:
                q_open[i] = 0.0
            for _ in range(40):
                a = act_of(q_open)
                if na == dof - 1:
                    a[-1] = 1.0
                env.step(a)
            info = u.evaluate()
            ok = bool(info["success"][0])
            results.append({"kind": kind, "ok": ok, "fn": fn,
                            "close_gap": held_close, "z_lift": z_after_lift})
            print(f"[{'✓' if ok else '✗'}] {kind:10s} ep{ep} 接触合力 {fn:6.1f}N "
                  f"闭合时果-指尖 {held_close*100:4.1f}cm 抬升后 z {z_after_lift:.2f}", flush=True)
            env.close()

    n = len(results)
    ok = sum(r["ok"] for r in results)
    print(f"\n{args.robot} 任务级抓放: {ok}/{n} = {ok / max(n,1) * 100:.1f}%  "
          f"用时 {(time.time() - t0) / 60:.1f} min")
    for k in args.kinds:
        sub = [r for r in results if r["kind"] == k]
        if sub:
            print(f"  {k:10s} {sum(r['ok'] for r in sub)}/{len(sub)}")
    out = Path(args.out) if args.out else ROOT / "runs" / f"{time.strftime('%Y-%m-%d-%H%M')}-dexpick-{args.robot}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"robot": args.robot, "offset": offset, "episodes": args.episodes,
         "summary": {"success": ok, "n": n}, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
