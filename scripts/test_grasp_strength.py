#!/usr/bin/env python
"""灵巧手 vs 平行夹爪：静态抓取强度对比（同一颗水果、同一位置）。

做法
----
1. 载入机器人（`xarm7_ability` 灵巧手 或 `panda_wristcam` 平行夹爪）；
2. 把水果放到"手指中心"（指尖质心 + 少量偏置）；
3. 闭合手指 / 夹爪，等 30 步；
4. 对水果施加**递增的向上拉力**，记录它脱离前后的最大拉力 = 抗滑脱力（抓取强度）；
5. 打印：接触力、指尖数、抓取强度。

用法:
    python scripts/test_grasp_strength.py --robot xarm7_ability --kind pear
    python scripts/test_grasp_strength.py --robot panda_wristcam --kind pear
"""
from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
import torch

import fruitsort.sim.tasks  # noqa: F401
from fruitsort.sim.assets.fruits import load_specs


def _make_is_grasping():
    """任意指尖有 >0.5N 接触即视作抓住（灵巧手没有内置 is_grasping）。"""
    def _is_grasping(self, obj=None, min_force: float = 0.5, **kw):
        n = self.robot.get_qpos().shape[0]
        out = torch.zeros(n, dtype=torch.bool)
        try:
            tips = [l for l in self.robot.links if "tip" in l.name or "_L2" in l.name]
            f = sum(float(t.get_net_contact_forces().abs().sum()) for t in tips)
            if f > min_force:
                out[:] = True
        except Exception:  # noqa: BLE001
            pass
        return out
    return _is_grasping


def patch_hand(uid: str):
    """给没有 tcp_pose 的灵巧手打补丁：用掌/前臂末端当"tcp"。"""
    if uid.startswith("xarm7"):
        from mani_skill.agents.robots.xarm.xarm7_ability import XArm7Ability

        def _tcp(self):
            lm = self.robot.links_map
            for n in ("palm", "hand_palm", "base_link", "hand_base", "right_hand"):
                if n in lm:
                    return lm[n].pose
            return self.robot.links[-1].pose

        XArm7Ability.tcp_pose = property(_tcp)
        XArm7Ability.is_grasping = _make_is_grasping()
    elif uid.startswith("dclaw"):
        from mani_skill.agents.robots.dclaw.dclaw import DClaw

        DClaw.tcp_pose = property(lambda self: self.robot.links_map["base_link"].pose)
        DClaw.is_grasping = _make_is_grasping()
    elif uid.startswith("allegro"):
        from mani_skill.agents.robots.allegro_hand.allegro import AllegroHandRight

        def _tcp(self):
            lm = self.robot.links_map
            for n in ("palm", "base_link", "hand_base"):
                if n in lm:
                    return lm[n].pose
            return self.robot.links[-1].pose

        AllegroHandRight.tcp_pose = property(_tcp)
        AllegroHandRight.is_grasping = _make_is_grasping()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="xarm7_ability")
    ap.add_argument("--kind", default="pear")
    ap.add_argument("--offset", type=float, default=0.045,
                    help="水果中心相对指尖质心的偏移（沿掌心法向）")
    ap.add_argument("--close-steps", type=int, default=40)
    ap.add_argument("--pull-max", type=float, default=30.0, help="最大拉力 (N)")
    ap.add_argument("--pull-steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    patch_hand(args.robot)
    env = gym.make(
        "FruitSort-v1", obs_mode="state", control_mode="pd_joint_pos", sim_backend="cpu",
        robot_uids=args.robot, active_kinds=("apple", "strawberry", "banana", "pear"),
        belt_speed_range=(0.0, 0.0), spawn_x=-0.55, spawn_spacing=0.35,
        instruction_kind=args.kind, render_mode="rgb_array",
    )
    obs, _ = env.reset(seed=args.seed)
    u = env.unwrapped
    robot = u.agent.robot
    dof = robot.get_qpos().shape[-1]
    names = [j.name for j in robot.get_active_joints()]
    q = robot.get_qpos()[0].cpu().numpy().copy()
    lim = robot.get_qlimits().cpu().numpy()[0]

    # ---- 手指 = 名字里含 finger 的 link 的末端（*tip / *_L2）----
    tip_names = [n for n in [l.name for l in robot.links]
                 if "tip" in n or "_L2" in n or "head" in n]
    if not tip_names:      # 平行夹爪：两指本身就是"指尖"
        tip_names = [n for n in [l.name for l in robot.links] if "finger" in n]
    tips = [robot.links_map[n] for n in tip_names]
    tip_p = torch.stack([t.pose.p[0] for t in tips]).mean(0).cpu().numpy()
    palm = u.agent.tcp_pose
    palm_p = palm.p[0].cpu().numpy()
    normal = palm_p - tip_p
    n_len = np.linalg.norm(normal)
    normal = normal / n_len if n_len > 1e-6 else np.array([0, 0, -1.0])

    # ---- 把水果放到指尖中心（沿掌心法向往外一点）----
    fruit = u.fruit_actors[args.kind]
    target = tip_p + normal * args.offset
    target[2] = max(target[2], 0.55)
    fruit.set_pose(__import__("mani_skill.utils.structs.pose", fromlist=["Pose"]).Pose.create_from_pq(
        torch.tensor([target], dtype=torch.float32)))
    fruit.set_linear_velocity(torch.zeros(1, 3))
    fruit.set_angular_velocity(torch.zeros(1, 3))

    # ---- 闭合姿态：手部关节朝"收拢"方向打到限位 ----
    arm_names = {f"joint{i}" for i in range(1, 8)}
    hand_idx = [i for i, n in enumerate(names) if n not in arm_names]
    close_q = q.copy()
    for i in hand_idx:
        lo, hi = lim[i]
        close_q[i] = hi if abs(hi) < abs(lo) else lo     # 朝 0 更近的一侧
    # 找让指尖更靠近掌心的一侧
    def span(qq):
        robot.set_qpos(torch.tensor(qq[None], dtype=torch.float32))
        p = torch.stack([t.pose.p[0] for t in tips]).mean(0).cpu().numpy()
        return float(np.linalg.norm(p - palm_p))
    base_span = span(q)
    robot.set_qpos(torch.tensor(q[None], dtype=torch.float32))
    best_q, best_span = q.copy(), base_span
    for sign in (+1, -1):
        qq = q.copy()
        for i in hand_idx:
            lo, hi = lim[i]
            qq[i] = hi if sign > 0 else lo
        s = span(qq)
        if s < best_span:
            best_q, best_span = qq, s

    # ---- 闭合 ----
    na = int(env.action_space.shape[0])
    def act_of(qq):
        if na == dof:
            return qq.astype(np.float32)
        if na == dof - 1:                 # 平行夹爪：7 臂 + 1 夹爪（-1=闭）
            return np.concatenate([qq[:7], np.array([-1.0])]).astype(np.float32)
        return qq[:na].astype(np.float32)
    for _ in range(args.close_steps):
        obs, *_ = env.step(act_of(best_q))
    fn = 0.0
    try:
        for t in tips:
            fn += float(t.get_net_contact_forces()[0].norm())
    except Exception:  # noqa: BLE001
        pass
    d_close = float(np.linalg.norm(fruit.pose.p[0].cpu().numpy() - best_q[:0].sum() - target))

    # ---- 递增拉力 ----
    p0 = fruit.pose.p[0].cpu().numpy().copy()
    slip_force, slipped_at = None, None
    for k in range(args.pull_steps):
        F = args.pull_max * (k + 1) / args.pull_steps
        fruit._bodies[0].add_force_torque(np.array([0.0, 0.0, F], np.float32), np.zeros(3, np.float32))
        obs, *_ = env.step(act_of(best_q))
        p = fruit.pose.p[0].cpu().numpy()
        if np.linalg.norm(p - p0) > 0.06:
            slip_force, slipped_at = F, k
            break
    print(f"{args.robot:20s} {args.kind:6s} | 指尖接触合力 {fn:6.1f} N | 闭合后指尖跨距 {best_span*100:.1f}cm "
          f"| 抗滑脱力 {'>' if slip_force is None else '='}{args.pull_max if slip_force is None else slip_force:.1f} N")
    img = env.render()
    arr = img.cpu().numpy() if hasattr(img, "cpu") else np.asarray(img)
    arr = arr[0] if arr.ndim == 4 else arr
    import cv2
    cv2.imwrite(f"/tmp/opencode/grasp_{args.robot}_{args.kind}.png",
                np.ascontiguousarray(arr[..., ::-1].astype(np.uint8)))
    env.close()


if __name__ == "__main__":
    main()
