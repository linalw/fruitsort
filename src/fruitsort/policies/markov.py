"""Markov 几何标注器：DAgger 用的"正确动作"来源。

只依赖当前状态（无 FSM 记忆）：
- 手里有水果 → 去对应料框
- 否则 → 去拦截指令指定的水果；对齐后下探、闭爪

它不保证自己闭环能成功（抓取时序很敏感），但作为**标签器**，
能给策略访问到的任意状态一个方向正确、平滑的目标。
"""
from __future__ import annotations

import numpy as np

from ..sim.conveyor import BIN_LAYOUT


def markov_label(u, instruction: str | None):
    """返回 (world_target[3], grip) ; grip: -1 闭 / +1 开"""
    tcp = u.agent.tcp_pose.p[0].cpu().numpy()
    kinds = list(u.active_kinds)

    # 1) 是否已经抓住某个水果（必须"贴近 + 夹爪已闭合"，否则会把闭爪前的下探误判成已抓住）
    held = None
    for k in kinds:
        p = u.fruit_actors[k].pose.p[0].cpu().numpy()
        if np.linalg.norm(p - tcp) < 0.08 and bool(u.agent.is_grasping(u.fruit_actors[k])[0]):
            held = k
            break
    if held is not None:
        bx, by = BIN_LAYOUT[held]
        over = abs(tcp[0] - bx) < 0.03 and abs(tcp[1] - by) < 0.03
        if over and tcp[2] < 0.56:
            return np.array([bx, by, 0.50], dtype=np.float32), 1.0
        return np.array([bx, by, 0.58 if not over else 0.52], dtype=np.float32), -1.0

    # 2) 选目标水果
    target = None
    if instruction in kinds:
        target = instruction
    else:
        cands = []
        for k in kinds:
            p, _ = (u.fruit_actors[k].pose.p[0].cpu().numpy(),
                    u.fruit_actors[k].linear_velocity[0].cpu().numpy())
            if p[2] > 0.2 and p[0] < u.pick_zone[1] + 0.1:
                cands.append((p[0], k))
        if cands:
            target = max(cands)[1]
    if target is None:
        return np.array([0.0, -0.05, 0.66], dtype=np.float32), 1.0

    p = u.fruit_actors[target].pose.p[0].cpu().numpy()
    v = u.fruit_actors[target].linear_velocity[0].cpu().numpy()
    x = float(np.clip(p[0] + v[0] * 0.10, u.pick_zone[0] - 0.02, u.pick_zone[1]))
    y = float(p[1])
    aligned = abs(tcp[0] - x) < 0.03 and abs(tcp[1] - y) < 0.03
    in_zone = u.pick_zone[0] <= p[0] <= u.pick_zone[1]
    if aligned and in_zone:
        if tcp[2] <= p[2] + 0.02:
            return np.array([x, y, p[2]], dtype=np.float32), -1.0
        return np.array([x, y, p[2]], dtype=np.float32), 1.0
    return np.array([x, y, p[2] + 0.10], dtype=np.float32), 1.0
