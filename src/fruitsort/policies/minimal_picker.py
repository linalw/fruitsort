"""极简抓取控制器（香蕉/梨调参结论的落地版）。

关键参数（来自抓取攻关实验）：
- lead=0.18 预瞄、depth=-0.01 下探、trigger=0.01 闭爪触发
- 抓后先以 1.5 cm/步垂直抬 5 cm，再走 via → 料框（避免甩飞）
"""
from __future__ import annotations

import numpy as np

from ..sim.conveyor import BIN_LAYOUT
from .scripted import HOME_EULER

MAX_STEP = 0.06


class MinimalPicker:
    def __init__(self, env, kind: str, lead: float = 0.18, depth: float = -0.01,
                 trigger: float = 0.01, approach_h: float = 0.10,
                 lift_dz: float = 0.05, lift_step: float = 0.015,
                 carry_z: float = 0.60, via=(0.0, -0.34)):
        self.u = env.unwrapped
        self.kind = kind
        self.lead, self.depth, self.trigger = lead, depth, trigger
        self.approach_h, self.lift_dz, self.lift_step = approach_h, lift_dz, lift_step
        self.carry_z, self.via = carry_z, via
        self.carrying = False
        self.lift_done = False
        self.via_done = False

    def _tcp(self):
        return self.u.agent.tcp_pose.p[0].cpu().numpy()

    def act(self):
        u = self.u
        actor = u.fruit_actors[self.kind]
        fp = actor.pose.p[0].cpu().numpy()
        fv = actor.linear_velocity[0].cpu().numpy()
        tcp = self._tcp()
        bx, by = BIN_LAYOUT[self.kind]

        if self.carrying:
            if not self.lift_done:
                tgt = np.array([tcp[0], tcp[1], tcp[2] + self.lift_step])
                if tcp[2] > (fp[2] + self.lift_dz):
                    self.lift_done = True
                grip = -1.0
            elif not self.via_done:
                tgt = np.array([self.via[0], self.via[1], self.carry_z])
                if abs(tcp[0] - self.via[0]) < 0.05 and abs(tcp[1] - self.via[1]) < 0.05:
                    self.via_done = True
                grip = -1.0
            else:
                over = abs(tcp[0] - bx) < 0.03 and abs(tcp[1] - by) < 0.03
                if over and tcp[2] < 0.55:
                    tgt = np.array([bx, by, 0.50])
                    grip = 1.0
                else:
                    tgt = np.array([bx, by, self.carry_z])
                    grip = -1.0
        else:
            xg = float(np.clip(fp[0] + fv[0] * self.lead, u.pick_zone[0] - 0.02, u.pick_zone[1]))
            aligned = abs(tcp[0] - xg) < 0.02 and abs(tcp[1] - fp[1]) < 0.02
            if aligned and u.pick_zone[0] <= fp[0] <= u.pick_zone[1]:
                tgt = np.array([xg, fp[1], fp[2] + self.depth])
                close = abs(tcp[0] - fp[0]) < self.trigger and abs(tcp[1] - fp[1]) < self.trigger
                grip = -1.0 if close else 1.0
                if close and bool(u.agent.is_grasping(actor)[0]):
                    self.carrying = True
            else:
                tgt = np.array([xg, fp[1], fp[2] + self.approach_h])
                grip = 1.0

        err = tgt - tcp
        d = float(np.linalg.norm(err))
        if d > MAX_STEP:
            tgt = tcp + err / d * MAX_STEP
        return np.array([tgt[1] + 0.5, -tgt[0], tgt[2], *HOME_EULER, grip], dtype=np.float32)
