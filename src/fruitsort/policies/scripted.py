"""脚本化专家策略：预测拦截 + 顶部抓取 + 入框。

这是整个系统的**性能下界与数据源**：
- 不依赖学习，只用仿真真值位姿与速度；
- 后续用视觉感知替换真值（same interface），再用它采集示教训练 ACT/DP/VLA。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..sim.conveyor import BIN_LAYOUT

# 机器人基座朝向：绕 z +90°，世界向量 → 基座系
_R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


# 机器人 home 位姿对应的 TCP 欧拉角（XYZ，rad），抓取全程保持夹爪竖直向下
HOME_EULER = (-3.126, -0.017, -1.606)


@dataclass
class PolicyConfig:
    lead_time: float = 0.18      # 预瞄时间（秒）
    above_height: float = 0.09   # 抓取前悬停高度
    grasp_depth: float = -0.01     # 相对水果中心的额外下探
    lift_height: float = 0.24
    carry_height: float = 0.60
    via_point: tuple = (0.0, -0.34)
    bin_height: float = 0.50
    max_step: float = 0.06       # 每步最大位移（米）
    close_steps: int = 12
    release_steps: int = 8
    ready_pose: tuple = (0.0, -0.05, 0.68)


class ScriptedSorter:
    """有限状态机：WAIT → TRACK → DESCEND → CLOSE → LIFT → MOVE → RELEASE。"""

    def __init__(self, env, cfg: PolicyConfig | None = None, verbose: bool = False, source=None, only_kind=None):
        self.env = env
        self.u = env.unwrapped
        self.cfg = cfg or PolicyConfig()
        self.verbose = verbose
        self.source = source  # None = 仿真真值；否则需提供 .get(kind) -> (pos, vel) | None
        self.only_kind = only_kind  # 指令：只抓这一类
        self.state = "WAIT"
        self.target: str | None = None
        self.done: set[str] = set()
        self.missed: set[str] = set()
        self.counter = 0
        self._grasp_ok = False
        self._via_done = False
        self._last_pv = None
        self.last_target = None
        self._state_steps = 0
        self._prev_state = None

    # ---------------- 工具 ---------------- #
    def _tcp(self) -> np.ndarray:
        return self.u.agent.tcp_pose.p[0].cpu().numpy()

    def _fruit(self, kind: str) -> tuple[np.ndarray, np.ndarray]:
        if self.source is not None:
            est = self.source.get(kind)
            if est is None:
                raise LookupError(kind)
            return est
        a = self.u.fruit_actors[kind]
        return a.pose.p[0].cpu().numpy(), a.linear_velocity[0].cpu().numpy()

    def _visible(self, kind: str) -> bool:
        if self.source is None:
            return True
        return self.source.get(kind) is not None

    def _action(self, target: np.ndarray, gripper: float = 1.0) -> np.ndarray:
        """绝对位姿控制（robot root 坐标系），单步限速避免猛冲。"""
        tgt = np.asarray(target, dtype=np.float64)
        tcp = self._tcp()
        err = tgt - tcp
        dist = float(np.linalg.norm(err))
        if dist > self.cfg.max_step:
            tgt = tcp + err / dist * self.cfg.max_step
        # 记录未限速的世界目标（BC 的回归标签更平滑）
        self.last_target = np.array([tgt[0], tgt[1], tgt[2], gripper], dtype=np.float32)
        # 世界系 → 机器人基座系（基座绕 z 旋转 +90°）
        x, y, z = tgt
        cmd = np.array([y + 0.5, -x, z], dtype=np.float32)
        return np.array([*cmd, *HOME_EULER, gripper], dtype=np.float32)

    def _home(self, gripper: float = 1.0) -> np.ndarray:
        return self._action(np.array(self.cfg.ready_pose), gripper)

    def _in_zone(self, kind: str, margin: float = 0.0) -> bool:
        if not self._visible(kind):
            return False
        p, _ = self._fruit(kind)
        return self.u.pick_zone[0] - margin <= p[0] <= self.u.pick_zone[1] + margin

    # ---------------- 主循环 ---------------- #
    def act(self) -> np.ndarray:
        cfg = self.cfg
        self.counter += 1
        if self.state != self._prev_state:
            self._prev_state = self.state
            self._state_steps = 0
        self._state_steps += 1

        # 看门狗：避免单个状态无限期卡住
        if self.state in ("TRACK", "DESCEND") and self._state_steps > 500:
            if self.target:
                self.missed.add(self.target)
            self.state = "WAIT"
            self.target = None
            return self._home()

        if self.state == "WAIT":
            # 选最靠前（x 最大）且未处理、未进入框的水果
            cands = []
            for k in self.u.active_kinds:
                if self.only_kind is not None and k != self.only_kind:
                    continue
                if k in self.done or k in self.missed:
                    continue
                if not self._visible(k):
                    continue
                p, _ = self._fruit(k)
                if p[2] < self.u.belt_top_z - 0.2:  # 已掉落
                    self.missed.add(k)
                    continue
                if p[0] < self.u.pick_zone[0] - 0.85:
                    continue
                cands.append((p[0], k))
            if not cands:
                return self._home()
            self.target = max(cands)[1]
            self.state = "TRACK"
            if self.verbose:
                print(f"[policy] target={self.target}")

        if self.target is None:
            return self._home()

        try:
            p, v = self._fruit(self.target)
            self._last_pv = (p, v)
        except LookupError:
            # 目标被机械臂遮挡：用最后一次估计 + 恒速外推继续执行
            if self._last_pv is None:
                self.state = "WAIT"
                self.target = None
                return self._home()
            p, v = self._last_pv
            p = p + v * 0.05
        x_pred = p[0] + v[0] * cfg.lead_time
        x_cmd = float(np.clip(x_pred, self.u.pick_zone[0] - 0.02, self.u.pick_zone[1]))
        y = p[1]

        if self.state == "TRACK":
            if self.target in self.missed or self.target in self.done:
                self.state = "WAIT"
                return self._home()
            if p[0] > self.u.pick_zone[1] + 0.10:
                self.missed.add(self.target)
                self.state = "WAIT"
                return self._home()
            tcp = self._tcp()
            if abs(x_cmd - tcp[0]) < 0.055 and abs(y - tcp[1]) < 0.055 and \
                    self.u.pick_zone[0] <= p[0] <= self.u.pick_zone[1]:
                self.state = "DESCEND"
                self.counter = 0
            return self._action([x_cmd, y, p[2] + cfg.above_height], gripper=1.0)

        if self.state == "DESCEND":
            if p[0] > self.u.pick_zone[1] + 0.12:
                self.missed.add(self.target)
                self.state = "WAIT"
                return self._home()
            # 下探预瞄随带速自适应：高速时目标要更靠前，否则永远追不上
            speed = float(self.u.belt_speed[0])
            lead_desc = 0.12 if speed < 0.15 else (0.20 if speed < 0.25 else 0.28)
            x_g = float(np.clip(p[0] + v[0] * lead_desc, self.u.pick_zone[0] - 0.02, self.u.pick_zone[1]))
            y_g = p[1]
            grasp_z = p[2] + cfg.grasp_depth - (0.01 if self.only_kind == "pear" else 0.0)
            tcp = self._tcp()
            # 香蕉/梨较薄，且下探会顶到带面：放宽触底判定，避免永远不闭爪
            tol = 0.03 if self.only_kind in ("banana", "pear") else 0.018
            if speed >= 0.15:
                tol = max(tol, 0.055)  # 高速：每步水果移动 1cm，容差必须放宽
            aligned = abs(x_g - tcp[0]) < tol and abs(y_g - tcp[1]) < tol
            # 高速预测闭爪：水果即将到达/刚过夹爪时闭爪，不等到"位姿完全对齐"
            dx = float(tcp[0] - p[0])   # >0: 夹爪在水果前方
            pred_close = (speed >= 0.15 and self.only_kind != "pear"
                          and tcp[2] <= grasp_z + 0.06
                          and -0.01 <= dx <= 0.08 and abs(y_g - tcp[1]) < 0.04)
            if tcp[2] <= grasp_z + (0.045 if self.only_kind in ("banana", "pear") else 0.015) and aligned or pred_close:
                self.state = "CLOSE"
                self.counter = 0
            return self._action([x_g, y_g, grasp_z], gripper=1.0)

        if self.state == "CLOSE":
            # 贴着水果走并闭合夹爪
            x_g = float(np.clip(p[0] + v[0] * 0.08, self.u.pick_zone[0] - 0.02, self.u.pick_zone[1]))
            # is_grasping 只看手指接触力，对梨/香蕉等常出现假阴性；
            # 用"物体是否贴着 TCP"作为补充判据。
            dist = float(np.linalg.norm(p - self._tcp()))
            self._grasp_ok = bool(self.u.agent.is_grasping(self.u.fruit_actors[self.target])[0]) or (
                self.counter >= 4 and dist < 0.06)
            if self._grasp_ok:
                self.state = "CARRY"
                self.counter = 0
                self._via_done = False
            elif self.counter > cfg.close_steps:
                # 抓空：若还在可达区内就重试
                self.state = "WAIT"
                if p[0] > self.u.pick_zone[1] - 0.05:
                    self.missed.add(self.target)
                self.counter = 0
            grip = -min(1.0, 0.3 + self.counter / 8.0)  # 渐进闭爪，减少冲击
            return self._action([x_g, p[1], p[2] + cfg.grasp_depth], gripper=grip)

        if self.state == "CARRY":
            bx, by = BIN_LAYOUT[self.target]
            tcp = self._tcp()
            p_now, _ = self._fruit(self.target)
            if float(np.linalg.norm(p_now - self._tcp())) > 0.10:
                self.state = "WAIT"
                if self.u.fruit_actors[self.target].pose.p[0, 0] > self.u.pick_zone[1] - 0.05:
                    self.missed.add(self.target)
                self.target = None
                return self._home()
            if self.counter > 90:  # 卡住保护：放弃这颗，避免整局僵死
                self.missed.add(self.target)
                self.state = "WAIT"
                self.target = None
                return self._home()
            # 先带前馈地抬离带面：水平跟一小段带速，减少"抓握 vs 带面拖拽"的相对滑移
            if tcp[2] < self.u.belt_top_z + 0.13 and self.counter < 40:
                vbelt = float(self.u.belt_speed[0])
                dz = 0.01 if self.only_kind in ("banana", "pear") else 0.02
                return self._action([tcp[0] + vbelt * 0.033 * 1.2, tcp[1], tcp[2] + dz], gripper=-1.0)
            vx, vy = cfg.via_point
            if abs(tcp[0] - vx) > 0.06 or abs(tcp[1] - vy) > 0.06:
                if not self._via_done:
                    return self._action([vx, vy, cfg.carry_height], gripper=-1.0)
            else:
                self._via_done = True
            if abs(tcp[0] - bx) < 0.04 and abs(tcp[1] - by) < 0.04:
                self.state = "LOWER"
                self.counter = 0
            return self._action([bx, by, cfg.carry_height], gripper=-1.0)

        if self.state == "LOWER":
            bx, by = BIN_LAYOUT[self.target]
            tcp = self._tcp()
            if self.counter > 60:
                self.missed.add(self.target)
                self.state = "WAIT"
                self.target = None
                return self._home()
            if abs(tcp[0] - bx) < 0.03 and abs(tcp[1] - by) < 0.03 and tcp[2] < cfg.bin_height + 0.03:
                self.state = "RELEASE"
                self.counter = 0
            return self._action([bx, by, cfg.bin_height], gripper=-1.0)

        if self.state == "RELEASE":
            bx, by = BIN_LAYOUT[self.target]
            if self.counter > cfg.release_steps:
                if bool(self.u._in_bin_xy(self.target)[0]):
                    self.done.add(self.target)
                else:
                    self.missed.add(self.target)
                self.state = "WAIT"
                self.target = None
            return self._action([bx, by, cfg.bin_height], gripper=1.0)

        return self._home()

    def run(self, max_steps: int = 900):
        self.env.reset(seed=0)
        info = {}
        for _ in range(max_steps):
            _, _, term, trunc, info = self.env.step(self.act())
            if bool(term or trunc):
                break
        return info
