#!/usr/bin/env python
"""连续作业评测：传送带不停、间隔随机、尺寸随机，看策略能否正常干活。

与 `eval_act_pure.py` 的区别
---------------------------
- 环境换成 `FruitSortStream-v1`：每类 3 个 slot（共 12 件在飞行中），
  上料间隔 `U(gap_min, gap_max)` 米、尺寸 `U(scale_min, scale_max)`（±15%）；
- **目标选择层**：策略本身只会"抓 kind 这一类的最近一件"，谁来抓、什么时候换由这里决定
  （按"最快离开工作区"排序 + 抓取中不切换）；
- 指标按件统计：delivered / dropped / missed / throughput / 每类成功率。

用法:
    python scripts/stream_eval.py --ckpt runs/act-wide-4c/ckpt.pt --belt 0.08 --seconds 90
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch

import fruitsort.sim.tasks  # noqa: F401
from fruitsort.learning.bc import build_obs
from fruitsort.policies.act_runner import ActPureRunner
from fruitsort.policies.scripted import HOME_EULER, ScriptedSorter

ROOT = Path(__file__).resolve().parent.parent
KINDS = ("apple", "strawberry", "banana", "pear")


class _Stub:
    """只给 build_obs 用的替身：提供 .pose.p / .linear_velocity 两个张量。"""

    def __init__(self, p, v):
        self.pose = type("P", (), {"p": torch.tensor([p], dtype=torch.float32)})
        self.linear_velocity = torch.tensor([v], dtype=torch.float32)


class StreamController:
    """在"多件在飞"的场景里给 ACT 选目标。

    约定：`u.fruit_actors[kind]` 指向**该类最近的一件**（保持训练时的观测结构），
    目标类的 `fruit_actors[kind]` 指向**当前目标件**（runner 的所有判据都读它）。
    """

    def __init__(self, env, runner: ActPureRunner, u, kinds=KINDS, grasp_x: float = -0.10,
                 only_kind: str | None = None, scripted: bool = False,
                 in_dist: bool = False, noise: float = 0.0, runners: dict | None = None,
                 single_runner=None, queue_thr: float = 0.30, engage_x: float = -0.45,
                 scripted_kinds=(), scripted_yaw: float = 0.0, objs: dict | None = None,
                 decline_yaw: float = 0.0, long_extent: float = 0.11,
                 kind_priority: dict | None = None, perception_targets: bool = False,
                 decline_kinds: tuple = (),
                 give_up_s: float = 12.0, src=None, trace_abandon: bool = False):
        self.env, self.runner, self.u = env, runner, u
        self.scripted = scripted
        self.in_dist = in_dist
        self.noise = float(noise)
        self.default_runner = runner
        # 按类路由：不同类别用不同模型（banana 用未被流数据稀释的版本）
        self.runners = dict(runners or {})
        # 按**队列状态**切换的双策略：单件模式（queue<=thr）用 single_runner，
        # 队列模式用默认 runner。切换只在"选新目标"时发生，避免搬运途中换模型。
        self.single_runner = single_runner
        self.queue_thr = float(queue_thr)
        # 只在**可达窗口**内接管目标：水果还在 x<-0.45 时，Panda 从基座 (0,-0.5) 过去
        # 距离≈0.85 m（臂展极限），命令位姿不可达 → PD 卡住、TCP 冻结（docs 6.26.35）。
        self.engage_x = float(engage_x)
        # 混合系统：这些类别交给规则专家（模型在队列里做不好的长条类），
        # 其余类别仍由模型驱动。只在"选新目标"时决定，搬运途中不换。
        self.engaged_ids = set()     # 被本臂"接管过"的件（= 我们的机会）
        self.scripted_kinds = tuple(scripted_kinds)
        self.use_scripted = False
        # 队列上下文录制时的手腕偏航（每集随机一个，与 record_demos 的 --grasp-yaw-range 同义）：
        # 让"腕部朝向"成为训练分布内的变量，执行期才可能按水果长轴对准。
        self.scripted_yaw = float(scripted_yaw)
        # 感知模式：目标选择/队列标志都读这个字典（而不是仿真真值）
        self.objs = objs
        # 长条目标（世界长度 > long_extent）只在"长轴与夹爪闭合方向够垂直"时接管：
        # 平口夹爪沿世界 y 闭合，香蕉长轴若也接近 y，只能夹住两端 → 必失败。
        # 现场语义：不接这些件，让给旁边朝向更合适的臂（记账里算 declined）。
        self.decline_yaw = float(decline_yaw)
        self.long_extent = float(long_extent)
        # 按类优先（数字越小越先接）：短周期/高可靠的类先做，长条留到空档
        self.kind_priority = dict(kind_priority or {})
        # ---- 真·纯感知目标选择（不读仿真槽位）----
        # 只用感知跟踪到的实例（每类"最靠前"那件）当候选；存活判据也来自感知
        # （丢检成幽灵/已离开工作区/接管超时 → 放弃），不再看 u._slots。
        self.decline_kinds = tuple(decline_kinds or ())
        self.perception_targets = bool(perception_targets)
        self.give_up_s = float(give_up_s)
        self.trace_abandon = bool(trace_abandon)
        self.src = src
        self.sorter = None
        self.kinds = tuple(kinds)
        self.only_kind = only_kind
        self.grasp_x = float(grasp_x)
        self.belt = float(getattr(u, "belt_speed", [0.08])[0]) if hasattr(u, "belt_speed") else 0.08
        self.target = None          # item dict
        self.switches = 0
        # 空档期预转：把腕部提前转到"下一件长条目标"的长轴朝向（不占抓取窗口）
        self.idle_align = True
        self._idle_yaw = 0.0
        self._idle_for = None
        self._yaw_cache = {}        # item_id -> 已对准的 yaw 指令（接管理想值）
        self._repoint()

    # ---------------- 内部 ----------------
    def _queue_level(self) -> float:
        """队列标志：场上 |x|<0.6 且在带面的水果数 /4（与训练/推理的特征同源）。"""
        n = 0
        for k in self.kinds:
            a = (self.objs or self.u.fruit_actors).get(k)
            if a is None:
                continue
            arr = a.pose.p[0].cpu().numpy() if hasattr(a, "pose") else np.zeros(3)
            if abs(float(arr[0])) < 0.6 and 0.40 < float(arr[2]) < 0.60:
                n += 1
        return n / 4.0

    @staticmethod
    def _actor_pos(a) -> np.ndarray:
        """兼容两种"物体"：仿真 actor(.pose.p) / 感知实例(.pos)。"""
        if hasattr(a, "pose"):
            return a.pose.p[0].cpu().numpy()
        return np.asarray(getattr(a, "pos", np.zeros(3)), dtype=np.float64)

    def _live(self):
        ks = (self.only_kind,) if self.only_kind else self.kinds
        if self.perception_targets and self.src is not None:
            tracks = getattr(self.src, "last_tracks", None) or {}
            out = []
            for k in ks:
                cands = [t for t in (tracks.get(k) or [])
                         if float(np.asarray(t.pos)[0]) >= -1.10]
                if not cands:
                    continue
                t = max(cands, key=lambda i: float(np.asarray(i.pos)[0]))
                out.append(dict(id=int(getattr(t, "track_id", -1)), kind=k,
                                actor=t, perception=True))
            return out
        items = []
        for k in ks:
            for slot in self.u._slots[k]:
                if slot["live"]:
                    items.append(slot["item"])
        return items

    def _repoint(self):
        """fruit_actors[kind] = 目标件本身 / 其余类按训练布局重映射的替身。

        模型训练时四类固定在 x 间隔 0.45 m 的布局上；连续作业里布局是随机的，
        直接把真实位置喂进去会让"接近"相位输出错误增量（实测 TCP 冻结）。
        这里把非目标类摆到"训练布局"（目标 x ± 0.45·Δi），让状态观测回到分布内。
        """
        tcp = self.u.agent.tcp_pose.p[0].cpu().numpy()
        if self.in_dist and self.target is not None:
            tk = self.target["kind"]
            tp = self.target["actor"].pose.p[0].cpu().numpy()
            belt = float(self.u.belt_speed[0])
            for j, k in enumerate(self.kinds):
                if k == tk:
                    self.u.fruit_actors[k] = self.target["actor"]
                    continue
                dx = 0.45 * (self.kinds.index(tk) - j)
                self.u.fruit_actors[k] = _Stub(
                    [tp[0] + dx, tp[1], tp[2]], [belt, 0.0, 0.0])
            return
        for k in self.kinds:
            if self.objs is not None:
                # 感知模式：不读产线槽位。非目标类用"感知到的位置"（替身），
                # 目标类由 runner.set_actors 注入；只有该类完全没检到时才留原位。
                _a = self.objs.get(k)
                if _a is not None and (self.target is None or self.target["kind"] != k):
                    self.u.fruit_actors[k] = _a
                continue
            live = [s["item"] for s in self.u._slots[k] if s["live"]]
            if not live:
                # 该类当前没有在飞件：必须给一个"合理"的替身（刚上料的位姿），
                # 否则 fruit_actors[k] 仍指向停在场外 z=-5 的 actor → 观测分布外，
                # 任何学习策略都会冻结（这正是此前"连续作业 0–17%"的真正原因）
                spec = self.u.specs[k]
                self.u.fruit_actors[k] = _Stub(
                    [self.u.stream_x, 0.0, self.u.belt_top_z + spec.half_height + 0.002],
                    [self.belt, 0.0, 0.0])
                continue
            if self.target is not None and self.target["kind"] == k:
                if self.objs is not None:
                    continue
                if self.objs is None:
                    self.u.fruit_actors[k] = self.target["actor"]
            else:
                nearest = min(live, key=lambda it: np.linalg.norm(
                    it["actor"].pose.p[0].cpu().numpy() - tcp))
                if self.objs is None:
                    self.u.fruit_actors[k] = nearest["actor"]

    def _finger_axis(self) -> float:
        pL = self.u.agent.robot.links_map["panda_leftfinger"].pose.p[0].cpu().numpy()
        pR = self.u.agent.robot.links_map["panda_rightfinger"].pose.p[0].cpu().numpy()
        d = pL - pR
        return float(np.arctan2(d[1], d[0]))

    def _idle_align_yaw(self) -> float:
        """空档（没有可接的件）时，把腕部转到"下一件香蕉"的长轴朝向。

        动机（v1.67）：抓取窗口内的原地预转要 1–2 s，而香蕉单件只允许 ~5 s。
        空档期手腕本来就闲着，用它提前转到位，窗口内就不用再转。
        """
        if not self.idle_align or self.objs is None:
            return 0.0
        a = self.objs.get("banana")
        yy = getattr(a, "yaw", None) if a is not None else None
        x = float(a.pose.p[0, 0]) if a is not None else -9.0
        if yy is None or not (-1.15 < x < -0.30):
            self._idle_yaw *= 0.9          # 没有目标：缓慢回正
            self._idle_for = None
            return self._idle_yaw
        psi = self._finger_axis()
        want = float(yy) + np.pi / 2.0
        d = psi - want
        d = float(np.arctan2(np.sin(2.0 * d), np.cos(2.0 * d)) / 2.0)
        if abs(np.degrees(d)) > 5.0:
            self._idle_yaw = float(np.clip(self._idle_yaw + 0.9 * d, -1.2, 1.2))
        return self._idle_yaw

    def _target_live(self) -> bool:
        it = self.target
        if not (self.perception_targets and self.objs is not None):
            return any(it is s["item"] for s in self.u._slots[it["kind"]] if s["live"])
        if self.runner is not None and getattr(self.runner, "grasping", False):
            return True                     # 正夹着：即使离开视野也继续
        # 这一件已经在框里（用**果实**位置判，不是 TCP）→ 已结束，立刻换目标。
        # 曾用"TCP 在框附近"做判据：手臂为了**别的件**经过料框时会把当前目标误判为"已送达"，
        # 于是丢弃→立刻重选同一件→runner.reset() 清空 hist 缓冲。实测 0.1 s 内连丢 5 次，
        # 策略每步都在"从零开始"，这是"四类混跑时苹果 0/4、单跑 6/6"的直接原因。
        try:
            from fruitsort.sim.conveyor import BIN_LAYOUT as _BL
            _bx, _by = _BL[it["kind"]]
            _ap = self._actor_pos(it["actor"])
            if float(np.hypot(_ap[0] - _bx, _ap[1] - _by)) < 0.16:
                return False
        except Exception:  # noqa: BLE001
            pass
        t0 = it.get("t_engage")
        if t0 is not None and (float(self.u._t) - float(t0)) > self.give_up_s:
            return False                    # 接管超时（现实中就该放弃了）
        return self._dead_reason(it) == ""

    def _dead_reason(self, it) -> str:
        """目标为什么该被放弃：空串=还活着；否则返回原因（诊断用）。"""
        if self.runner is not None and getattr(self.runner, "grasping", False):
            return ""
        try:
            from fruitsort.sim.conveyor import BIN_LAYOUT as _BL
            _bx, _by = _BL[it["kind"]]
            _ap = self._actor_pos(it["actor"])
            if float(np.hypot(_ap[0] - _bx, _ap[1] - _by)) < 0.16:
                return "at_bin"
        except Exception:  # noqa: BLE001
            pass
        t0 = it.get("t_engage")
        if t0 is not None and (float(self.u._t) - float(t0)) > self.give_up_s:
            return "timeout"
        a = self.objs.get(it["kind"])
        if a is None:
            return "lost"
        x = float(a.pose.p[0, 0])
        if x <= -1.05:
            return "ghost"
        if x > 0.45:
            return "passed"
        return ""

    def _in_bin(self, p) -> bool:
        """该点是否落在某个料框口内（料框是环境夹具，不是物体真值）。"""
        try:
            from fruitsort.sim.conveyor import BIN_LAYOUT
        except Exception:  # noqa: BLE001
            return False
        for _k, (bx, by) in BIN_LAYOUT.items():
            if float(np.hypot(float(p[0]) - bx, float(p[1]) - by)) < 0.17:
                return True
        return False

    def _pick_target(self):
        """选**最接近理想拦截点**的一件（不是最紧急的）。

        实测：按"x 最大"选会一路追到工作区边缘（x>0.3，训练分布外），
        模型只会悬停不闭爪；按"离拦截点多近"选能留出下探时间。
        """
        tcp = self.u.agent.tcp_pose.p[0].cpu().numpy()
        urg, late = [], []
        for it in self._live():
            if it["kind"] in self.decline_kinds:
                continue
            p = self._actor_pos(it["actor"])
            if np.linalg.norm(p - tcp) < 0.05:   # 就在指尖上（可能正抓着）
                continue
            # 已经在料框里（送达后的"幽灵"轨迹还在跟踪器里活几秒）→ 别再选它，
            # 否则"选中→at_bin 丢弃→再选中"每步一次，把 runner 的历史清空
            if self._in_bin(p):
                continue
            if self.objs is not None:
                # 感知模式下不要追"场外幽灵"：该类当前没有检测（替身停在 clip_x）时，
                # 说明还看不见它，接管只会让手臂朝 -1.1 m 走（不可达 → 冻结）
                est = self.objs.get(it["kind"])
                if est is None or float(est.pose.p[0, 0]) <= -1.05:
                    continue
                if self.decline_yaw > 0:
                    ext = float(getattr(est, "extent", 0.0) or 0.0)
                    yy = getattr(est, "yaw", None)
                    if ext > self.long_extent and yy is not None:
                        # 夹爪闭合方向≈世界 y：|sin(yaw)| 大 → 长轴贴着闭合方向 → 夹不住
                        if abs(float(np.sin(yy))) > self.decline_yaw:
                            self.declined_yaw = getattr(self, "declined_yaw", 0) + 1
                            continue
            if self.engage_x <= p[0] <= -0.05:   # 已进入可达窗口：优先最靠前的一件
                urg.append((p[0], it))
            elif -0.05 < p[0] <= 0.15:           # 已经比较靠后：兜底
                late.append((p[0], it))
        if urg:
            if self.kind_priority:
                return min(urg, key=lambda t: (self.kind_priority.get(t[1]["kind"], 9.0),
                                               -t[0]))[1]
            return max(urg, key=lambda t: t[0])[1]
        if late:
            return max(late, key=lambda t: t[0])[1]
        return None

    # ---------------- 主循环 ----------------
    def act(self, obs):
        self._repoint()
        # 目标失效（已被回收/送达/漏过）→ 换目标
        if self.target is not None and not self._target_live():
            if getattr(self, "trace_abandon", False):
                _r = self._dead_reason(self.target)
                try:
                    _x = float(self._actor_pos(self.target["actor"])[0])
                except Exception:  # noqa: BLE001
                    _x = None
                print(f"    [abandon] t={float(self.u._t):5.1f} "
                      f"{str(self.target.get('kind')):10s} why={_r} "
                      f"x={'-' if _x is None else format(_x, '+.2f')} "
                      f"t_eng={self.target.get('t_engage')}", flush=True)
            self.target = None
            # 目标结束（送达/掉落/漏过）：清掉执行器状态，否则 phase 会停在"搬运"
            self.runner.reset()
        if self.target is None:
            self.target = self._pick_target()
            if self.target is not None:
                # 纯感知目标没有真值 id：用"估计位置最近的场上件"做**诊断**映射，
                # 仅用于机会内记账（模型输入不含真值）
                if self.target.get("perception") and self.objs is not None:
                    est = self.objs.get(self.target["kind"])
                    if est is not None:
                        p_est = est.pose.p[0].cpu().numpy()
                        best = None
                        for k2 in self.kinds:
                            for s2 in self.u._slots[k2]:
                                if not s2["live"]:
                                    continue
                                q = s2["item"]["actor"].pose.p[0].cpu().numpy()
                                dd = float(np.linalg.norm(q[:2] - p_est[:2]))
                                if best is None or dd < best[0]:
                                    best = (dd, s2["item"]["id"])
                        if best is not None and best[0] < 0.25:
                            self.target["diag_id"] = best[1]
                self.engaged_ids.add(self.target.get("diag_id", self.target["id"]))
                if "t_engage" not in self.target:
                    self.target["t_engage"] = float(self.u._t)
                    # 接管时的果实 x（诊断"接得太晚"）：记账到场上件，便于逐件复盘
                    try:
                        self.target["x_engage"] = float(
                            self._actor_pos(self.target["actor"])[0])
                    except Exception:  # noqa: BLE001
                        self.target["x_engage"] = None
                    _d = self.target.get("diag_id", self.target.get("id"))
                    if _d in self.u.stream_stats.get("items", {}):
                        # 记账：感知目标不是场上件对象，t_engage 必须显式同步，
                        # 否则 handle_s（单件耗时）永远是 0、逐件时长无法复盘
                        self.u.stream_stats["items"][_d]["x_engage"] = self.target["x_engage"]
                        self.u.stream_stats["items"][_d]["t_engage"] = self.target["t_engage"]
                q = self._queue_level()
                if q <= self.queue_thr and self.single_runner is not None:
                    # 单件模式：用"干净单件数据"训练的模型（apple 8/8、banana 7/8）
                    self.runner = self.single_runner
                elif self.target["kind"] in self.runners:
                    # 队列模式：优先按类专用模型（各类自带流上下文数据）
                    self.runner = self.runners[self.target["kind"]]
                self.runner.kind = self.target["kind"]
                self.runner.reset()
                _yc = self._yaw_cache.get(self.target["id"])
                if _yc is None and abs(getattr(self, "_idle_yaw", 0.0)) > 1e-3:
                    # 非长条件：也用空档期的偏航，让腕部保持（不回 HOME）
                    _yc = float(self._idle_yaw)
                if _yc is not None and getattr(self.runner, "yaw_align", False):
                    self.runner._yaw_target = float(_yc)   # 空档期已转到位：窗口内不用再转
                    self.runner._yaw_cmd = float(_yc)      # 不在对准范围时由 hold 分支保持
                # 切模型后必须重新注入感知替身（否则新 runner 这一步读到仿真真值）
                if self.objs is not None:
                    self.runner.set_actors(self.objs)
                self.switches += 1
                if self.objs is None:
                    self.u.fruit_actors[self.target["kind"]] = self.target["actor"]
                self.use_scripted = (self.scripted
                                     or self.target["kind"] in self.scripted_kinds)
                if self.use_scripted:
                    cfg_yaw = None
                    if self.scripted_yaw:
                        from fruitsort.policies.scripted import PolicyConfig
                        cfg_yaw = PolicyConfig(grasp_yaw=self.scripted_yaw)
                    self.sorter = ScriptedSorter(
                        self.env, cfg=cfg_yaw, only_kind=self.target["kind"])
        if self.target is None:
            # 没活可干：回到待命位姿、张开手（基座系：x=y+0.5, y=-x, z=z）
            # 顺便把腕部预转到"下一件香蕉"的长轴朝向（空档期不占抓取窗口）
            iy = self._idle_align_yaw()
            a = (self.objs or {}).get("banana")
            if self.objs is not None and a is not None and abs(iy) > 1e-3:
                for it in self._live():
                    if it["kind"] != "banana":
                        continue
                    if -1.15 < float(self._actor_pos(it["actor"])[0]) < -0.30:
                        self._yaw_cache[it["id"]] = iy
                        break
            return np.array([0.5, 0.0, 0.55, HOME_EULER[0], HOME_EULER[1],
                             HOME_EULER[2] + iy, 1.0], dtype=np.float32)
        if self.objs is not None and getattr(self.runner, "suction", False):
            # 吸盘：把"已吸附"同步给场景（停带 + 不被回收判据误判）
            it = self.target
            if it is not None:
                it["held"] = bool(getattr(self.runner, "attached", False))
        # 吸盘释放/放回：用 target['actor'] 执行（runner 只发请求）
        _rn = self.runner
        if self.target is not None and (getattr(_rn, "_release_to", None) is not None
                                        or getattr(_rn, "_belt_put", None) is not None):
            import torch as _th
            from mani_skill.utils.structs.pose import Pose as _Pose
            _a = self.target["actor"]
            _rel = getattr(_rn, "_release_to", None)
            _put = getattr(_rn, "_belt_put", None)
            if _rel is not None:
                _p = [float(_rel[0]), float(_rel[1]), 0.08]
                _rn._release_to = None
            else:
                _p = [float(_put[0]), float(_put[1]), float(_put[2])]
                _rn._belt_put = None
            try:
                _a.set_pose(_Pose.create_from_pq(_th.tensor([_p], dtype=_th.float32)))
                _a.set_linear_velocity(_th.zeros(1, 3))
                _a.set_angular_velocity(_th.zeros(1, 3))
            except Exception:  # noqa: BLE001
                pass
        if self.use_scripted:
            act = self.sorter.act()
            if self.noise > 0:
                act = act.copy()
                act[:3] += np.random.normal(0, self.noise, 3).astype(np.float32)
            return act
        return self.runner.act(obs)


def parse_map(spec: str) -> dict:
    """'banana=-0.01,pear=0.015' -> {'banana': -0.01, 'pear': 0.015}"""
    out = {}
    for kv in (spec or "").split(","):
        kv = kv.strip()
        if not kv or "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        out[k.strip()] = float(v)
    return out


WIDTH_MAP = {"apple": 0.062, "strawberry": 0.034, "banana": 0.044, "pear": 0.068}


def _pack_images(obs, roi=None, xy=None):
    """与 record_demos.py 一致：128×128 JPEG；给了 roi 时额外返回 ROI 裁剪。"""
    sd = obs["sensor_data"]
    oh_full = sd["overhead"]["rgb"][0].cpu().numpy()
    roi_b = b""
    if roi is not None and xy is not None:
        _r = roi.crop(oh_full, xy)
        roi_b = cv2.imencode(".jpg", _r[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
    oh = cv2.resize(oh_full, (128, 128), interpolation=cv2.INTER_AREA)
    wr = None
    if "hand_camera" in sd:
        wr = sd["hand_camera"]["rgb"][0].cpu().numpy()
        wr = cv2.resize(wr, (128, 128), interpolation=cv2.INTER_AREA)
    return (cv2.imencode(".jpg", oh[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes(),
            cv2.imencode(".jpg", wr[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
            if wr is not None else b"", roi_b)


_KIND_BGR = {"apple": (60, 60, 220), "strawberry": (90, 60, 235),
             "banana": (60, 195, 240), "pear": (90, 180, 90)}
_ROI_HELPER = [None]


def _head_panel(obs, tracks, tgt_xy, tgt_kind, proj, panel_w: int = 620, masks=None):
    """头部相机画面 + 纯感知检测结果 → 画中画面板。

    存在的理由：演示视频的第三人称视角无法证明"没读仿真真值"。这里直接贴**模型唯一的
    图像输入**（overhead RGB-D）和它的检测输出，观众可以自己核对"框住的确实是感知算出来的"。
    """
    try:
        rgb = obs["sensor_data"]["overhead"]["rgb"][0].cpu().numpy()
    except Exception:  # noqa: BLE001
        return None
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rgb = np.ascontiguousarray(rgb[..., ::-1])      # RGB → BGR（下面统一按 BGR 画）
    h0, w0 = rgb.shape[:2]
    ph = int(round(panel_w * h0 / w0))
    img = cv2.resize(rgb, (panel_w, ph), interpolation=cv2.INTER_AREA).copy()

    # 世界 (x,y) → 面板像素。`proj` 由调用方给（v1 用带面单应，v2 用相机模型）
    def _px(xy):
        u, v = proj(xy)
        return float(u) * panel_w / w0, float(v) * ph / h0

    ox, oy = _px((-1.10, 0.0))
    ux, uy = _px((-0.10, 0.0))
    ppm = max(1.0, float(np.hypot(ux - ox, uy - oy)) / 1.0)

    # 分割掩码（v2 感知才会给）：画轮廓 + 类别，直接展示"掩码从哪来"
    if masks:
        _sx, _sy = panel_w / float(w0), ph / float(h0)
        for seg, kind, conf in masks:
            _m = (np.asarray(seg.mask).astype(np.uint8) * 255)
            cnts, _ = cv2.findContours(_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnts = [np.round(c.reshape(-1, 2) * np.array([_sx, _sy])).astype(np.int32)
                    for c in cnts]
            col = _KIND_BGR.get(kind, (200, 200, 200))
            cv2.drawContours(img, cnts, -1, col, 2, cv2.LINE_AA)
            x0, y0, x1, y1 = seg.bbox
            cv2.putText(img, f"{kind[:2]} {conf:.1f}",
                        (int(x0 * _sx), max(12, int(y0 * _sy) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, f"{kind[:2]} {conf:.1f}",
                        (int(x0 * _sx), max(12, int(y0 * _sy) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

    n_tr = 0
    for kind, insts in (tracks or {}).items():
        for ins in insts:
            if getattr(ins, "lost", 0) > 3:
                continue
            n_tr += 1
            u, v = _px(ins.pos[:2])
            r = int(max(7, 0.5 * float(getattr(ins, "extent", 0.06)) * ppm))
            col = _KIND_BGR.get(kind, (200, 200, 200))
            cv2.circle(img, (int(u), int(v)), r, col, 2, cv2.LINE_AA)
            cv2.putText(img, kind[:2], (int(u) - 8, int(v) - r - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    if tgt_xy is not None:
        u, v = _px(tgt_xy)
        col = _KIND_BGR.get(tgt_kind or "", (255, 255, 255))
        cv2.drawMarker(img, (int(u), int(v)), col, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
        cv2.circle(img, (int(u), int(v)), 19, (255, 255, 255), 1, cv2.LINE_AA)
        txt = f"TARGET {tgt_kind}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx, ty = int(u) + 22, int(v) + 18
        if tx + tw + 6 > panel_w:
            tx = int(u) - tw - 26
        cv2.rectangle(img, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 6), (0, 0, 0), -1)
        cv2.putText(img, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    # 标题条：强调这是"模型输入"，不是仿真真值
    bar = img.copy()
    cv2.rectangle(bar, (0, 0), (panel_w, 26), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.55, img, 0.45, 0, img)
    cv2.putText(img, f"overhead RGB-D 1280x960  |  model input, no GT  |  tracks={n_tr}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

    # ROI 缩略图（模型真正看到的目标窗口）+ 掩码叠加，放大 156×156 以便看清
    pw_, ph_ = panel_w, ph
    if tgt_xy is not None:
        try:
            u0, v0 = _px(tgt_xy)
            u1, _ = _px((float(tgt_xy[0]) + 0.5, float(tgt_xy[1])))
            half = float(np.clip(abs(u1 - u0) / 2.0, 8.0, 0.45 * min(pw_, ph_)))
            x0, x1 = max(0, int(u0 - half)), min(pw_, int(u0 + half))
            y0, y1 = max(0, int(v0 - half)), min(ph_, int(v0 + half))
            crop = img[y0:y1, x0:x1]
            if crop.size and (y1 - y0) >= 8 and (x1 - x0) >= 8:
                roi = cv2.resize(crop, (156, 156), interpolation=cv2.INTER_NEAREST)
                tx0, ty0 = panel_w - 164, ph - 164
                cv2.rectangle(img, (tx0 - 2, ty0 - 2), (tx0 + 158, ty0 + 158), (255, 255, 255), 2)
                img[ty0:ty0 + 156, tx0:tx0 + 156] = roi
                cv2.putText(img, "ROI 0.5m x156", (tx0 - 4, ty0 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        except Exception:  # noqa: BLE001
            pass

    return img


def _save_demo(out_dir: Path, idx: int, buf: dict, seed: int) -> None:
    if len(buf["obs"]) < 20:
        return
    payload = dict(obs=np.stack(buf["obs"]), action=np.stack(buf["act"]),
                   seed=seed, kinds=np.array(KINDS), instruction=np.array(buf["kind"]),
                   overhead_jpg=np.array(buf["oh"], dtype=object),
                   wrist_jpg=np.array(buf["wr"], dtype=object))
    if buf.get("roi"):
        payload["roi_jpg"] = np.array(buf["roi"], dtype=object)
    np.savez_compressed(out_dir / f"ep_{idx:04d}.npz", **payload)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/act-wide-4c/ckpt.pt")
    ap.add_argument("--belt", type=float, default=0.08)
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--slots", type=int, default=3, help="每类候选件数（共 4×slots 件）")
    ap.add_argument("--spawn-x", type=float, default=-0.85, help="上料点 x（米）")
    ap.add_argument("--gap-min", type=float, default=0.60,
                    help="相邻两件的间距下限（米）")
    ap.add_argument("--gap-max", type=float, default=1.20, help="间距上限（米）")
    ap.add_argument("--scale-min", type=float, default=0.85)
    ap.add_argument("--scale-max", type=float, default=1.15)
    ap.add_argument("--control-freq", type=int, default=30,
                    help="与部署评测一致（act-wide-4c 的 87.5%%/96.9%% 都是 30 Hz）")
    ap.add_argument("--max-step-map", default="",
                    help="按类接近步长，如 'apple=0.12,strawberry=0.12,pear=0.10,banana=0.06'")
    ap.add_argument("--carry-step-map", default="",
                    help="按类搬运步长，如 'apple=0.04,pear=0.03,banana=0.02'")
    ap.add_argument("--max-step", type=float, default=0.06,
                    help="接近段每步最大位移（米/步）：调大可加速机械臂，代价是精度/稳定性")
    ap.add_argument("--fb", type=float, default=0.5)
    ap.add_argument("--grip-ema", type=float, default=0.6)
    ap.add_argument("--carry-latch", action="store_true", default=True)
    ap.add_argument("--carry-max-step", type=float, default=0.02)
    ap.add_argument("--engage-x", type=float, default=-0.45,
                    help="只在 x>=该值时接管目标（更远的水果超出臂展，命令不可达会冻结）")
    ap.add_argument("--grasp-x", type=float, default=-0.10, help="理想拦截点 x（米）")
    ap.add_argument("--in-dist-obs", action="store_true",
                    help="把非目标类按训练布局重映射（让状态观测回到分布内）")
    ap.add_argument("--grasp-yaw-range", type=float, nargs=2, default=None,
                    help="录制时每集随机手腕偏航（rad）：'队列上下文 + 随机腕部'合并成一批数据")
    ap.add_argument("--yaw-align", action="store_true",
                    help="验证性实验：用仿真真值 yaw 旋转手腕，使夹爪闭合方向垂直于水果长轴")
    ap.add_argument("--yaw-align-off", type=float, default=0.0,
                    help="长轴假设偏移（0 或 π/2），用于试两种本地轴")
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
    ap.add_argument("--close-commit", type=float, default=0.0,
                    help="闭爪承诺：离果心 z 低于该值且水平对齐时强制闭爪并冻结水平目标（0=关闭）")
    ap.add_argument("--close-commit-xy", type=float, default=0.030)
    ap.add_argument("--close-commit-map", default="",
                    help="按类闭爪承诺阈值，如 'apple=0.025,strawberry=0.025'")
    ap.add_argument("--grasp-proxy", default="auto", choices=["auto", "always", "never"],
                    help="抓取近似判据（离得近+闭爪）：auto=仅当查不到真实接触时用")
    ap.add_argument("--suction", action="store_true",
                    help="吸盘/软末端：模型说吸且贴近果心即附着，搬运跟 TCP，到框上方释放")
    ap.add_argument("--suction-radius", type=float, default=0.10)
    ap.add_argument("--suction-carry", default="scripted", choices=["scripted", "policy"],
                    help="吸住后怎么搬到框：scripted=直接奔框（推荐）/policy=用策略搬运段")
    ap.add_argument("--grasp-verify", action="store_true",
                    help="只碰到水果顶端不算抓住（要求指面到达果身高度或已被抬起）")
    ap.add_argument("--lead-long", type=float, default=0.0,
                    help="长条目标的拦截点前移（秒）。实测有害：长条会被推到尖端去抓（默认关）")
    ap.add_argument("--lead-long-extent", type=float, default=0.11)
    ap.add_argument("--lead-long-taper", type=float, default=0.15,
                    help="前移在距果心该距离内线性衰减到 0（0=不衰减）")
    ap.add_argument("--intercept", type=float, default=0.0,
                    help="拦截式下探的垂直速度（m/s，0=关闭）：按'下探耗时×带速'预测落点")
    ap.add_argument("--close-win-xy", type=float, default=0.03)
    ap.add_argument("--close-win-z", type=float, default=0.06)
    ap.add_argument("--release-gate", type=float, default=0.0,
                    help="放料门：只在料框中心该距离内才允许开爪（米；0=关）")
    ap.add_argument("--release-wait", type=int, default=120,
                    help="放料门最长拦截步数（超过就放开，避免抱着果子跑整局）")
    ap.add_argument("--grip-hyst", type=float, default=0.0,
                    help="夹爪指令迟滞（logit 双阈值）：>0 时 |logit|<h 维持上一步，抑制闭爪抖动")
    ap.add_argument("--close-z-off-map", default="",
                    help="闭爪门触发时的下探高度偏移（果心+该值），如 'strawberry=-0.005'")
    ap.add_argument("--close-z-gate-map", default="",
                    help="按类闭爪高度门，如 'apple=0.009,strawberry=0.006,pear=0.011'")
    ap.add_argument("--close-z-gate", type=float, default=0.0,
                    help="闭爪高度门（米）：离果心高于该值时禁止闭爪，改为下探（0=关闭）")
    ap.add_argument("--descend-commit", type=int, default=0,
                    help="下探承诺步数：模型一旦判定下探，就保持该相位 N 步，避免 0/1 抖动")
    ap.add_argument("--z-bias-map", default="",
                    help="按类抓取高度偏置，如 'pear=0.015,banana=-0.01'")
    ap.add_argument("--yaw-map", default="",
                    help="按类手腕偏航（rad），如 'banana=0.6'（让夹爪闭合方向对齐长轴）")
    ap.add_argument("--width-g", type=float, default=None,
                    help="固定的夹爪闭合指令值（位置等价力控；gap≈0.030+0.050g）")
    ap.add_argument("--close-ramp", type=float, default=0.0,
                    help="渐进闭爪：每步 -close_ramp（0=一步到位）")
    ap.add_argument("--width-setpoint", action="store_true",
                    help="闭爪到按类的宽度设定点(LeTac 式)，而不是合到底")
    ap.add_argument("--stall-guard", type=float, default=0.0,
                    help="停滞守卫的最小接近步长（米，0=关闭）：指令位移≈0 且水果仍在窗口内时"
                         "强制朝水果迈这一步，用于打破\"接近相位塌陷→TCP 冻结\"")
    ap.add_argument("--single-ckpt", default="",
                    help="按队列状态切换的双策略：单件模式(queue<=--queue-thr)用该模型")
    ap.add_argument("--queue-thr", type=float, default=0.30)
    ap.add_argument("--ckpt-map", default="",
                    help="按类路由模型：'banana=runs/act-noisy/ckpt.pt'（逗号分隔多条）")
    ap.add_argument("--scripted-kinds", default="",
                    help="混合系统：这些类（逗号分隔）交给规则专家，其余由模型驱动，"
                         "如 'banana,pear'")
    ap.add_argument("--kind-priority", default="",
                    help="按类优先接件，如 'apple=0,strawberry=1,pear=2,banana=3'（空=只按 x）")
    ap.add_argument("--side-grasp", default="",
                    help="这些类改成'侧向进入'抓取（如 strawberry）")
    ap.add_argument("--side-offset", type=float, default=0.09)
    ap.add_argument("--decline-kinds", default="",
                    help="这些类一律不接管（明确让给同伴），如 banana")
    ap.add_argument("--decline-yaw", type=float, default=0.64,
                    help="长条目标：|sin(长轴)| 超过该值就不接管（长轴贴住闭合方向夹不住；0=关）")
    ap.add_argument("--long-extent", type=float, default=0.11)
    ap.add_argument("--grasp-axis-offset", type=float, default=0.0,
                    help="抓取点沿长轴偏移（米，只作用于长条目标；感知 yaw 给方向）")
    ap.add_argument("--grasp-axis-extent", type=float, default=0.12)
    ap.add_argument("--hand-cam", action="store_true",
                    help="感知里再加腕部相机（头部被手臂挡住时补最后 5 cm；现实中可得）")
    ap.add_argument("--perception-targets", action="store_true",
                    help="目标选择/存活判据全部来自感知跟踪（不读仿真槽位）；接管 12 s 未完成即放弃")
    ap.add_argument("--give-up-s", type=float, default=12.0)
    ap.add_argument("--near-extent-min", type=float, default=0.05,
                    help="near-hold 允许的最小世界长度（默认 0.05：草莓 0.043 被排除）")
    ap.add_argument("--near-extent-max", type=float, default=0.11)
    ap.add_argument("--near-hold", action=argparse.BooleanOptionalAction, default=True,
                    help="感知：丢检且夹爪压到目标上方时位置取 TCP（只对紧凑水果生效；默认开）")
    ap.add_argument("--perception", default="oracle",
                    choices=["oracle", "vision", "rgbd", "rgbdv2", "fused"],
                    help="oracle=仿真真值；fused=俯视+侧视 RGB-D 感知（无真值）")
    ap.add_argument("--hand-fallback", action="store_true", default=True,
                    help="头部丢检时用腕部相机补 xy（z 沿用历史）；v1.100 起默认开")
    ap.add_argument("--no-hand-fallback", dest="hand_fallback", action="store_false",
                    help="关掉腕部兜底（对照实验用）")
    ap.add_argument("--camera-calib", default="overhead",
                    help="相机模型名（assets/calib/camera_<name>.json），--perception rgbdv2 用")
    ap.add_argument("--seg-backend", default="classic",
                    choices=["classic", "learned", "hybrid"],
                    help="rgbdv2 的分割后端（learned 需 --seg-ckpt）")
    ap.add_argument("--seg-ckpt", default="runs/seg-unet/seg.pt")
    ap.add_argument("--seg-thresh", type=float, default=0.5)
    ap.add_argument("--recognizer", default="",
                    help="识别头权重（scripts/train_recognizer.py 产出）；空=规则识别")
    ap.add_argument("--policy", default="act", choices=["act", "scripted"],
                    help="act=部署策略；scripted=规则专家（测场景上限）")
    ap.add_argument("--only-kind", default=None,
                    help="只抓这一类（诊断用：隔离'指令切换'的影响）")
    ap.add_argument("--allow-ghost-frames", action="store_true",
                    help="录制时允许'缺类'帧（部署端同样会喂 ghost 观测）。关闭时只录四类齐全帧，"
                         "产量会低一个数量级（实测 4/60s vs 20/60s）")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="录制时在专家动作上加的高斯噪声（米）——专家主导 + 小扰动，"
                         "比收集模型失败状态安全得多")
    ap.add_argument("--video", default="",
                    help="输出演示视频（mp4）。画面叠加：时间/带速/目标类别/送达·漏检·掉落/相位")
    ap.add_argument("--no-video-inset", action="store_true",
                    help="视频里不贴头部 RGB-D 画中画（默认贴：证明模型输入只有相机）")
    ap.add_argument("--video-fps", type=int, default=0,
                    help="视频帧率，0 = 跟随 control-freq（与实时一致）")
    ap.add_argument("--record", default="", help="把专家在连续流里的成功片段录成训练数据（目录）")
    ap.add_argument("--roi-m", type=float, default=0.0,
                    help="录制时额外保存目标 ROI 裁剪（米；0=不存）")
    ap.add_argument("--trace-abandon", action="store_true",
                    help="打印每次放弃目标的原因（at_bin/timeout/lost/ghost/passed）")
    ap.add_argument("--trace-release", action="store_true",
                    help="记录每次开爪（放下）瞬间 TCP 到本类料框中心的距离，用于判断'放偏了'")
    ap.add_argument("--trace-drop", action="store_true",
                    help="记录每次掉落瞬间的上下文（相位/夹持/到框距离/高度），用于定位掉落原因")
    ap.add_argument("--trace-close", action="store_true",
                    help="记录每次闭爪瞬间 TCP 与目标的偏差（拦截时序诊断：正=落在果实后面）")
    ap.add_argument("--trace-items", action="store_true",
                    help="逐件记账：每 20 步打印 (相位, 到框距离, 是否附着)，并在每局末打印每件结果")
    ap.add_argument("--seed", type=int, default=40000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    args.video_inset = not args.no_video_inset

    env = gym.make(
        "FruitSortStream-v1", obs_mode="rgbd", control_mode="pd_ee_pose", sim_backend="cpu",
        robot_uids="panda_wristcam", active_kinds=KINDS,
        belt_speed_range=(args.belt, args.belt), spawn_x=args.spawn_x,
        slots_per_kind=args.slots, gap_range=(args.gap_min, args.gap_max),
        scale_range=(args.scale_min, args.scale_max),
        always_stock=bool(args.record),
        render_mode=("rgb_array" if args.video else None),
        sim_config=dict(control_freq=args.control_freq,
                        sim_freq=120 if args.control_freq == 30 else 100))
    src = None
    if args.perception == "fused":
        from fruitsort.perception.adapter import FusedSource
        src = FusedSource(env.unwrapped, dt=1.0 / args.control_freq,
                          use_hand=args.hand_cam, near_hold=args.near_hold,
                          near_extent_min=args.near_extent_min,
                          near_extent=args.near_extent_max,
                          grasp_axis_offset=args.grasp_axis_offset,
                          grasp_axis_extent=args.grasp_axis_extent)
    elif args.perception == "rgbd":
        from fruitsort.perception.adapter import RGBDSource
        src = RGBDSource(env.unwrapped, dt=1.0 / args.control_freq,
                         near_hold=args.near_hold,
                         near_extent_min=args.near_extent_min,
                         near_extent=args.near_extent_max,
                         grasp_axis_offset=args.grasp_axis_offset,
                         grasp_axis_extent=args.grasp_axis_extent,
                         use_hand=args.hand_cam)
    elif args.perception == "rgbdv2":
        from fruitsort.perception.adapter import RGBDv2Source
        from fruitsort.perception.recognition import FeatureRecognizer
        _rec = None
        if args.recognizer:
            _rec = FeatureRecognizer.load(args.recognizer)
            print(f"识别头: {args.recognizer}", flush=True)
        _segm = None
        if args.seg_backend in ("learned", "hybrid"):
            from fruitsort.perception.learned_seg import HybridSegmenter, LearnedSegmenter
            _cls = HybridSegmenter if args.seg_backend == "hybrid" else LearnedSegmenter
            _segm = _cls(args.seg_ckpt, thresh=args.seg_thresh)
            _dev = getattr(getattr(_segm, "learned", _segm), "device", "?")
            print(f"分割后端: {args.seg_backend} ({args.seg_ckpt}) thresh={args.seg_thresh} "
                  f"device={_dev}", flush=True)
        src = RGBDv2Source(env.unwrapped, dt=1.0 / args.control_freq,
                           camera=args.camera_calib, recognizer=_rec, seg=_segm,
                           use_hand_fallback=args.hand_fallback,
                           near_hold=args.near_hold,
                           near_extent_min=args.near_extent_min,
                           near_extent=args.near_extent_max,
                           grasp_axis_offset=args.grasp_axis_offset,
                           grasp_axis_extent=args.grasp_axis_extent,
                           use_hand=args.hand_cam)
    elif args.perception == "vision":
        from fruitsort.perception.adapter import VisionSource
        src = VisionSource(env.unwrapped, dt=1.0 / args.control_freq)

    def _mk(ckpt, kind):
        return ActPureRunner(env, ckpt, kind, fb=args.fb, grip_ema=args.grip_ema,
                             max_step=args.max_step,
                             max_step_map=parse_map(args.max_step_map),
                             carry_step_map=parse_map(args.carry_step_map),
                             carry_latch=args.carry_latch,
                             carry_max_step=args.carry_max_step,
                             stall_guard=args.stall_guard,
                             descend_commit=args.descend_commit,
                             grip_hyst=args.grip_hyst,
                             release_gate=args.release_gate, release_wait=args.release_wait,
                             close_z_gate=args.close_z_gate,
                             lead_long=args.lead_long,
                             lead_long_extent=args.lead_long_extent,
                             lead_long_taper=args.lead_long_taper,
                             close_commit=args.close_commit,
                             close_commit_xy=args.close_commit_xy,
                             close_z_gate_map=parse_map(args.close_z_gate_map),
                             close_z_off_map=parse_map(args.close_z_off_map),
                             close_commit_map=parse_map(args.close_commit_map),
                             grasp_verify=args.grasp_verify,
                             grasp_proxy=args.grasp_proxy,
                             suction=args.suction,
                             suction_radius=args.suction_radius,
                             suction_carry=args.suction_carry,
                             side_grasp_kinds=tuple(k.strip() for k in args.side_grasp.split(",") if k.strip()),
                             side_offset=args.side_offset,
                             yaw_align=args.yaw_align,
                             yaw_align_off=args.yaw_align_off,
                             yaw_align_extent=args.yaw_align_extent,
                             yaw_align_rot_frac=args.yaw_align_rot_frac,
                             yaw_align_fast=args.yaw_align_fast,
                             yaw_hold_frac=args.yaw_hold_frac,
                             yaw_hold_cap=args.yaw_hold_cap,
                             yaw_hold_fade=args.yaw_hold_fade,
                             intercept=args.intercept,
                             close_win_xy=args.close_win_xy,
                             close_win_z=args.close_win_z,
                             grasp_z_bias=parse_map(args.z_bias_map),
                             yaw_map=parse_map(args.yaw_map),
                             width_map=(dict(WIDTH_MAP) if args.width_setpoint else None),
                             width_g=args.width_g, close_ramp=args.close_ramp)

    if os.environ.get("FRUITSORT_AUDIT_GT"):
        from fruitsort.perception.audit import install as _gt_install
        _gt_install(env.unwrapped)
        print("[GT 审计] 已挂上真值计数器（fruit_actors / _slots / is_grasping）", flush=True)

    runner = _mk(args.ckpt, "apple")
    runners = {}
    if args.ckpt_map:
        for kv in args.ckpt_map.split(","):
            _k, _path = kv.split("=", 1)
            runners[_k.strip()] = _mk(_path, _k.strip())
        print("按类路由:", list(runners), flush=True)
    single_runner = None
    if args.single_ckpt:
        single_runner = _mk(args.single_ckpt, "apple")
        print(f"队列切换: 单件模式={args.single_ckpt} (queue<={args.queue_thr})", flush=True)
    yaw_rng = np.random.default_rng(args.seed + 7)
    ep_yaw = 0.0
    ctrl = StreamController(env, runner, env.unwrapped, grasp_x=args.grasp_x,
                            objs=(None if src is None else {}),
                            only_kind=args.only_kind, scripted=(args.policy == "scripted"),
                            in_dist=args.in_dist_obs, noise=args.noise, runners=runners,
                            single_runner=single_runner, queue_thr=args.queue_thr,
                            engage_x=args.engage_x,
                            scripted_kinds=[k.strip() for k in args.scripted_kinds.split(",") if k.strip()],
                            decline_yaw=args.decline_yaw, long_extent=args.long_extent,
                            kind_priority=parse_map(args.kind_priority),
                            decline_kinds=tuple(k.strip() for k in args.decline_kinds.split(",") if k.strip()),
                            perception_targets=args.perception_targets,
                            give_up_s=args.give_up_s, src=src,
                            trace_abandon=args.trace_abandon)

    rec_dir = Path(args.record) if args.record else None
    if rec_dir is not None:
        rec_dir.mkdir(parents=True, exist_ok=True)
    n_saved = 0
    max_steps = int(args.seconds * args.control_freq)
    runs = []
    t0 = time.time()
    _vw = {"w": None}          # 跨局共用：每局重建会截断文件（实测 3×90s 只剩 90s）
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        u = env.unwrapped
        runner.reset()
        if args.grasp_yaw_range is not None:
            ep_yaw = float(yaw_rng.uniform(*args.grasp_yaw_range))
        ctrl = StreamController(env, runner, u, grasp_x=args.grasp_x,
                                objs=(None if src is None else {}),
                                only_kind=args.only_kind, scripted=(args.policy == "scripted"),
                                in_dist=args.in_dist_obs, noise=args.noise, runners=runners,
                                single_runner=single_runner, queue_thr=args.queue_thr,
                                engage_x=args.engage_x,
                                scripted_kinds=[k.strip() for k in args.scripted_kinds.split(",") if k.strip()],
                                scripted_yaw=ep_yaw,
                                decline_yaw=args.decline_yaw, long_extent=args.long_extent,
                            kind_priority=parse_map(args.kind_priority),
                            decline_kinds=tuple(k.strip() for k in args.decline_kinds.split(",") if k.strip()),
                            perception_targets=args.perception_targets,
                            give_up_s=args.give_up_s, src=src,
                            trace_abandon=args.trace_abandon)
        writer = _vw["w"]
        if args.video:
            _v = Path(args.video)
            _v.parent.mkdir(parents=True, exist_ok=True)
            _v = _v.with_suffix(".mp4")
        buf = {"obs": [], "act": [], "oh": [], "wr": [], "roi": [], "kind": None, "id": None}

        def flush():
            nonlocal n_saved, buf
            it = u.stream_stats["items"].get(buf["id"] or -1)
            if (rec_dir is not None and it is not None
                    and it.get("outcome") == "delivered"):
                _save_demo(rec_dir, n_saved, buf, args.seed + ep)
                n_saved += 1
        if args.video:
            _v = Path(args.video)
            _v.parent.mkdir(parents=True, exist_ok=True)
            _v = _v.with_suffix(".mp4")
        buf = {"obs": [], "act": [], "oh": [], "wr": [], "roi": [], "kind": None, "id": None}
        _prev_grip = 1.0
        _t_phase1 = None
        _t_phase2 = None
        _n_ev = 0
        _last_pos = {}

        for t in range(max_steps):
            if src is not None:
                objs = src.update(obs, held_kind=(ctrl.runner.kind if ctrl.runner.grasping else None))
                ctrl.objs = objs
                ctrl.runner.set_actors(objs)
            act = ctrl.act(obs)
            it = ctrl.target
            if args.trace_items and it is not None and t % 20 == 0:
                _tcp = u.agent.tcp_pose.p[0].cpu().numpy()
                try:
                    from fruitsort.sim.conveyor import BIN_LAYOUT as _BL
                    _bx, _by = _BL[it["kind"]]
                    _bd = float(np.hypot(_tcp[0] - _bx, _tcp[1] - _by))
                except Exception:  # noqa: BLE001
                    _bd = float("nan")
                _rn = ctrl.runner
                print(f"    [trk] t={t:4d} id={it['id']:3d} {it['kind']:10s} "
                      f"ph={_rn.phase_state} att={getattr(_rn,'attached',False)} "
                      f"bd={_bd*100:5.1f}cm tcp=({_tcp[0]:+.2f},{_tcp[1]:+.2f},{_tcp[2]:.2f}) "
                      f"engaged_at={it.get('t_engage', -1):.1f}s", flush=True)
            if it is None or it["id"] != buf["id"]:
                flush()
                if it is not None:
                    buf["id"], buf["kind"] = it["id"], it["kind"]
            # 只用"四类都在场"的帧录制：缺类时我们喂的是 ghost 观测，
            # 图像里没有这颗水果、状态里却有 → 图文不一致会污染训练数据
            all_live = all(any(sl["live"] for sl in u._slots[k]) for k in KINDS)
            if rec_dir is not None and not all_live and not args.allow_ghost_frames:
                flush()          # 缺类 → 结束当前段，等四类齐全再开新段
                buf["id"] = None
            if buf["id"] is not None and ctrl.scripted and rec_dir is not None:
                buf["obs"].append(build_obs(u, buf["kind"]))
                # 专家的 last_target 是 4 维绝对目标（与 record_demos 一致）
                buf["act"].append(np.asarray(ctrl.sorter.last_target, dtype=np.float32))
                _xy = None
                if args.roi_m > 0 and buf["kind"] is not None and ctrl.target is not None:
                    _xy = ctrl.target["actor"].pose.p[0].cpu().numpy()[:2]
                if args.roi_m > 0 and not hasattr(ctrl, "_roi_helper"):
                    from fruitsort.perception.roi import RoiCrop
                    ctrl._roi_helper = RoiCrop(width_m=args.roi_m, out_size=128)
                oh, wr, roi_b = _pack_images(
                    obs, getattr(ctrl, "_roi_helper", None), _xy)
                buf["oh"].append(oh); buf["wr"].append(wr)
                if roi_b:
                    buf["roi"].append(roi_b)
            obs, _, term, trunc, info = env.step(act)
            if args.video:
                frame = env.render()
                arr = frame.cpu().numpy() if hasattr(frame, "cpu") else np.asarray(frame)
                arr = arr[0] if arr.ndim == 4 else arr
                img = np.ascontiguousarray(arr[..., ::-1].astype(np.uint8))   # RGB→BGR
                if writer is None:
                    h, w = img.shape[:2]
                    writer = cv2.VideoWriter(str(_v), cv2.VideoWriter_fourcc(*"mp4v"),
                                             float(args.video_fps or args.control_freq), (w, h))
                    _vw["w"] = writer
                st_v = u.stream_stats
                tgt_v = ctrl.target["kind"] if ctrl.target is not None else "-"
                ph = ctrl.runner.phase_state if ctrl.runner is not None else -1
                lines = [f"ep={ep + 1}/{args.episodes}   t={u._t:5.1f}s   belt={args.belt:.2f} m/s"
                         f"   gap={args.gap_min}-{args.gap_max} m",
                         f"target={tgt_v}   phase={ph}   grip={'closed' if ctrl.runner.prev_grip < 0 else 'open'}",
                         f"delivered={st_v['delivered']}   missed={st_v['missed']}   dropped={st_v['dropped']}"
                         f"   spawned={st_v['spawned']}",
                         f"engaged={len(getattr(ctrl, 'engaged_ids', ()))}   "
                         f"handed_to_peers={max(0, st_v['missed'] - 0)}   "
                         f"step_mode={'per-class' if args.max_step_map else 'uniform'}"]
                hh, ww = img.shape[:2]
                fs = max(0.6, ww / 1920.0 * 1.0)
                lh = int(round(34 * fs))
                pad = int(round(10 * fs))
                overlay = img.copy()
                cv2.rectangle(overlay, (0, 0), (int(ww * 0.42), lh * len(lines) + 2 * pad),
                              (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
                for i, txt in enumerate(lines):
                    org = (pad, pad + lh * (i + 1) - int(8 * fs))
                    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs,
                                (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs,
                                (255, 255, 255), 2, cv2.LINE_AA)
                if args.video_inset and src is not None and getattr(src, "vs", None) is not None:
                    _k = ctrl.runner.kind if ctrl.runner is not None else None
                    _est = (ctrl.objs or {}).get(_k) if _k else None
                    _txy = None
                    if _est is not None:
                        try:
                            _txy = _est.pose.p[0].cpu().numpy()[:2]
                        except Exception:  # noqa: BLE001
                            _txy = None
                    _mk = getattr(getattr(src, "vs", None), "last_masks", None)
                    _hom = getattr(getattr(src, "vs", None), "homography", None)
                    if _hom is not None:
                        def _proj(xy, _H=_hom.H):
                            p = _H @ np.array([float(xy[0]), float(xy[1]), 1.0])
                            return float(p[0] / p[2]), float(p[1] / p[2])
                    else:
                        _cam = getattr(getattr(src, "vs", None), "camera", None)
                        _bz = float(getattr(u, "belt_top_z", 0.45))

                        def _proj(xy, _c=_cam, _z=_bz):
                            uv = _c.project(np.array([[xy[0], xy[1], _z]]))[0]
                            return float(uv[0]), float(uv[1])
                    panel = _head_panel(obs, getattr(src, "last_tracks", None),
                                        _txy, _k, _proj, masks=_mk)
                    if panel is not None:
                        ph_, pw_ = panel.shape[:2]
                        m = int(round(ww * 0.008))
                        y0, x0 = hh - ph_ - m, ww - pw_ - m
                        if y0 > 0 and x0 > 0:
                            cv2.rectangle(img, (x0 - 2, y0 - 2),
                                          (x0 + pw_ + 2, y0 + ph_ + 2), (90, 90, 90), 2)
                            img[y0:y0 + ph_, x0:x0 + pw_] = panel
                writer.write(img)
            _rn = ctrl.runner
            if args.trace_drop:
                try:
                    _evs = u.stream_stats["events"]
                except Exception:  # noqa: BLE001
                    _evs = []
                while _n_ev < len(_evs):
                    _te, _ke, _what, _iid = _evs[_n_ev]
                    _n_ev += 1
                    if _what != "dropped":
                        continue
                    _tcp = u.agent.tcp_pose.p[0].cpu().numpy()
                    try:
                        from fruitsort.sim.conveyor import BIN_LAYOUT as _BL
                        _bx, _by = _BL[_ke]
                    except Exception:  # noqa: BLE001
                        _bx, _by = 0.0, -0.3
                    _lp = _last_pos.get(_iid)
                    _lp = None if _lp is None else np.asarray(_lp)
                    _att = getattr(_rn, "attached", False)
                    _gr = getattr(_rn, "grasping", False)
                    print(f"    [drop] t={u._t:5.1f} {_ke:10s} id={_iid:3d} "
                          f"ph={-1 if _rn is None else _rn.phase_state} grasp={int(_gr)} "
                          f"tcp=({_tcp[0]:+.2f},{_tcp[1]:+.2f},{_tcp[2]:.2f}) "
                          f"last=({'-' if _lp is None else f'{_lp[0]:+.2f}'.strip()},"
                          f"{'-' if _lp is None else f'{_lp[1]:+.2f}'.strip()},"
                          f"{'-' if _lp is None else f'{_lp[2]:.2f}'.strip()}) "
                          f"bin=({_bx:+.2f},{_by:+.2f}) "
                          f"d_bin={float(np.hypot(_tcp[0]-_bx, _tcp[1]-_by))*100:4.1f}cm", flush=True)
            if args.trace_drop:
                for _k2 in u.active_kinds:
                    for _s2 in u._slots[_k2]:
                        if _s2["live"]:
                            _last_pos[_s2["item"]["id"]] = _s2["item"]["actor"].pose.p[0].cpu().numpy()
            if (args.trace_close or args.trace_release) and _rn is not None:
                _kind_t = ctrl.target["kind"] if ctrl.target is not None else None
                _ph = _rn.phase_state
                if _ph == 1 and _t_phase1 is None:
                    _t_phase1 = float(u._t)
                if _ph == 2 and _t_phase2 is None:
                    _t_phase2 = float(u._t)
                _g = float(getattr(_rn, "prev_grip", 1.0))
                if (args.trace_release and _g > 0 >= _prev_grip and _kind_t is not None):
                    try:
                        from fruitsort.sim.conveyor import BIN_LAYOUT as _BL3
                        _bx3, _by3 = _BL3[_kind_t]
                    except Exception:  # noqa: BLE001
                        _bx3, _by3 = 0.0, -0.3
                    _tcp3 = u.agent.tcp_pose.p[0].cpu().numpy()
                    print(f"    [open] t={u._t:5.1f} {_kind_t:10s} ph={_ph} "
                          f"grasp={int(getattr(_rn, 'grasping', False))} "
                          f"tcp=({_tcp3[0]:+.2f},{_tcp3[1]:+.2f},{_tcp3[2]:.2f}) "
                          f"d_bin={float(np.hypot(_tcp3[0]-_bx3, _tcp3[1]-_by3))*100:4.1f}cm "
                          f"dy={(_tcp3[1]-_by3)*100:+5.1f}cm", flush=True)
                if _g < 0 <= _prev_grip and _kind_t is not None and ctrl.objs:
                    _a = ctrl.objs.get(_kind_t)
                    if _a is not None:
                        _tcp = u.agent.tcp_pose.p[0].cpu().numpy()
                        _fp = _a.pose.p[0].cpu().numpy()
                        _d = _tcp - _fp
                        print(f"    [close] t={u._t:5.1f} {_kind_t:10s} ph={_ph} "
                              f"fruit_x={_fp[0]:+.3f} d=({_d[0]*100:+5.1f},{_d[1]*100:+5.1f},"
                              f"{_d[2]*100:+5.1f})cm |d|={np.linalg.norm(_d)*100:4.1f}cm "
                              f"descend={0.0 if _t_phase1 is None else u._t - _t_phase1:4.2f}s "
                              f"att={getattr(_rn, 'attached', False)}", flush=True)
                _prev_grip = _g
            if bool(term) or bool(trunc):
                break
        flush()
        st = u.stream_stats
        # "机会内"口径：只算本臂接管过的件（其余漏检视为让给旁边的工作臂）
        items = st.get("items", {})
        eng = [it for it in items.values() if it.get("id") in getattr(ctrl, "engaged_ids", set())]
        eng_done = [it for it in eng if it.get("outcome") == "delivered"]
        eng_res = [it for it in eng if it.get("outcome") in ("delivered", "dropped", "missed")]
        declined = [it for it in items.values()
                    if it.get("outcome") == "missed" and it.get("id") not in getattr(ctrl, "engaged_ids", set())]
        handle = [it["t_end"] - it["t_engage"] for it in eng
                  if it.get("outcome") == "delivered" and it.get("t_end") and it.get("t_engage")]
        st["handle_s"] = float(np.mean(handle)) if handle else 0.0
        st["engaged"] = len(eng)
        st["engaged_by_kind"] = {}
        for k in KINDS:
            e_k = [it for it in eng if it.get("kind") == k]
            d_k = [it for it in e_k if it.get("outcome") == "delivered"]
            st["engaged_by_kind"][k] = (len(e_k), len(d_k))
        st["engaged_delivered"] = len(eng_done)
        st["declined"] = len(declined)
        resolved = st["delivered"] + st["dropped"] + st["missed"]
        row = dict(ep=ep, belt=args.belt, seconds=round(u._t, 1),
                   spawned=st["spawned"], delivered=st["delivered"],
                   dropped=st["dropped"], missed=st["missed"], picked=st["picked"],
                   switches=ctrl.switches,
                   success_rate=(st["delivered"] / resolved) if resolved else 0.0,
                   engaged=st["engaged"], engaged_delivered=st["engaged_delivered"],
                   engaged_success=(len(eng_done) / len(eng_res)) if eng_res else 0.0,
                   declined=st["declined"], handle_s=st["handle_s"],
                   throughput_per_min=60.0 * st["delivered"] / max(u._t, 1e-6),
                   per_kind=st["per_kind"])
        runs.append(row)
        print(f"[ep{ep}] {row['seconds']:.0f}s 上料 {row['spawned']} | 送达 {row['delivered']} "
              f"| 机会内 {row['engaged_delivered']}/{row['engaged']} "
              f"= {(row['engaged_delivered']/row['engaged']*100) if row['engaged'] else 0:.0f}% "
              f"| 让给同伴 {row['declined']} | 单件 {row['handle_s']:.1f}s "
              f"掉落 {row['dropped']} 漏过 {row['missed']} | 抓取过 {row['picked']} | "
              f"成功率 {row['success_rate']*100:.1f}% | 吞吐 {row['throughput_per_min']:.2f} 件/min "
              f"| 换目标 {row['switches']}", flush=True)
        print("      每类: " + "  ".join(
            f"{k} {v['delivered']}/{v['spawned']}(drop {v['dropped']}) "
            for k, v in row["per_kind"].items()), flush=True)
        ebk = st.get("engaged_by_kind", {})
        if args.trace_items:
            for _it in st.get("items", {}).values():
                _te, _tx = _it.get("t_engage"), _it.get("t_end")
                _xe = _it.get("x_engage")
                print(f"    [item] id={_it.get('id')} {_it.get('kind'):10s} "
                      f"outcome={_it.get('outcome')} engage={_te} end={_tx} "
                      f"x_engage={'-' if _xe is None else f'{_xe:+.2f}'} "
                      f"dur={'-' if (_te is None or _tx is None) else f'{_tx-_te:.1f}s'}", flush=True)
        print("      机会内每类: " + "  ".join(
            f"{k} {ebk.get(k, (0, 0))[1]}/{ebk.get(k, (0, 0))[0]}" for k in KINDS), flush=True)

    n = len(runs)
    agg = {k: sum(r[k] for r in runs) for k in
           ("spawned", "delivered", "dropped", "missed", "picked")}
    resolved = agg["delivered"] + agg["dropped"] + agg["missed"]
    print(f"\n== 连续作业 @{args.belt} m/s，间隔 {args.gap_min}–{args.gap_max} m，"
          f"尺寸 ±{int((args.scale_max-1)*100)}% ==")
    print(f"总上料 {agg['spawned']} | 送达 {agg['delivered']} | 掉落 {agg['dropped']} | "
          f"漏过 {agg['missed']} | 成功率 {agg['delivered']/max(resolved,1)*100:.1f}% | "
          f"吞吐 {np.mean([r['throughput_per_min'] for r in runs]):.2f} 件/min")
    eng = sum(r["engaged"] for r in runs); engd = sum(r["engaged_delivered"] for r in runs)
    print(f"机会内成功率: {engd}/{eng} = {(engd / eng * 100) if eng else 0:.1f}% | "
          f"让给同伴（未接管即漏检）: {sum(r['declined'] for r in runs)} 件")
    for k in KINDS:
        d = sum(r["per_kind"][k]["delivered"] for r in runs)
        s = sum(r["per_kind"][k]["spawned"] for r in runs)
        print(f"  {k:10s} {d}/{s}" + (f" = {d/s*100:.0f}%" if s else ""))
    out = Path(args.out) if args.out else ROOT / "runs" / f"{time.strftime('%Y-%m-%d-%H%M')}-stream-eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"args": vars(args), "runs": runs, "aggregate": agg}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    if rec_dir is not None:
        print(f"录制 {n_saved} 条成功片段 → {rec_dir}")
    if os.environ.get("FRUITSORT_AUDIT_GT"):
        from fruitsort.perception import audit as _gt_audit
        _gt_audit.report()
    print(f"结果写入 {out}/metrics.json | 用时 {(time.time()-t0)/60:.1f} min")
    if _vw["w"] is not None:
        _vw["w"].release(); _vw["w"] = None
        print(f"视频写入 {Path(args.video).with_suffix('.mp4')}")
    env.close()


if __name__ == "__main__":
    main()
