"""把感知结果包装成"和仿真真值同接口"的物体源，供策略与执行层使用。

设计目的：让 `ActPureRunner` / `build_obs` 完全不区分"真值"与"感知"——
它们只读 `obj.pose.p` 和 `obj.linear_velocity`。这样：

* oracle 模式 = 直接用 `u.fruit_actors`；
* vision 模式 = 用这里的 `VisionSource.update(obs)` 返回的替身字典。

现实中替换感知实现（YOLO/SAM + 深度 + 跟踪）时，只要保持同样的输出即可。
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from ..sim.assets.fruits import load_specs
from .rgbd import RGBDPerception
from .vision import BeltHomography, VisionState

KINDS = ("apple", "strawberry", "banana", "pear")


class _Stub:
    """最小物体接口：pose.p / linear_velocity（形状 (1,3)）。

    `raw_p` 是**未做"跟随夹爪"平滑**的估计，用于抓取确认（避免自证）。
    """

    def __init__(self, p, v, raw_p=None):
        self.pose = SimpleNamespace(p=torch.tensor([p], dtype=torch.float32))
        self.linear_velocity = torch.tensor([v], dtype=torch.float32)
        self.raw_p = np.asarray(raw_p if raw_p is not None else p, dtype=np.float64)


class VisionSource:
    """overhead RGB → 每类一个 (pos, vel) 替身。"""

    def __init__(self, u, dt: float = 1 / 30, cam_h: float = 2.0, kinds=KINDS,
                 spawn_x: float = -0.85, clip_x: float = -1.10):
        specs = load_specs()
        self.u = u
        self.kinds = tuple(kinds)
        self.specs = specs
        self.spawn_x = float(spawn_x)
        self.clip_x = float(clip_x)
        self.plane_z = float(getattr(u, "belt_top_z", 0.45))
        self.vs = VisionState(
            BeltHomography.load(cam_h=cam_h),
            {k: specs[k].half_height for k in self.kinds},
            dt=dt, kinds=self.kinds)
        # 未见过的类：先按"刚上料"的替身占位（保持与训练一致的 active=1），
        # 一旦被检测到就切到实测位置。
        self.last = {k: (np.array([self.clip_x, 0.0,
                                   self.plane_z + specs[k].half_height], dtype=np.float64),
                         np.zeros(3, dtype=np.float64))
                     for k in self.kinds}

    def update(self, obs, held_kind: str | None = None) -> dict:
        rgb = obs["sensor_data"]["overhead"]["rgb"][0].cpu().numpy()
        self.vs.update(rgb, self.kinds)
        out = {}
        try:
            belt = float(self.u.belt_speed[0])
        except Exception:  # noqa: BLE001
            belt = 0.08
        try:
            tcp = self.u.agent.tcp_pose.p[0].cpu().numpy().astype(np.float64)
        except Exception:  # noqa: BLE001
            tcp = None
        for k in self.kinds:
            got = self.vs.get(k)
            if got is None:
                # 视野外（或短暂被挡）：用"刚上料"占位（与训练侧 view-clip 一致）
                p = np.array([self.clip_x, 0.0,
                              self.plane_z + self.specs[k].half_height], dtype=np.float64)
                v = np.array([belt, 0.0, 0.0])
            else:
                p, v = got
                if float(p[0]) < self.clip_x:
                    p = np.array([self.clip_x, 0.0, p[2]], dtype=np.float64)
                    v = np.array([belt, 0.0, 0.0])
            # 已确认夹住（由指端接触力判定）：位置跟着 TCP
            # （头顶相机此时多半被手臂挡住，不能靠检测）
            raw = p.copy()
            if held_kind == k and tcp is not None:
                p = np.array([tcp[0], tcp[1], tcp[2] - 0.02], dtype=np.float64)
                v = np.zeros(3, dtype=np.float64)
            self.last[k] = (p, v)
            out[k] = _Stub(p, v, raw_p=raw)
        return out


class RGBDSource(VisionSource):
    """RGB-D 版本：多实例跟踪，每类暴露"最靠前（离机械臂最近）"的实例。

    与真值同接口，因此 runner / build_obs 不需要任何改动。
    """

    def __init__(self, u, dt: float = 1 / 30, cam_h: float = 2.0, kinds=KINDS,
                 clip_x: float = -1.10):
        specs = load_specs()
        self.u = u
        self.kinds = tuple(kinds)
        self.specs = specs
        self.clip_x = float(clip_x)
        self.plane_z = float(getattr(u, "belt_top_z", 0.45))
        self.vs = RGBDPerception(u, dt=dt, cam_h=cam_h, kinds=self.kinds)
        self.last = {k: (np.array([self.clip_x, 0.0,
                                   self.plane_z + specs[k].half_height], dtype=np.float64),
                         np.zeros(3, dtype=np.float64))
                     for k in self.kinds}
        self.miss = {k: 99 for k in self.kinds}

    def update(self, obs, held_kind: str | None = None) -> dict:
        got = self.vs.update(obs)
        try:
            belt = float(self.u.belt_speed[0])
        except Exception:  # noqa: BLE001
            belt = 0.08
        try:
            tcp = self.u.agent.tcp_pose.p[0].cpu().numpy().astype(np.float64)
        except Exception:  # noqa: BLE001
            tcp = None
        out = {}
        for k in self.kinds:
            cands = [it for it in got.get(k, []) if it.pos[0] >= self.clip_x]
            if cands:
                it = max(cands, key=lambda i: i.pos[0])       # 最靠前的一件
                p = np.array(it.pos, dtype=np.float64)
                v = np.array(it.vel, dtype=np.float64)
                v[2] = 0.0
                self.miss[k] = 0
            elif self.miss[k] < 12:
                # 短暂丢检（遮挡/反光）：沿用上一次并外推，别跳回"场外幽灵"
                p, v = self.last[k]
                p = p.copy(); p[0] += v[0] * self.vs.dt
                self.miss[k] += 1
            else:
                p = np.array([self.clip_x, 0.0,
                              self.plane_z + self.specs[k].half_height], dtype=np.float64)
                v = np.array([belt, 0.0, 0.0])
                self.miss[k] = 99
            raw = p.copy()
            if held_kind == k and tcp is not None:
                p = np.array([tcp[0], tcp[1], tcp[2] - 0.02], dtype=np.float64)
                v = np.zeros(3, dtype=np.float64)
            self.last[k] = (p, v)
            out[k] = _Stub(p, v, raw_p=raw)
        return out


class FusedSource(RGBDSource):
    """俯视（精确）+ 侧视（补远端 x<−1.0）融合。"""

    def __init__(self, u, dt: float = 1 / 30, cam_h: float = 2.0, kinds=KINDS,
                 clip_x: float = -1.10):
        from .side import SideCamera
        specs = load_specs()
        self.u = u
        self.kinds = tuple(kinds)
        self.specs = specs
        self.clip_x = float(clip_x)
        self.plane_z = float(getattr(u, "belt_top_z", 0.45))
        self.vs = RGBDPerception(u, dt=dt, cam_h=cam_h, kinds=self.kinds,
                                 extra_detectors=[SideCamera(u)])
        self.last = {k: (np.array([self.clip_x, 0.0,
                                   self.plane_z + specs[k].half_height], dtype=np.float64),
                         np.zeros(3, dtype=np.float64))
                     for k in self.kinds}
        self.miss = {k: 99 for k in self.kinds}
