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

    def __init__(self, p, v, raw_p=None, yaw=None):
        # np.array([...]) 再 as_tensor：避免 "list of ndarrays" 的慢路径（每步 4 次 × 数千步）
        self.pose = SimpleNamespace(p=torch.as_tensor(
            np.array([p], dtype=np.float32)))
        self.linear_velocity = torch.as_tensor(np.array([v], dtype=np.float32))
        self.raw_p = np.array(raw_p if raw_p is not None else p, dtype=np.float64)
        # 长轴朝向（感知提供；None 表示未知 → 执行层可退回真值四元数或不做对准）
        self.yaw = None if yaw is None else float(yaw)


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
        # 单应视觉没有朝向估计（只能退回四元数/不估计）
        self.yaw = {k: None for k in self.kinds}

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
            # 带速先验（现实中由变频器/编码器给出，见审计表第 1 行）：
            # 水果贴在带面上时速度就是带速，用差分会把 1–2 cm 的定位抖动放大成
            # 十几 cm/s 的速度噪声（训练标签里速度恒为带速）→ 策略会犹豫不敢闭爪。
            on_belt = abs(float(p[2]) - (self.plane_z + self.specs[k].half_height)) < 0.02
            if on_belt and held_kind != k:
                v = np.array([belt, 0.0, 0.0], dtype=np.float64)
            else:
                v = np.clip(v, -0.30, 0.30)
            raw = p.copy()
            yy = None
            if held_kind == k and tcp is not None:
                p = np.array([tcp[0], tcp[1], tcp[2] - 0.02], dtype=np.float64)
                v = np.zeros(3, dtype=np.float64)
            self.last[k] = (p, v)
            self.yaw[k] = yy
            out[k] = _Stub(p, v, raw_p=raw, yaw=yy)
        return out


class RGBDSource(VisionSource):
    """RGB-D 版本：多实例跟踪，每类暴露"最靠前（离机械臂最近）"的实例。

    与真值同接口，因此 runner / build_obs 不需要任何改动。
    """

    def __init__(self, u, dt: float = 1 / 30, cam_h: float = 2.0, kinds=KINDS,
                 clip_x: float = -1.10, hold_max: int = 300, hold_x_max: float = 0.30):
        specs = load_specs()
        self.u = u
        self.kinds = tuple(kinds)
        self.specs = specs
        self.clip_x = float(clip_x)
        self.dt = float(dt)
        # 遮挡/丢检时"沿带速外推"的时长与范围：水果在带面上不可能瞬移，只要它
        # 还该在工作区（x<=hold_x_max）就继续跟，别一跳回"场外幽灵"（否则策略
        # 会去够一个不存在的东西，整件丢掉）
        self.hold_max = int(hold_max)
        self.hold_x_max = float(hold_x_max)
        self.plane_z = float(getattr(u, "belt_top_z", 0.45))
        self.vs = RGBDPerception(u, dt=dt, cam_h=cam_h, kinds=self.kinds)
        self.last = {k: (np.array([self.clip_x, 0.0,
                                   self.plane_z + specs[k].half_height], dtype=np.float64),
                         np.zeros(3, dtype=np.float64))
                     for k in self.kinds}
        self.miss = {k: 99 for k in self.kinds}
        self.yaw = {k: None for k in self.kinds}

    def update(self, obs, held_kind: str | None = None) -> dict:
        got = self.vs.update(obs)
        self.last_tracks = got
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
                yy = float(getattr(it, "yaw", float("nan")))
                if not np.isfinite(yy):
                    yy = self.yaw[k]                    # 该视角没给朝向 → 沿用上次
            elif (self.miss[k] < self.hold_max
                  and float(self.last[k][0][0]) <= self.hold_x_max):
                # 短暂丢检（遮挡/反光）：沿**带速**外推（比最后一帧的瞬时速度可靠），
                # 保留 y/z，直到它本该走出工作区或超时
                p0 = self.last[k][0]
                p = np.array([float(p0[0]) + belt * self.dt, float(p0[1]), float(p0[2])],
                             dtype=np.float64)
                v = np.array([belt, 0.0, 0.0], dtype=np.float64)
                yy = self.yaw[k]
                self.miss[k] += 1
            else:
                p = np.array([self.clip_x, 0.0,
                              self.plane_z + self.specs[k].half_height], dtype=np.float64)
                v = np.array([belt, 0.0, 0.0])
                yy = None
                self.miss[k] = 99
            raw = p.copy()
            if held_kind == k and tcp is not None:
                p = np.array([tcp[0], tcp[1], tcp[2] - 0.02], dtype=np.float64)
                v = np.zeros(3, dtype=np.float64)
            self.last[k] = (p, v)
            self.yaw[k] = yy
            out[k] = _Stub(p, v, raw_p=raw, yaw=yy)
        return out


class FusedSource(RGBDSource):
    """俯视（精确）+ 侧视（补远端 x<−1.0）融合。"""

    def __init__(self, u, dt: float = 1 / 30, cam_h: float = 2.0, kinds=KINDS,
                 clip_x: float = -1.10, hold_max: int = 300, hold_x_max: float = 0.30):
        from .side import SideCamera
        specs = load_specs()
        self.u = u
        self.kinds = tuple(kinds)
        self.specs = specs
        self.clip_x = float(clip_x)
        self.dt = float(dt)
        self.hold_max = int(hold_max)
        self.hold_x_max = float(hold_x_max)
        self.plane_z = float(getattr(u, "belt_top_z", 0.45))
        # 腕部相机（hand.py）已评估：最终下探段确实能看见被手臂挡住的目标，
        # 但精度只有 1–3 cm（俯视 0.2 cm），混进来反而让单发 apple 4/6 → 3/6、
        # 掉落 2 → 3（`grasp_verify` 被 2–3 cm 的 z 误差否掉）。默认不启用。
        self.vs = RGBDPerception(u, dt=dt, cam_h=cam_h, kinds=self.kinds,
                                 extra_detectors=[SideCamera(u)])
        self.last = {k: (np.array([self.clip_x, 0.0,
                                   self.plane_z + specs[k].half_height], dtype=np.float64),
                         np.zeros(3, dtype=np.float64))
                     for k in self.kinds}
        self.miss = {k: 99 for k in self.kinds}
        self.yaw = {k: None for k in self.kinds}
