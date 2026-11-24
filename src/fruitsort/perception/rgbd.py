"""RGB-D 感知：多实例检测 + 3D 定位 + 跟踪（不依赖仿真真值）。

与 `vision.py` 的区别
---------------------
`vision.py` 只用 RGB + 单应，并假设"每类只有一个实例、z = 带面 + 半高"。
这里补上两点（也对应审计文档里的待办 2、3）：

1. **用深度**：每个像素反投影得到世界 z，物体中心 z = 顶面 z − 半高，
   因此被夹起/堆叠时 z 也是对的（不再假设贴在带面上）；
2. **多实例跟踪**：可同时跟踪同类多件（匈牙利匹配 + 匀速预测 + 外观/尺寸代价），
   供上层的"目标选择"按实例挑选。

实现细节
--------
- 相机：`overhead` 的 `get_model_matrix()`（GL 约定）给出相机位姿，取 z 作为高度；
  深度图是毫米 → 米；x,y 用已验证的单应（`assets/calib/overhead_homography.npy`，误差 0.3 cm）。
- 分割：HSV 饱和度 + 世界高度（> 带面 + 8 mm）→ 去掉灰色带面/机械臂/料框。
- 分类：颜色族 + 世界尺寸（红：苹果/草莓按尺寸；黄：香蕉/梨按世界长度）。
- 跟踪：状态 = (kind, pos3, vel3, hits, last_t)；匹配代价 = 3D 距离 + 类别不一致惩罚。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..sim.assets.fruits import load_specs
from .vision import BeltHomography, family_of

KINDS = ("apple", "strawberry", "banana", "pear")
RED, YELLOW, GREEN = "red", "yellow", "green"


@dataclass
class Instance:
    kind: str
    pos: np.ndarray          # (3,) 世界系，物体中心
    vel: np.ndarray          # (3,)
    area: int = 0
    hue: float = 0.0
    extent: float = 0.0      # 世界长度（m），用于区分香蕉/梨
    hits: int = 1
    last_t: int = 0
    lost: int = 0
    track_id: int = -1

    def predict(self, dt: float, steps: int = 1) -> np.ndarray:
        return self.pos + self.vel * dt * steps


class RGBDPerception:
    """overhead RGB-D → 每类若干实例（多目标）。"""

    def __init__(self, u, dt: float = 1 / 30, cam_h: float = 2.0, kinds=KINDS,
                 min_area: int = 40, max_lost: int = 20,
                 vel_alpha: float = 0.4, extra_detectors=()):
        self.u = u
        self.kinds = tuple(kinds)
        self.specs = load_specs()
        self.half = {k: self.specs[k].half_height for k in self.kinds}
        self.homography = BeltHomography.load(cam_h=cam_h)
        self.belt_z = float(getattr(u, "belt_top_z", 0.45))
        self.dt = float(dt)
        self.min_area = int(min_area)
        self.max_lost = int(max_lost)
        self.vel_alpha = float(vel_alpha)

        cam = u._sensors["overhead"].camera
        self.cam_z = float(cam.get_model_matrix()[0, 2, 3].cpu().numpy())
        self.k = float(cam.get_intrinsic_matrix()[0, 0, 0].cpu().numpy())
        self.cx = float(cam.get_intrinsic_matrix()[0, 0, 2].cpu().numpy())
        self.cy = float(cam.get_intrinsic_matrix()[0, 1, 2].cpu().numpy())

        # 额外视角（如 side 相机）的检测器：只补俯视看不到的远端
        self.extra_detectors = list(extra_detectors)
        self.t = 0
        self.tracks: list[Instance] = []
        self._next_id = 0

    # ------------------------------------------------------------------ #
    def _detect(self, rgb: np.ndarray, depth_m: np.ndarray) -> list[Instance]:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        z = self.cam_z - depth_m                      # 每个像素的世界高度
        mask = ((hsv[..., 1] > 70) & (hsv[..., 2] > 50)
                & (z > self.belt_z + 0.008) & (z < self.belt_z + 0.35)).astype(np.uint8)
        if mask.sum() == 0:
            return []
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, cent = cv2.connectedComponentsWithStats(mask, 8)
        out: list[Instance] = []
        for i in range(1, n):
            area = int(st[i, cv2.CC_STAT_AREA])
            if area < self.min_area:
                continue
            m = lab == i
            hue = float(np.median(hsv[..., 0][m]))
            fam = family_of(hue)
            if fam is None:
                continue
            uv = cent[i]
            xy = self.homography.pixel_to_world(uv)      # 先在带面(z=0.45)上反投影
            if not (-1.35 <= xy[0] <= 0.60 and abs(xy[1]) <= 0.22):
                continue                               # 视野外/料框
            w = int(st[i, cv2.CC_STAT_WIDTH]); h = int(st[i, cv2.CC_STAT_HEIGHT])
            p0 = self.homography.pixel_to_world(uv)
            p1 = self.homography.pixel_to_world(uv + np.array([w, h]) / 2.0)
            extent = float(np.linalg.norm(p1 - p0))
            kind = self._classify(fam, extent, area)
            if kind is None:
                continue
            z_top = float(np.median(z[m]))
            z_c = max(self.belt_z + self.half[kind] * 0.6, z_top - self.half[kind])
            # 注：x,y 用带面标定的单应；物体比带面高 3–5 cm 时的视差 <4 mm，
            # 实测加视差修正反而略差（z 估计本身有 3–5 mm 偏差），故不做二次修正。
            out.append(Instance(kind=kind, pos=np.array([xy[0], xy[1], z_c]),
                                vel=np.zeros(3), area=area, hue=hue, extent=extent))
        return out

    def _classify(self, fam: str, extent: float, area: int) -> str | None:
        """颜色族 + 世界尺寸分类（同类多件时也一致）。"""
        if fam == RED:
            return "apple" if extent > 0.045 else "strawberry"
        if fam == YELLOW:
            return "banana" if extent > 0.115 else "pear"
        if fam == GREEN:
            return "pear"
        return None

    # ------------------------------------------------------------------ #
    def _associate(self, dets: list[Instance]) -> list[Instance]:
        """匈牙利匹配（3D 预测位置 + 类别）→ 更新轨迹，返回当前帧的实例列表。"""
        self.t += 1
        preds = [tr.predict(self.dt, tr.lost + 1) for tr in self.tracks]
        nt, nd = len(self.tracks), len(dets)
        if nt == 0:
            for det in dets:
                det.track_id = self._next_id; self._next_id += 1
                det.last_t = self.t
                self.tracks.append(det)
            return dets
        big = 1e3
        cost = np.full((nt, nd), big)
        for i, tr in enumerate(self.tracks):
            for j, det in enumerate(dets):
                c = float(np.linalg.norm(preds[i][:2] - det.pos[:2]))
                if tr.kind != det.kind:
                    c += 0.25
                cost[i, j] = c
        try:
            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment(cost)
        except Exception:                                # noqa: BLE001
            rows, cols = np.array([]), np.array([])
        used_det = set()
        out: list[Instance] = []
        for i, j in zip(rows, cols):
            if cost[i, j] > 0.25:
                continue
            tr, det = self.tracks[i], dets[j]
            v = (det.pos - tr.pos) / max(self.dt * (tr.lost + 1), 1e-6)
            tr.vel = self.vel_alpha * v + (1 - self.vel_alpha) * tr.vel
            tr.pos = det.pos
            tr.kind, tr.area, tr.hue, tr.extent = det.kind, det.area, det.hue, det.extent
            tr.hits += 1; tr.last_t = self.t; tr.lost = 0
            out.append(tr); used_det.add(j)
        for j, det in enumerate(dets):
            if j in used_det:
                continue
            det.track_id = self._next_id; self._next_id += 1
            det.last_t = self.t
            self.tracks.append(det); out.append(det)
        for tr in self.tracks:
            if tr.last_t != self.t:
                tr.lost += 1
        self.tracks = [tr for tr in self.tracks if tr.lost <= self.max_lost]
        return out

    # ------------------------------------------------------------------ #
    def update(self, obs, min_hits: int = 1):
        """返回 {kind: [Instance, ...]}（按 x 从大到小 = 离机械臂最近优先）。"""
        rgb = obs["sensor_data"]["overhead"]["rgb"][0].cpu().numpy()
        depth = obs["sensor_data"]["overhead"]["depth"][0].cpu().numpy()[..., 0] / 1000.0
        dets = self._detect(rgb, depth)
        for det_src in self.extra_detectors:
            try:
                for d in det_src.detect(obs):
                    # 与俯视检测去重（同一件在重叠区只保留一次）
                    if any(np.linalg.norm(d["pos"][:2] - o.pos[:2]) < 0.08 for o in dets):
                        continue
                    dets.append(Instance(kind=d["kind"], pos=np.asarray(d["pos"], float),
                                         vel=np.zeros(3), area=int(d.get("area", 0)),
                                         hue=float(d.get("hue", 0.0)),
                                         extent=float(d.get("extent", 0.0))))
            except Exception:  # noqa: BLE001
                pass
        inst = self._associate(dets)
        out: dict[str, list[Instance]] = {k: [] for k in self.kinds}
        for it in inst:
            if it.hits >= min_hits and it.kind in out:
                out[it.kind].append(it)
        for k in out:
            out[k].sort(key=lambda i: -i.pos[0])
        return out

    def best(self, kind: str) -> Instance | None:
        live = [t for t in self.tracks if t.kind == kind and t.lost <= self.max_lost]
        if not live:
            return None
        return max(live, key=lambda i: (i.hits, i.pos[0]))
