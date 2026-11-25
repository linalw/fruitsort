"""RGB-D 感知：颜色/形状分类 + 平面单应反投影 + 多目标跟踪。

不依赖深度图（传送带高度已知），只用一个固定外部相机 + 一次标定：
- `scripts/calibrate_camera.py` 把已知世界坐标的标记放到带面上，拟合单应矩阵；
- 运行时对饱和色块做连通域，按颜色族 + 形状先验 + 上一帧位置做全局分配；
- 输出每个类别平滑后的位置与速度，接口与仿真真值一致（`get(kind) -> (pos, vel)`）。

这一层可整体替换为 SAM/YOLO + 手眼标定，上层策略不变。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[3]
CALIB_DIR = ROOT / "assets" / "calib"

RED = "red"
YELLOW = "yellow"
GREEN = "green"

# 仿真里梨与香蕉都落在黄色色相区（PBR 着色后 H≈25–30），
# 因此 pear 同时接受 yellow/green，最终靠长宽比区分。
FAMILY = {
    "apple": (RED,),
    "strawberry": (RED,),
    "banana": (YELLOW,),
    "pear": (YELLOW, GREEN),
}

# 形状先验（cost 增量，越小越匹配）
def shape_cost(kind: str, b: "Blob") -> float:
    if kind == "banana":
        return 1.5 * max(0.0, 1.45 - b.aspect)       # 细长更像香蕉
    if kind == "pear":
        return 1.5 * max(0.0, b.aspect - 1.30)       # 圆润更像梨
    if kind == "apple":
        return 1.0 * max(0.0, 1.0 - b.area / 260.0)  # 面积大更像苹果
    if kind == "strawberry":
        return 1.0 * max(0.0, b.area / 260.0 - 1.0)  # 面积小更像草莓
    return 0.0


@dataclass
class Blob:
    uv: np.ndarray
    area: int
    w: int
    h: int
    hue: float
    sat: float
    val: float

    @property
    def aspect(self) -> float:
        return max(self.w, self.h) / max(1, min(self.w, self.h))


@dataclass
class Detection:
    xy: np.ndarray
    blob: Blob
    family: str


def detect_blobs(rgb: np.ndarray, min_area: int = 12, min_sat: int = 65) -> list[Blob]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = ((hsv[..., 1] > min_sat) & (hsv[..., 2] > 40)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        m = lab == i
        hue, sat, val = hsv[m].mean(axis=0)
        out.append(Blob(uv=cent[i], area=int(area), w=int(w), h=int(h),
                        hue=float(hue), sat=float(sat), val=float(val)))
    return out


def family_of(hue: float) -> str | None:
    if hue <= 10 or hue >= 168:
        return RED
    if 16 <= hue <= 34:
        return YELLOW
    if 35 <= hue <= 62:
        return GREEN
    return None


class BeltHomography:
    """世界 (x, y) ↔ 像素 (u, v) 的平面单应。"""

    def __init__(self, H: np.ndarray, plane_z: float = 0.45, cam_xy=(0.0, -0.05), cam_h: float = 2.0):
        self.H = H
        self.H_inv = np.linalg.inv(H)
        self.plane_z = plane_z
        self.cam_xy = np.array(cam_xy, dtype=np.float64)
        self.cam_h = cam_h

    @classmethod
    def load(cls, path: Path | None = None, **kw) -> "BeltHomography":
        if path is None:
            import os as _os
            # 允许给不同相机各存一份标定（真机 Gemini 335 vs 旧的 640×480 窄视场）
            path = CALIB_DIR / _os.environ.get("FRUITSORT_CALIB", "overhead_homography.npy")
        if not path.exists():
            raise FileNotFoundError(
                f"缺少标定文件 {path}，先运行: python scripts/calibrate_camera.py")
        return cls(np.load(path), **kw)

    def pixel_to_world(self, uv: np.ndarray, dz: float = 0.0) -> np.ndarray:
        p = self.H_inv @ np.array([uv[0], uv[1], 1.0])
        p = p[:2] / p[2]
        # 高度差 dz 造成的视差修正（相机近似垂直向下）
        return self.cam_xy + (p - self.cam_xy) * (self.cam_h - dz) / self.cam_h


class VisionState:
    """多目标跟踪 + 速度估计。"""

    def __init__(self, calib: BeltHomography, half_heights: dict[str, float],
                 kinds=("apple", "strawberry", "banana", "pear"),
                 min_x: float = -1.25, max_x: float = 0.60, max_abs_y: float = 0.22,
                 vel_alpha: float = 0.35, max_lost: int = 15, dt: float = 0.05):
        self.calib = calib
        self.half_heights = half_heights
        self.kinds = tuple(kinds)
        self.min_x, self.max_x, self.max_abs_y = min_x, max_x, max_abs_y
        self.vel_alpha = vel_alpha
        self.max_lost = max_lost
        self.dt = dt
        self.pos: dict[str, np.ndarray] = {}
        self.vel: dict[str, np.ndarray] = {}
        self.lost: dict[str, int] = {k: 99 for k in self.kinds}
        self.last_seen: dict[str, int] = {k: -1 for k in self.kinds}
        self.t = 0

    # ---------------- 检测 ---------------- #
    def _detections(self, rgb: np.ndarray, active: tuple[str, ...]) -> list[Detection]:
        out = []
        for b in detect_blobs(rgb):
            fam = family_of(b.hue)
            if fam is None:
                continue
            xy = self.calib.pixel_to_world(b.uv, self.half_heights.get("apple", 0.03))
            if not (self.min_x <= xy[0] <= self.max_x and abs(xy[1]) <= self.max_abs_y):
                continue  # 料框/背景色块
            out.append(Detection(xy=xy, blob=b, family=fam))
        return out

    # ---------------- 分配 ---------------- #
    def _cost(self, kind: str, det: Detection) -> float:
        if det.family not in FAMILY[kind]:
            return 1e6
        cost = shape_cost(kind, det.blob)
        if kind in self.pos:
            # 丢太久（>max_lost）就不再强绑旧轨迹：允许按新位置重新接管，
            # 否则目标一旦被手臂遮挡错过一次，就永远追不回来（实测會被"幽灵位"锁死）。
            horizon = min(self.lost[kind], self.max_lost)
            pred = self.pos[kind] + self.vel.get(kind, np.zeros(2)) * self.dt * horizon
            w = 2.0 if self.lost[kind] > self.max_lost else 6.0
            cost += w * float(np.linalg.norm(det.xy - pred))
        if self.lost[kind] > self.max_lost:
            cost += 0.2  # 允许新目标，但优先延续旧轨迹
        return cost

    def update(self, rgb: np.ndarray, active: tuple[str, ...] | None = None) -> dict[str, Detection]:
        self.t += 1
        active = tuple(active or self.kinds)
        dets = self._detections(rgb, active)
        # 全局分配（Hungarian）+ 每类一个"空检测"候选
        kinds = list(active)
        nk, nd = len(kinds), len(dets)
        big = 1e6
        cost = np.full((nk, nd + nk), 0.8, dtype=np.float64)  # 空检测代价
        for i, kind in enumerate(kinds):
            for j, det in enumerate(dets):
                cost[i, j] = self._cost(kind, det)
        rows, cols = linear_sum_assignment(cost)
        assigned: dict[str, Detection] = {}
        for i, j in zip(rows, cols):
            if j < nd and cost[i, j] < 4.5:
                assigned[kinds[i]] = dets[j]

        # 黄色族消歧（香蕉 vs 梨）：用**世界长度**而不是进料顺序。
        # 俯视下香蕉长 ~0.16–0.20 m、梨 ~0.08–0.10 m，深度已知即可从像素长度换算。
        yellows = [d for d in dets if d.family == YELLOW]
        if yellows and ("banana" in kinds or "pear" in kinds):
            def _len(d):
                p0 = self.calib.pixel_to_world(d.blob.uv)
                p1 = self.calib.pixel_to_world(d.blob.uv + np.array([d.blob.w, d.blob.h]) / 2.0)
                return float(np.linalg.norm(p1 - p0))
            yellows.sort(key=lambda d: -_len(d))
            assigned.pop("banana", None)
            assigned.pop("pear", None)
            rest = list(yellows)
            for kind in ("banana", "pear"):
                if kind not in kinds or not rest:
                    continue
                pick = None
                for d in rest:
                    long_enough = _len(d) > 0.115
                    if (kind == "banana") == long_enough:
                        pick = d
                        break
                if pick is None:
                    pick = rest[0]
                assigned[kind] = pick
                rest.remove(pick)

        for kind in self.kinds:
            if kind in assigned:
                xy = assigned[kind].xy
                if kind in self.pos:
                    v = (xy - self.pos[kind]) / self.dt
                    self.vel[kind] = self.vel_alpha * v + (1 - self.vel_alpha) * self.vel.get(kind, np.zeros(2))
                self.pos[kind] = xy
                self.lost[kind] = 0
                self.last_seen[kind] = self.t
            else:
                self.lost[kind] = min(self.lost[kind] + 1, 99)
        return assigned

    def get(self, kind: str):
        """返回 (pos3, vel3)。

        短暂遮挡（机械臂挡住水果）时用恒速外推继续给出估计，超过 max_lost 帧才放弃。
        这正是 AHEAD / TEMPO 里"预测穿越遮挡"的最小实现。
        """
        if kind not in self.pos or self.lost.get(kind, 99) > self.max_lost:
            return None
        v = self.vel.get(kind, np.zeros(2))
        xy = self.pos[kind] + v * self.dt * self.lost.get(kind, 0)
        p = np.array([xy[0], xy[1], self.calib.plane_z + self.half_heights.get(kind, 0.03)])
        return p, np.array([v[0], v[1], 0.0])

    def stale(self, kind: str) -> bool:
        return kind not in self.pos or self.lost.get(kind, 99) > self.max_lost
