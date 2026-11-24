"""侧视相机（`side`）的 3D 检测：补俯视相机的远端盲区。

为什么需要
----------
`overhead` 在带面高度上的覆盖约为 x ≥ −1.1 m；而连续作业的上料点在 x≈−0.85…−1.9，
更远的水果在俯视图里直接出画（实测 pear @ x=−1.73 的像素在画面外）。
`side` 相机（1.35,−1.30,1.15，斜看带面）能看到全部四类（含 x=−1.73）。

与俯视相机的差别
----------------
- **不用单应**（斜视 + 平面假设误差大），而是用深度逐像素反投影到 3D；
- 深度图单位为毫米；`depth<=0` 视为无效（真实相机有空洞，这里也做同样的防御）；
- 分类用**3D 包围盒对角线长度**（香蕉 ~0.18 m vs 梨 ~0.09 m），不受斜视透视影响；
- 侧视会被机械臂遮挡：若某连通域的中位深度明显小于该方向的"带面距离"，说明被挡住，丢弃。
"""
from __future__ import annotations

import cv2
import numpy as np

from ..sim.assets.fruits import load_specs
from .vision import family_of

KINDS = ("apple", "strawberry", "banana", "pear")


class SideCamera:
    """`side` 相机 → 3D 实例列表（不含跟踪，由上层统一分配）。"""

    # 可见表面 → 中心的 z 修正比例：斜视只看到上表面，实测 0.4 最接近真值；
    # 正上方看（腕部相机）应取 1.0（中心 = 顶面 − 半高）
    z_center_frac: float = 0.4

    def __init__(self, u, cam_name: str = "side", min_area: int = 18):
        cam = u._sensors[cam_name].camera
        K = cam.get_intrinsic_matrix()[0].cpu().numpy()
        M = cam.get_model_matrix()[0].cpu().numpy()      # GL: cam→world
        self.R, self.C = M[:3, :3], M[:3, 3]
        self.fx, self.fy = float(K[0, 0]), float(K[1, 1])
        self.cx, self.cy = float(K[0, 2]), float(K[1, 2])
        self.belt_z = float(getattr(u, "belt_top_z", 0.45))
        self.specs = load_specs()
        self.half = {k: self.specs[k].half_height for k in KINDS}
        self.min_area = int(min_area)

    # ------------------------------------------------------------------ #
    def _unproject(self, uu, vv, d):
        """像素 + 深度(m) → 世界坐标（GL 相机系：x 右、y 上、z 后）。"""
        x_cv = (uu - self.cx) * d / self.fx
        y_gl = -(vv - self.cy) * d / self.fy
        p = np.stack([x_cv, y_gl, -d], axis=-1)
        return p @ self.R.T + self.C

    def detect(self, obs) -> list:
        rgb = obs["sensor_data"][self.cam_name_rgb()]["rgb"][0].cpu().numpy()
        depth = obs["sensor_data"][self.cam_name_rgb()]["depth"][0].cpu().numpy()[..., 0] / 1000.0
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        valid = depth > 0.05
        z = self._unproject_world_z(depth)
        mask = (valid & (hsv[..., 1] > 70) & (hsv[..., 2] > 50)
                & (z > self.belt_z + 0.01) & (z < self.belt_z + 0.35)).astype(np.uint8)
        if mask.sum() == 0:
            return []
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, cent = cv2.connectedComponentsWithStats(mask, 8)
        out = []
        for i in range(1, n):
            area = int(st[i, cv2.CC_STAT_AREA])
            if area < self.min_area:
                continue
            m = lab == i
            vs, us = np.nonzero(m)
            d = depth[vs, us]
            pts = self._unproject(us.astype(np.float64), vs.astype(np.float64), d)
            # 侧视被机械臂遮挡：blob 的深度显著小于同方向带面深度 → 丢掉
            span = pts.max(0) - pts.min(0)
            if float(np.linalg.norm(span)) > 0.45:
                continue
            hue = float(np.median(hsv[..., 0][m]))
            fam = family_of(hue)
            if fam is None:
                continue
            extent = float(np.linalg.norm(span[[0, 1]]))
            kind = self._classify(fam, extent)
            if kind is None:
                continue
            # 长轴朝向：3D 点云主轴投影到 xy（斜视只看到上表面，但长轴方向仍可靠）
            c0 = pts.mean(axis=0)
            d0 = pts - c0
            wv, vec = np.linalg.eigh(d0.T @ d0 / max(len(d0), 1))
            e1 = vec[:, int(np.argmax(wv))]
            yaw = float(np.arctan2(e1[1], e1[0]))
            pos = np.median(pts, axis=0)
            z_c = max(self.belt_z + self.half[kind] * 0.6,
                      float(pos[2]) - self.half[kind] * self.z_center_frac)
            out.append(dict(kind=kind, pos=np.array([pos[0], pos[1], z_c]),
                            extent=extent, area=area, hue=hue, yaw=yaw))
        return out

    def cam_name_rgb(self) -> str:
        return "side"

    def _unproject_world_z(self, depth):
        h, w = depth.shape
        uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        d = np.maximum(depth, 1e-6)
        x_cv = (uu - self.cx) * d / self.fx
        y_gl = -(vv - self.cy) * d / self.fy
        return self.R[2, 0] * x_cv + self.R[2, 1] * y_gl - self.R[2, 2] * d + self.C[2]

    @staticmethod
    def _classify(fam: str, extent: float):
        if fam == "red":
            return "apple" if extent > 0.055 else "strawberry"
        if fam == "yellow":
            return "banana" if extent > 0.090 else "pear"
        if fam == "green":
            return "pear"
        return None
