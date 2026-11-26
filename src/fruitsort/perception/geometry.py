"""掩码 + 深度 → 3D 几何（中心、长轴、短轴、高度、朝向）。

和旧实现的区别
--------------
旧：`x,y` 用带面单应（要求 nadir）、`z = cam_z − depth`（要求光轴垂直）。
新：先用 `CameraModel.unproject` 把**每个掩码像素**变成世界点，再在 3D 上做统计：

* 中心 = 掩码内世界点的**中位数**（抗飞点）；
* 高度 = 95 分位 z − 带面 z（分位数比 max 稳，深度边缘常有飞点）；
* 长轴/短轴/朝向 = 世界 xy 平面的 PCA + 1/99 分位（和旧版一致，便于对比）。

全程不假设相机竖直，所以**倾斜安装也能用**——这是"真机可用"的关键一步。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ObjGeometry:
    center: np.ndarray        # (3,) 世界系，物体中心（z = 带面 + 高/2）
    extent: float             # 长轴（m）
    width: float              # 短轴（m）
    height: float             # 高出带面（m）
    yaw: float                # 长轴朝向（rad，世界系，mod π）
    n_points: int
    plane_z: float
    pca_ratio: float = 1.0    # 长/短轴比（识别"长条 vs 圆形"用）


def robust_plane_z(points: np.ndarray, prior_z: float, tol: float = 0.05,
                   search: float = 0.10) -> float:
    """估计带面高度：在先验 z 附近取点，取中位数。

    真机上更稳的做法是对整幅点云做 RANSAC 平面拟合；仿真里带面高度是已知的，
    这里用它做先验，只修正小偏差（例如带面被压陷、标定漂移）。
    """
    z = points[..., 2]
    m = np.isfinite(z) & (np.abs(z - prior_z) < search)
    if m.sum() < 50:
        return float(prior_z)
    # 只取"接近平面"的那一层：直方图峰值
    vals = z[m]
    hist, edges = np.histogram(vals, bins=40)
    k = int(np.argmax(hist))
    c = 0.5 * (edges[k] + edges[k + 1])
    near = vals[np.abs(vals - c) < tol]
    return float(np.median(near) if near.size else c)


def obb_from_mask(pts_world: np.ndarray, mask: np.ndarray, plane_z: float,
                  min_points: int = 30) -> ObjGeometry | None:
    """掩码 → 3D 有向包围盒。`pts_world` 与 `mask` 同形（H,W,3）/（H,W）。"""
    p = pts_world[mask]
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) < min_points:
        return None
    xy = p[:, :2]
    c = np.median(xy, axis=0)
    d = xy - c
    if len(d) < 3:
        return None
    w_, vec = np.linalg.eigh((d.T @ d) / len(d))
    e1 = vec[:, int(np.argmax(w_))]
    e2 = np.array([-e1[1], e1[0]])
    u_, v_ = d @ e1, d @ e2
    lo, hi = np.percentile(u_, [1, 99])
    vlo, vhi = np.percentile(v_, [1, 99])
    extent = float(hi - lo)
    width = float(vhi - vlo)
    yaw = float(np.arctan2(e1[1], e1[0]))
    xy_c = c + e1 * 0.5 * (lo + hi) + e2 * 0.5 * (vlo + vhi)
    z_top = float(np.percentile(p[:, 2], 95))
    height = float(max(0.0, z_top - plane_z))
    center = np.array([xy_c[0], xy_c[1], plane_z + 0.5 * height], dtype=np.float64)
    return ObjGeometry(center=center, extent=extent, width=width, height=height,
                       yaw=yaw, n_points=int(len(p)), plane_z=float(plane_z),
                       pca_ratio=float(extent / max(width, 1e-6)))


def split_plausible(g: ObjGeometry, tiny: float = 0.03, huge: float = 0.30,
                    max_width: float = 0.16) -> bool:
    """掩码是否像一个真实水果（剔除料框边、细线、粘连成一大坨的）。"""
    return (tiny <= g.extent <= huge and 0.012 <= g.width <= max_width
            and g.height >= 0.010)
