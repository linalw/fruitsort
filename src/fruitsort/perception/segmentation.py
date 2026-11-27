"""图像分割：从 RGB-D 得到**每个物体的掩码**。

设计
----
分割器的输入是"像素 + 该像素的世界坐标"（由 `CameraModel.unproject` 给出），
输出是一组掩码。这样分割**不依赖俯视/平面假设**——只要相机的 K/T 标定对了，
相机怎么装都能用；真机上换成学习式分割（SAM/U-Net）时，只需实现同一个 `segment()`。

当前实现 `DepthColorSegmenter`：
1. **高度门**：世界 z 高于带面（`above_belt`）、低于 `max_height` → 去掉带面/地面；
2. **颜色门**：HSV 饱和度/亮度 → 去掉灰色机械臂、料框、阴影；
3. 形态学开闭 + 连通域，面积过小的丢掉；
4. **粘连分离**：距离变换找局部极大值做标记，分水岭把贴在一起的两颗切开
   （仿真里同框多件很常见，不切就会把两颗当成一颗超长物体）。

`family_of()` 只做"颜色族"粗分类（红/黄/绿），真正的类别由 `recognition` 决定。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

RED, YELLOW, GREEN = "red", "yellow", "green"


def family_of(hue: float) -> str | None:
    """OpenCV HSV 的 hue（0–179）→ 颜色族。"""
    if hue <= 10 or hue >= 168:
        return RED
    if 16 <= hue <= 34:
        return YELLOW
    if 35 <= hue <= 62:
        return GREEN
    return None


@dataclass
class SegMask:
    mask: np.ndarray            # bool (H,W)
    bbox: tuple[int, int, int, int]   # x0,y0,x1,y1
    area: int
    family: str | None
    hue: float
    sat: float
    val: float
    split_of: int = -1          # 由哪个大连通域切开（诊断用）
    score: float = 1.0          # 该掩码的置信度（经典=1.0；学习式=掩码内平均概率）


class DepthColorSegmenter:
    """深度门 + 颜色门 + 分水岭分离。纯 CPU，1280×960 约 8 ms。"""

    name = "depth_color"

    def __init__(self, above_belt: float = 0.008, max_height: float = 0.35,
                 sat_min: int = 70, val_min: int = 50,
                 min_area: int = 60, open_k: int = 3, close_k: int = 5,
                 split: bool = True, split_ratio: float = 1.55,
                 split_peak_frac: float = 0.80, erode: int = 0):
        self.above_belt = float(above_belt)
        self.max_height = float(max_height)
        self.sat_min = int(sat_min)
        self.val_min = int(val_min)
        self.min_area = int(min_area)
        self.open_k = int(open_k)
        self.close_k = int(close_k)
        self.split = bool(split)
        self.split_ratio = float(split_ratio)
        self.split_peak_frac = float(split_peak_frac)
        self.erode = int(erode)

    # ------------------------------------------------------------------ #
    def _gate(self, rgb: np.ndarray, pts_world: np.ndarray, belt_z: float) -> tuple[np.ndarray, np.ndarray]:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        z = pts_world[..., 2]
        ok_depth = np.isfinite(z)
        above = ok_depth & (z > belt_z + self.above_belt) & (z < belt_z + self.max_height)
        color = (hsv[..., 1] >= self.sat_min) & (hsv[..., 2] >= self.val_min)
        m = (above & color).astype(np.uint8)
        if self.open_k > 0:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((self.open_k, self.open_k), np.uint8))
        if self.close_k > 0:
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((self.close_k, self.close_k), np.uint8))
        return m, hsv

    def _split_touching(self, lab: np.ndarray, n: int,
                        stats: np.ndarray) -> list[tuple[np.ndarray, int]]:
        """把粘连的连通域按"距离变换的峰"切开。

        判据：距离变换里**有几个峰**。圆形水果只有一个峰；贴在一起的有多个。
        切法：对每个掩码像素，归给最近的峰中心（Voronoi）——比 `cv2.watershed`
        稳定得多（分水岭在深度粘连时会把大部分像素判给同一个盆地）。
        切开后每块必须 ≥ 总面积的 25%，否则视为一颗有凹坑的果子，不切。
        """
        out: list[tuple[np.ndarray, int]] = []
        for i in range(1, n):
            comp = (lab == i)
            area = int(comp.sum())
            if not self.split or area < self.min_area:
                out.append((comp, i))
                continue
            dt = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 5)
            tmax = float(dt.max())
            if tmax < 2.0:
                out.append((comp, i))
                continue
            npk, pk_lab, pk_stats, pk_cent = cv2.connectedComponentsWithStats(
                (dt >= self.split_peak_frac * tmax).astype(np.uint8), 8)
            if npk <= 2:
                out.append((comp, i))
                continue
            centers = pk_cent[1:npk]                      # (K,2) 各峰的像素质心
            ys, xs = np.nonzero(comp)
            pix = np.stack([xs, ys], axis=1).astype(np.float64)
            d2 = ((pix[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)   # (N,K)
            assign = d2.argmin(axis=1)
            pieces = []
            for j in range(len(centers)):
                m = np.zeros_like(comp)
                sel = pix[assign == j].astype(int)
                if len(sel) == 0:
                    continue
                m[sel[:, 1], sel[:, 0]] = True
                if int(m.sum()) >= max(self.min_area, 0.25 * area):
                    pieces.append(m)
            if len(pieces) < 2:
                out.append((comp, i))
            else:
                out.extend((p, i) for p in pieces)
        return out

    # ------------------------------------------------------------------ #
    def segment(self, rgb: np.ndarray, pts_world: np.ndarray, belt_z: float) -> list[SegMask]:
        m, hsv = self._gate(rgb, pts_world, belt_z)
        if m.sum() == 0:
            return []
        if self.erode > 0:
            m = cv2.erode(m, np.ones((2 * self.erode + 1,) * 2, np.uint8))
        return masks_from_binary(m, hsv, self)


def masks_from_binary(m: np.ndarray, hsv: np.ndarray, cfg) -> list[SegMask]:
    """二值图 → `SegMask` 列表（连通域 + 粘连分离 + 外观统计）。

    经典分割与学习式分割共用这一段：谁产生二值图都能接上。
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    pieces = cfg._split_touching(lab, n, stats)
    out: list[SegMask] = []
    for mask, origin in pieces:
        area = int(mask.sum())
        if area < cfg.min_area:
            continue
        ys, xs = np.nonzero(mask)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        hue = float(np.median(hsv[..., 0][mask]))
        sat = float(np.median(hsv[..., 1][mask]))
        val = float(np.median(hsv[..., 2][mask]))
        out.append(SegMask(mask=mask, bbox=bbox, area=area,
                           family=family_of(hue), hue=hue, sat=sat, val=val,
                           split_of=origin))
    return out
