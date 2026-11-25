"""目标为中心的 ROI 裁剪（头部相机）。

为什么需要
----------
真机头部相机（Gemini 335）要覆盖 ~4 m 的带面；把整幅图压到 96×96 后，一颗苹果只剩
**1–2 个像素**，CNN 分支等于瞎的（v1.76 实测：宽视场 + 1280×960 下策略在新相机上全崩）。
把"以目标为中心、0.5 m 见方"的窗口裁出来再缩到 96×96，果子回到 **~12 px**，
与旧窄视场（2.7 px@96）相比反而更清楚，而且与视场大小解耦。

约定
----
- 世界 (x,y) → 像素用带面单应（`BeltHomography.H`，world→pixel）；
- 窗口宽度 `width_m` 固定（默认 0.5 m），输出 `out_size`（默认 128，与录制/训练一致）；
- 目标出画时用 replicate 边界填充，保证形状稳定。
"""
from __future__ import annotations

import cv2
import numpy as np

from .vision import BeltHomography


class RoiCrop:
    def __init__(self, width_m: float = 0.50, out_size: int = 128):
        self.hom = BeltHomography.load()
        self.width_m = float(width_m)
        self.out = int(out_size)

    def _px(self, xy) -> tuple[float, float]:
        p = self.hom.H @ np.array([float(xy[0]), float(xy[1]), 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    def crop(self, rgb: np.ndarray, xy) -> np.ndarray:
        """rgb: (H,W,3) RGB；xy: 目标世界 (x,y) → (out,out,3)。"""
        h, w = rgb.shape[:2]
        u, v = self._px(xy)
        u2, _ = self._px((float(xy[0]) + self.width_m, float(xy[1])))
        half = max(6.0, abs(u2 - u) / 2.0)
        u0, u1 = int(round(u - half)), int(round(u + half))
        v0, v1 = int(round(v - half)), int(round(v + half))
        pl, pt = max(0, -u0), max(0, -v0)
        pr, pb = max(0, u1 - w), max(0, v1 - h)
        img = rgb
        if pl or pt or pr or pb:
            img = cv2.copyMakeBorder(rgb, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
        u0 += pl; u1 += pl; v0 += pt; v1 += pt
        c = img[max(0, v0):max(1, v1), max(0, u0):max(1, u1)]
        if c.size == 0 or c.shape[0] < 4 or c.shape[1] < 4:
            c = img
        return cv2.resize(c, (self.out, self.out), interpolation=cv2.INTER_AREA)
