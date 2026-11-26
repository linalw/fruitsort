"""学习式分割后端：RGB(+高度) → 掩码，接口与 `DepthColorSegmenter` 完全一致。

为什么需要它
------------
经典分割靠阈值（饱和度 ≥70、高于带面 8 mm）。真机换灯、换皮带颜色、果皮反光时，
这些阈值必须重调，而且**同色系只能靠尺寸**区分。`LearnedSegmenter` 用一个小 U-Net
直接从图像学掩码，训练时加光度/几何增强，换光照不需要重调阈值。

目前的做法（务必如实标注）
--------------------------
仿真里没有人工标注的分割数据集，所以训练标签来自**经典分割的输出**（蒸馏）。
这能证明"学习式后端可以替换经典后端、并在此之上加鲁棒性"，但**不能**证明它比经典
在干净数据上更准。真机上应当用几十帧人工标注 + 同样的训练脚本重训。

网络
----
输入 4 通道（RGB + 高出带面的高度），输出 1 通道 logits；下采样 2 倍（640×480），
卷积 3 层 + 跳连（U-Net 精简版，约 0.12 M 参数），CPU 前向 ~12 ms。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .segmentation import DepthColorSegmenter, masks_from_binary


class _UNetSmall:
    """4 → 1 的小 U-Net（torch 懒加载，避免 import 时依赖 torch）。"""

    def __init__(self, w: int = 24):
        import torch.nn as nn
        self.net = nn.Sequential()          # 占位，真正结构在 build()
        self.w = w

    @staticmethod
    def build(w: int = 24):
        import torch
        import torch.nn as nn

        class Block(nn.Module):
            def __init__(self, cin, cout):
                super().__init__()
                self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
                self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
                # 不用 BN/GN：它们按 (H,W) 统计，patch 训练 + 全帧推理时统计量完全不同，
                # 实测会让全帧输出整体翻成"全背景"（patch loss 0.005、全帧 IoU 0）
                self.bn = nn.Identity()
                self.act = nn.ReLU(inplace=True)

            def forward(self, x):
                return self.act(self.bn(self.c2(self.act(self.c1(x)))))

        class UNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.e1 = Block(4, w)
                self.e2 = Block(w, 2 * w)
                self.e3 = Block(2 * w, 4 * w)
                self.u2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, stride=2)
                self.d2 = Block(4 * w, 2 * w)
                self.u1 = nn.ConvTranspose2d(2 * w, w, 2, stride=2)
                self.d1 = Block(2 * w, w)
                self.out = nn.Conv2d(w, 1, 1)
                self.pool = nn.MaxPool2d(2)

            def forward(self, x):
                e1 = self.e1(x)                      # H
                e2 = self.e2(self.pool(e1))          # H/2
                e3 = self.e3(self.pool(e2))          # H/4
                d2 = self.d2(torch.cat([self.u2(e3), e2], 1))
                d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
                return self.out(d1)

        return UNet()


class LearnedSegmenter:
    """学习式分割：与 DepthColorSegmenter 同接口（可直接替换）。"""

    name = "unet"

    def __init__(self, ckpt: str | Path, device: str | None = None, thresh: float = 0.5,
                 down: int = 2, min_area: int = 60, split: bool = True,
                 height_scale: float = 0.20):
        import torch
        ck = torch.load(Path(ckpt), map_location="cpu", weights_only=False)
        self.net = _UNetSmall.build(int(ck.get("w", 24)))
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        # 全帧 1280×960 在 CPU 上 ~0.8 s/帧；GPU 上 0.3 ms → 默认用 GPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.net.to(self.device)
        self.thresh = float(thresh)
        self.down = int(down)
        self.height_scale = float(height_scale)
        self.torch = torch
        # 复用经典分割的后处理（面积过滤/粘连分离/外观统计）
        self._post = DepthColorSegmenter(min_area=min_area, split=split)

    # ------------------------------------------------------------------ #
    def _input(self, rgb: np.ndarray, pts_world: np.ndarray, belt_z: float) -> np.ndarray:
        h = np.clip((pts_world[..., 2] - belt_z) / self.height_scale, 0.0, 1.0)
        h = np.nan_to_num(h, nan=0.0)
        img = rgb.astype(np.float32) / 255.0
        x = np.concatenate([img, h[..., None].astype(np.float32)], axis=2)   # H,W,4
        if self.down > 1:
            x = cv2.resize(x, (x.shape[1] // self.down, x.shape[0] // self.down),
                           interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None])              # 1,4,h,w

    def segment(self, rgb: np.ndarray, pts_world: np.ndarray, belt_z: float):
        x = self.torch.from_numpy(self._input(rgb, pts_world, belt_z)).to(self.device)
        with self.torch.no_grad():
            logit = self.net(x)
        prob = self.torch.sigmoid(logit)[0, 0].cpu().numpy()
        prob = cv2.resize(prob, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        m = (prob >= self.thresh).astype(np.uint8)
        if m.sum() == 0:
            return []
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        return masks_from_binary(m, hsv, self._post)

    # ---------------- 训练支持 ----------------
    @staticmethod
    def build_net(w: int = 24):
        return _UNetSmall.build(w)
