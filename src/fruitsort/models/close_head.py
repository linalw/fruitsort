"""学习式"何时闭爪"分类头（scripts/train_close_head.py 训练）。

输入：与策略相同的 49 维观测（`learning.bc.build_obs`）。
输出：当前帧"专家会在接下来 ~0.1 s 内闭爪"的概率。
"""
from __future__ import annotations

import torch
from torch import nn


class CloseHead(nn.Module):
    def __init__(self, d_in: int = 49):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DescendHead(nn.Module):
    """DAgger 式两分类：0=继续下降、1=现在闭爪（scripts/train_descend_head.py）。"""

    def __init__(self, d_in: int = 49):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)
