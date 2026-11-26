"""物体识别：掩码+几何 → 类别（apple / strawberry / banana / pear）。

两种后端
--------
* `RuleRecognizer`：颜色族（红/黄/绿）+ 世界尺寸。零样本可用，是真机冷启动的兜底，
  阈值与旧版一致，便于和历史数字对比；
* `FeatureRecognizer`：小型 MLP（8 维特征 → 4 类），由 `scripts/train_recognizer.py`
  在带标注的掩码上训练。真机上用同样接口换数据即可微调（甚至换成 ROI 上的 CNN）。

接口只有一句：`predict(feat) -> (kind, prob)`，`feat` 是 `features(...)` 的输出。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

KINDS = ("apple", "strawberry", "banana", "pear")
FEAT_DIM = 8
FEAT_NAMES = ("extent", "width", "height", "pca_ratio", "area_px", "hue", "sat", "val")


@dataclass
class Feat:
    extent: float
    width: float
    height: float
    pca_ratio: float
    area_px: float
    hue: float
    sat: float
    val: float

    def vec(self) -> np.ndarray:
        return np.array([self.extent, self.width, self.height, self.pca_ratio,
                         self.area_px, self.hue, self.sat, self.val], dtype=np.float32)


def features(g, seg) -> Feat:
    """几何 + 掩码外观 → 8 维特征。"""
    return Feat(extent=float(g.extent), width=float(g.width), height=float(g.height),
                pca_ratio=float(g.pca_ratio), area_px=float(seg.area),
                hue=float(seg.hue), sat=float(seg.sat), val=float(seg.val))


class RuleRecognizer:
    """颜色族 + 世界尺寸（旧版阈值的等价实现）。"""

    name = "rules"

    def __init__(self, apple_min: float = 0.045, banana_min: float = 0.115):
        self.apple_min = float(apple_min)
        self.banana_min = float(banana_min)

    def predict(self, f: Feat) -> tuple[str | None, float]:
        from .segmentation import family_of
        fam = family_of(f.hue)
        if fam is None:
            return None, 0.0
        if fam == "red":
            return ("apple", 0.9) if f.extent > self.apple_min else ("strawberry", 0.9)
        if fam == "yellow":
            return ("banana", 0.9) if f.extent > self.banana_min else ("pear", 0.9)
        return "pear", 0.85              # green

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        out = []
        for row in np.atleast_2d(X):
            g = type("G", (), dict(extent=row[0], width=row[1], height=row[2],
                                   pca_ratio=row[3]))()
            s = type("S", (), dict(area=row[4], hue=row[5], sat=row[6], val=row[7]))()
            k, _ = self.predict(features(g, s))
            out.append(-1 if k is None else KINDS.index(k))
        return np.array(out, dtype=np.int64)


class FeatureRecognizer:
    """8 维特征上的小 MLP（torch）。缺失权重时不可用 → 上层自动回退到规则。"""

    name = "mlp"

    def __init__(self, mu: np.ndarray, sd: np.ndarray, state: dict, hidden: int = 48):
        self.mu = np.asarray(mu, dtype=np.float32)
        self.sd = np.maximum(np.asarray(sd, dtype=np.float32), 1e-6)
        import torch
        import torch.nn as nn
        self.torch = torch
        self.net = nn.Sequential(
            nn.Linear(FEAT_DIM, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, len(KINDS)))
        self.net.load_state_dict(state)
        self.net.eval()

    def predict_batch(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = (np.atleast_2d(np.asarray(X, dtype=np.float32)) - self.mu) / self.sd
        with self.torch.no_grad():
            logits = self.net(self.torch.from_numpy(X))
            prob = self.torch.softmax(logits, dim=-1).numpy()
        return prob.argmax(axis=1), prob.max(axis=1)

    def predict(self, f: Feat) -> tuple[str | None, float]:
        idx, p = self.predict_batch(f.vec()[None])
        return KINDS[int(idx[0])], float(p[0])

    # ---------------- 存取 ----------------
    def save(self, path: Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(dict(mu=self.mu, sd=self.sd, state=self.net.state_dict(),
                             feat=FEAT_NAMES, kinds=KINDS), path)

    @classmethod
    def load(cls, path: Path) -> "FeatureRecognizer":
        import torch
        ck = torch.load(Path(path), map_location="cpu", weights_only=False)
        return cls(ck["mu"], ck["sd"], ck["state"])
