"""相机模型：内参 + 外参，用于"深度图 → 世界点"。

为什么要单独一层
----------------
旧实现把 `x,y` 交给**带面单应**（只对 nadir + 固定高度成立），把 `z` 算成
`cam_z − depth`（同样只对光轴垂直成立）。这两条在真机上都不成立：

* 真机相机几乎不可能严格垂直；
* 深度图给的是**沿光轴的 z**，不是世界 z。相机一斜，`cam_z − depth` 就不再是高度。

所以这里改成标准针孔模型：`p_cam = K⁻¹·[u,v,1]·d`，`p_world = R·p_cam + t`。
外参可以是仿真里拟合出来的，也可以是真机上用手眼/棋盘格标定出来的，**格式相同**。

标定文件格式（JSON，真机照填即可）::

    {"name": "overhead", "width": 1280, "height": 960,
     "K": [[fx,0,cx],[0,fy,cy],[0,0,1]],
     "T": [[r00,r01,r02,tx],[...],[...],[0,0,0,1]],   # world <- camera
     "reproj_px": 0.4}
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CALIB_DIR = Path(__file__).resolve().parents[3] / "assets" / "calib"


@dataclass
class CameraModel:
    K: np.ndarray                  # (3,3) 内参
    T: np.ndarray                  # (4,4) world <- camera（相机在世界系里的位姿）
    width: int = 0
    height: int = 0
    name: str = "overhead"
    reproj_px: float = float("nan")

    # ---------------- 派生量 ----------------
    @property
    def R(self) -> np.ndarray:
        return self.T[:3, :3]

    @property
    def t(self) -> np.ndarray:
        return self.T[:3, 3]

    @property
    def center(self) -> np.ndarray:
        """相机光心在世界系的位置。"""
        return self.t.copy()

    def ray_dir(self, u, v) -> np.ndarray:
        """像素 (u,v) 的单位视线方向（世界系，指向场景）。"""
        p = np.linalg.inv(self.K) @ np.array([u, v, 1.0])
        d = self.R @ p
        n = np.linalg.norm(d)
        return d / (n if n > 1e-12 else 1.0)

    # ---------------- 投影 / 反投影 ----------------
    def unproject(self, depth_m: np.ndarray) -> np.ndarray:
        """深度图 (H,W) 米 → 世界点 (H,W,3)。深度非正/NaN 处返回 NaN。"""
        d = np.asarray(depth_m, dtype=np.float64)
        h, w = d.shape[:2]
        uu, vv = np.meshgrid(np.arange(w), np.arange(h))
        pix = np.stack([uu.ravel(), vv.ravel(), np.ones(h * w)], axis=0)   # (3,N)
        rays = np.linalg.inv(self.K) @ pix                                  # (3,N) 归一化平面
        pts_cam = rays * d.reshape(1, -1)                                   # 乘深度
        pts_w = self.R @ pts_cam + self.t.reshape(3, 1)
        pts_w = pts_w.T.reshape(h, w, 3)
        bad = ~np.isfinite(d) | (d <= 0.0) | (d > 50.0)
        pts_w[bad] = np.nan
        return pts_w

    def project(self, pts_world: np.ndarray) -> np.ndarray:
        """世界点 (N,3) → 像素 (N,2)；在相机背后返回 NaN。"""
        p = np.asarray(pts_world, dtype=np.float64).reshape(-1, 3)
        c = (p - self.t) @ self.R                     # R^T (p - t)
        z = c[:, 2]
        uv = (self.K @ c.T).T
        out = uv[:, :2] / np.where(np.abs(z) > 1e-9, z, np.nan)[:, None]
        out[z <= 1e-6] = np.nan
        return out

    # ---------------- 保存 / 读取 ----------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(
            name=self.name, width=int(self.width), height=int(self.height),
            K=np.asarray(self.K).tolist(), T=np.asarray(self.T).tolist(),
            reproj_px=None if not np.isfinite(self.reproj_px) else float(self.reproj_px),
        ), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CameraModel":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        rp = d.get("reproj_px")
        return cls(K=np.array(d["K"], dtype=np.float64), T=np.array(d["T"], dtype=np.float64),
                   width=int(d.get("width", 0)), height=int(d.get("height", 0)),
                   name=str(d.get("name", "overhead")),
                   reproj_px=float("nan") if rp is None else float(rp))

    @classmethod
    def load_or_none(cls, name: str = "overhead") -> "CameraModel | None":
        p = CALIB_DIR / f"camera_{name}.json"
        return cls.load(p) if p.exists() else None
