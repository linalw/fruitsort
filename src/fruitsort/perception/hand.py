"""腕部相机（`hand_camera`）：补"手臂悬在水果上方 → 俯视被自己挡住"的盲区。

为什么需要
----------
俯视相机是纯感知的主力（0.5 cm），但机械臂一旦伸到目标正上方，目标就被手臂挡住：
丢检后只能按带速外推，而策略此时正要闭爪 —— 位置每帧都在"往前滑"，
于是出现"悬停不敢闭爪"（实测单发 apple 单件耗时 12 s，真值只要 6.5 s）。
腕部相机挂在手腕上，**它永远看得见自己正下方的那个水果**，正好补这一段。

实现
----
复用 `SideCamera` 的深度反投影与颜色/尺寸分类（3D 包围盒，与视角无关），只加三条约束：
1. 只报**腕部附近**（TCP 水平距离 < `max_dist`）的目标；
2. 只报**带面通道内**（|y|<0.25、x∈[−1.25,0.60]）的目标 —— 料框在 |y|≥0.28，
   这条把"把手伸进料框时看到框壁/框里的水果"排除掉；
3. 只报带面附近高度（z < belt+0.22）的目标。
"""
from __future__ import annotations

import numpy as np

from .side import SideCamera

KINDS = ("apple", "strawberry", "banana", "pear")


class HandCamera(SideCamera):
    """`hand_camera` → 3D 实例列表（正下方、带面通道内）。"""

    z_center_frac = 1.0        # 正上方看：中心 = 顶面 − 半高

    def __init__(self, u, cam_name: str = "hand_camera", min_area: int = 25,
                 max_dist: float = 0.35):
        super().__init__(u, cam_name=cam_name, min_area=min_area)
        self.u = u
        self.max_dist = float(max_dist)

    def _refresh(self) -> None:
        """腕部相机随手臂动：每帧都必须重新取位姿/内参。"""
        cam = self.u._sensors[self.cam_name_rgb()].camera
        M = cam.get_model_matrix()[0].cpu().numpy()
        self.R, self.C = M[:3, :3], M[:3, 3]
        K = cam.get_intrinsic_matrix()[0].cpu().numpy()
        self.fx, self.fy = float(K[0, 0]), float(K[1, 1])
        self.cx, self.cy = float(K[0, 2]), float(K[1, 2])

    def detect(self, obs) -> list:
        self._refresh()
        out = super().detect(obs)
        try:
            tcp = self.u.agent.tcp_pose.p[0].cpu().numpy().astype(np.float64)
        except Exception:  # noqa: BLE001
            return out
        keep = []
        for d in out:
            p = np.asarray(d["pos"], dtype=np.float64)
            if abs(float(p[1])) > 0.25 or not (-1.25 <= float(p[0]) <= 0.60):
                continue
            if float(p[2]) > self.belt_z + 0.22:
                continue
            if float(np.linalg.norm(p[:2] - tcp[:2])) > self.max_dist:
                continue
            keep.append(d)
        return keep

    def cam_name_rgb(self) -> str:
        return "hand_camera"
