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
                 max_dist: float = 0.35, z_from_belt: bool = True, calib: str = "hand"):
        super().__init__(u, cam_name=cam_name, min_area=min_area)
        self.u = u
        self.max_dist = float(max_dist)
        # 单目深度在最后 5 cm 的 z 误差 1–2 cm，而"果实在带上"这件事是确定的：
        # 贴带面时 z 直接用 带面 + 半高（±0.5 cm），比测出来的深度更稳
        self.z_from_belt = bool(z_from_belt)
        # 手眼标定（真机必需）：给了就只用 TCP 位姿 + T_tcp_cam 算相机位姿，
        # 不再依赖仿真的 get_model_matrix()。仿真里由 scripts/fit_hand_eye.py 生成。
        self.calib = None
        if calib:
            try:
                import json
                from .camera import CALIB_DIR
                f = CALIB_DIR / f"camera_{calib}.json"
                if f.exists():
                    d = json.loads(f.read_text(encoding="utf-8"))
                    self.calib = dict(K=np.array(d["K"], dtype=np.float64),
                                      T_tcp_cam=np.array(d["T_tcp_cam"], dtype=np.float64),
                                      width=int(d.get("width", 0)),
                                      height=int(d.get("height", 0)))
            except Exception:  # noqa: BLE001
                self.calib = None

    def _refresh(self) -> None:
        """腕部相机随手臂动：每帧重取位姿。优先用**手眼标定 + TCP**（真机可用）。"""
        if self.calib is not None:
            p = self.u.agent.tcp_pose.p[0].cpu().numpy().astype(np.float64)
            q = self.u.agent.tcp_pose.q[0].cpu().numpy().astype(np.float64)
            w, x, y, z = q
            R = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ])
            T_tcp = np.eye(4); T_tcp[:3, :3] = R; T_tcp[:3, 3] = p
            T_cam = T_tcp @ self.calib["T_tcp_cam"]
            self.R, self.C = T_cam[:3, :3], T_cam[:3, 3]
            K = self.calib["K"]
            self.fx, self.fy = float(K[0, 0]), float(K[1, 1])
            self.cx, self.cy = float(K[0, 2]), float(K[1, 2])
            return
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
            if self.z_from_belt and float(p[2]) < self.belt_z + 0.10:
                d = dict(d)
                d["pos"] = np.array([p[0], p[1], self.belt_z + self.half[d["kind"]]])
            keep.append(d)
        return keep

    def cam_name_rgb(self) -> str:
        return "hand_camera"
