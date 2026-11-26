"""感知流水线：RGB-D → 分割掩码 → 3D 几何 → 类别 → 多目标跟踪。

这是"真机可用"的那条链路，输入只有相机数据 + 相机模型：

    rgb (H,W,3 uint8) + depth_m (H,W float) + CameraModel + 带面高度先验
        → DepthColorSegmenter            # 每件一个掩码
        → obb_from_mask                  # 中心/长轴/短轴/高度/朝向
        → Rule|FeatureRecognizer         # 类别
        → Tracker                        # 跨帧 ID + 速度

输出与 `RGBDPerception.update()` 完全同构（`{kind: [Instance, ...]}`），
所以 `RGBDSource` / `ActPureRunner` / `build_obs` 一行都不用改。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import CameraModel
from .geometry import ObjGeometry, obb_from_mask, robust_plane_z, split_plausible
from .recognition import KINDS, Feat, RuleRecognizer, features
from .rgbd import Instance
from .segmentation import DepthColorSegmenter


class Tracker:
    """匈牙利匹配 + 匀速预测（与旧版 `RGBDPerception._associate` 同语义）。"""

    def __init__(self, dt: float = 1 / 30, max_lost: int = 20, vel_alpha: float = 0.4,
                 gate_m: float = 0.25):
        self.dt = float(dt)
        self.max_lost = int(max_lost)
        self.vel_alpha = float(vel_alpha)
        self.gate_m = float(gate_m)
        self.tracks: list[Instance] = []
        self._next_id = 0
        self.t = 0

    def update(self, dets: list[Instance]) -> list[Instance]:
        self.t += 1
        preds = [tr.predict(self.dt, tr.lost + 1) for tr in self.tracks]
        if not self.tracks:
            for d in dets:
                d.track_id = self._next_id; self._next_id += 1; d.last_t = self.t
                self.tracks.append(d)
            return list(dets)
        cost = np.full((len(self.tracks), len(dets)), 1e3)
        for i, tr in enumerate(self.tracks):
            for j, d in enumerate(dets):
                c = float(np.linalg.norm(preds[i][:2] - d.pos[:2]))
                if tr.kind != d.kind:
                    c += 0.25
                cost[i, j] = c
        try:
            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment(cost)
        except Exception:                       # noqa: BLE001
            rows = cols = np.array([], dtype=int)
        used, out = set(), []
        for i, j in zip(rows, cols):
            if cost[i, j] > self.gate_m:
                continue
            tr, d = self.tracks[i], dets[j]
            v = (d.pos - tr.pos) / max(self.dt * (tr.lost + 1), 1e-6)
            tr.vel = self.vel_alpha * v + (1 - self.vel_alpha) * tr.vel
            tr.pos, tr.kind, tr.area = d.pos, d.kind, d.area
            tr.hue, tr.extent, tr.area = d.hue, d.extent, d.area
            if np.isfinite(d.yaw) and np.isfinite(tr.yaw):
                dy = float(np.arctan2(np.sin(2 * (d.yaw - tr.yaw)), np.cos(2 * (d.yaw - tr.yaw))))
                tr.yaw = float(tr.yaw + 0.4 * dy / 2.0)
            elif np.isfinite(d.yaw):
                tr.yaw = d.yaw
            tr.hits += 1; tr.last_t = self.t; tr.lost = 0
            out.append(tr); used.add(j)
        for j, d in enumerate(dets):
            if j in used:
                continue
            d.track_id = self._next_id; self._next_id += 1; d.last_t = self.t
            self.tracks.append(d); out.append(d)
        for tr in self.tracks:
            if tr.last_t != self.t:
                tr.lost += 1
        self.tracks = [tr for tr in self.tracks if tr.lost <= self.max_lost]
        return out


@dataclass
class PipelineStats:
    masks: int = 0
    rejected_shape: int = 0
    rejected_class: int = 0
    tracks: int = 0
    plane_z: float = 0.0


class PerceptionPipeline:
    """真机接口：`update_rgbd(rgb, depth_m)`；仿真里用 `update(obs)` 适配。"""

    def __init__(self, camera: CameraModel, belt_z: float, dt: float = 1 / 30,
                 segmenter: DepthColorSegmenter | None = None,
                 recognizer=None, max_lost: int = 20,
                 min_confidence: float = 0.25, plane_prior: float | None = None,
                 depth_repair: bool = True, repair_iter: int = 3,
                 mask_fill: bool = True):
        self.camera = camera
        self.belt_z = float(belt_z)
        self.plane_prior = float(plane_prior if plane_prior is not None else belt_z)
        self.seg = segmenter or DepthColorSegmenter()
        self.rec = recognizer or RuleRecognizer()
        self.min_confidence = float(min_confidence)
        self.tracker = Tracker(dt=dt, max_lost=max_lost)
        self.last_stats = PipelineStats()
        self.last_masks: list[tuple[object, str, float]] = []   # (SegMask, kind, conf) 供可视化
        self.last_debug: list[dict] = []
        # 深度修复：Gemini/RealSense 在深色果皮、反光、遮挡边缘会成片丢深度，
        # 掩码随之"缺角/中空"，长轴和高度都会偏小 → 误分类、误抓。
        self.depth_repair = bool(depth_repair)
        self.repair_iter = int(repair_iter)
        self.mask_fill = bool(mask_fill)
        self.last_repair_frac = 0.0

    # ------------------------------------------------------------------ #
    @staticmethod
    def _repair_depth(d: np.ndarray, iters: int = 3) -> tuple[np.ndarray, float]:
        """把无效深度按邻域最小值补上（迭代膨胀），并返回"坏了多少"。

        用最小值而不是均值：深度是"最近的表面"，边缘的混合像素通常比真实表面
        **更远**（打到背景），取近的那个更接近物体。
        """
        bad = ~np.isfinite(d) | (d <= 0.0)
        frac = float(bad.mean())
        if frac <= 1e-4:
            return d, 0.0
        dd = np.where(bad, np.inf, d).astype(np.float32)
        k = np.ones((3, 3), np.uint8)
        for _ in range(max(1, iters)):
            mn = cv2.erode(np.where(np.isinf(dd), 1e3, dd).astype(np.float32), k)
            fill = bad & np.isfinite(mn) & (mn < 1e3)
            dd[fill] = mn[fill]
            bad = bad & ~fill
            if not bad.any():
                break
        out = np.where(np.isinf(dd), np.nan, dd).astype(np.float64)
        return out, frac

    @staticmethod
    def _fill_mask(mask: np.ndarray) -> np.ndarray:
        """补掩码内部空洞（深度丢点造成的"中空"）。"""
        m = mask.astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return mask
        out = np.zeros_like(m)
        cv2.drawContours(out, cnts, -1, 1, -1)
        return out.astype(bool)

    # ------------------------------------------------------------------ #
    def update_rgbd(self, rgb: np.ndarray, depth_m: np.ndarray) -> dict[str, list[Instance]]:
        h, w = depth_m.shape[:2]
        if self.camera.width and (self.camera.width != w or self.camera.height != h):
            raise ValueError(f"相机模型是 {self.camera.width}×{self.camera.height}，"
                             f"输入是 {w}×{h}：缩放或重标定后再用")
        if self.depth_repair:
            depth_m, self.last_repair_frac = self._repair_depth(depth_m, self.repair_iter)
        else:
            self.last_repair_frac = float((~np.isfinite(depth_m) | (depth_m <= 0)).mean())
        pts = self.camera.unproject(depth_m)
        plane_z = robust_plane_z(pts, self.plane_prior)
        masks = self.seg.segment(rgb, pts, plane_z)

        dets: list[Instance] = []
        dbg: list[dict] = []
        accepted: list[tuple[object, str, float]] = []
        n_rej_shape = n_rej_cls = 0
        for seg in masks:
            if self.mask_fill:
                seg.mask = self._fill_mask(seg.mask)
                seg.area = int(seg.mask.sum())
            g = obb_from_mask(pts, seg.mask, plane_z)
            if g is None or not split_plausible(g):
                n_rej_shape += 1
                continue
            f: Feat = features(g, seg)
            kind, conf = self.rec.predict(f)
            if kind is None or conf < self.min_confidence:
                n_rej_cls += 1
                continue
            dets.append(Instance(kind=kind, pos=g.center.copy(), vel=np.zeros(3),
                                 area=seg.area, hue=seg.hue, extent=g.extent, yaw=g.yaw))
            accepted.append((seg, kind, float(conf)))
            dbg.append(dict(kind=kind, conf=float(conf), extent=g.extent, width=g.width,
                            height=g.height, yaw=g.yaw, bbox=seg.bbox, area=seg.area))

        tracked = self.tracker.update(dets)
        out: dict[str, list[Instance]] = {k: [] for k in KINDS}
        for it in tracked:
            if it.kind in out:
                out[it.kind].append(it)
        for k in out:
            out[k].sort(key=lambda i: -i.pos[0])      # 离机械臂近的在前（与旧版一致）
        self.last_stats = PipelineStats(masks=len(masks), rejected_shape=n_rej_shape,
                                        rejected_class=n_rej_cls, tracks=len(tracked),
                                        plane_z=plane_z)
        self.last_debug = dbg
        self.last_masks = accepted
        return out

    def update(self, obs) -> dict[str, list[Instance]]:
        """仿真适配：从 SAPIEN obs 取 RGB-D。真机上直接用 `update_rgbd`。"""
        sd = obs["sensor_data"]["overhead"]
        rgb = sd["rgb"][0].cpu().numpy()
        depth = sd["depth"][0].cpu().numpy()
        depth = depth[..., 0] if depth.ndim == 3 else depth
        depth_m = depth.astype(np.float64) / 1000.0
        return self.update_rgbd(rgb, depth_m)
