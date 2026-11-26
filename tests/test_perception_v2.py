"""感知 v2 单元测试（不需要仿真，秒级）。

覆盖：相机投影/反投影闭环、分割与粘连分离、3D 几何、规则识别。
"""
from __future__ import annotations

import numpy as np
import pytest

from fruitsort.perception.camera import CameraModel
from fruitsort.perception.geometry import obb_from_mask, robust_plane_z
from fruitsort.perception.recognition import Feat, RuleRecognizer, features
from fruitsort.perception.segmentation import DepthColorSegmenter, family_of


def _cam_nadir(h=2.45, f=640.0):
    K = np.array([[f, 0, 320.0], [0, f, 240.0], [0, 0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])   # 朝下
    T[:3, 3] = [0.0, 0.0, h]
    return CameraModel(K=K, T=T, width=640, height=480)


def test_camera_roundtrip():
    cam = _cam_nadir()
    pts = np.array([[0.0, 0.0, 0.45], [0.3, -0.1, 0.52], [-0.5, 0.2, 0.45]])
    uv = cam.project(pts)
    assert np.isfinite(uv).all()
    # 用该像素的深度反投影，应回到原点（深度 = 相机系 z）
    dc = (pts - cam.t) @ cam.R
    depth = dc[:, 2]
    back = []
    for (u, v), d in zip(uv, depth):
        p_cam = np.linalg.inv(cam.K) @ np.array([u, v, 1.0]) * d
        back.append(cam.R @ p_cam + cam.t)
    back = np.array(back)
    assert np.allclose(back, pts, atol=1e-6)


def test_unproject_shape_and_plane():
    cam = _cam_nadir()
    depth = np.full((480, 640), 2.0)          # 2 m 远的平面 → z = 2.45 - 2.0 = 0.45
    pts = cam.unproject(depth)
    assert pts.shape == (480, 640, 3)
    assert abs(np.nanmedian(pts[..., 2]) - 0.45) < 1e-6


def test_robust_plane_z():
    cam = _cam_nadir()
    depth = np.full((480, 640), 2.0)
    pts = cam.unproject(depth)
    pts[..., 2] += np.random.default_rng(0).normal(0, 0.002, pts[..., 2].shape)
    assert abs(robust_plane_z(pts, 0.45) - 0.45) < 0.01


def _synth_scene(cam, blobs):
    """在带面上方放若干个球，渲染成 (rgb, depth, 掩码列表)。"""
    h, w = cam.height, cam.width
    depth = np.full((h, w), np.nan)
    rgb = np.zeros((h, w, 3), np.uint8)
    masks = []
    for (x, y, r, color) in blobs:
        z_top = 0.45 + 0.06
        uv = cam.project(np.array([[x, y, z_top]]))[0]
        rr = int(r * cam.K[0, 0] / (cam.t[2] - z_top))
        yy, xx = np.mgrid[0:h, 0:w]
        m = ((xx - uv[0]) ** 2 + (yy - uv[1]) ** 2) <= rr ** 2
        depth[m] = cam.t[2] - z_top            # 顶面深度
        rgb[m] = color
        masks.append(m)
    # 带面：2.0 m
    depth[np.isnan(depth)] = cam.t[2] - 0.45
    return rgb, depth, masks


def test_segment_two_objects():
    cam = _cam_nadir()
    red = (200, 30, 30)
    yel = (200, 200, 30)
    rgb, depth, _ = _synth_scene(cam, [(-0.20, 0.0, 0.03, red), (0.20, 0.0, 0.03, yel)])
    pts = cam.unproject(depth)
    segs = DepthColorSegmenter(min_area=40, split=False).segment(rgb, pts, 0.45)
    assert len(segs) == 2
    fams = sorted(s.family for s in segs)
    assert fams == ["red", "yellow"]
    for s in segs:
        g = obb_from_mask(pts, s.mask, 0.45)
        assert g is not None
        assert abs(g.height - 0.06) < 0.015
        assert 0.03 < g.extent < 0.10


def test_segment_splits_touching_pair():
    cam = _cam_nadir()
    red = (200, 30, 30)
    # 两个球挨着（圆心相距 1.2 个半径）→ 应被分水岭切开
    rgb, depth, _ = _synth_scene(cam, [(-0.028, 0.0, 0.03, red), (0.028, 0.0, 0.03, red)])
    pts = cam.unproject(depth)
    joined = DepthColorSegmenter(min_area=40, split=False).segment(rgb, pts, 0.45)
    split = DepthColorSegmenter(min_area=40, split=True, split_peak_frac=0.80).segment(rgb, pts, 0.45)
    assert len(joined) == 1, "两个球应连成一块"
    assert len(split) >= 2, "分水岭应把它们切开"


def test_geometry_of_known_box():
    cam = _cam_nadir()
    # 直接在带面上造一个 0.10 × 0.04 的长条点云
    from fruitsort.perception.geometry import ObjGeometry  # noqa: F401
    xs = np.linspace(-0.05, 0.05, 60)
    ys = np.linspace(-0.02, 0.02, 30)
    X, Y = np.meshgrid(xs, ys)
    pts = np.zeros((480, 640, 3))
    # 用掩码索引一个假的点集
    mask = np.zeros((480, 640), bool)
    flat = np.stack([X.ravel(), Y.ravel(), np.full(X.size, 0.48)], axis=1)
    idx = np.arange(len(flat))
    pts = np.full((480, 640, 3), np.nan)
    pts.reshape(-1, 3)[:len(flat)] = flat
    mask.reshape(-1)[:len(flat)] = True
    g = obb_from_mask(pts, mask, 0.45)
    assert g is not None
    assert abs(g.extent - 0.10) < 0.01
    assert abs(g.width - 0.04) < 0.01
    assert abs(g.height - 0.03) < 0.005
    assert g.pca_ratio > 2.0


def test_rules_by_size():
    r = RuleRecognizer()
    f = lambda e, w, hue: Feat(extent=e, width=w, height=0.03, pca_ratio=e / w,
                               area_px=300, hue=hue, sat=150, val=180)
    assert r.predict(f(0.065, 0.060, 0))[0] == "apple"
    assert r.predict(f(0.042, 0.040, 2))[0] == "strawberry"
    assert r.predict(f(0.170, 0.030, 25))[0] == "banana"
    assert r.predict(f(0.090, 0.070, 30))[0] == "pear"
    assert r.predict(f(0.090, 0.070, 45))[0] == "pear"
    assert family_of(100) is None      # 青/非果色


def test_features_vector():
    cam = _cam_nadir()
    rgb, depth, _ = _synth_scene(cam, [(0.0, 0.0, 0.03, (200, 30, 30))])
    pts = cam.unproject(depth)
    seg = DepthColorSegmenter(min_area=40).segment(rgb, pts, 0.45)[0]
    g = obb_from_mask(pts, seg.mask, 0.45)
    v = features(g, seg).vec()
    assert v.shape == (8,) and np.isfinite(v).all()
