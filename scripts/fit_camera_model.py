#!/usr/bin/env python
"""拟合相机模型（内参 K + 外参 T），输出 `assets/calib/camera_<name>.json`。

原理：把苹果依次放到**已知世界坐标**（x/y 网格 + 两个高度），检测其像素质心，
用针孔模型 `u,v = K·R⁻¹(p−t)` 做最小二乘，同时解 fx, fy, cx, cy 与 6 自由度位姿。

真机上不需要这个脚本（用 `cv2.calibrateCamera` + 手眼标定），只要产出**同样格式**的
JSON 即可；这里是给仿真/自检用的，顺便验证"深度反投影"这条路在数值上成立。

用法:
    python scripts/fit_camera_model.py                 # 默认 overhead
    FRUITSORT_HEAD_RES=1280x960 FRUITSORT_HEAD_FOV=1.62 python scripts/fit_camera_model.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils.structs.pose import Pose

import fruitsort.sim.tasks  # noqa: F401
from fruitsort.perception.camera import CALIB_DIR, CameraModel
from fruitsort.perception.vision import detect_blobs

# x 网格 × y × 两个高度：高度差是分离"焦距"与"相机高度"的关键
XS = (-1.30, -0.95, -0.60, -0.25, 0.10, 0.45)
YS = (-0.12, 0.00, 0.12)
DZ = (0.030, 0.150, 0.300)   # 三个高度：破 f 与相机高度的退化解
FAR = (-9.0, 0.0, 2.0)


def _place(actor, x, y, z):
    actor.set_pose(Pose.create_from_pq(torch.tensor([[x, y, z]], dtype=torch.float32)))
    actor.set_linear_velocity(torch.zeros(1, 3))
    actor.set_angular_velocity(torch.zeros(1, 3))


def _rot(rvec):
    return cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="overhead")
    ap.add_argument("--out", default="")
    ap.add_argument("--free-z", action="store_true",
                    help="不固定相机高度（默认固定：真机上相机高度是量得出来的，"
                         "放开会与焦距退化，实测带面 z 偏 1.5 cm）")
    args = ap.parse_args()

    env = gym.make("FruitSort-v1", obs_mode="rgbd", control_mode="pd_ee_pose",
                   sim_backend="cpu", with_bins=False, belt_speed_range=(0.0, 0.0))
    obs, _ = env.reset(seed=0)
    u = env.unwrapped
    belt = float(u.belt_top_z)
    kinds = list(u.active_kinds)
    if "apple" not in kinds:
        raise SystemExit("需要 apple 参与（用色块姿态拟合）")

    W: list[np.ndarray] = []
    P: list[np.ndarray] = []
    for x in XS:
        for y in YS:
            for dz in DZ:
                for k in kinds:
                    _place(u.fruit_actors[k], FAR[0], FAR[1], FAR[2]) if k != "apple" else None
                _place(u.fruit_actors["apple"], x, y, belt + u.specs["apple"].half_height + dz)
                obs, *_ = env.step(np.zeros(7))
                rgb = obs["sensor_data"]["overhead"]["rgb"][0].cpu().numpy()
                blobs = [b for b in detect_blobs(rgb, min_area=40)]
                if len(blobs) != 1:
                    continue
                W.append([x, y, belt + u.specs["apple"].half_height + dz])
                P.append(blobs[0].uv)
    cam = u._sensors["overhead"].camera
    M = cam.get_model_matrix()
    M = M[0].cpu().numpy() if getattr(M, "ndim", 2) == 3 else np.asarray(M)
    env.close()

    W = np.array(W, dtype=np.float64)
    P = np.array(P, dtype=np.float64)
    h, w = rgb.shape[:2]
    print(f"标定点 {len(W)} 个，图像 {w}×{h}，相机位置(来自仿真) {np.round(M[:3,3],3)}")

    # ---- 初值：像素/米 → 焦距；相机朝下，绕光轴做粗搜（仿真里图像相对带面是斜的）----
    # 像素/米：只取"水平相距 > 0.1 m 且同高"的点对，避免 z 方向的点在分母里为 0
    ratios = []
    for i in range(len(W)):
        for j in range(i + 1, len(W)):
            dxy = float(np.linalg.norm(W[i, :2] - W[j, :2]))
            if dxy > 0.1 and abs(W[i, 2] - W[j, 2]) < 1e-6:
                ratios.append(float(np.linalg.norm(P[i] - P[j]) / dxy))
    ppm = float(np.median(ratios)) if ratios else 300.0
    eye = M[:3, 3]
    f0 = float(ppm * (eye[2] - belt))
    best = None
    for th in np.linspace(0, 2 * np.pi, 48, endpoint=False):
        Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        R = Rz @ np.diag([1.0, -1.0, -1.0])       # 相机 x 向右、y 向下、z 向前(=朝下)
        rvec = cv2.Rodrigues(R)[0].ravel()
        K = np.array([[f0, 0, w / 2], [0, f0, h / 2], [0, 0, 1]], dtype=np.float64)
        prj = CameraModel(K=K, T=np.eye(4), width=w, height=h)
        prj.T = np.eye(4); prj.T[:3, :3] = R; prj.T[:3, 3] = eye
        err = np.nanmean(np.linalg.norm(prj.project(W) - P, axis=1))
        if best is None or err < best[0]:
            best = (err, rvec, R)

    rvec0, R0 = best[1], best[2]
    print(f"粗搜：roll 最优重投影 {best[0]:.1f} px，f0={f0:.0f}")

    # ---- 精修：rvec(3) + tvec(3) + fx,fy,cx,cy ----
    from scipy.optimize import least_squares

    free_z = bool(args.free_z)

    def unpack(p):
        rvec = p[:3]
        if free_z:
            tvec = p[3:6]; rest = p[6:10]
        else:
            tvec = np.array([p[3], p[4], float(eye[2])]); rest = p[5:9]
        fx, fy, cx, cy = rest
        return rvec, tvec, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def resid(p):
        rvec, tvec, K = unpack(p)
        cm = CameraModel(K=K, T=np.eye(4), width=w, height=h)
        cm.T[:3, :3] = _rot(rvec); cm.T[:3, 3] = tvec
        return (cm.project(W) - P).ravel()

    if free_z:
        p0 = np.concatenate([rvec0, eye, [f0, f0, w / 2, h / 2]])
        lo = np.concatenate([[-np.inf] * 6, [50, 50, 0, 0]])
        hi = np.concatenate([[np.inf] * 6, [5000, 5000, w, h]])
    else:
        # 固定相机高度（真机上量得出来）：破掉 (f, tz) 的退化解，
        # 否则带面 z 会偏 1.5 cm，而整条 z 链路都建在它上面
        p0 = np.concatenate([rvec0, eye[:2], [f0, f0, w / 2, h / 2]])
        lo = np.concatenate([[-np.inf] * 5, [50, 50, 0, 0]])
        hi = np.concatenate([[np.inf] * 5, [5000, 5000, w, h]])
    sol = least_squares(resid, p0, bounds=(lo, hi), loss="soft_l1", f_scale=2.0, max_nfev=4000)
    rvec, tvec, K = unpack(sol.x)
    T = np.eye(4); T[:3, :3] = _rot(rvec); T[:3, 3] = tvec
    cm = CameraModel(K=K, T=T, width=w, height=h, name=args.name)
    err = np.linalg.norm(cm.project(W) - P, axis=1)
    cm.reproj_px = float(np.mean(err))

    # 自检：光轴是否朝下、带面处的像素/米
    axis = cm.R @ np.array([0, 0, 1.0])
    ppm_fit = float(np.linalg.norm(cm.project(np.array([[0.0, 0.0, belt + 0.045],
                                                        [0.5, 0.0, belt + 0.045]]))[1]
                                   - cm.project(np.array([[0.0, 0.0, belt + 0.045]]))[0]))
    print(f"✓ fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    print(f"  相机中心 {np.round(cm.center,3)}  光轴 {np.round(axis,3)} (z 分量应≈-1)")
    print(f"  重投影 mean={err.mean():.2f} px  max={err.max():.2f} px   带面 {ppm_fit:.0f} px/m")

    out = Path(args.out) if args.out else CALIB_DIR / f"camera_{args.name}.json"
    cm.save(out)
    print(f"  保存到 {out}")


if __name__ == "__main__":
    main()
