#!/usr/bin/env python
"""机位对比台：相机放低 / 上游 / 对面，各自换来什么、代价是什么。

三个问题一次量完（都不改策略、不跑闭环）：

1. **放低** → 每米多少像素、苹果/草莓各占几像素、带面还盖不盖得住；
   同时给出"离面高度视差"的偏置（nadir 下偏置 ∝ 果高/相机高，越低越大）。
2. **上游** → 覆盖率（图像四角反投影到带面）、可达窗余量、预瞄秒数。
3. **对面** → 把机械臂停在果实正上方，数"果实还剩多少像素可见"。
   nadir 下这个比值与机位在带子哪一侧无关，倾斜才有关。

用法:
    python scripts/camera_bench.py                # 跑全部机位
    python scripts/camera_bench.py --poses lower  # 只跑名字里含 lower 的
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils.structs.pose import Pose

import fruitsort.sim.tasks  # noqa: F401
from fruitsort.perception.vision import detect_blobs
from fruitsort.policies.scripted import HOME_EULER

ROOT = Path(__file__).resolve().parent.parent
FOV, RES = "1.62", "1280x960"

# 可达窗：真值口径（5 seeds×3 局）里手臂实际接管的 x 范围
REACH = (-0.45, 0.15)
# 需要"看着"的范围：可达窗下游留 0.35 m 余量 + 上游 0.9 m（≈11 s 预瞄 @0.08 m/s）
WANT = (-0.80, 1.05)

POSES = {
    # 名字: (eye, look)   look 只给 (x,y)，z 取带面
    "base_nadir_245": ((0.00, -0.05, 2.45), (0.00, -0.05)),
    "lower_nadir_200": ((0.00, -0.05, 2.00), (0.00, -0.05)),
    "lower_nadir_180": ((0.00, -0.05, 1.80), (0.00, -0.05)),
    "lower_nadir_160": ((0.00, -0.05, 1.60), (0.00, -0.05)),
    "upstream_180": ((0.45, -0.05, 1.80), (0.45, -0.05)),
    "side_tilt_opp": ((0.00, 1.05, 1.75), (0.00, -0.10)),
    "side_tilt_same": ((0.00, -1.20, 1.75), (0.00, -0.10)),
}

# y 必须拉开：只在带子中线附近取点会让单应在 y 方向病态，外推到图像四角会飞掉
CAL_PTS = [(-1.60, 0.16), (-1.60, -0.16), (-1.10, 0.16), (-1.10, -0.16),
           (-0.55, 0.16), (-0.55, -0.16), (-0.55, 0.00),
           (0.00, 0.16), (0.00, -0.16), (0.00, 0.00),
           (0.50, 0.16), (0.50, -0.16)]
TEST_PTS = [(-1.30, 0.00), (-0.75, 0.05), (-0.20, -0.05), (0.35, 0.00)]


def _set_env(pose) -> None:
    eye, look = pose
    os.environ.update({
        "FRUITSORT_HEAD_FOV": FOV, "FRUITSORT_HEAD_RES": RES,
        "FRUITSORT_HEAD_POS": ",".join(f"{v:.3f}" for v in eye),
        "FRUITSORT_HEAD_LOOK": f"{look[0]:.3f},{look[1]:.3f}",
    })
    os.environ.pop("FRUITSORT_HEAD_Z", None)


def _place(actor, x, y, z):
    actor.set_pose(Pose.create_from_pq(torch.tensor([[x, y, z]], dtype=torch.float32)))
    actor.set_linear_velocity(torch.zeros(1, 3))
    actor.set_angular_velocity(torch.zeros(1, 3))


def _only(u, kinds, keep: str, xy, H: float) -> None:
    """只留 keep 这一类在场（其余移到视野外），保证检测到的是唯一色块。"""
    for k in kinds:
        _place(u.fruit_actors[k],
               xy[0], xy[1], H + u.specs[k].half_height + 0.002) if k == keep \
            else _place(u.fruit_actors[k], -9.0, 0.0, H + 0.6)


def _head_rgb(obs) -> np.ndarray:
    return obs["sensor_data"]["overhead"]["rgb"][0].cpu().numpy()


def _red_px(rgb: np.ndarray) -> int:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = (((hsv[..., 0] <= 10) | (hsv[..., 0] >= 168))
         & (hsv[..., 1] > 80) & (hsv[..., 2] > 50))
    return int(m.sum())


def _act(world_xyz, grip: float = 1.0) -> np.ndarray:
    """世界系目标 → 基座系位姿动作（与 scripted.py 同约定）。"""
    x, y, z = world_xyz
    return np.array([y + 0.5, -x, z, *HOME_EULER, grip], dtype=np.float32)


def bench(name: str, pose, out_dir: Path) -> dict:
    _set_env(pose)
    env = gym.make("FruitSort-v1", obs_mode="rgbd", control_mode="pd_ee_pose",
                   sim_backend="cpu", with_bins=False, belt_speed_range=(0.0, 0.0))
    obs, _ = env.reset(seed=0)
    u = env.unwrapped
    kinds = list(u.active_kinds)
    H = u.belt_top_z
    res: dict = {"pose": name, "eye": pose[0], "look": pose[1]}

    # ---------------- 1. 自标定（把苹果放到已知点，拟合单应） ----------------
    wpts, ppts = [], []
    for (x, y) in CAL_PTS:
        _only(u, kinds, "apple", (x, y), H)
        obs, *_ = env.step(np.zeros(7))
        blobs = [b for b in detect_blobs(_head_rgb(obs), min_area=40)]
        if len(blobs) != 1:
            continue
        wpts.append((x, y)); ppts.append(blobs[0].uv)
    if len(wpts) < 5:
        env.close()
        res["ok"] = False
        res["calib_pts"] = len(wpts)
        return res
    Hm, _ = cv2.findHomography(np.array(wpts), np.array(ppts), 0)
    Hi = np.linalg.inv(Hm)
    res.update(ok=True, calib_pts=len(wpts))

    # ---------------- 2. 光学：px/m、覆盖率 ----------------
    def px_of(xy):
        p = Hm @ np.array([xy[0], xy[1], 1.0]); return p[:2] / p[2]

    ox, oy = px_of((-1.0, 0.0)); ux, uy = px_of((0.0, 0.0))
    ppm = float(np.hypot(ux - ox, uy - oy))                     # 每米像素
    res["px_per_m"] = round(ppm, 1)

    h_cam = float(u._sensors["overhead"].camera.get_model_matrix()[0, 2, 3])
    h_plane = h_cam - H
    # 覆盖率：图像四角反投影到带面，取 x 范围（只算 |y|<0.25 的带面内点）
    _w, _h = (int(v) for v in RES.split("x"))
    # 铺满整幅像素再反投影：带子在图像里是斜的，只看四个角会漏掉整条带面
    uu, vv = np.meshgrid(np.arange(0, _w, 16), np.arange(0, _h, 16))
    pix = np.stack([uu.ravel(), vv.ravel()]).astype(float)
    p3 = Hi @ np.vstack([pix, np.ones(pix.shape[1])])
    world = (p3[:2] / p3[2]).T
    world = world[np.isfinite(world).all(axis=1) & (np.abs(world) < 6).all(axis=1)]
    onbelt = world[np.abs(world[:, 1]) < 0.25]
    x_cov = (float(onbelt[:, 0].min()), float(onbelt[:, 0].max())) if len(onbelt) else (0, 0)
    res["x_cover"] = [round(x_cov[0], 2), round(x_cov[1], 2)]
    res["x_cover_want"] = [REACH[0] - 0.35, REACH[1] + 0.90]
    res["covers_want"] = bool(x_cov[0] <= REACH[0] - 0.20 and x_cov[1] >= REACH[1] + 0.60)
    res["lookahead_s"] = round(max(0.0, x_cov[1] - REACH[1]) / 0.08, 1)
    res["apple_px"] = round(0.065 * ppm, 1)
    res["straw_px"] = round(0.043 * ppm, 1)
    # nadir 视差：果实半高 3 cm 在 x=±1.0 m 处的 xy 偏置
    res["parallax_cm_at_1m"] = round(100 * 1.0 * 0.03 / max(h_plane, 1e-3), 2) if pose[0][0] == pose[1][0] and pose[0][1] == pose[1][1] else None

    # ---------------- 3. 定位精度（静态、单件） ----------------
    errs = []
    for (x, y) in TEST_PTS:
        _only(u, kinds, "apple", (x, y), H)
        obs, *_ = env.step(np.zeros(7))
        rgb = _head_rgb(obs)
        blobs = [b for b in detect_blobs(rgb, min_area=40)
                 if np.hypot(*(b.uv - px_of((x, y)))) < 60]
        if not blobs:
            errs.append(None); continue
        p = Hi @ np.array([*blobs[0].uv, 1.0]); wxy = p[:2] / p[2]
        errs.append(float(np.hypot(wxy[0] - x, wxy[1] - y)))
    ok = [e for e in errs if e is not None]
    res["acc_cm"] = round(100 * float(np.mean(ok)), 2) if ok else None
    res["acc_miss"] = sum(1 for e in errs if e is None)

    # ---------------- 4. 遮挡：机械臂停在果实正上方 ----------------
    _only(u, kinds, "apple", (0.0, 0.0), H)
    for _ in range(4):
        obs, *_ = env.step(_act((0.0, -0.55, 1.10)))
    px_free = _red_px(_head_rgb(obs))
    for _ in range(90):
        obs, *_ = env.step(_act((0.0, 0.0, H + 0.055)))
    px_arm = _red_px(_head_rgb(obs))
    res["apple_px_free"] = px_free
    res["apple_px_under_arm"] = px_arm
    res["visible_under_arm"] = round(px_arm / max(px_free, 1), 3)
    env.close()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", default="", help="名字子串过滤，逗号分隔")
    ap.add_argument("--out", default="runs/2026-11-26-camera-bench/sweep.json")
    args = ap.parse_args()
    keys = [k for k in POSES if not args.poses or any(s in k for s in args.poses.split(","))]

    rows = []
    for k in keys:
        print(f"== {k}  {POSES[k]}", flush=True)
        r = bench(k, POSES[k], Path(args.out).parent)
        rows.append(r)
        if r.get("ok"):
            print(f"   px/m={r['px_per_m']:6.1f}  苹果 {r['apple_px']}px 草莓 {r['straw_px']}px"
                  f"  覆盖 x=[{r['x_cover'][0]:+.2f},{r['x_cover'][1]:+.2f}]"
                  f"  预瞄 {r['lookahead_s']}s  精度 {r['acc_cm']}cm"
                  f"  臂下可见 {r['visible_under_arm']:.0%}", flush=True)
        else:
            print(f"   ✗ 标定失败（有效点 {r.get('calib_pts')}）", flush=True)

    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {p}")


if __name__ == "__main__":
    main()
