#!/usr/bin/env python
"""对比感知后端：旧版（HSV+单应，nadir 假设）vs 新版（掩码+深度反投影+识别）。

真值只用于**打分**（这是评测，不是模型输入）：把每帧的检出与仿真里真实水果的位置
做最近邻匹配，统计检出率 / 分类错误 / 3D 误差 / 误检数。

用法:
    python scripts/bench_perception.py                     # 默认机位，两种后端对比
    python scripts/bench_perception.py --camera camera_overhead_tilt
    python scripts/bench_perception.py --frames 60 --episodes 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import fruitsort.sim.tasks  # noqa: F401
from fruitsort.perception.camera import CALIB_DIR, CameraModel
from fruitsort.perception.pipeline import PerceptionPipeline
from fruitsort.perception.rgbd import RGBDPerception

KINDS = ("apple", "strawberry", "banana", "pear")


def _gt(u) -> list[dict]:
    out = []
    for k in u.active_kinds:
        for s in u._slots[k]:
            if not s["live"]:
                continue
            p = s["item"]["actor"].pose.p[0].cpu().numpy()
            if -1.35 <= p[0] <= 0.60 and abs(p[1]) < 0.25 and p[2] > 0.40:
                out.append(dict(kind=k, pos=np.asarray(p, dtype=np.float64), id=s["item"]["id"]))
    return out


def _flatten(det: dict) -> list[dict]:
    out = []
    for k, lst in det.items():
        for it in lst or []:
            p = np.asarray(it.pos, dtype=np.float64)
            # 只比较与真值相同的窗口，否则"视野外本来就没标真值的水果"会被算成误检
            if not (-1.35 <= p[0] <= 0.60 and abs(p[1]) < 0.25):
                continue
            out.append(dict(kind=k, pos=p))
    return out


def score(gts: list[dict], dets: list[dict], match_m: float = 0.08) -> dict:
    used = set()
    hit = cls_ok = 0
    errs, e_xy, e_z, ext_err = [], [], [], []
    for g in gts:
        best, bd = None, 1e9
        for j, d in enumerate(dets):
            if j in used:
                continue
            dd = float(np.linalg.norm(d["pos"] - g["pos"]))   # 含 z
            if dd < bd:
                best, bd = j, dd
        if best is not None and bd <= match_m:
            used.add(best)
            hit += 1
            cls_ok += int(dets[best]["kind"] == g["kind"])
            errs.append(bd)
            e_xy.append(float(np.linalg.norm(dets[best]["pos"][:2] - g["pos"][:2])))
            e_z.append(float(abs(dets[best]["pos"][2] - g["pos"][2])))
    fp = len(dets) - len(used)
    return dict(n_gt=len(gts), hit=hit, cls_ok=cls_ok, fp=fp,
                err_cm=[100 * e for e in errs],
                e_xy_cm=[100 * e for e in e_xy], e_z_cm=[100 * e for e in e_z])


def summarize(rows: list[dict], name: str) -> dict:
    n_gt = sum(r["n_gt"] for r in rows)
    hit = sum(r["hit"] for r in rows)
    cls = sum(r["cls_ok"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    errs = np.array([e for r in rows for e in r["err_cm"]], dtype=float)
    exy = np.array([e for r in rows for e in r["e_xy_cm"]], dtype=float)
    ez = np.array([e for r in rows for e in r["e_z_cm"]], dtype=float)
    det = dict(name=name, frames=len(rows), n_gt=n_gt, hit=hit,
               recall=(hit / n_gt if n_gt else 0.0),
               cls_acc=(cls / max(hit, 1)), fp_per_frame=(fp / max(len(rows), 1)),
               err_mean_cm=float(errs.mean()) if errs.size else None,
               err_p95_cm=float(np.percentile(errs, 95)) if errs.size else None,
               xy_mean_cm=float(exy.mean()) if exy.size else None,
               z_mean_cm=float(ez.mean()) if ez.size else None,
               z_p95_cm=float(np.percentile(ez, 95)) if ez.size else None)
    return det


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="overhead")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--match-m", type=float, default=0.08)
    ap.add_argument("--out", default="")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="随机丢弃多少比例的深度像素（模拟反光/深色果皮的洞）")
    ap.add_argument("--noise-mm", type=float, default=0.0, help="深度高斯噪声（毫米）")
    ap.add_argument("--no-repair", action="store_true", help="关掉 v2 的深度修复/掩码补洞")
    ap.add_argument("--noise-seed", type=int, default=1234)
    ap.add_argument("--seg", default="classic", choices=["classic", "learned"],
                    help="v2 的分割后端：经典(颜色+深度) 或 学习式(U-Net)")
    ap.add_argument("--seg-ckpt", default="runs/seg-unet/seg.pt")
    ap.add_argument("--seg-thresh", type=float, default=0.5)
    ap.add_argument("--brightness", type=float, default=1.0,
                    help="光照强度缩放（模拟换灯/曝光），1.0=原始")
    args = ap.parse_args()

    cam = CameraModel.load_or_none(args.camera)
    if cam is None:
        raise SystemExit(f"缺少 {CALIB_DIR}/camera_{args.camera}.json，先跑 scripts/fit_camera_model.py")

    env = gym.make("FruitSortStream-v1", obs_mode="rgbd", control_mode="pd_ee_pose",
                   sim_backend="cpu", robot_uids="panda_wristcam", active_kinds=KINDS,
                   belt_speed_range=(0.08, 0.08), slots_per_kind=args.slots,
                   gap_range=(1.2, 1.6))
    u = env.unwrapped
    old = RGBDPerception(u, dt=1 / 30)
    _seg = None
    if args.seg == "learned":
        from fruitsort.perception.learned_seg import LearnedSegmenter
        _seg = LearnedSegmenter(args.seg_ckpt, thresh=args.seg_thresh)
        print(f"分割后端: learned ({args.seg_ckpt})")
    new = PerceptionPipeline(camera=cam, belt_z=float(u.belt_top_z), dt=1 / 30,
                             segmenter=_seg,
                             depth_repair=not args.no_repair,
                             mask_fill=not args.no_repair)
    rng = np.random.default_rng(args.noise_seed)

    def degrade(depth_m):
        d = depth_m.copy()
        if args.noise_mm > 0:
            d = d + rng.normal(0, args.noise_mm / 1000.0, d.shape)
        if args.dropout > 0:
            bad = rng.random(d.shape) < args.dropout
            d[bad] = np.nan
        return d

    rows_old, rows_new = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=70000 + ep)
        old.tracks.clear(); new.tracker.tracks.clear()
        for t in range(args.frames + 25):
            obs, *_ = env.step(np.zeros(7))
            if t < 25:
                continue
            if args.brightness != 1.0:
                _r = obs["sensor_data"]["overhead"]["rgb"][0].cpu().numpy().astype(np.float32)
                _r = np.clip(_r * args.brightness, 0, 255).astype(np.uint8)
                obs["sensor_data"]["overhead"]["rgb"] = torch.as_tensor(
                    _r[None], dtype=torch.uint8)
            if args.dropout > 0 or args.noise_mm > 0:
                _d = obs["sensor_data"]["overhead"]["depth"][0].cpu().numpy()
                _d = (_d[..., 0] if _d.ndim == 3 else _d).astype(np.float64) / 1000.0
                _d = degrade(_d)
                obs["sensor_data"]["overhead"]["depth"] = torch.as_tensor(
                    (np.nan_to_num(_d, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0)[None, ..., None],
                    dtype=torch.float32)
            gts = _gt(u)
            if not gts:
                continue
            rows_old.append(score(gts, _flatten(old.update(obs)), args.match_m))
            rows_new.append(score(gts, _flatten(new.update(obs)), args.match_m))

    so = summarize(rows_old, "v1 (HSV+单应)")
    sn = summarize(rows_new, "v2 (掩码+反投影)")
    print(f"\n机位标定 {args.camera}  帧 {len(rows_old)}  真值件均 {np.mean([r['n_gt'] for r in rows_old]):.1f}")
    hdr = (f"{'后端':22s} {'召回':>6s} {'分类准':>7s} {'误检/帧':>8s} "
           f"{'3D':>7s} {'xy':>7s} {'z':>7s} {'z_p95':>7s}")
    print(hdr); print("-" * len(hdr))
    for s in (so, sn):
        def f(x):
            return "-" if x is None else f"{x:.2f}"
        print(f"{s['name']:22s} {s['recall']*100:5.1f}% {s['cls_acc']*100:6.1f}% "
              f"{s['fp_per_frame']:8.2f} {f(s['err_mean_cm']):>7s} {f(s['xy_mean_cm']):>7s} "
              f"{f(s['z_mean_cm']):>7s} {f(s['z_p95_cm']):>7s}")

    if args.out:
        p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(camera=args.camera, v1=so, v2=sn), ensure_ascii=False, indent=2),
                     encoding="utf-8")
        print(f"写入 {p}")


if __name__ == "__main__":
    main()
