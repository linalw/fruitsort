#!/usr/bin/env python
"""诊断：同一批帧上，经典分割 vs 学习式分割 的掩码差在哪。

为什么要这个
------------
学习式后端在静态 bench 上"100% 召回 / 100% 分类 / 3D 0.21 cm"，但闭环是 0/1、0/2
（经典 3/4、3/4）。闭环里多出来的东西只有两样：
  1. **手臂遮挡**（bench 的帧里机械臂基本不在果子上方）；
  2. 执行层两道门吃的是 **extent / height / center**（分类阈值 45/115 mm、
     闭爪高度门 1.2 cm、near-hold 的 0.03–0.11 m）。
所以这里逐帧对比这两个量，并按"是否在手臂下方"分组统计。

用法:
    python scripts/diag_seg_backends.py --seconds 40 --seeds 60002 --out runs/diag
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

import fruitsort.sim.tasks  # noqa: F401
from fruitsort.perception.camera import CameraModel
from fruitsort.perception.learned_seg import LearnedSegmenter
from fruitsort.perception.pipeline import PerceptionPipeline
from fruitsort.perception.segmentation import DepthColorSegmenter
from fruitsort.policies.scripted import ScriptedSorter

KINDS = ("apple", "strawberry", "banana", "pear")


def _dets(pipe: PerceptionPipeline) -> list[dict]:
    return [d for d in pipe.last_debug]


def _match(a: list[dict], b: list[dict], max_m: float = 0.05):
    """按中心最近邻配对（同一件在两个后端的掩码）。"""
    pairs, used = [], set()
    for i, da in enumerate(a):
        best, bd = None, 1e9
        for j, db in enumerate(b):
            if j in used:
                continue
            d = float(np.hypot(da["center"][0] - db["center"][0],
                               da["center"][1] - db["center"][1]))
            if d < bd:
                best, bd = j, d
        if best is not None and bd <= max_m:
            used.add(best)
            pairs.append((da, b[best], bd))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="overhead")
    ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--seeds", default="60002")
    ap.add_argument("--seg-thresh", type=float, default=0.75)
    ap.add_argument("--seg-dilate", type=int, default=0)
    ap.add_argument("--backend", default="learned", choices=["learned", "hybrid"])
    ap.add_argument("--seg-ckpt", default="runs/seg-unet/seg.pt")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cam = CameraModel.load_or_none(args.camera)
    if cam is None:
        raise SystemExit(f"缺少 camera_{args.camera}.json")
    env = gym.make("FruitSortStream-v1", obs_mode="rgbd", control_mode="pd_ee_pose",
                   sim_backend="cpu", robot_uids="panda_wristcam", active_kinds=KINDS,
                   belt_speed_range=(0.08, 0.08), slots_per_kind=2, gap_range=(1.2, 2.0))
    u = env.unwrapped
    cls = PerceptionPipeline(camera=cam, belt_z=float(u.belt_top_z), dt=1 / 30,
                             segmenter=DepthColorSegmenter())
    if args.backend == "hybrid":
        from fruitsort.perception.learned_seg import HybridSegmenter
        lrn_seg = HybridSegmenter(args.seg_ckpt, thresh=args.seg_thresh,
                                  dilate=args.seg_dilate)
    else:
        lrn_seg = LearnedSegmenter(args.seg_ckpt, thresh=args.seg_thresh,
                                   dilate=args.seg_dilate)
    lrn = PerceptionPipeline(camera=cam, belt_z=float(u.belt_top_z), dt=1 / 30,
                             segmenter=lrn_seg)
    _dev = getattr(getattr(lrn_seg, "learned", lrn_seg), "device", "?")
    print(f"backend={args.backend} thresh={args.seg_thresh} device={_dev}")

    rows = []
    n_cls = n_lrn = 0
    for seed in [int(s) for s in args.seeds.split(",")]:
        obs, _ = env.reset(seed=seed)
        cls.tracker.tracks.clear()
        lrn.tracker.tracks.clear()
        sorter = ScriptedSorter(env)
        n = int(args.seconds * 30)
        for t in range(n):
            act = sorter.act()
            obs, *_ = env.step(act)
            d_cls = cls.update(obs)
            d_lrn = lrn.update(obs)
            tcp = u.agent.tcp_pose.p[0].cpu().numpy()
            a = _dets(cls)
            b = _dets(lrn)
            n_cls += len(a); n_lrn += len(b)
            for da, db, dist in _match(a, b):
                under_arm = float(np.hypot(da["center"][0] - tcp[0],
                                          da["center"][1] - tcp[1])) < 0.16
                rows.append(dict(seed=seed, t=t, under_arm=under_arm,
                                 kind_c=da["kind"], kind_l=db["kind"],
                                 ext_c=da["extent"], ext_l=db["extent"],
                                 wid_c=da["width"], wid_l=db["width"],
                                 h_c=da["height"], h_l=db["height"],
                                 z_c=da["center"][2], z_l=db["center"][2],
                                 area_c=da["area"], area_l=db["area"],
                                 xy_c=da["center"][:2], xy_l=db["center"][:2]))
    env.close()

    if not rows:
        raise SystemExit("没有配对样本")
    R = rows
    ext_d = np.array([r["ext_l"] - r["ext_c"] for r in R]) * 1000
    h_d = np.array([r["h_l"] - r["h_c"] for r in R]) * 1000
    z_d = np.array([r["z_l"] - r["z_c"] for r in R]) * 1000
    xy_d = np.array([np.linalg.norm(r["xy_l"] - r["xy_c"]) for r in R]) * 1000
    agree = np.mean([r["kind_c"] == r["kind_l"] for r in R])
    ratio = np.array([r["area_l"] / max(r["area_c"], 1) for r in R])

    print(f"\n经典检出 {n_cls}  学习式检出 {n_lrn}  配对 {len(R)} 对（{len(set(r['seed'] for r in R))} seeds）"
          f"  手臂下方 {sum(r['under_arm'] for r in R)}")
    print(f"  学习式相对经典的召回 {len(R)/max(n_cls,1)*100:5.1f}%   "
          f"误报 {max(0, n_lrn - len(R))}")
    print(f"  分类一致率 {agree*100:5.1f}%   面积比 中位 {np.median(ratio):.2f}")
    for name, arr in (("Δ长轴", ext_d), ("Δ高度", h_d), ("Δ中心z", z_d), ("Δ中心xy", xy_d)):
        print(f"  {name:8s} 中位 {np.median(arr):+6.2f} mm  "
              f"P5/P95 {np.percentile(arr,5):+6.2f}/{np.percentile(arr,95):+6.2f} mm")

    # 阈值敏感度：有多少件因为 Δ 跨过了分类阈值 / 闭爪门
    thr_flip = 0
    for r in R:
        for thr in (0.045, 0.115):
            if (r["ext_c"] > thr) != (r["ext_l"] > thr):
                thr_flip += 1
                break
    gate_flip = int(np.sum(np.abs(z_d) > 12.0))
    for ua in (False, True):
        sub = [r for r in R if r["under_arm"] == ua]
        if not sub:
            continue
        ag = np.mean([r["kind_c"] == r["kind_l"] for r in sub])
        ar = np.median([r["area_l"] / max(r["area_c"], 1) for r in sub])
        ex = np.median([(r["ext_l"] - r["ext_c"]) * 1000 for r in sub])
        print(f"  under_arm={int(ua):d}: n={len(sub):4d} 分类一致 {ag*100:5.1f}% "
              f"面积比 {ar:.2f} Δ长轴 {ex:+5.1f} mm")
    print(f"  跨过分类阈值(45/115mm) 的件: {thr_flip}/{len(R)} = {thr_flip/len(R)*100:.1f}%")
    print(f"  Δ中心z > 12 mm（会动到闭爪门）: {gate_flip}/{len(R)} = {gate_flip/len(R)*100:.1f}%")

    if args.out:
        p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
        summary = dict(n=len(R), agree=float(agree),
                       ext_mm=[float(np.median(ext_d)), float(np.percentile(ext_d, 5)),
                               float(np.percentile(ext_d, 95))],
                       h_mm=[float(np.median(h_d)), float(np.percentile(h_d, 5)),
                             float(np.percentile(h_d, 95))],
                       z_mm=[float(np.median(z_d)), float(np.percentile(z_d, 5)),
                             float(np.percentile(z_d, 95))],
                       thr_flip=thr_flip, gate_flip=gate_flip,
                       area_ratio=float(np.median(ratio)),
                       under_arm=int(sum(r["under_arm"] for r in R)))
        p.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        R_json = [{k: (list(v) if isinstance(v, np.ndarray) else v) for k, v in r.items()} for r in R]
        (p.parent / "pairs.json").write_text(json.dumps(R_json, ensure_ascii=False), encoding="utf-8")
        print(f"写入 {p}")


if __name__ == "__main__":
    main()
