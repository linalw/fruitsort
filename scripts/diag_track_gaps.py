#!/usr/bin/env python
"""轨迹连续性诊断：目标"闪断"多久？

为什么不用检出率
----------------
ByteTrack 式的两阶段关联**不增加检出**，它只是用低分候选把已有轨迹续上。
所以要看的是"**每个真值件在每一帧是否还有活着的轨迹**"，而不是"这一帧有没有检出"。

用法:
    python scripts/diag_track_gaps.py --seconds 40 --seeds 60002,60003
"""
from __future__ import annotations

import argparse
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


def _mk(cam, u, seg, two_stage):
    p = PerceptionPipeline(camera=cam, belt_z=float(u.belt_top_z), segmenter=seg,
                           two_stage=two_stage)
    p.tcp_provider = lambda: u.agent.tcp_pose.p[0].cpu().numpy()[:2]
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="overhead")
    ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--seeds", default="60002")
    ap.add_argument("--seg-thresh", type=float, default=0.60)
    ap.add_argument("--low", type=float, default=0.30)
    ap.add_argument("--seg-ckpt", default="runs/seg-unet/seg.pt")
    args = ap.parse_args()

    cam = CameraModel.load_or_none(args.camera)
    if cam is None:
        raise SystemExit(f"缺少 camera_{args.camera}.json")
    env = gym.make("FruitSortStream-v1", obs_mode="rgbd", control_mode="pd_ee_pose",
                   sim_backend="cpu", robot_uids="panda_wristcam", active_kinds=KINDS,
                   belt_speed_range=(0.08, 0.08), slots_per_kind=2, gap_range=(1.2, 2.0))
    u = env.unwrapped
    cls = _mk(cam, u, DepthColorSegmenter(), two_stage=False)
    lrn_1 = _mk(cam, u, LearnedSegmenter(args.seg_ckpt, thresh=args.seg_thresh, dilate=0), False)
    lrn_2 = _mk(cam, u, LearnedSegmenter(args.seg_ckpt, thresh=args.seg_thresh, dilate=0), True)
    lrn_2.low_score = args.low

    names = ["classic", "learned(1阶段)", f"learned(2阶段 low={args.low})"]
    pipes = [cls, lrn_1, lrn_2]
    stat = {n: dict(seen=0, covered=0, gaps=[], cur=0) for n in names}

    for seed in [int(s) for s in args.seeds.split(",")]:
        obs, _ = env.reset(seed=seed)
        for p in pipes:
            p.tracker.tracks.clear()
        sorter = ScriptedSorter(env)
        for _ in range(int(args.seconds * 30)):
            obs, *_ = env.step(sorter.act())
            gts = []
            for k in u.active_kinds:
                for sl in u._slots[k]:
                    if not sl["live"]:
                        continue
                    q = sl["item"]["actor"].pose.p[0].cpu().numpy()
                    if -1.30 <= q[0] <= 0.55 and abs(q[1]) < 0.25 and q[2] > 0.40:
                        gts.append(np.asarray(q, dtype=np.float64))
            if not gts:
                continue
            for n, p in zip(names, pipes):
                p.update(obs)
                trk = [t for t in p.tracker.tracks if t.lost <= 2]
                for g in gts:
                    st = stat[n]
                    st["seen"] += 1
                    ok = any(float(np.linalg.norm(t.pos[:2] - g[:2])) < 0.05 for t in trk)
                    if ok:
                        st["covered"] += 1
                        if st["cur"]:
                            st["gaps"].append(st["cur"])
                        st["cur"] = 0
                    else:
                        st["cur"] += 1
    env.close()

    print(f"\n帧×件 样本按后端分组（每类相同），seeds={args.seeds}")
    for n in names:
        st = stat[n]
        gaps = np.array(st["gaps"]) if st["gaps"] else np.array([0])
        print(f"  {n:26s} 轨迹覆盖 {st['covered']/max(st['seen'],1)*100:5.1f}%   "
              f"断档次数 {len(st['gaps']):4d}   断档中位 {np.median(gaps):4.1f} 帧   "
              f"P90 {np.percentile(gaps, 90):4.1f} 帧")


if __name__ == "__main__":
    main()
