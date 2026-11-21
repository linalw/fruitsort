#!/usr/bin/env python
"""过滤 DAgger 数据：只保留"目标水果仍在可抓窗口内"的帧。

为什么
------
连续流 DAgger 里，模型会一路悬停到水果越过可抓窗口（x>0.3），此时规则专家仍给出
"去拦截点"的绝对目标（方向朝后/朝上、幅度大）——这类样本占了数据大头，
微调后不仅没改善连续作业，还把单发能力打崩（docs 6.26.6）。

过滤规则
--------
`build_obs` 的布局：[tcp 7][qpos 9][belt 1][4 × (pos 3 + vel 3 + active + held)]，
目标类别由 npz 里的 `instruction` 指定。只保留目标位置 x ∈ [x_min, x_max] 的帧，
并把连续保留的帧切成 ≥`min_len` 的子段（保持时序/相位语义）。

用法:
    python scripts/filter_dagger.py --src dagger-a dagger-b --dst dagger-f --x-min -0.62 --x-max 0.22
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KINDS = ("apple", "strawberry", "banana", "pear")


def fruit_x(obs_row: np.ndarray, kind: str) -> float:
    base = 17 + 8 * KINDS.index(kind)
    return float(obs_row[base])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="源数据集目录（可多个）")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--x-min", type=float, default=-0.62)
    ap.add_argument("--x-max", type=float, default=0.22)
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--clip-step", type=float, default=0.06,
                    help="标签增量裁剪到部署步长（-1 = 不裁剪）。DAgger 里专家目标"
                         "可能距当前 TCP 0.4 m，直接当标签会让位置头输出不可执行的跳变")
    args = ap.parse_args()

    root = Path("/media/ubuntu/66A27065A2703B9D/data/fruitsort/datasets")
    dst = Path(args.dst)
    if not dst.is_absolute():
        dst = root / dst
    dst.mkdir(parents=True, exist_ok=True)

    n_in = n_out = 0
    frames_in = frames_out = 0
    for src in args.src:
        d = Path(src) if Path(src).is_absolute() else root / src
        for f in sorted(d.glob("*.npz")):
            z = np.load(f, allow_pickle=True)
            obs, act = z["obs"], z["action"]
            kind = str(z["instruction"])
            frames_in += len(obs)
            n_in += 1
            keep = np.array([args.x_min <= fruit_x(o, kind) <= args.x_max for o in obs])
            if args.clip_step and args.clip_step > 0:
                # 标签 = 当前 TCP + clip(专家目标 − 当前 TCP, ±step)，与部署一致
                tcp = obs[:, :3]
                d = act[:, :3] - tcp
                act = act.copy()
                act[:, :3] = tcp + np.clip(d, -args.clip_step, args.clip_step)
            if not keep.any():
                continue
            # 切成连续段
            idx = np.flatnonzero(keep)
            splits = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
            for seg in splits:
                if len(seg) < args.min_len:
                    continue
                np.savez_compressed(
                    dst / f"ep_{n_out:04d}.npz",
                    obs=obs[seg], action=act[seg], seed=z["seed"],
                    kinds=z["kinds"], instruction=z["instruction"],
                    overhead_jpg=z["overhead_jpg"][seg],
                    wrist_jpg=z["wrist_jpg"][seg])
                n_out += 1
                frames_out += len(seg)
    print(json.dumps({"src_segments": n_in, "src_frames": frames_in,
                      "kept_segments": n_out, "kept_frames": frames_out,
                      "dst": str(dst)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
