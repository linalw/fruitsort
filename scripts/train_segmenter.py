#!/usr/bin/env python
"""训练学习式分割后端（U-Net 精简版）：RGB+高度 → 掩码。

标签来源（重要，别误读）
------------------------
仿真里没有人工分割标注，这里用**经典分割的输出作教师**（蒸馏）。它证明的是
"学习式后端可以替换经典后端、并能通过增强获得光照鲁棒性"，**不是**"比经典更准"。
真机应换成几十帧人工标注 + 同一脚本。

用法:
    python scripts/train_segmenter.py --frames 60 --out runs/seg-unet/seg.pt
    python scripts/bench_perception.py --seg learned --seg-ckpt runs/seg-unet/seg.pt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

KINDS = ("apple", "strawberry", "banana", "pear")


def _augment(rgb: np.ndarray, h: np.ndarray, rng: np.random.Generator):
    """光度增强：真机换灯/曝光/色温时，阈值法会崩，增强后的 CNN 不会。"""
    b = float(rng.uniform(0.55, 1.45))          # 亮度
    c = float(rng.uniform(0.80, 1.20))          # 对比度
    g = float(rng.uniform(0.75, 1.35))          # gamma
    out = np.clip(rgb.astype(np.float32) * b, 0, 255)
    mean = out.mean()
    out = np.clip((out - mean) * c + mean, 0, 255) / 255.0
    out = np.power(out, 1.0 / g) * 255.0
    if rng.random() < 0.5:                       # 轻微色温偏移（R/B 通道增益）
        out[..., 0] = np.clip(out[..., 0] * float(rng.uniform(0.85, 1.15)), 0, 255)
        out[..., 2] = np.clip(out[..., 2] * float(rng.uniform(0.85, 1.15)), 0, 255)
    hh = h + rng.normal(0, 0.004, h.shape).astype(np.float32)
    return out.astype(np.uint8), hh


def collect(frames: int, seeds: list[int], camera: str, down: int) -> tuple[np.ndarray, np.ndarray]:
    import gymnasium as gym

    import fruitsort.sim.tasks  # noqa: F401
    from fruitsort.perception.camera import CameraModel
    from fruitsort.perception.segmentation import DepthColorSegmenter

    cam = CameraModel.load_or_none(camera)
    if cam is None:
        raise SystemExit(f"缺少 camera_{camera}.json")
    seg = DepthColorSegmenter()
    X, Y = [], []
    env = gym.make("FruitSortStream-v1", obs_mode="rgbd", control_mode="pd_ee_pose",
                   sim_backend="cpu", robot_uids="panda_wristcam", active_kinds=KINDS,
                   belt_speed_range=(0.05, 0.12), slots_per_kind=2, gap_range=(0.8, 2.4))
    u = env.unwrapped
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        for t in range(frames + 20):
            obs, *_ = env.step(np.zeros(7))
            if t < 20:
                continue
            sd = obs["sensor_data"]["overhead"]
            rgb = sd["rgb"][0].cpu().numpy()
            d = sd["depth"][0].cpu().numpy()
            d = (d[..., 0] if d.ndim == 3 else d).astype(np.float64) / 1000.0
            pts = cam.unproject(d)
            masks = seg.segment(rgb, pts, float(u.belt_top_z))
            if not masks:
                continue
            lab = np.zeros(rgb.shape[:2], np.uint8)
            for m in masks:
                lab[m.mask] = 1
            h = np.clip((pts[..., 2] - float(u.belt_top_z)) / 0.20, 0, 1)
            h = np.nan_to_num(h, nan=0.0).astype(np.float32)
            if down > 1:
                small = cv2.resize(rgb, (rgb.shape[1] // down, rgb.shape[0] // down),
                                   interpolation=cv2.INTER_AREA)
                hh = cv2.resize(h, (h.shape[1] // down, h.shape[0] // down),
                                interpolation=cv2.INTER_AREA)
                lab = cv2.resize(lab, (lab.shape[1] // down, lab.shape[0] // down),
                                 interpolation=cv2.INTER_NEAREST)
            else:
                small, hh = rgb, h
            X.append(np.dstack([small.astype(np.float16), hh.astype(np.float16)]))
            Y.append(lab)
    env.close()
    return np.stack(X), np.stack(Y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--seeds", default="81001,81002,81003")
    ap.add_argument("--camera", default="overhead")
    ap.add_argument("--down", type=int, default=2)
    ap.add_argument("--w", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patch", type=int, default=128, help="训练 patch 边长（推理仍是全帧卷积）")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default="runs/seg-unet/seg.pt")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()
    X, Y = collect(args.frames, seeds, args.camera, args.down)
    print(f"样本 {len(X)}  形状 {X.shape}  正像素占比 {Y.mean()*100:.2f}%  "
          f"采集 {(time.time()-t0)/60:.1f} min")
    n = len(X)
    idx = np.random.default_rng(0).permutation(n)
    n_tr = int(0.8 * n)
    tr, va = idx[:n_tr], idx[n_tr:]

    import torch
    import torch.nn as nn
    from fruitsort.perception.learned_seg import _UNetSmall

    torch.manual_seed(0)
    net = _UNetSmall.build(args.w)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    # 正像素只有 0.13%：BCE 会退化成"全背景"。用 pos_weight + Dice 双管齐下。
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([120.0], device=dev))

    def dice_loss(logit, target, eps=1.0):
        p = torch.sigmoid(logit)
        num = 2.0 * (p * target).sum(dim=(1, 2, 3)) + eps
        den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
        return 1.0 - (num / den).mean()

    def to_t(i):
        # 注意：**只有 RGB 除 255**，高度通道本来就是 0–1（之前整块除 255 → 验证集输入全错，
        # 全帧 IoU 恒为 0，看起来像"模型崩了"，其实只是指标算错）
        x = X[i].astype(np.float32).transpose(2, 0, 1)
        x[:3] /= 255.0
        return torch.from_numpy(x).to(dev), torch.from_numpy(Y[i].astype(np.float32))[None].to(dev)

    rng = np.random.default_rng(1)
    best = (-1.0, None)
    P = args.patch

    def sample_patch(i):
        """以目标为中心裁 P×P；20% 概率裁随机背景（教它别乱报）。"""
        rgb = X[i][..., :3].astype(np.float32) / 255.0
        h = X[i][..., 3].astype(np.float32)
        lab = Y[i]
        H, W = lab.shape
        ys, xs = np.nonzero(lab)
        if len(ys) and rng.random() < 0.75:
            # 关键：物体在 patch 里的位置**随机**（之前总是居中 → 网络学出"只看中心"的
            # 先验，全帧推理时 IoU 直接掉到 0）
            k = rng.integers(len(ys))
            cy, cx = int(ys[k]), int(xs[k])
            y0 = cy - int(rng.integers(0, P))
            x0 = cx - int(rng.integers(0, P))
        else:
            y0 = int(rng.integers(0, max(1, H - P)))
            x0 = int(rng.integers(0, max(1, W - P)))
        y0 = int(np.clip(y0, 0, max(0, H - P)))
        x0 = int(np.clip(x0, 0, max(0, W - P)))
        return (np.dstack([rgb[y0:y0 + P, x0:x0 + P], h[y0:y0 + P, x0:x0 + P]]).transpose(2, 0, 1),
                lab[y0:y0 + P, x0:x0 + P].astype(np.float32)[None])

    for ep in range(args.epochs):
        net.train()
        perm = rng.permutation(tr)
        tot = 0.0
        for b in range(0, len(perm), args.batch):
            xs, ys = [], []
            for i in perm[b:b + args.batch]:
                xp, yp = sample_patch(int(i))
                xs.append(xp); ys.append(yp)
            x = torch.from_numpy(np.stack(xs)).to(dev)
            y = torch.from_numpy(np.stack(ys)).to(dev)
            opt.zero_grad()
            logit = net(x)
            loss = bce(logit, y) + dice_loss(logit, y)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(xs)
        if (ep + 1) % 2 == 0 or ep == 0:
            net.eval()
            ious = []
            with torch.no_grad():
                for i in va[:6]:
                    x, y = to_t(i)
                    p = (torch.sigmoid(net(x[None]))[0, 0] > 0.5).float()
                    inter = float((p * y).sum()); uni = float(((p + y) > 0).sum())
                    ious.append(inter / max(uni, 1.0))
            _iou = float(np.mean(ious))
            print(f"  epoch {ep+1:2d}  loss {tot/len(perm):.3f}  全帧 val IoU {_iou*100:.1f}%")
            if _iou > best[0]:
                best = (_iou, {k: v.detach().clone() for k, v in net.state_dict().items()})

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    state = best[1] if best[1] is not None else net.state_dict()
    torch.save(dict(model=state, w=args.w, down=args.down, camera=args.camera, patch=P,
                    val_iou=best[0]), out)
    print(f"保存到 {out}  best val IoU {best[0]*100:.1f}%  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
