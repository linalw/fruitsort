#!/usr/bin/env python
"""训练"何时闭爪"分类头（训练侧显式监督）。

动机（docs/throughput.md §8–9）：下探段的抖动是模型逐 chunk 重决策的固有节奏，
runner 侧 11 种门/去抖/时机修法全部无收益。这里改为**从专家示教里直接学"该闭爪的时刻"**：

- 标签：第 t 帧"张开"、且 t+1..t+k 内出现"张开→闭合"（k=2 帧 = 0.1 s @20 Hz）→ 正样本；
- 特征：与策略相同的 49 维观测（`build_obs`，感知替身注入后一致）；
- 模型：49→128→64→1 的 MLP + 标准化，`pos_weight` 处理类不平衡；
- 推理：`--close-head` 打开后，在 `phase==1 且未夹住且未闭合` 时若 p≥阈值就**补一个闭爪**
  （只加不抑制，和 `--close-timing` 的接入口一致，但判据来自示教而不是几何）。

用法:
    python scripts/train_close_head.py --out runs/close-head --epochs 6
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent


def collect(data_dirs, k: int = 2):
    X, Y = [], []
    for d in data_dirs:
        for f in sorted(glob.glob(str(Path(d) / "ep_*.npz"))):
            z = np.load(f, allow_pickle=True)
            obs = z["obs"].astype(np.float32)
            a = z["action"]
            if obs.ndim != 2 or obs.shape[1] != 49:
                continue
            closed = a[:, 3] < 0
            # 只在"张开"帧打标签
            fut = np.zeros(len(closed), dtype=bool)
            for j in range(1, k + 1):
                fut[:-j] |= closed[j:]
            open_now = ~closed
            y = (open_now & fut).astype(np.float32)
            X.append(obs[open_now])
            Y.append(y[open_now])
    return np.concatenate(X), np.concatenate(Y)


class Head(nn.Module):
    def __init__(self, d_in: int = 49):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="*", default=None,
                    help="数据集目录；缺省用 data/datasets 下所有含闭爪的目录")
    ap.add_argument("--out", default="runs/close-head")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k", type=int, default=2, help="提前 k 帧作为正样本（20 Hz）")
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    if args.data:
        dirs = args.data
    else:
        dirs = sorted(p for p in glob.glob(str(ROOT / "data/datasets/*"))
                      if glob.glob(str(Path(p) / "ep_0000.npz")))
    t0 = time.time()
    X, Y = collect(dirs, args.k)
    n, d = X.shape
    print(f"样本 {n} 帧 / {d} 维，正样本 {int(Y.sum())} ({Y.mean()*100:.2f}%)，"
          f"目录 {len(dirs)}，耗时 {time.time()-t0:.1f}s", flush=True)

    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    nv = int(n * args.val_frac)
    vi, ti = idx[:nv], idx[nv:]
    mu, sd = X[ti].mean(0), X[ti].std(0) + 1e-6

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Head(d).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos_w = torch.tensor((1 - Y[ti].mean()) / max(Y[ti].mean(), 1e-6), device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    Xt = torch.from_numpy((X - mu) / sd).to(dev)
    Yt = torch.from_numpy(Y).to(dev)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(ti), device=dev)
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            # 注意：perm 是"训练集内部"的下标，映射回 ti
            sel = torch.from_numpy(ti[b.cpu().numpy()]).to(dev)
            opt.zero_grad()
            out = model(Xt[sel])
            loss = lossf(out, Yt[sel])
            loss.backward()
            opt.step()
            tot += float(loss) * len(sel)
        # 验证 AUC
        model.eval()
        with torch.no_grad():
            sel = torch.from_numpy(vi).to(dev)
            pv = torch.sigmoid(model(Xt[sel])).cpu().numpy()
        yv = Y[vi]
        order = np.argsort(pv)
        ranks = np.empty(len(pv)); ranks[order] = np.arange(1, len(pv) + 1)
        npos = yv.sum(); nneg = len(yv) - npos
        auc = ((ranks[yv == 1].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1))
        print(f"epoch {ep+1}/{args.epochs} loss {tot/max(len(ti),1):.4f} "
              f"valAUC {auc:.3f} (n={nv}, pos={int(npos)})", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "mu": mu, "sd": sd, "d_in": d,
                "k": args.k}, out / "close.pt")
    json.dump({"dirs": dirs, "samples": int(n), "pos": int(Y.sum()),
               "val_auc": float(auc), "k": args.k},
              open(out / "metrics.json", "w"), ensure_ascii=False, indent=2)
    print("saved", out / "close.pt", flush=True)


if __name__ == "__main__":
    main()
