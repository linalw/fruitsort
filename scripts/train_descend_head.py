#!/usr/bin/env python
"""在**策略自己到达的状态**上学习"继续下降 vs 现在闭爪"（DAgger 式重标注）。

与 `train_close_head.py` 的区别（那是 v1.121，离线 AUC 0.998 但闭环更差）：

- 数据来自**策略自跑**的状态（`stream_eval.py --dump-states`），不是专家示教 ——
  因此包含"悬停"这种专家从没到过的状态；
- 标签由专家规则在该状态上给出（专家只看几何：够低 + 够准 → 闭；够准但还高 → 继续降）；
- 输出是**互斥两分类**：`descend` / `close`。推理时 `descend` 会**同时**把竖直目标压到果心
  并强制张开（不再被模型的抖动闭爪打断），`close` 才闭爪。

用法：
    python scripts/train_descend_head.py --states /tmp/opencode/states/*.npz --out runs/descend-head
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

# 专家几何门（与 runner 默认一致：close_z_gate=0.012、close_commit_xy=0.03）
Z_CLOSE = 0.012
XY_TOL = 0.030
XY_FAR = 0.060


def load_states(patterns):
    X, Y = [], []
    files = []
    for pat in patterns:
        files += sorted(glob.glob(pat))
    for f in files:
        d = np.load(f, allow_pickle=False)
        _o = d["obs"].astype(np.float32)
        if "tac" in d.files:                      # v1.124 起转储带触觉 → 并联成 58 维
            _t = d["tac"].astype(np.float32)
            if _t.shape[0] == _o.shape[0]:
                _o = np.concatenate([_o, _t], axis=1)
        X.append(_o)
        dz = d["dz"].astype(np.float32)
        dxy = d["dxy"].astype(np.float32)
        ok = dxy <= XY_FAR
        close = ok & (dz <= Z_CLOSE) & (dz >= -0.005)
        descend = ok & (dz > Z_CLOSE)
        y = np.where(close, 1, np.where(descend, 0, -1))
        Y.append(y.astype(np.int8))
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    keep = Y >= 0
    return X[keep], Y[keep], files


class Head(nn.Module):
    def __init__(self, d_in: int = 49):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", nargs="+", required=True)
    ap.add_argument("--out", default="runs/descend-head")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    args = ap.parse_args()

    X, Y, files = load_states(args.states)
    n, d = X.shape
    print(f"策略状态 {n} 帧（{len(files)} 个文件）：descend {int((Y == 0).sum())} / "
          f"close {int((Y == 1).sum())}", flush=True)

    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    nv = int(n * args.val_frac)
    vi, ti = idx[:nv], idx[nv:]
    mu, sd = X[ti].mean(0), X[ti].std(0) + 1e-6

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Head(d).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    cnt = np.bincount(Y[ti], minlength=2).astype(np.float32)
    w = torch.tensor(cnt.sum() / (2 * np.maximum(cnt, 1)), device=dev)
    lossf = nn.CrossEntropyLoss(weight=w)

    Xt = torch.from_numpy((X - mu) / sd).to(dev)
    Yt = torch.from_numpy(Y.astype(np.int64)).to(dev)
    best = -1.0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(ti), device=dev)
        for i in range(0, len(perm), args.batch):
            sel = torch.from_numpy(ti[perm[i:i + args.batch].cpu().numpy()]).to(dev)
            opt.zero_grad()
            loss = lossf(model(Xt[sel]), Yt[sel])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            sel = torch.from_numpy(vi).to(dev)
            pred = model(Xt[sel]).argmax(-1).cpu().numpy()
        acc = float((pred == Y[vi]).mean())
        if ep % 10 == 9 or ep == args.epochs - 1:
            print(f"epoch {ep+1}/{args.epochs} val_acc {acc:.4f}", flush=True)
        best = max(best, acc)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "mu": mu, "sd": sd, "d_in": d,
                "z_close": Z_CLOSE, "xy_tol": XY_TOL}, out / "descend.pt")
    json.dump({"files": files, "samples": int(n),
               "descend": int((Y == 0).sum()), "close": int((Y == 1).sum()),
               "val_acc": best}, open(out / "metrics.json", "w"),
              ensure_ascii=False, indent=2)
    print("saved", out / "descend.pt", flush=True)


if __name__ == "__main__":
    main()
