#!/usr/bin/env python
"""训练行为克隆基线（MLP + 动作分块）。

用法:
    python scripts/train_bc.py --data data/datasets/bc-pair --out runs/bc-pair --epochs 60
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fruitsort.learning.bc import ChunkBC, Normalizer, build_windows, load_episodes

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, nargs="+", help="一个或多个数据目录")
    ap.add_argument("--out", default="")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--hist", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_dirs = []
    for d in args.data:
        p = Path(d)
        data_dirs.append(p if p.is_absolute() else ROOT / p)
    episodes = []
    for d in data_dirs:
        episodes += load_episodes(d)
    n_val = max(1, int(len(episodes) * args.val_frac))
    train_eps, val_eps = episodes[:-n_val], episodes[-n_val:]

    obs_tr, act_tr = build_windows(train_eps, args.chunk, args.hist)
    obs_va, act_va = build_windows(val_eps, args.chunk, args.hist)
    norm = Normalizer(obs_tr, act_tr.reshape(-1, act_tr.shape[-1]))

    x_tr = torch.from_numpy(norm.obs(obs_tr)).float()
    y_tr = torch.from_numpy(norm.act(act_tr)).float()
    x_va = torch.from_numpy(norm.obs(obs_va)).float()
    y_va = torch.from_numpy(norm.act(act_va)).float()

    model = ChunkBC(chunk=args.chunk, hidden=args.hidden, hist=args.hist)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = torch.nn.MSELoss()
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=args.batch, shuffle=True, drop_last=True)

    tag = time.strftime("%Y-%m-%d-%H%M")
    out = Path(args.out) if args.out else ROOT / "runs" / f"{tag}-bc"
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    for epoch in range(args.epochs):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(xb); n += len(xb)
        sched.step()
        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(x_va), y_va))
        history.append({"epoch": epoch, "train": tot / max(n, 1), "val": val})
        if val < best:
            best = val
            torch.save({"model": model.state_dict(), "normalizer": norm.state_dict(),
                        "chunk": args.chunk, "hidden": args.hidden, "hist": args.hist,
                        "obs_dim": 49, "act_dim": 7, "val_loss": val,
                        "episodes": len(train_eps)}, out / "ckpt.pt")
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d} train {tot / max(n, 1):.4f} val {val:.4f}")

    (out / "metrics.json").write_text(json.dumps({
        "data": [str(d) for d in data_dirs], "train_episodes": len(train_eps), "val_episodes": len(val_eps),
        "samples": len(obs_tr), "best_val": best, "chunk": args.chunk,
        "history": history,
    }, indent=2), encoding="utf-8")
    print(f"\n✓ 最佳 val loss {best:.4f}，模型: {out / 'ckpt.pt'}")


if __name__ == "__main__":
    main()
