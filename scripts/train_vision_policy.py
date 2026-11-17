#!/usr/bin/env python
"""训练多视角时序策略（v0.5）。

用法:
    python scripts/train_vision_policy.py --data data/datasets/vla-pair --epochs 30
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fruitsort.learning.bc import Normalizer
from fruitsort.learning.image_dataset import ImageDemoDataset
from fruitsort.models.vision_policy import MultiViewTemporalPolicy, PolicyConfig

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--out", default="")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hist", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--future", type=int, default=3)
    ap.add_argument("--future-weight", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--delay-prob", type=float, default=0.0, help="延迟增强概率（替换最后一帧）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_dirs = []
    for d in args.data:
        p = Path(d)
        data_dirs.append(p if p.is_absolute() else ROOT / p)

    ds = ImageDemoDataset(data_dirs, hist=args.hist, chunk=args.chunk, future=args.future,
                          delay_prob=args.delay_prob)
    n_val = max(1, int(len(ds) * 0.15))
    train_idx = list(range(0, len(ds) - n_val))
    val_idx = list(range(len(ds) - n_val, len(ds)))

    # 用训练集统计量做标准化（状态与动作）
    sample_idx = train_idx[:: max(1, len(train_idx) // 2000)]
    states = np.stack([ds[i]["state"].numpy() for i in sample_idx]).reshape(-1, 49)
    acts = np.stack([ds[i]["act"].numpy() for i in sample_idx]).reshape(-1, 4)[:, :3]
    fut = np.stack([ds[i]["future"].numpy() for i in sample_idx]).reshape(-1, 24)
    norm_state = Normalizer(states, states)
    norm_act = Normalizer(acts, acts)
    norm_fut = Normalizer(fut, fut)

    class Wrap(torch.utils.data.Dataset):
        def __init__(self, base, idx):
            self.base, self.idx = base, idx

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            s = self.base[self.idx[i]]
            s["state"] = torch.from_numpy(norm_state.obs(s["state"].numpy()).astype(np.float32))
            a = s["act"].numpy().copy()
            a[:, :3] = norm_act.act(a[:, :3])
            s["act"] = torch.from_numpy(a.astype(np.float32))
            s["future"] = torch.from_numpy(norm_fut.act(s["future"].numpy()).astype(np.float32))
            return s

    dl_tr = DataLoader(Wrap(ds, train_idx), batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0)
    dl_va = DataLoader(Wrap(ds, val_idx), batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, persistent_workers=args.workers > 0)

    cfg = PolicyConfig(hist=args.hist, chunk=args.chunk, future_steps=args.future)
    model = MultiViewTemporalPolicy(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    mse = torch.nn.MSELoss()

    tag = time.strftime("%Y-%m-%d-%H%M")
    out = Path(args.out) if args.out else ROOT / "runs" / f"{tag}-vpolicy"
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    hist_log = []
    for ep in range(args.epochs):
        model.train()
        tot, na = 0.0, 0
        for b in dl_tr:
            oh = b["oh"].to(args.device); wr = b["wr"].to(args.device)
            st = b["state"].to(args.device); kind = b["kind"].to(args.device)
            act = b["act"].to(args.device); fut = b["future"].to(args.device)
            pred_a, pred_f, sel = model(oh, wr, st, kind)
            idx_map = torch.tensor([0, 0, 1, 2, 3], device=kind.device)
            fidx = idx_map[kind]
            loss = mse(pred_a[:, :, :3], act[:, :, :3]) \
                + 2.0 * torch.nn.functional.binary_cross_entropy_with_logits(pred_a[:, :, 3], act[:, :, 3], pos_weight=torch.tensor(4.5, device=act.device)) \
                + 0.5 * torch.nn.functional.cross_entropy(sel, fidx) \
                + args.future_weight * mse(pred_f, fut)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(oh); na += len(oh)
        sched.step()
        model.eval()
        vt, vn = 0.0, 0
        with torch.no_grad():
            for b in dl_va:
                oh = b["oh"].to(args.device); wr = b["wr"].to(args.device)
                st = b["state"].to(args.device); kind = b["kind"].to(args.device)
                pred_a, pred_f, sel = model(oh, wr, st, kind)
                va = b["act"].to(args.device)
                idx_map = torch.tensor([0, 0, 1, 2, 3], device=kind.device)
                fidx = idx_map[kind]
                vl = mse(pred_a[:, :, :3], va[:, :, :3]) \
                    + 2.0 * torch.nn.functional.binary_cross_entropy_with_logits(pred_a[:, :, 3], va[:, :, 3], pos_weight=torch.tensor(4.5, device=va.device)) \
                    + 0.5 * torch.nn.functional.cross_entropy(sel, fidx) \
                    + args.future_weight * mse(pred_f, b["future"].to(args.device))
                vt += float(vl) * len(oh); vn += len(oh)
        val = vt / max(vn, 1)
        hist_log.append({"epoch": ep, "train": tot / max(na, 1), "val": val})
        if val < best:
            best = val
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                        "norm_state": norm_state.state_dict(), "norm_act": norm_act.state_dict(),
                        "norm_fut": norm_fut.state_dict(), "val": val}, out / "ckpt.pt")
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:3d} train {tot / max(na, 1):.4f} val {val:.4f}", flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    (out / "metrics.json").write_text(json.dumps({
        "data": [str(d) for d in data_dirs], "samples": len(ds), "train": len(train_idx), "val": len(val_idx),
        "params": n_params, "best_val": best, "cfg": cfg.__dict__, "history": hist_log,
    }, indent=2), encoding="utf-8")
    print(f"\n✓ 参数量 {n_params/1e6:.2f}M，最佳 val {best:.4f}，模型 {out/'ckpt.pt'}")


if __name__ == "__main__":
    main()
