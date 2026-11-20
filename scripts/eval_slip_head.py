#!/usr/bin/env python
"""离线评估滑移辅助头（在**未参与训练**的失败局上）。

标签是「即将滑移的起始」（t+5..t+20 内会开始滑，且当前没滑），
所以这个 AUROC 衡量的是**预测能力**，不是"读观测里的 v_slip"。

用法:
    python scripts/eval_slip_head.py --ckpt runs/act-ta-onslip/ckpt.pt \
        --data data/datasets/sdagger2h-apple ... --stride 4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from fruitsort.learning.bc import Normalizer
from fruitsort.learning.image_dataset import ImageDemoDataset
from fruitsort.models.act_policy import ActConfig, ActPolicy

ROOT = Path(__file__).resolve().parent.parent


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    pos, neg = y.sum(), (1 - y).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    return float(np.trapz(tp / pos, fp / neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max-windows", type=int, default=4000)
    args = ap.parse_args()

    ck = torch.load(Path(args.ckpt), map_location="cpu", weights_only=False)
    cfg = ActConfig(**ck["cfg"])
    model = ActPolicy(cfg).to("cpu")
    model.load_state_dict(ck["model"]); model.eval()
    ns = Normalizer.from_state(ck["norm_state"])

    dirs = [Path(d) if Path(d).is_absolute() else ROOT / d for d in args.data]
    use_tac = int(getattr(cfg, "tactile_dim", 0) or 0) > 0
    ds = ImageDemoDataset(dirs, hist=cfg.hist, chunk=cfg.chunk, stride=args.stride,
                          use_phase=True, delta_actions=True, aux_dirs=dirs,
                          split_tactile=use_tac)
    idx = list(range(0, len(ds), max(1, len(ds) // args.max_windows)))
    ps, ys, eps = [], [], []
    with torch.no_grad():
        for i in idx:
            s = ds[i]
            st = ns.obs(s["state"].numpy()[None]).astype(np.float32)
            tac = s["tactile"][None] if "tactile" in s else None
            model(s["oh"][None], s["wr"][None], torch.from_numpy(st),
                  s["kind"][None].long(), use_prior=True, tactile=tac)
            ps.append(float(torch.sigmoid(model.last_slip).mean().item()))
            ys.append(float(s["slip"]))
            eps.append(int(ds.index[i][0]))
    ps, ys = np.array(ps), np.array(ys)
    pos = ys > 0.5
    print(f"窗口 {len(ys)}｜正样本 {int(pos.sum())}（{pos.mean() * 100:.1f}%）｜剧集 {len(set(eps))}")
    if pos.sum() and (~pos).sum():
        print(f"AUROC = {auroc(ps, ys):.3f}｜正样本平均概率 {ps[pos].mean():.3f} vs 负样本 {ps[~pos].mean():.3f}")
        for th in (0.3, 0.5, 0.7):
            pred = ps > th
            tp = int((pred & pos).sum())
            fp = int((pred & ~pos).sum())
            print(f"  阈值 {th}: 召回 {tp / pos.sum():.2f}  误报率 {fp / max((~pos).sum(), 1):.2f}  "
                  f"触发率 {pred.mean():.2f}")
    else:
        print("正/负样本缺失，无法算 AUROC")
    out = ROOT / "runs" / f"{time.strftime('%Y-%m-%d-%H%M')}-slip-head"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"ckpt": args.ckpt, "data": args.data, "windows": len(ys),
         "positives": int(pos.sum()), "auroc": auroc(ps, ys)}, ensure_ascii=False, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
