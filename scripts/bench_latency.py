#!/usr/bin/env python
"""测量策略推理延迟与参数量。

用法:
    python scripts/bench_latency.py --ckpt runs/xxx/ckpt.pt
    python scripts/bench_latency.py --random --img-size 96
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from fruitsort.models.vision_policy import MultiViewTemporalPolicy, PolicyConfig


def bench(model, device, batch, iters=200, warmup=30, hist=4, img=96):
    model.eval()
    oh = torch.rand(batch, hist, 3, img, img, device=device)
    wr = torch.rand(batch, hist, 3, img, img, device=device)
    st = torch.randn(batch, hist, 49, device=device)
    kind = torch.zeros(batch, dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(oh, wr, st, kind)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(oh, wr, st, kind)
            if device.type == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return {
        "batch": batch,
        "mean_ms": statistics.mean(ts),
        "p50_ms": ts[len(ts) // 2],
        "p95_ms": ts[int(len(ts) * 0.95)],
        "fps": 1000.0 / statistics.mean(ts) * batch,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--hist", type=int, default=4)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    if args.ckpt:
        ck = torch.load(Path(args.ckpt), map_location="cpu", weights_only=False)
        cfg = PolicyConfig(**ck["cfg"])
        model = MultiViewTemporalPolicy(cfg)
        model.load_state_dict(ck["model"])
    else:
        cfg = PolicyConfig(hist=args.hist, img_size=args.img_size)
        model = MultiViewTemporalPolicy(cfg)

    n = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n:,} ({n/1e6:.2f}M)  img={cfg.img_size} hist={cfg.hist} d={cfg.d_model} layers={cfg.n_layers}")

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.insert(0, torch.device("cuda"))
    for dev in devices:
        m = model.to(dev)
        for bs in (1, 8):
            r = bench(m, dev, bs, iters=args.iters, hist=cfg.hist, img=cfg.img_size)
            print(f"  {str(dev):5s} batch={bs:2d}  mean {r['mean_ms']:.2f} ms  "
                  f"p95 {r['p95_ms']:.2f} ms  {r['fps']:.0f} fps")


if __name__ == "__main__":
    main()
