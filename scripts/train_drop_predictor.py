#!/usr/bin/env python
"""独立的「掉落预测器」：只用触觉 + 少量几何特征，不碰主策略。

为什么单独做
------------
实测（docs/model-architecture.md 6.19）：把触觉拼进 state 或做成 token 都会拖累
闭环成功率（3/3 → 1/3 → 0/3），力通道进一步恶化。但**掉落预测本身是有效的**
（留出集 AUROC 0.78）。所以把预测拆成独立的轻量 MLP：
- 主策略保持 `act-wide-4c`（0.08 87.5% / 0.15 96.9% 不变）
- 预测器只在推理端做"要不要干预"的决策

输入：最近 5 帧触觉（5×9=45）+ 当前几何（物体-夹爪距离、是否贴住、带速）= 48 维
标签：当前贴着夹爪、且 5~20 帧内物体离开 >10cm

用法:
    python scripts/train_drop_predictor.py --data data/datasets/dwt-* data/datasets/sdagger3-*
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
ALL = ("apple", "strawberry", "banana", "pear")
HIST = 5
TAC = 9


def load_episodes(dirs):
    """返回 [(feat (T,48), label (T,), episode_key)]。"""
    out = []
    for d in dirs:
        d = Path(d) if Path(d).is_absolute() else ROOT / d
        for f in sorted(d.glob("ep_*.npz")):
            data = np.load(f, allow_pickle=True)
            if "overhead_jpg" in data and "obs" not in data:
                continue
            obs = data["obs"].astype(np.float32)
            if obs.shape[1] < 58:
                continue
            instr = str(data["instruction"])
            j = ALL.index(instr) if instr in ALL else 0
            p_obj = obs[:, 17 + 8 * j:17 + 8 * j + 3]
            d_obj = np.linalg.norm(p_obj - obs[:, :3], axis=1)
            held = obs[:, 17 + 8 * j + 7]
            tac = obs[:, 49:58]
            T = len(obs)
            # 标签：贴住 且 5~20 帧内离开 >10cm
            label = np.zeros(T, dtype=np.float32)
            for i in range(T):
                lo, hi = i + 5, min(i + 21, T)
                if hi > lo and d_obj[i] < 0.06 and held[i] > 0.5 and (d_obj[lo:hi] > 0.10).any():
                    label[i] = 1.0
            # 特征：触觉历史(5×9) + [距离, 贴住, 带速]
            feat = np.zeros((T, HIST * TAC + 3), dtype=np.float32)
            for i in range(T):
                idx = np.clip(np.arange(i - HIST + 1, i + 1), 0, T - 1)
                feat[i, :HIST * TAC] = tac[idx].reshape(-1)
                feat[i, HIST * TAC:] = [d_obj[i], held[i], obs[i, 16]]
            out.append((feat, label, f"{d.name}/{f.name}", instr))
    return out


class DropNet(nn.Module):
    def __init__(self, dim=HIST * TAC + 3, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="示教 + 失败片段目录（支持通配）")
    ap.add_argument("--holdout-kinds", nargs="+", default=["pear", "banana"],
                    help="这些类的失败片段留出做测试")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="runs/dropnet")
    args = ap.parse_args()

    dirs = []
    for pat in args.data:
        dirs += sorted(Path().glob(pat)) if "*" in pat else [Path(pat)]
    eps = load_episodes(dirs)
    if not eps:
        raise SystemExit("没有可用数据（需要带触觉的 obs，宽度 ≥58）")

    # 按剧集留出：holdout 类里、路径含 sdagger 的失败片段做测试
    tr, va = [], []
    for feat, label, key, instr in eps:
        is_hold = instr in args.holdout_kinds and "sdagger" in key
        (va if is_hold else tr).append((feat, label))
    Xtr = np.concatenate([f for f, _ in tr]); ytr = np.concatenate([l for _, l in tr])
    Xva = np.concatenate([f for f, _ in va]) if va else Xtr
    yva = np.concatenate([l for _, l in va]) if va else ytr
    print(f"训练帧 {len(Xtr)}（正样本 {ytr.mean() * 100:.2f}%）｜"
          f"测试帧 {len(Xva)}（正样本 {yva.mean() * 100:.2f}%）")

    mu, sd = Xtr.mean(0), np.maximum(Xtr.std(0), 1e-6)
    model = DropNet(Xtr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    pw = torch.tensor(max(1.0, float((1 - ytr.mean()) / max(ytr.mean(), 1e-3))))
    Xtr_t = torch.from_numpy(((Xtr - mu) / sd).astype(np.float32))
    ytr_t = torch.from_numpy(ytr)
    bs = 512
    for ep in range(args.epochs):
        perm = torch.randperm(len(Xtr_t))
        tot = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(Xtr_t[idx]), ytr_t[idx], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:3d} loss {tot / len(perm):.4f}", flush=True)

    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(((Xva - mu) / sd).astype(np.float32)))).numpy()
    order = np.argsort(-p); y = yva[order]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    pos, neg = y.sum(), (1 - y).sum()
    auc = float(np.trapz(tp / pos, fp / neg)) if pos and neg else float("nan")
    print(f"留出 AUROC = {auc:.3f}｜正样本均值 {p[yva > 0.5].mean() if pos else float('nan'):.3f} "
          f"负样本 {p[yva < 0.5].mean() if neg else float('nan'):.3f}")
    if pos and neg:
        for th in (0.3, 0.5, 0.7):
            pred = p > th
            print(f"  阈值 {th}: 召回 {int((pred & (yva > 0.5)).sum()) / pos:.2f} "
                  f"误报率 {int((pred & (yva < 0.5)).sum()) / neg:.2f} 触发率 {pred.mean():.2f}")

    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "mean": mu, "std": sd, "hist": HIST,
                "tac": TAC, "auroc": auc}, out / "ckpt.pt")
    (out / "metrics.json").write_text(json.dumps(
        {"data": args.data, "frames_train": len(Xtr), "frames_test": len(Xva),
         "pos_rate": float(ytr.mean()), "auroc": auc,
         "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"✓ 模型 {out/'ckpt.pt'}")


if __name__ == "__main__":
    main()
