#!/usr/bin/env python
"""训练 ACT-lite chunked 策略（`src/fruitsort/models/act_policy.py`）。

与 `train_vision_policy.py` 的差别
---------------------------------
- 模型换成 ACT 结构（4 层编码器 + 4 层解码器 + K 个 query），6.9M 参数（v0.5 是 0.41M）
- 动作块 K=16、历史 T=8
- 加 CVAE 隐变量（posterior 编码专家动作块，推理用先验 z=0）
- bf16 autocast 训练（5090 上 ~2.5k samples/s）
- 标注 delta 动作模式，评测脚本自动读取

用法:
    python scripts/train_act.py --data data/datasets/suction-030 data/datasets/suction-015 \
        --out runs/act-suction --epochs 80 --batch 64
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fruitsort.learning.bc import Normalizer
from fruitsort.learning.image_dataset import ImageDemoDataset
from fruitsort.models.act_policy import ActConfig, ActPolicy, count_params

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--out", default="")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--hist", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--enc-layers", type=int, default=4)
    ap.add_argument("--dec-layers", type=int, default=4)
    ap.add_argument("--grip-weight", type=float, default=2.0)
    ap.add_argument("--kl-weight", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--delay-prob", type=float, default=0.1, help="把最后一帧替换成上一帧的概率（模拟延迟）")
    ap.add_argument("--no-aug", action="store_true", help="关闭亮度/对比度抖动")
    ap.add_argument("--val-frac", type=float, default=0.15, help="按类别分层留出验证 episode 比例")
    ap.add_argument("--no-cvae", action="store_true", help="关闭 CVAE 隐变量（纯确定性 ACT）")
    ap.add_argument("--delta-actions", action="store_true", help="预测相对当前 TCP 的增量")
    ap.add_argument("--phase", action="store_true",
                    help="给状态追加 1-hot 相位（接近/下探/搬运），obs_dim 49→52，"
                         "并训练相位辅助头（策略自己预测相位）")
    ap.add_argument("--phase-weight", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 个样本（冒烟测试用）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_dirs = [Path(d) if Path(d).is_absolute() else ROOT / d for d in args.data]

    ds = ImageDemoDataset(data_dirs, hist=args.hist, chunk=args.chunk, stride=args.stride,
                          delay_prob=args.delay_prob, delta_actions=args.delta_actions,
                          use_phase=args.phase)
    obs_dim = 52 if args.phase else 49

    # 分层划分：每组（同一数据集 × 同一指令）留出最后若干个 episode。
    # 现有脚本按窗口顺序切 15%，结果 val 100% 是 pear（文件按类别顺序保存），
    # 既看不出整体泛化，也会让 best-ckpt 选择偏向 pear。
    per_kind: dict[str, list[int]] = {}
    for i, (ei, _t) in enumerate(ds.index):
        per_kind.setdefault(ds.episodes[ei]["instruction"], []).append(i)
    train_idx, val_idx = [], []
    for kind, idxs in per_kind.items():
        eps = sorted({ds.index[i][0] for i in idxs})
        k = max(1, int(round(len(eps) * args.val_frac)))
        hold = set(eps[-k:])
        for i in idxs:
            (val_idx if ds.index[i][0] in hold else train_idx).append(i)
    train_idx.sort(); val_idx.sort()
    if args.limit:
        train_idx = train_idx[: args.limit]
        val_idx = val_idx[: max(8, args.limit // 4)]
    print("每类窗口/验证:", {k: (sum(1 for i in v), sum(1 for i in v if i in set(val_idx)))
                              for k, v in per_kind.items()})

    # 训练集统计量（子采样估计，避免全量 18k 窗口的 IO）
    sample_idx = train_idx[:: max(1, len(train_idx) // 2000)]
    states = np.stack([ds[i]["state"].numpy() for i in sample_idx]).reshape(-1, obs_dim)
    acts = np.stack([ds[i]["act"].numpy() for i in sample_idx]).reshape(-1, 4)
    norm_state = Normalizer(states, states)
    norm_act = Normalizer(acts[:, :3], acts[:, :3])
    pos_frac = float(acts[:, 3].mean())
    pos_weight = torch.tensor((1 - pos_frac) / max(pos_frac, 1e-3), device=args.device)
    print(f"样本 {len(ds)}（train {len(train_idx)} / val {len(val_idx)}），"
          f"吸附帧占比 {pos_frac:.3f} → pos_weight {pos_weight.item():.2f}")

    class Wrap(torch.utils.data.Dataset):
        def __init__(self, base, idx):
            self.base, self.idx = base, idx

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            s = self.base[self.idx[i]]
            oh, wr = s["oh"], s["wr"]
            if not args.no_aug and torch.rand(()) < 0.5:
                # 亮度/对比度抖动：rgbd 渲染的色彩随机化很弱，闭环时亮度会随视角变化
                sc = 1.0 + (torch.rand((), dtype=torch.float32) - 0.5) * 0.30
                sh = (torch.rand((), dtype=torch.float32) - 0.5) * 0.16
                oh = (oh * sc + sh).clamp_(0, 1)
                wr = (wr * sc + sh).clamp_(0, 1)
            s["oh"], s["wr"] = oh, wr
            s["state"] = torch.from_numpy(
                norm_state.obs(s["state"].numpy()).astype(np.float32))
            a = s["act"].numpy().copy()
            a[:, :3] = norm_act.act(a[:, :3])
            s["act"] = torch.from_numpy(a.astype(np.float32))
            return s

    dl_tr = DataLoader(Wrap(ds, train_idx), batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, drop_last=True,
                       persistent_workers=args.workers > 0, pin_memory=True)
    dl_va = DataLoader(Wrap(ds, val_idx), batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, persistent_workers=args.workers > 0,
                       pin_memory=True)

    cfg = ActConfig(hist=args.hist, chunk=args.chunk, d_model=args.d_model,
                    enc_layers=args.enc_layers, dec_layers=args.dec_layers,
                    dropout=args.dropout, use_cvae=not args.no_cvae, obs_dim=obs_dim,
                    n_phase=3 if args.phase else 0)
    model = ActPolicy(cfg).to(args.device)
    n_params = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    mse = torch.nn.MSELoss()

    def compute_loss(pred, mu, logvar, act):
        l_xyz = mse(pred[:, :, :3], act[:, :, :3])
        l_grip = torch.nn.functional.binary_cross_entropy_with_logits(
            pred[:, :, 3], act[:, :, 3], pos_weight=pos_weight)
        loss = l_xyz + args.grip_weight * l_grip
        l_kl = torch.zeros((), device=args.device)
        if mu is not None:
            l_kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
            loss = loss + args.kl_weight * l_kl
        return loss, l_xyz, l_grip, l_kl

    tag = time.strftime("%Y-%m-%d-%H%M")
    out = Path(args.out) if args.out else ROOT / "runs" / f"{tag}-act"
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    best, hist_log = float("inf"), []
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tot, na, acc = 0.0, 0, {}
        for b in dl_tr:
            oh = b["oh"].to(args.device, non_blocking=True)
            wr = b["wr"].to(args.device, non_blocking=True)
            st = b["state"].to(args.device, non_blocking=True)
            kind = b["kind"].to(args.device, non_blocking=True)
            act = b["act"].to(args.device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=args.device.startswith("cuda")):
                if args.phase:
                    pred, mu, logvar, logits = model(oh, wr, st, kind, act_target=act,
                                                     phase_out=True)
                else:
                    pred, mu, logvar = model(oh, wr, st, kind, act_target=act)
                    logits = None
                loss, l_xyz, l_grip, l_kl = compute_loss(pred.float(), mu, logvar, act)
                l_ph = torch.zeros((), device=args.device)
                if logits is not None:
                    l_ph = torch.nn.functional.cross_entropy(
                        logits.float().reshape(-1, logits.shape[-1]),
                        b["phase_chunk"].to(args.device).reshape(-1))
                    loss = loss + args.phase_weight * l_ph
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(oh)
            na += len(oh)
            for k, v in (("xyz", l_xyz), ("grip", l_grip), ("kl", l_kl), ("phase", l_ph)):
                acc[k] = acc.get(k, 0.0) + float(v.detach()) * len(oh)
        sched.step()

        model.eval()
        vt, vn, vacc = 0.0, 0, {}
        with torch.no_grad():
            for b in dl_va:
                oh = b["oh"].to(args.device); wr = b["wr"].to(args.device)
                st = b["state"].to(args.device); kind = b["kind"].to(args.device)
                act = b["act"].to(args.device)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=args.device.startswith("cuda")):
                    if args.phase:
                        pred, _, _, vlogits = model(oh, wr, st, kind, use_prior=True,
                                                    phase_out=True)
                    else:
                        pred, _, _ = model(oh, wr, st, kind, use_prior=True)
                        vlogits = None
                    vl, v_xyz, v_grip, _ = compute_loss(pred.float(), None, None, act)
                    if vlogits is not None:
                        v_ph = torch.nn.functional.cross_entropy(
                            vlogits.float().reshape(-1, vlogits.shape[-1]),
                            b["phase_chunk"].to(args.device).reshape(-1))
                        vl = vl + args.phase_weight * v_ph
                        vacc["phase"] = vacc.get("phase", 0.0) + float(v_ph) * len(oh)
                vt += float(vl) * len(oh); vn += len(oh)
                for k, v in (("xyz", v_xyz), ("grip", v_grip)):
                    vacc[k] = vacc.get(k, 0.0) + float(v) * len(oh)
        val, dt = vt / max(vn, 1), time.time() - t0
        hist_log.append({
            "epoch": ep, "train": tot / max(na, 1), "val": val, "sec": round(dt, 1),
            "train_xyz": acc.get("xyz", 0) / max(na, 1), "train_grip": acc.get("grip", 0) / max(na, 1),
            "train_kl": acc.get("kl", 0) / max(na, 1),
            "val_xyz": vacc.get("xyz", 0) / max(vn, 1), "val_grip": vacc.get("grip", 0) / max(vn, 1),
        })
        if val < best:
            best = val
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                        "norm_state": norm_state.state_dict(),
                        "norm_act": norm_act.state_dict(),
                        "delta_actions": bool(args.delta_actions),
                        "params": n_params, "val": val}, out / "ckpt.pt")
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:3d} train {tot/max(na,1):.4f} "
                  f"(xyz {acc.get('xyz',0)/max(na,1):.4f} grip {acc.get('grip',0)/max(na,1):.4f} "
                  f"kl {acc.get('kl',0)/max(na,1):.4f} ph {acc.get('phase',0)/max(na,1):.4f}) "
                  f"val {val:.4f} (ph {vacc.get('phase',0)/max(vn,1):.4f}) {dt:.1f}s", flush=True)

    (out / "metrics.json").write_text(json.dumps({
        "model": "ActPolicy", "data": [str(d) for d in data_dirs],
        "samples": len(ds), "train": len(train_idx), "val": len(val_idx),
        "params": n_params, "best_val": best, "cfg": cfg.__dict__,
        "delta_actions": bool(args.delta_actions), "phase": bool(args.phase),
        "args": vars(args),
        "history": hist_log,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 参数量 {n_params/1e6:.2f}M，最佳 val {best:.4f}，模型 {out/'ckpt.pt'}")


if __name__ == "__main__":
    main()
