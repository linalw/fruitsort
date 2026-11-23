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
    ap.add_argument("--flow", action="store_true",
                    help="xyz 用 flow matching（rectified flow）采样，替代回归头；夹爪仍 logit")
    ap.add_argument("--window-val", action="store_true",
                    help="窗口级随机切分（train/val 来自同一批 episode）——可学性门槛用")
    ap.add_argument("--reuse-norm", action="store_true",
                    help="微调时沿用 init ckpt 的归一化器（实测会发散，仅作诊断用；默认按新数据重拟合）")
    ap.add_argument("--freeze", default="none", choices=["none", "head", "decoder"],
                    help="冻结主干只训适配器：head=只训动作/相位头；"
                         "decoder=只训 decoder+头（视觉与编码器冻结）。用于避免小样本微调打崩原策略")
    ap.add_argument("--init", default="",
                    help="从已有 ckpt 初始化权重（微调用；仍会按 --data 重新拟合归一化）")
    ap.add_argument("--no-cvae", action="store_true", help="关闭 CVAE 隐变量（纯确定性 ACT）")
    ap.add_argument("--delta-actions", action="store_true", help="预测相对当前 TCP 的增量")
    ap.add_argument("--geom", action="store_true",
                    help="观测加抓取几何（TCP 相对目标水果的 dx,dy,dz,|d|）")
    ap.add_argument("--phase", action="store_true",
                    help="给状态追加 1-hot 相位（接近/下探/搬运），obs_dim 49→52，"
                         "并训练相位辅助头（策略自己预测相位）")
    ap.add_argument("--phase-weight", type=float, default=0.5)
    ap.add_argument("--descend-boost", type=int, default=1,
                    help="下探/闭爪窗过采样倍数（相位=1 的窗口重复 N 次）")
    ap.add_argument("--no-tactile", action="store_true",
                    help="消融：丢掉观测里的 9 维触觉（同一份数据，干净 A/B）")
    ap.add_argument("--slip-head", action="store_true", help="加辅助头（预测未来 0.5s 内是否掉落）")
    ap.add_argument("--tactile-token", action="store_true",
                    help="触觉作为独立 token（9 维单独编码），而不是拼进 state")
    ap.add_argument("--force-head", action="store_true",
                    help="动作加第 5 维 = 目标夹持力（力作为动作通道）")
    ap.add_argument("--aux-data", nargs="*", default=[],
                    help="失败/滑脱片段目录：只参与滑移辅助损失，不参与动作 BC 损失")
    ap.add_argument("--aux-boost", type=int, default=1, help="失败片段过采样倍数")
    ap.add_argument("--slip-weight", type=float, default=0.3)
    ap.add_argument("--force-weight", type=float, default=1.0)
    ap.add_argument("--boost-kind", default="", help="弱类过采样：类名，如 pear")
    ap.add_argument("--kind-boost", type=int, default=1, help="该类的重复倍数")
    ap.add_argument("--carry-boost", type=int, default=1,
                    help="料框附近减速窗过采样倍数（目标 xy 距料框 <12cm 的窗口重复 N 次）")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 个样本（冒烟测试用）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_dirs = [Path(d) if Path(d).is_absolute() else ROOT / d for d in args.data]

    aux_dirs = [Path(d) if Path(d).is_absolute() else ROOT / d for d in args.aux_data]
    ds = ImageDemoDataset(list(data_dirs) + aux_dirs, hist=args.hist, chunk=args.chunk,
                          stride=args.stride, delay_prob=args.delay_prob,
                          delta_actions=args.delta_actions, use_phase=args.phase,
                          use_geom=args.geom,
                          descend_boost=args.descend_boost, carry_boost=args.carry_boost,
                          boost_kind=args.boost_kind, kind_boost=args.kind_boost,
                          drop_tactile=args.no_tactile, aux_dirs=aux_dirs,
                          split_tactile=args.tactile_token,
                          force_label=args.force_head)
    base_dim = int(ds[0]["state"].shape[-1]) - (3 if args.phase else 0)
    obs_dim = base_dim + (3 if args.phase else 0)

    # 分层划分：每组（同一数据集 × 同一指令）留出最后若干个 episode。
    # 现有脚本按窗口顺序切 15%，结果 val 100% 是 pear（文件按类别顺序保存），
    # 既看不出整体泛化，也会让 best-ckpt 选择偏向 pear。
    per_kind: dict[str, list[int]] = {}
    for i, (ei, _t) in enumerate(ds.index):
        per_kind.setdefault(ds.episodes[ei]["instruction"], []).append(i)
    train_idx, val_idx = [], []
    for kind, idxs in per_kind.items():
        eps = sorted({ds.index[i][0] for i in idxs})
        # 只从**主数据**里留验证集：aux 数据是 is_aux，若按全量留，
        # 最后 k 个 episode 恰好全是 aux → val 永远为空（曾导致 val 全 0、无法选模型）
        eps_main = [e for e in eps if not ds.episodes[e].get("aux_only", False)] or eps
        # 打乱后再留出：按文件顺序取最后 k 个会让验证集整个落在"最后一个数据目录"上
        # （实测 val 因此变成纯 DAgger 段、val 79 而 train 0.004）
        eps_main = list(np.random.default_rng(args.seed).permutation(sorted(eps_main)))
        k = max(1, int(round(len(eps_main) * args.val_frac)))
        hold = set(eps_main[-k:])
        for i in idxs:
            is_aux = ds.episodes[ds.index[i][0]].get("aux_only", False)
            (train_idx if (is_aux or ds.index[i][0] not in hold) else val_idx).append(i)
    if args.aux_boost > 1:
        extra = [i for i in train_idx if ds.episodes[ds.index[i][0]].get("aux_only", False)]
        train_idx = train_idx + extra * (args.aux_boost - 1)
    train_idx.sort(); val_idx.sort()
    if args.limit:
        train_idx = train_idx[: args.limit]
        val_idx = val_idx[: max(8, args.limit // 4)]
    if args.window_val:
        rng_w = np.random.default_rng(args.seed)
        perm = rng_w.permutation(len(train_idx))
        k = max(1, int(round(len(perm) * args.val_frac)))
        val_idx = sorted(train_idx[i] for i in perm[:k])
        train_idx = sorted(train_idx[i] for i in perm[k:])
        print(f"[window-val] 同一批 episode 的窗口随机切分：val {len(val_idx)} 窗口", flush=True)
    print("每类窗口/验证:", {k: (sum(1 for i in v), sum(1 for i in v if i in set(val_idx)))
                              for k, v in per_kind.items()})

    # 训练集统计量（子采样估计，避免全量 18k 窗口的 IO）
    sample_idx = train_idx[:: max(1, len(train_idx) // 2000)]
    states = np.stack([ds[i]["state"].numpy() for i in sample_idx]).reshape(-1, obs_dim)
    acts = np.stack([ds[i]["act"].numpy() for i in sample_idx]).reshape(-1, 4)
    norm_state = Normalizer(states, states)
    norm_act = Normalizer(acts[:, :3], acts[:, :3])
    # 微调时必须沿用 init ckpt 的归一化器：编码器是用原统计量训练的，
    # 按新数据重拟合会让输入/输出尺度整体漂移——这是此前四次微调"打崩原策略"的根因
    if args.init and args.reuse_norm:
        _ck0 = torch.load(args.init, map_location="cpu", weights_only=False)
        if "norm_state" in _ck0 and "norm_act" in _ck0:
            norm_state = Normalizer.from_state(_ck0["norm_state"])
            norm_act = Normalizer.from_state(_ck0["norm_act"])
            print("沿用 init ckpt 的归一化器（--refit-norm 可强制重拟合）", flush=True)
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
            # 数据集混用：连续流数据没有 slip 标签，补零保持 batch 结构一致
            if "slip" not in s:
                s["slip"] = torch.zeros((), dtype=torch.float32)
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
                    n_phase=3 if args.phase else 0, use_flow=bool(args.flow),
                    n_slip=1 if args.slip_head else 0,
                    tactile_dim=9 if args.tactile_token else 0,
                    use_force_head=bool(args.force_head))
    model = ActPolicy(cfg).to(args.device)
    if args.init:
        _ck = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(_ck["model"])
        print(f"从 {args.init} 初始化权重（微调）", flush=True)
    n_params = count_params(model)
    if args.freeze != "none":
        trainable_sets = {
            "head": ("act_head", "phase_head", "slip_head"),
            "decoder": ("decoder", "dec_norm", "query_pos",
                        "act_head", "phase_head", "slip_head"),
        }[args.freeze]
        n_tr = 0
        for name, p_ in model.named_parameters():
            keep = name.startswith(trainable_sets)
            p_.requires_grad = bool(keep)
            n_tr += p_.numel() if keep else 0
        print(f"冻结主干（{args.freeze}）：可训练参数 {n_tr/1e6:.2f}M / 总 {n_params/1e6:.2f}M",
              flush=True)
    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    mse = torch.nn.MSELoss()

    def compute_loss(pred, mu, logvar, act, w=None):
        xyz_p = ((pred[:, :, :3] - act[:, :, :3]) ** 2).mean(dim=(1, 2))
        grip_p = torch.nn.functional.binary_cross_entropy_with_logits(
            pred[:, :, 3], act[:, :, 3], pos_weight=pos_weight, reduction="none").mean(dim=1)
        if w is not None and float(w.sum()) > 0:
            l_xyz = (xyz_p * w).sum() / w.sum().clamp(min=1.0)
            l_grip = (grip_p * w).sum() / w.sum().clamp(min=1.0)
        else:
            l_xyz, l_grip = xyz_p.mean(), grip_p.mean()
        l_force = torch.zeros((), device=args.device)
        if pred.shape[-1] > 4 and act.shape[-1] > 4:
            fp = ((pred[:, :, 4] - act[:, :, 4]) ** 2).mean(dim=1)
            if w is not None and float(w.sum()) > 0:
                l_force = (fp * w).sum() / w.sum().clamp(min=1.0)
            else:
                l_force = fp.mean()
        loss = l_xyz + args.grip_weight * l_grip + args.force_weight * l_force
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
                tac = b["tactile"].to(args.device) if "tactile" in b else None
                if args.flow:
                    mem = model.encode(oh, wr, st, kind, tactile=tac)
                    l_flow, grip_logit, logits = model.flow_loss(mem, st, act)
                    w_bc = (1.0 - b["aux_only"].to(args.device).float()) if "aux_only" in b else None
                    grip_t = (act[:, :, 3] < 0).float()
                    l_grip = torch.nn.functional.binary_cross_entropy_with_logits(
                        grip_logit.float(), grip_t, pos_weight=pos_weight, reduction="none").mean(1)
                    l_grip = (l_grip * w_bc).sum() / w_bc.sum().clamp(min=1.0) if w_bc is not None else l_grip.mean()
                    l_xyz = l_flow
                    l_kl = torch.zeros((), device=args.device)
                    loss = l_xyz + args.grip_weight * l_grip
                else:
                    if args.phase:
                        pred, mu, logvar, logits = model(oh, wr, st, kind, act_target=act,
                                                         phase_out=True, tactile=tac)
                    else:
                        pred, mu, logvar = model(oh, wr, st, kind, act_target=act, tactile=tac)
                        logits = None
                    w_bc = (1.0 - b["aux_only"].to(args.device).float()) if "aux_only" in b else None
                    loss, l_xyz, l_grip, l_kl = compute_loss(pred.float(), mu, logvar, act, w_bc)
                l_slip = torch.zeros((), device=args.device)
                if args.slip_head and "slip" in b:
                    sl = b["slip"].to(args.device).float()
                    p_slip = model.last_slip.float()
                    # 正样本极少 → 用 pos_weight 平衡
                    pw = torch.tensor(max(1.0, float((1 - sl.mean()) / sl.mean().clamp(min=1e-3))),
                                      device=args.device)
                    l_slip = torch.nn.functional.binary_cross_entropy_with_logits(p_slip, sl,
                                                                                 pos_weight=pw)
                    loss = loss + args.slip_weight * l_slip
                l_ph = torch.zeros((), device=args.device)
                if logits is not None:
                    ce = torch.nn.functional.cross_entropy(
                        logits.float().reshape(-1, logits.shape[-1]),
                        b["phase_chunk"].to(args.device).reshape(-1), reduction="none")
                    ce = ce.view(len(ce) // logits.shape[1], logits.shape[1]).mean(dim=1)
                    l_ph = (ce * w_bc).sum() / w_bc.sum().clamp(min=1.0) if w_bc is not None else ce.mean()
                    loss = loss + args.phase_weight * l_ph
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(oh)
            na += len(oh)
            for k, v in (("xyz", l_xyz), ("grip", l_grip), ("kl", l_kl), ("phase", l_ph),
                         ("slip", l_slip)):
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
                    vtac = b["tactile"].to(args.device) if "tactile" in b else None
                    if args.phase:
                        pred, _, _, vlogits = model(oh, wr, st, kind, use_prior=True,
                                                    phase_out=True, tactile=vtac)
                    else:
                        pred, _, _ = model(oh, wr, st, kind, use_prior=True, tactile=vtac)
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
                  f"slip {acc.get('slip',0)/max(na,1):.3f}) val {val:.4f} "
                  f"(ph {vacc.get('phase',0)/max(vn,1):.4f}) {dt:.1f}s", flush=True)

    (out / "metrics.json").write_text(json.dumps({
        "model": "ActPolicy", "data": [str(d) for d in data_dirs],
        "samples": len(ds), "train": len(train_idx), "val": len(val_idx),
        "params": n_params, "best_val": best, "cfg": cfg.__dict__,
        "delta_actions": bool(args.delta_actions), "phase": bool(args.phase),
        "tactile": bool(not args.no_tactile and base_dim >= 58), "slip_head": bool(args.slip_head),
        "args": vars(args),
        "history": hist_log,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 参数量 {n_params/1e6:.2f}M，最佳 val {best:.4f}，模型 {out/'ckpt.pt'}")


if __name__ == "__main__":
    main()
