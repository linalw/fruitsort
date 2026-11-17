#!/usr/bin/env python
"""采集「纯模型 vs teacher」的逐步对照数据，训练残差控制器（Res-Refiner）。

采集协议（DAgger 风味：
- 执行**纯模型**闭环：不加近距伺服、不加几何夹爪、不加确定性搬运，只有
  `pd_ee_pose` + 模型输出的世界目标 xyz / 夹爪 logit；
- 每一步都让 `HybridTeacher` 在**同一条观测上**给出它自己的目标与夹爪，
  作为该状态的监督标签；
- 失败的 episode 全部保留（这些正是纯模型会踩到的状态）。

标签：`Δxyz = teacher_tgt - model_tgt`（回归时按 max_pos 截断幅度），
夹爪：`teacher_grip ∈ {0,1}`（1 = 闭合），对 `model_logit + Δgrip` 做 BCE。

用法:
    source scripts/env.sh maniskill3
    python scripts/train_residual.py --episodes 10 --out runs/residual-v1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

import fruitsort.sim.tasks  # noqa: F401
from fruitsort.learning.bc import Normalizer, build_obs
from fruitsort.learning.residual import (ResidualConfig, ResidualRefiner, featurize,
                                         onehot, save_residual)
from fruitsort.models.vision_policy import MultiViewTemporalPolicy, PolicyConfig
from fruitsort.policies.hybrid_teacher import HybridTeacher
from fruitsort.policies.scripted import HOME_EULER

ROOT = Path(__file__).resolve().parent.parent
INSTR_ID = {"all": 0, "apple": 1, "strawberry": 2, "banana": 3, "pear": 4}
ID_KIND = {0: "apple", 1: "strawberry", 2: "banana", 3: "pear"}
KIND_SLOT = {"apple": 0, "strawberry": 1, "banana": 2, "pear": 3}
MAX_STEP = 0.06


def prep_img(x, size):
    if x.shape[0] != size:
        x = cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
    return (x.astype(np.float32) / 255.0).transpose(2, 0, 1)


def to_cmd(tgt, tcp, grip):
    tgt = np.asarray(tgt, dtype=np.float64).copy()
    err = tgt - tcp
    d = float(np.linalg.norm(err))
    if d > MAX_STEP:
        tgt = tcp + err / d * MAX_STEP
    return np.array([tgt[1] + 0.5, -tgt[0], tgt[2], *HOME_EULER, float(grip)], dtype=np.float32)


def make_env(args, ep: int):
    env = gym.make(
        "FruitSort-v1", obs_mode="rgbd", control_mode="pd_ee_pose", sim_backend="cpu",
        robot_uids="panda_wristcam", active_kinds=tuple(args.kinds),
        belt_speed_range=(args.belt_speed, args.belt_speed),
        spawn_x=args.spawn_x, spawn_spacing=args.spawn_spacing,
        instruction_kind=None if args.instruction == "all" else args.instruction)
    obs, _ = env.reset(seed=args.seed + ep)
    return env, obs


def load_policy(ckpt, device):
    ck = torch.load(Path(ckpt), map_location="cpu", weights_only=False)
    cfg = PolicyConfig(**ck["cfg"])
    model = MultiViewTemporalPolicy(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, Normalizer.from_state(ck["norm_state"]), Normalizer.from_state(ck["norm_act"])


@torch.no_grad()
def select_kind(model, cfg, ns, obs, u, instruction, device):
    sd = obs["sensor_data"]
    oh = prep_img(sd["overhead"]["rgb"][0].cpu().numpy(), cfg.img_size)
    wr = prep_img(sd["hand_camera"]["rgb"][0].cpu().numpy(), cfg.img_size)
    st = build_obs(u)
    _, _, sel = model(
        torch.from_numpy(np.stack([oh] * cfg.hist)[None]).float().to(device),
        torch.from_numpy(np.stack([wr] * cfg.hist)[None]).float().to(device),
        torch.from_numpy(ns.obs(np.stack([st] * cfg.hist).astype(np.float32))[None]).float().to(device),
        torch.tensor([INSTR_ID[instruction]], device=device))
    kind = ID_KIND.get(int(sel.argmax(-1).item()), "apple")
    if kind not in u.fruit_actors:
        kind = instruction if instruction in u.fruit_actors else list(u.fruit_actors)[0]
    return kind


# ---------------------------------------------------------------------- #
# 1) 采集
# ---------------------------------------------------------------------- #
def collect(args, model, cfg, ns, na, device) -> list[dict]:
    out_dir = Path(args.out) / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    eps = []
    for ep in range(args.episodes):
        env, obs = make_env(args, ep)
        u = env.unwrapped
        kind = select_kind(model, cfg, ns, obs, u, args.instruction, device)
        teacher = HybridTeacher(env, args.ckpt, args.instruction, device=device)
        teacher.selected = kind  # 选择对齐：同一个模型、同一个观测 → 同一颗水果

        oh_h, wr_h, st_h = [], [], []
        rec = {k: [] for k in ("state", "model_tgt", "model_logit", "teacher_tgt", "teacher_grip")}
        info, t = {}, 0
        for t in range(args.max_steps):
            sd = obs["sensor_data"]
            oh_h.append(prep_img(sd["overhead"]["rgb"][0].cpu().numpy(), cfg.img_size))
            wr_h.append(prep_img(sd["hand_camera"]["rgb"][0].cpu().numpy(), cfg.img_size))
            st_h.append(build_obs(u).astype(np.float32))
            for h in (oh_h, wr_h, st_h):
                if len(h) > cfg.hist:
                    h.pop(0)
            while len(oh_h) < cfg.hist:
                oh_h.insert(0, oh_h[0]); wr_h.insert(0, wr_h[0]); st_h.insert(0, st_h[0])

            tcp = u.agent.tcp_pose.p[0].cpu().numpy().astype(np.float64)
            # --- 纯模型前向（与 eval_vision_geo --near 0 完全一致）---
            with torch.no_grad():
                pred, _, _ = model(
                    torch.from_numpy(np.stack(oh_h)[None]).float().to(device),
                    torch.from_numpy(np.stack(wr_h)[None]).float().to(device),
                    torch.from_numpy(ns.obs(np.stack(st_h).astype(np.float32))[None]).float().to(device),
                    torch.tensor([INSTR_ID[args.instruction]], device=device))
            p4 = pred[0, 0].cpu().numpy()
            model_tgt = np.clip(na.unact(p4[:3]), -2, 2).astype(np.float64)
            model_logit = float(p4[3])

            # --- teacher 在同一条观测上打标（DAgger）---
            _, label = teacher.act(obs)
            teacher_tgt = np.asarray(label[:3], dtype=np.float64)
            teacher_grip = 1.0 if float(label[3]) < 0 else 0.0

            rec["state"].append(ns.obs(st_h[-1][None])[0].astype(np.float32))
            rec["model_tgt"].append(model_tgt.astype(np.float32))
            rec["model_logit"].append(np.float32(model_logit))
            rec["teacher_tgt"].append(teacher_tgt.astype(np.float32))
            rec["teacher_grip"].append(np.float32(teacher_grip))

            grip = -1.0 if model_logit > 0 else 1.0   # 纯模型夹爪：logit 符号
            obs, _, term, trunc, info = env.step(to_cmd(model_tgt, tcp, grip))
            if bool(info.get("success") or info.get("fail")):
                break

        ok = bool(info.get("success", False))
        arrs = {k: np.asarray(v, dtype=np.float32) for k, v in rec.items()}
        arrs["kind_slot"] = np.full(len(arrs["state"]), KIND_SLOT.get(kind, 0), dtype=np.int64)
        arrs["success"] = np.array(ok)
        np.savez_compressed(out_dir / f"ep_{ep:03d}.npz", **arrs)
        d = np.linalg.norm(arrs["teacher_tgt"] - arrs["model_tgt"], axis=1)
        print(f"[{'✓' if ok else '✗'}] ep {ep:02d} kind={kind} steps={t + 1} "
              f"|Δxyz| mean={d.mean():.3f} m  p90={np.percentile(d, 90):.3f}  "
              f"teacher_close={arrs['teacher_grip'].mean() * 100:.0f}%  "
              f"model_close={(arrs['model_logit'] > 0).mean() * 100:.0f}%")
        eps.append({"episode": ep, "success": ok, "steps": t + 1, "kind": kind,
                    "n": int(len(d)), "delta_mean": float(d.mean())})
        env.close()
    return eps


# ---------------------------------------------------------------------- #
# 2) 训练
# ---------------------------------------------------------------------- #
def clip_norm(x: torch.Tensor, m: float) -> torch.Tensor:
    n = x.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return x * (n.clamp_max(m) / n)


def train(args, eps_meta: list[dict], device) -> dict:
    ds = sorted((Path(args.out) / "dataset").glob("ep_*.npz"))
    data = [np.load(f) for f in ds]
    n_ep = len(data)
    n_val = max(1, int(round(n_ep * args.val_frac)))
    val_idx = set(range(n_ep - n_val, n_ep))  # 按 episode 划分，最后 n_val 条做验证

    def stack(idx):
        st = np.concatenate([data[i]["state"] for i in idx])
        mt = np.concatenate([data[i]["model_tgt"] for i in idx])
        ml = np.concatenate([data[i]["model_logit"] for i in idx])
        tt = np.concatenate([data[i]["teacher_tgt"] for i in idx])
        tg = np.concatenate([data[i]["teacher_grip"] for i in idx])
        oh = np.stack([onehot(int(data[i]["kind_slot"][0])) for i in idx for _ in range(len(data[i]["state"]))])
        return st, mt, ml, tt, tg, oh

    tr = [i for i in range(n_ep) if i not in val_idx]
    te = [i for i in range(n_ep) if i in val_idx]
    st_tr, mt_tr, ml_tr, tt_tr, tg_tr, oh_tr = stack(tr)
    st_te, mt_te, ml_te, tt_te, tg_te, oh_te = stack(te)
    print(f"训练样本 {len(st_tr)}（{len(tr)} ep）/ 验证 {len(st_te)}（{len(te)} ep）")

    cfg = ResidualConfig(obs_dim=st_tr.shape[1], n_kinds=oh_tr.shape[1],
                         hidden=args.hidden, max_pos=args.max_pos, max_grip=args.max_grip)
    net = ResidualRefiner(cfg)
    net.fit_normalizer(featurize(st_tr, mt_tr, oh_tr))
    net = net.to(device)

    t = {k: torch.from_numpy(v.astype(np.float32)).to(device)
         for k, v in dict(st=st_tr, mt=mt_tr, ml=ml_tr, tt=tt_tr, tg=tg_tr, oh=oh_tr).items()}
    v = {k: torch.from_numpy(v.astype(np.float32)).to(device)
         for k, v in dict(st=st_te, mt=mt_te, ml=ml_te, tt=tt_te, tg=tg_te, oh=oh_te).items()}
    d_tgt_tr = clip_norm(t["tt"] - t["mt"], args.max_pos)
    d_tgt_te = clip_norm(v["tt"] - v["mt"], args.max_pos)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    n = len(st_tr)
    hist = []
    best = None
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            dxyz, dgrip = net(t["st"][idx], t["mt"][idx], t["oh"][idx])
            loss = F.mse_loss(dxyz, d_tgt_tr[idx]) + args.grip_w * F.binary_cross_entropy_with_logits(
                t["ml"][idx] + dgrip, t["tg"][idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        net.eval()
        with torch.no_grad():
            dxyz, dgrip = net(v["st"], v["mt"], v["oh"])
            l_xyz = float(F.mse_loss(dxyz, d_tgt_te))
            l_grip = float(F.binary_cross_entropy_with_logits(v["ml"] + dgrip, v["tg"]))
            cos = float(F.cosine_similarity(dxyz, d_tgt_te, dim=-1).mean())
            acc = float(((v["ml"] + dgrip > 0).float() == v["tg"]).float().mean())
            base_acc = float(((v["ml"] > 0).float() == v["tg"]).float().mean())
        hist.append({"epoch": ep, "train": tot / n, "val_xyz": l_xyz, "val_grip": l_grip,
                     "val_cos": cos, "vel_grip_acc": acc, "base_grip_acc": base_acc})
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"ep {ep:02d} train={tot / n:.5f} val_xyz={l_xyz:.5f} "
                  f"grip={l_grip:.3f} cos={cos:.2f} acc={acc:.3f}（基线 {base_acc:.3f}）")
        if best is None or l_xyz + args.grip_w * l_grip < best[0]:
            best = (l_xyz + args.grip_w * l_grip, {k: vv.detach().clone() for k, vv in net.state_dict().items()})
    net.load_state_dict(best[1])
    net.eval()

    meta = {"ckpt": str(args.ckpt), "params": net.n_params, "cfg": vars(net.cfg) if hasattr(net.cfg, "__dict__") else None,
            "dataset": {"episodes": n_ep, "train": len(st_tr), "val": len(st_te),
                        "train_eps": tr, "val_eps": te},
            "history": hist, "best_val": best[0],
            "delta_tgt_norm_mean": float(d_tgt_tr.norm(dim=-1).mean()),
            "teacher_close_rate": float(tg_tr.mean())}
    save_residual(Path(args.out) / "ckpt.pt", net, meta)
    (Path(args.out) / "metrics.json").write_text(json.dumps(
        {**{k: v for k, v in meta.items() if k != "history"}, "episodes": eps_meta},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"残差参数量 {net.n_params}；ckpt → {Path(args.out) / 'ckpt.pt'}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/vpolicy-4c-v3/ckpt.pt")
    ap.add_argument("--kinds", nargs="+", default=["apple", "strawberry"])
    ap.add_argument("--instruction", default="apple",
                    choices=["all", "apple", "strawberry", "banana", "pear"])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--belt-speed", type=float, default=0.06)
    ap.add_argument("--spawn-x", type=float, default=-0.75)
    ap.add_argument("--spawn-spacing", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=450)
    ap.add_argument("--out", default=f"runs/{time.strftime('%Y-%m-%d-%H%M')}-residual")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--max-pos", type=float, default=0.03)
    ap.add_argument("--max-grip", type=float, default=3.0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grip-w", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    model, cfg, ns, na = load_policy(args.ckpt, args.device)
    t0 = time.time()
    eps_meta = collect(args, model, cfg, ns, na, args.device)
    print(f"采集耗时 {time.time() - t0:.0f}s，纯模型成功率 "
          f"{sum(e['success'] for e in eps_meta)}/{len(eps_meta)}")
    train(args, eps_meta, args.device)


if __name__ == "__main__":
    main()
