#!/usr/bin/env python
"""训练"识别头"：掩码特征 → 类别（8 维 → 4 类的小 MLP）。

数据从仿真里采：跑传送带，用**真值位置**给每个掩码打标签（标注，不是模型输入），
存成 (特征, 标签)。真机上把这一步换成"人工标一批 + 微调"即可，接口不变。

用法:
    python scripts/train_recognizer.py --frames 120 --out runs/seg-head/recognizer.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

KINDS = ("apple", "strawberry", "banana", "pear")


def collect(frames: int, seed: int, camera: str) -> tuple[np.ndarray, np.ndarray]:
    import gymnasium as gym

    import fruitsort.sim.tasks  # noqa: F401
    from fruitsort.perception.camera import CameraModel
    from fruitsort.perception.geometry import obb_from_mask
    from fruitsort.perception.pipeline import PerceptionPipeline
    from fruitsort.perception.recognition import features

    cam = CameraModel.load_or_none(camera)
    if cam is None:
        raise SystemExit(f"缺少 camera_{camera}.json，先跑 scripts/fit_camera_model.py")

    env = gym.make("FruitSortStream-v1", obs_mode="rgbd", control_mode="pd_ee_pose",
                   sim_backend="cpu", robot_uids="panda_wristcam", active_kinds=KINDS,
                   belt_speed_range=(0.05, 0.12), slots_per_kind=2, gap_range=(1.0, 2.2))
    u = env.unwrapped
    pipe = PerceptionPipeline(camera=cam, belt_z=float(u.belt_top_z), dt=1 / 30)
    pipe.min_confidence = 0.0              # 收集阶段不过滤，让识别头自己学
    X, y = [], []
    obs, _ = env.reset(seed=seed)
    for t in range(frames + 30):
        obs, *_ = env.step(np.zeros(7))
        if t < 30:
            continue
        gt = []
        for k in u.active_kinds:
            for s in u._slots[k]:
                if not s["live"]:
                    continue
                p = s["item"]["actor"].pose.p[0].cpu().numpy()
                if -1.35 <= p[0] <= 0.60 and abs(p[1]) < 0.25 and p[2] > 0.40:
                    gt.append((k, np.asarray(p, dtype=np.float64)))
        if not gt:
            continue
        # 直接用流水线的中间结果（掩码 + 几何），不走识别
        sd = obs["sensor_data"]["overhead"]
        rgb = sd["rgb"][0].cpu().numpy()
        d = sd["depth"][0].cpu().numpy()
        d = (d[..., 0] if d.ndim == 3 else d).astype(np.float64) / 1000.0
        pts = cam.unproject(d)
        from fruitsort.perception.geometry import robust_plane_z
        pz = robust_plane_z(pts, float(u.belt_top_z))
        for seg in pipe.seg.segment(rgb, pts, pz):
            g = obb_from_mask(pts, seg.mask, pz)
            if g is None:
                continue
            c = g.center
            best, bd = None, 1e9
            for k, p in gt:
                dd = float(np.linalg.norm(c[:2] - p[:2]))
                if dd < bd:
                    best, bd = k, dd
            if best is None or bd > 0.08:
                continue                       # 对不上真值的掩码不参与训练
            X.append(features(g, seg).vec())
            y.append(KINDS.index(best))
    env.close()
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--seed", type=int, default=80000)
    ap.add_argument("--camera", default="overhead")
    ap.add_argument("--out", default="runs/seg-head/recognizer.pt")
    ap.add_argument("--epochs", type=int, default=400)
    args = ap.parse_args()

    X, y = collect(args.frames, args.seed, args.camera)
    if len(X) < 50:
        raise SystemExit(f"样本太少（{len(X)}）")
    n_tr = int(0.75 * len(X))
    perm = np.random.default_rng(0).permutation(len(X))
    X, y = X[perm], y[perm]
    Xtr, ytr, Xva, yva = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6

    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(8, 48), nn.GELU(), nn.Linear(48, 48), nn.GELU(), nn.Linear(48, 4))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    lossf = nn.CrossEntropyLoss()
    xt = torch.from_numpy((Xtr - mu) / sd); yt = torch.from_numpy(ytr)
    xv = torch.from_numpy((Xva - mu) / sd); yv = torch.from_numpy(yva)
    for ep in range(args.epochs):
        net.train(); opt.zero_grad()
        loss = lossf(net(xt), yt); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        tr_acc = (net(xt).argmax(1) == yt).float().mean().item()
        va_acc = (net(xv).argmax(1) == yv).float().mean().item()

    # 规则识别在同一批数据上的准确率（基线）
    from fruitsort.perception.recognition import RuleRecognizer
    rule = RuleRecognizer()
    rule_pred = rule.predict_batch(Xva)
    rule_acc = float((rule_pred == yva).mean())

    print(f"\n样本 {len(X)}（训练 {len(Xtr)} / 验证 {len(Xva)}）")
    print(f"训练集 acc {tr_acc*100:.1f}%   验证集 acc {va_acc*100:.1f}%   规则基线 {rule_acc*100:.1f}%")
    with torch.no_grad():
        pred = net(xv).argmax(1).numpy()
    print("混淆（行=真值，列=预测）:")
    cm = np.zeros((4, 4), int)
    for a, b in zip(yva, pred):
        cm[a, b] += 1
    print("        " + "".join(f"{k[:4]:>7s}" for k in KINDS))
    for i, k in enumerate(KINDS):
        print(f"{k:8s}" + "".join(f"{v:7d}" for v in cm[i]))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(mu=mu, sd=sd, state=net.state_dict(),
                    feat=("extent", "width", "height", "pca_ratio", "area_px", "hue", "sat", "val"),
                    kinds=KINDS), out)
    print(f"\n保存到 {out}")


if __name__ == "__main__":
    main()
