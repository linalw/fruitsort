#!/usr/bin/env python
"""评测残差控制器：纯模型闭环 vs 模型 + 学习式残差。

两者都**不使用**任何近距伺服 / 几何夹爪 / 确定性搬运，只把世界坐标目标
转成 `pd_ee_pose` 动作；残差只贡献 ≤3 cm 的目标偏移和夹爪 logit 偏置。

用法:
    source scripts/env.sh maniskill3
    python scripts/eval_residual.py --residual runs/residual-v1/ckpt.pt --episodes 5
    python scripts/eval_residual.py --residual none --episodes 5      # 只跑基线
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

import fruitsort.sim.tasks  # noqa: F401
from fruitsort.eval import format_summary, summarize
from fruitsort.learning.bc import Normalizer, build_obs
from fruitsort.learning.residual import load_residual, onehot
from fruitsort.models.vision_policy import MultiViewTemporalPolicy, PolicyConfig
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


def classify(info, grasped_ever, closed_ever, min_dist, steps, max_steps) -> str | None:
    if info.get("success"):
        return None
    if info.get("dropped"):
        return "dropped"
    if not grasped_ever:
        if not closed_ever:
            return "never_closed" if min_dist < 0.08 else "never_reached"
        return "closed_missed"
    if steps >= max_steps - 1:
        return "carry_place_timeout"
    return "other"


def run_group(use_residual, args, model, cfg, ns, na, residual) -> list[dict]:
    dev = args.device
    tag = "模型+残差" if use_residual else "纯模型"
    results = []
    lat_model, lat_res = [], []
    for ep in range(args.episodes):
        env = gym.make(
            "FruitSort-v1", obs_mode="rgbd", control_mode="pd_ee_pose", sim_backend="cpu",
            robot_uids="panda_wristcam", active_kinds=tuple(args.kinds),
            belt_speed_range=(args.belt_speed, args.belt_speed),
            spawn_x=args.spawn_x, spawn_spacing=args.spawn_spacing,
            instruction_kind=None if args.instruction == "all" else args.instruction)
        obs, _ = env.reset(seed=args.seed + ep)
        u = env.unwrapped

        sd0 = obs["sensor_data"]
        _oh = prep_img(sd0["overhead"]["rgb"][0].cpu().numpy(), cfg.img_size)
        _wr = prep_img(sd0["hand_camera"]["rgb"][0].cpu().numpy(), cfg.img_size)
        _st = build_obs(u)
        with torch.no_grad():
            _, _, _sel = model(
                torch.from_numpy(np.stack([_oh] * cfg.hist)[None]).float().to(dev),
                torch.from_numpy(np.stack([_wr] * cfg.hist)[None]).float().to(dev),
                torch.from_numpy(ns.obs(np.stack([_st] * cfg.hist).astype(np.float32))[None]).float().to(dev),
                torch.tensor([INSTR_ID[args.instruction]], device=dev))
        kind = ID_KIND.get(int(_sel.argmax(-1).item()), "apple")
        if kind not in u.fruit_actors:
            kind = args.instruction if args.instruction in u.fruit_actors else list(u.fruit_actors)[0]
        slot = KIND_SLOT.get(kind, 0)

        oh_h, wr_h, st_h = [], [], []
        info, t = {}, 0
        grasped_ever, closed_ever, min_dist = False, False, 9.9
        dsum = np.zeros(3); dmax = 0.0; n_res = 0
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
            actor = u.fruit_actors[kind]
            fp = actor.pose.p[0].cpu().numpy().astype(np.float64)
            dist = float(np.linalg.norm(fp - tcp))
            min_dist = min(min_dist, dist)
            g_now = bool(u.agent.is_grasping(actor)[0])
            grasped_ever = grasped_ever or g_now

            t_a = time.perf_counter()
            st_np = ns.obs(np.stack(st_h).astype(np.float32))[None]
            with torch.no_grad():
                pred, _, _ = model(
                    torch.from_numpy(np.stack(oh_h)[None]).float().to(dev),
                    torch.from_numpy(np.stack(wr_h)[None]).float().to(dev),
                    torch.from_numpy(st_np).float().to(dev),
                    torch.tensor([INSTR_ID[args.instruction]], device=dev))
            if dev == "cuda":
                torch.cuda.synchronize()
            lat_model.append((time.perf_counter() - t_a) * 1e3)

            p4 = pred[0, 0].cpu().numpy()
            model_tgt = np.clip(na.unact(p4[:3]), -2, 2).astype(np.float64)
            model_logit = float(p4[3])

            tgt, logit = model_tgt, model_logit
            if use_residual:
                t_b = time.perf_counter()
                with torch.no_grad():
                    s = torch.from_numpy(st_np[0, -1][None]).float().to(dev)
                    mt = torch.from_numpy(model_tgt.astype(np.float32)[None]).float().to(dev)
                    oh = torch.from_numpy(onehot(slot)[None]).float().to(dev)
                    dxyz, dgrip = residual(s, mt, oh)
                if dev == "cuda":
                    torch.cuda.synchronize()
                lat_res.append((time.perf_counter() - t_b) * 1e3)
                dxyz_np = dxyz[0].cpu().numpy().astype(np.float64)
                tgt = model_tgt + dxyz_np
                logit = model_logit + float(dgrip[0].item())
                dsum += dxyz_np; dmax = max(dmax, float(np.linalg.norm(dxyz_np))); n_res += 1

            grip = -1.0 if logit > 0 else 1.0
            closed_ever = closed_ever or (grip < 0 and dist < 0.10)
            obs, _, term, trunc, info = env.step(to_cmd(tgt, tcp, grip))
            if bool(info.get("success") or info.get("fail")):
                break

        ok = bool(info.get("success", False))
        failure = classify(info, grasped_ever, closed_ever, min_dist, t + 1, args.max_steps)
        results.append({"success": ok, "duration_s": (t + 1) / 20.0, "failure": failure,
                        "steps": t + 1, "grasped": grasped_ever, "min_dist": round(min_dist, 3),
                        "residual_mean_m": (dsum / max(n_res, 1)).round(4).tolist() if n_res else None,
                        "residual_max_m": round(dmax, 4) if n_res else None})
        print(f"[{'✓' if ok else '✗'}] {tag} ep {ep:02d} steps={t + 1} "
              f"grasp={grasped_ever} min_d={min_dist:.3f} fail={failure}")
        env.close()

    if lat_model:
        print(f"  延迟 {tag}: 模型 {np.mean(lat_model):.2f} ms"
              + (f" + 残差 {np.mean(lat_res):.3f} ms"
                 f"（p95 {np.percentile(lat_res, 95):.3f}）" if lat_res else ""))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/vpolicy-4c-v3/ckpt.pt")
    ap.add_argument("--residual", default="runs/residual-v1/ckpt.pt",
                    help="残差 ckpt；none 表示只跑基线")
    ap.add_argument("--kinds", nargs="+", default=["apple", "strawberry"])
    ap.add_argument("--instruction", default="apple",
                    choices=["all", "apple", "strawberry", "banana", "pear"])
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--belt-speed", type=float, default=0.08)
    ap.add_argument("--spawn-x", type=float, default=-0.55)
    ap.add_argument("--spawn-spacing", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=400)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(Path(args.ckpt), map_location="cpu", weights_only=False)
    cfg = PolicyConfig(**ck["cfg"])
    model = MultiViewTemporalPolicy(cfg).to(args.device)
    model.load_state_dict(ck["model"]); model.eval()
    ns = Normalizer.from_state(ck["norm_state"]); na = Normalizer.from_state(ck["norm_act"])

    residual = None
    if args.residual.lower() not in ("none", ""):
        residual, rck = load_residual(args.residual, args.device)
        print(f"残差: {args.residual}  参数量 {sum(p.numel() for p in residual.parameters())}  "
              f"max_pos={residual.cfg.max_pos} m, max_grip={residual.cfg.max_grip}")

    out = {"ckpt": str(args.ckpt), "residual": str(args.residual), "instruction": args.instruction,
           "episodes": args.episodes, "belt_speed": args.belt_speed, "spawn_x": args.spawn_x,
           "seed": args.seed, "groups": {}}
    base = run_group(False, args, model, cfg, ns, na, None)
    out["groups"]["model_only"] = {"summary": summarize(base), "episodes": base}
    if residual is not None:
        res = run_group(True, args, model, cfg, ns, na, residual)
        out["groups"]["model_residual"] = {"summary": summarize(res), "episodes": res}

    print("\n===== 纯模型（无任何几何伺服/脚本执行器）=====")
    print(format_summary(out["groups"]["model_only"]["summary"]))
    if residual is not None:
        print("===== 模型 + 学习式残差 =====")
        print(format_summary(out["groups"]["model_residual"]["summary"]))

    d = ROOT / "runs" / f"{time.strftime('%Y-%m-%d-%H%M')}-residual-eval"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {d / 'metrics.json'}")


if __name__ == "__main__":
    main()
