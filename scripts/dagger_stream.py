#!/usr/bin/env python
"""连续流里的 DAgger 采集：**学生跑闭环**，专家给访问到的状态打标。

为什么需要
    act-dual2 的 val 已经到了 0.036，但闭环里手臂在离果 5–7 cm 处悬停（trace 见 docs §30）。
    这类"离线好、闭环崩"的问题标准解法是 DAgger：让学生去访问它自己会到的状态，
    由专家在这些状态上给出"该下探/该闭爪/该搬框"的标签。

标签规则（与 `dagger_act.py` 同一套马尔可夫规则，无记忆）：
    - 已抓住：搬到对应料框上方（z=0.60、闭爪），到框后下降（z=0.50、开爪）
    - 未抓住：追拦截点（果位 + v*0.10）；x/y 对齐且在抓取区 → 下探到果心并闭爪
    - 否则：抬到果心 +10 cm 接近

数据里保存的是**感知状态**（`build_obs(..., actors=objs)`）+ 全局/ROI/腕部三路图，
与部署同源；`instruction` 固定为 `--kind`（一局只追这一类，其余类照常在场当上下文）。

用法:
    python scripts/dagger_stream.py --ckpt runs/act-dual2/ckpt.pt --kind apple --episodes 3 \
        --out data/datasets/dagger-apple
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np

import fruitsort.sim.tasks  # noqa: F401
from fruitsort.learning.bc import build_obs
from fruitsort.perception.adapter import RGBDSource
from fruitsort.perception.roi import RoiCrop
from fruitsort.policies.act_runner import ActPureRunner
from fruitsort.sim.conveyor import BIN_LAYOUT

ROOT = Path(__file__).resolve().parent.parent
KINDS = ("apple", "strawberry", "banana", "pear")


def _load_ctrl():
    spec = importlib.util.spec_from_file_location("se", str(ROOT / "scripts" / "stream_eval.py"))
    se = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(se)
    return se


def jpg(x):
    return cv2.imencode(".jpg", x[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def expert_label(actor, kind, u) -> np.ndarray:
    """马尔可夫专家：返回 4 维世界目标 (x,y,z) + 夹爪 (+1 开 / -1 闭)。

    `actor` 可能是感知实例（.pos/.vel）或仿真 actor（.pose.p/.linear_velocity）；
    接触力判定一律用**真值 actor**（相当于真机指端 F/T）。
    """
    tcp = u.agent.tcp_pose.p[0].cpu().numpy().astype(np.float64)
    if hasattr(actor, "pose"):
        fp = actor.pose.p[0].cpu().numpy().astype(np.float64)
        fv = actor.linear_velocity[0].cpu().numpy().astype(np.float64)
    else:
        fp = np.asarray(getattr(actor, "pos"), dtype=np.float64)
        fv = np.asarray(getattr(actor, "vel", np.zeros(3)), dtype=np.float64)
    bx, by = BIN_LAYOUT[kind]
    _real = u.fruit_actors.get(kind)
    # 感知模式下槽位可能被替身占用（没有 _bodies）→ 跳过接触查询
    if _real is not None and not hasattr(_real, "_bodies"):
        _real = None
    holding = bool(u.agent.is_grasping(_real)[0]) if _real is not None else False
    if holding:
        over = abs(tcp[0] - bx) < 0.06 and abs(tcp[1] - by) < 0.06
        if over and tcp[2] < 0.58:
            return np.array([bx, by, 0.50, 1.0], dtype=np.float32)
        return np.array([bx, by, 0.60, -1.0], dtype=np.float32)
    x = float(np.clip(fp[0] + fv[0] * 0.10, u.pick_zone[0] - 0.02, u.pick_zone[1]))
    aligned = abs(tcp[0] - x) < 0.025 and abs(tcp[1] - fp[1]) < 0.025
    if aligned and u.pick_zone[0] <= fp[0] <= u.pick_zone[1]:
        return np.array([x, fp[1], fp[2], -1.0], dtype=np.float32)
    return np.array([x, fp[1], fp[2] + 0.10, 1.0], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kind", required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--belt", type=float, default=0.08)
    ap.add_argument("--gap", type=float, nargs=2, default=(3.0, 4.0))
    ap.add_argument("--seed", type=int, default=63000)
    ap.add_argument("--perception", default="rgbd", choices=["rgbd", "fused", "oracle"])
    ap.add_argument("--roi-m", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    se = _load_ctrl()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    roi = RoiCrop(width_m=args.roi_m, out_size=128) if args.roi_m > 0 else None
    n_saved = 0
    for ep in range(args.episodes):
        env = gym.make("FruitSortStream-v1", obs_mode="rgbd", control_mode="pd_ee_pose",
                       sim_backend="cpu", robot_uids="panda_wristcam", active_kinds=KINDS,
                       belt_speed_range=(args.belt, args.belt), slots_per_kind=1,
                       gap_range=tuple(args.gap), scale_range=(0.85, 1.15), spawn_x=-0.85,
                       sim_config=dict(control_freq=30, sim_freq=120))
        obs, _ = env.reset(seed=args.seed + ep)
        u = env.unwrapped
        src = None
        if args.perception == "rgbd":
            src = RGBDSource(u, dt=1 / 30)
        elif args.perception == "fused":
            from fruitsort.perception.adapter import FusedSource
            src = FusedSource(u, dt=1 / 30)
        runner = ActPureRunner(env, args.ckpt, args.kind, fb=0.7, grip_ema=0.7,
                               carry_latch=True, carry_max_step=0.02, grasp_verify=True,
                               max_step_map={"apple": 0.12, "strawberry": 0.12,
                                             "pear": 0.10, "banana": 0.06},
                               carry_step_map={"apple": 0.04, "strawberry": 0.04,
                                               "pear": 0.03, "banana": 0.02})
        ctrl = se.StreamController(env, runner, u, objs=(None if src is None else {}),
                                   only_kind=args.kind, in_dist=False, scripted_kinds=[],
                                   engage_x=-0.45, perception_targets=(src is not None),
                                   decline_yaw=0.0, src=src)
        rec_o, rec_a, rec_oh, rec_roi, rec_wr = [], [], [], [], []
        for t in range(int(args.seconds * 30)):
            objs = None
            if src is not None:
                objs = src.update(obs, held_kind=(ctrl.runner.kind if ctrl.runner.grasping else None))
                ctrl.objs = objs
                ctrl.runner.set_actors(objs)
            act = ctrl.act(obs)                      # 学生动作（会进入自己的失败状态）
            it = ctrl.target
            if it is not None and it["kind"] == args.kind:
                rec_o.append(build_obs(u, args.kind, actors=(objs if objs is not None else None)))
                rec_a.append(expert_label(it["actor"], args.kind, u))
                sd = obs["sensor_data"]
                oh = sd["overhead"]["rgb"][0].cpu().numpy()
                rec_oh.append(jpg(cv2.resize(oh, (128, 128), interpolation=cv2.INTER_AREA)))
                if roi is not None:
                    _a = it["actor"]
                    xy = (_a.pose.p[0].cpu().numpy()[:2] if hasattr(_a, "pose")
                          else np.asarray(getattr(_a, "pos"))[:2])
                    rec_roi.append(jpg(roi.crop(oh, xy)))
                wr = sd["hand_camera"]["rgb"][0].cpu().numpy()
                rec_wr.append(jpg(cv2.resize(wr, (128, 128), interpolation=cv2.INTER_AREA)))
            obs, _, term, trunc, info = env.step(act)
            if term or trunc:
                break
        if rec_o:
            payload = dict(obs=np.stack(rec_o).astype(np.float32),
                           action=np.stack(rec_a).astype(np.float32),
                           seed=args.seed + ep, kinds=np.array(KINDS),
                           instruction=np.array(args.kind),
                           overhead_jpg=np.array(rec_oh, dtype=object),
                           wrist_jpg=np.array(rec_wr, dtype=object))
            if rec_roi:
                payload["roi_jpg"] = np.array(rec_roi, dtype=object)
            np.savez_compressed(out / f"ep_{ep:04d}.npz", **payload)
            n_saved += 1
        st = u.stream_stats
        print(f"[ep{ep}] 送达 {st['delivered']} 漏过 {st['missed']} 掉落 {st['dropped']} "
              f"| 记录 {len(rec_o)} 帧", flush=True)
        env.close()
    print(f"✓ {n_saved} 局 DAgger 数据 → {out}")


if __name__ == "__main__":
    main()
