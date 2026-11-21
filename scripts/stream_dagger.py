#!/usr/bin/env python
"""在连续作业里做 DAgger：**模型跑 rollout，专家给标签**。

为什么需要
----------
`act-wide-4c` 是单发策略：连续作业时它抓起第一件后就"冻结"（见 docs 6.26.3）。
纯专家示教（stream_eval --record）只包含专家自己的状态分布，模型**会去到的**
那些错误状态（刚放完货还在料框上方、目标正在逃逸、多件混杂）从来没被标注过。

DAgger 的做法：让**模型**驱动环境，每一步都用规则专家对**当前状态**给出动作标签，
这样标签就覆盖了模型自己的状态分布。

用法:
    python scripts/stream_dagger.py --ckpt runs/act-wide-4c/ckpt.pt \
        --seconds 300 --out data/fruitsort/datasets/dagger-1 --belt 0.08
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np

import fruitsort.sim.tasks  # noqa: F401
from fruitsort.learning.bc import build_obs
from fruitsort.policies.act_runner import ActPureRunner
from fruitsort.policies.scripted import ScriptedSorter

ROOT = Path(__file__).resolve().parent.parent
KINDS = ("apple", "strawberry", "banana", "pear")


def prep_images(obs):
    sd = obs["sensor_data"]
    oh = sd["overhead"]["rgb"][0].cpu().numpy()
    oh = cv2.resize(oh, (128, 128), interpolation=cv2.INTER_AREA)
    wr = sd["hand_camera"]["rgb"][0].cpu().numpy()
    wr = cv2.resize(wr, (128, 128), interpolation=cv2.INTER_AREA)
    return (cv2.imencode(".jpg", oh[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes(),
            cv2.imencode(".jpg", wr[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes())


class TargetPicker:
    """目标选择（与 stream_eval.py 一致）：'还来得及'窗口内选最靠前的一件。"""

    def __init__(self, u, kinds=KINDS):
        self.u, self.kinds = u, tuple(kinds)
        self.target = None

    def live(self):
        out = []
        for k in self.kinds:
            for slot in self.u._slots[k]:
                if slot["live"]:
                    out.append(slot["item"])
        return out

    def repoint(self):
        tcp = self.u.agent.tcp_pose.p[0].cpu().numpy()
        for k in self.kinds:
            live = [s["item"] for s in self.u._slots[k] if s["live"]]
            if not live:
                continue
            if self.target is not None and self.target["kind"] == k:
                self.u.fruit_actors[k] = self.target["actor"]
            else:
                self.u.fruit_actors[k] = min(
                    live, key=lambda it: np.linalg.norm(
                        it["actor"].pose.p[0].cpu().numpy() - tcp))["actor"]

    def pick(self):
        tcp = self.u.agent.tcp_pose.p[0].cpu().numpy()
        urg, late = [], []
        for it in self.live():
            p = it["actor"].pose.p[0].cpu().numpy()
            if np.linalg.norm(p - tcp) < 0.05:
                continue
            if -0.60 <= p[0] <= -0.05:
                urg.append((p[0], it))
            elif -0.05 < p[0] <= 0.15:
                late.append((p[0], it))
        if urg:
            return max(urg, key=lambda t: t[0])[1]
        if late:
            return max(late, key=lambda t: t[0])[1]
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/act-wide-4c/ckpt.pt", help="驱动 rollout 的模型")
    ap.add_argument("--out", required=True)
    ap.add_argument("--belt", type=float, default=0.08)
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--gap-min", type=float, default=0.6)
    ap.add_argument("--gap-max", type=float, default=1.2)
    ap.add_argument("--scale-min", type=float, default=0.85)
    ap.add_argument("--scale-max", type=float, default=1.15)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--control-freq", type=int, default=30)
    ap.add_argument("--min-frames", type=int, default=20)
    ap.add_argument("--seed", type=int, default=61000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = gym.make(
        "FruitSortStream-v1", obs_mode="rgbd", control_mode="pd_ee_pose", sim_backend="cpu",
        robot_uids="panda_wristcam", active_kinds=KINDS,
        belt_speed_range=(args.belt, args.belt), slots_per_kind=args.slots,
        gap_range=(args.gap_min, args.gap_max), scale_range=(args.scale_min, args.scale_max),
        sim_config=dict(control_freq=args.control_freq,
                        sim_freq=120 if args.control_freq == 30 else 100))
    runner = ActPureRunner(env, args.ckpt, "apple", fb=0.5, grip_ema=0.6,
                           carry_latch=True, carry_max_step=0.02)
    obs, _ = env.reset(seed=args.seed)
    u = env.unwrapped
    picker = TargetPicker(u)
    buf = {"obs": [], "act": [], "oh": [], "wr": [], "kind": None, "id": None}
    n_saved = 0
    n_frames = 0

    def flush():
        nonlocal n_saved, n_frames, buf
        if len(buf["obs"]) >= args.min_frames and buf["id"] is not None:
            np.savez_compressed(
                out / f"ep_{n_saved:04d}.npz",
                obs=np.stack(buf["obs"]), action=np.stack(buf["act"]),
                seed=args.seed, kinds=np.array(KINDS),
                instruction=np.array(buf["kind"]),
                overhead_jpg=np.array(buf["oh"], dtype=object),
                wrist_jpg=np.array(buf["wr"], dtype=object))
            n_saved += 1
            n_frames += len(buf["obs"])
        buf = {"obs": [], "act": [], "oh": [], "wr": [], "kind": None, "id": None}

    t0 = time.time()
    max_steps = int(args.seconds * args.control_freq)
    for t in range(max_steps):
        picker.repoint()
        if picker.target is not None and not any(
                picker.target is s["item"] for s in u._slots[picker.target["kind"]] if s["live"]):
            picker.target = None
        if picker.target is None:
            picker.target = picker.pick()
            if picker.target is not None:
                runner.kind = picker.target["kind"]
                runner.reset()
                u.fruit_actors[picker.target["kind"]] = picker.target["actor"]
                sorter = ScriptedSorter(env, only_kind=picker.target["kind"])

        if picker.target is None:
            act = np.array([0.5, 0.0, 0.55, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            obs, _, term, trunc, info = env.step(act)
            if bool(term) or bool(trunc):
                break
            continue

        # 专家对**当前状态**给标签（FSM 是反应式的，直接读当前场景）
        _ = sorter.act()
        label = np.asarray(sorter.last_target, dtype=np.float32)   # [x, y, z, grip]

        if picker.target["id"] != buf["id"]:
            flush()
            buf["id"], buf["kind"] = picker.target["id"], picker.target["kind"]

        buf["obs"].append(build_obs(u, buf["kind"]))
        buf["act"].append(label)
        oh, wr = prep_images(obs)
        buf["oh"].append(oh)
        buf["wr"].append(wr)

        act = runner.act(obs)          # **模型**驱动
        obs, _, term, trunc, info = env.step(act)
        if bool(term) or bool(trunc):
            break
    flush()

    st = u.stream_stats
    print(f"DAgger 采集完成：{n_saved} 条片段 / {n_frames} 帧 → {out}")
    print(f"  rollout 上料 {st['spawned']} 送达 {st['delivered']} 漏过 {st['missed']} "
          f"掉落 {st['dropped']} | 用时 {(time.time()-t0)/60:.1f} min")
    env.close()


if __name__ == "__main__":
    main()
