#!/usr/bin/env python
"""用脚本专家采集示教数据（BC / ACT / DP 的训练集）。

观测（49 维）：
    tcp_pose(7) + qpos(9) + belt_speed(1)
    + 4 类 × [pos(3), vel(3), active(1), held(1)]
动作（7 维）：pd_ee_pose [x, y, z, ex, ey, ez, gripper]

用法:
    python scripts/record_demos.py --kinds apple strawberry --episodes 40
    python scripts/record_demos.py --kinds apple strawberry banana pear --episodes 20 --belt-speed 0.04
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np

import fruitsort.sim.tasks  # noqa: F401
from fruitsort.policies.scripted import ScriptedSorter

ROOT = Path(__file__).resolve().parent.parent
ALL = ("apple", "strawberry", "banana", "pear")


def obs_vector(u) -> np.ndarray:
    tcp = u.agent.tcp_pose.raw_pose[0].cpu().numpy()          # (7) p + q
    qpos = u.agent.robot.get_qpos()[0].cpu().numpy()           # (9)
    belt = np.array([float(u.belt_speed[0])], dtype=np.float32)
    feats = []
    for k in ALL:
        if k in u.fruit_actors:
            a = u.fruit_actors[k]
            p = a.pose.p[0].cpu().numpy()
            v = a.linear_velocity[0].cpu().numpy()
            held = float(np.linalg.norm(p - tcp[:3]) < 0.08)
            active = 1.0
        else:
            p = np.zeros(3); v = np.zeros(3); held = 0.0; active = 0.0
        feats.append(np.concatenate([p, v, [active, held]]))
    return np.concatenate([tcp, qpos, belt, *feats]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["apple", "strawberry"])
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--belt-speed", type=float, default=None, help="缺省 0.04–0.08 随机")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0, help="执行动作上加的高斯噪声（米），用于鲁棒化")
    ap.add_argument("--robot", default="panda")
    ap.add_argument("--images", action="store_true", help="同时保存 overhead + 腕部图像（JPEG 字节）")
    ap.add_argument("--instruction", default="all", choices=["all", "apple", "strawberry", "banana", "pear"],
                    help="指令：只抓某一类（默认全部）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tag = time.strftime("%Y%m%d-%H%M")
    out = Path(args.out) if args.out else ROOT / "data" / "datasets" / f"fruitsort-{tag}"
    out.mkdir(parents=True, exist_ok=True)

    speed_range = ((args.belt_speed, args.belt_speed) if args.belt_speed is not None
                   else (0.04, 0.08))
    env = gym.make(
        "FruitSort-v1", obs_mode="rgbd" if args.images else "state",
        control_mode="pd_ee_pose", sim_backend="cpu", robot_uids=args.robot,
        active_kinds=tuple(args.kinds), belt_speed_range=speed_range,
        instruction_kind=None if args.instruction == "all" else args.instruction,
    )

    rng = np.random.default_rng(args.seed + 999)
    records = []
    n_ok = 0
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        policy = ScriptedSorter(env, only_kind=None if args.instruction == 'all' else args.instruction)
        u = env.unwrapped
        obs_list, act_list = [], []
        over_list, wrist_list = [], []
        info = {}
        for t in range(args.max_steps):
            o = obs_vector(u)
            if args.images:
                sd = obs["sensor_data"]
                oh = sd["overhead"]["rgb"][0].cpu().numpy()
                oh = cv2.resize(oh, (128, 128), interpolation=cv2.INTER_AREA)
                over_list.append(cv2.imencode(".jpg", oh[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes())
                if "hand_camera" in sd:
                    wr = sd["hand_camera"]["rgb"][0].cpu().numpy()
                    wr = cv2.resize(wr, (128, 128), interpolation=cv2.INTER_AREA)
                    wrist_list.append(cv2.imencode(".jpg", wr[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes())
            a = policy.act()
            obs_list.append(o)
            act_list.append(policy.last_target)
            if args.noise > 0:
                a = a.copy()
                a[:3] += rng.normal(0, args.noise, 3).astype(np.float32)
            obs, _, term, trunc, info = env.step(a)
            if bool(info.get("success") or info.get("fail")):
                break
            if t > 900:
                break
        ok = bool(info.get("success", False))
        n_ok += ok
        if ok:
            ep_file = out / f"ep_{ep:04d}.npz"
            payload = dict(obs=np.stack(obs_list), action=np.stack(act_list),
                           seed=args.seed + ep, kinds=np.array(args.kinds),
                           instruction=np.array(args.instruction))
            if args.images:
                payload["overhead_jpg"] = np.array(over_list, dtype=object)
                payload["wrist_jpg"] = np.array(wrist_list, dtype=object)
            np.savez_compressed(ep_file, **payload)
            records.append({"file": ep_file.name, "steps": len(obs_list),
                            "belt_speed": float(u.belt_speed[0])})
        print(f"[{'✓' if ok else '✗'}] ep {ep:03d} steps={len(obs_list)} ok={n_ok}/{ep + 1}", flush=True)

    meta = {
        "kinds": args.kinds,
        "episodes": len(records),
        "success_rate": n_ok / args.episodes,
        "obs_dim": 49,
        "action_dim": 4,
        "belt_speed_range": speed_range,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "episodes_meta": records,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    env.close()
    print(f"\n✓ 保存 {len(records)} 条成功示教到 {out}（成功率 {n_ok}/{args.episodes}）")


if __name__ == "__main__":
    main()
