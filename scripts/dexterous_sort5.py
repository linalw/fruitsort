#!/usr/bin/env python
"""Panda + D'Claw：用现成的 `ScriptedSorter`（pd_ee_pose）跑任务级分拣。

思路
----
合并 URDF 里加了一个虚拟 TCP `panda_tcp`，与 Panda 原 `panda_hand_tcp` **完全同位**
（FK 验证过）。所以 `pd_ee_pose` 的位姿映射与平行夹爪一致，可以直接复用
`ScriptedSorter` 的状态机；只需两处适配：

1. **TCP 高度补偿**：D'Claw 的指尖在虚拟 TCP 下方 0.186 m，所以命令的 TCP 要抬高同样的量，
   指尖才会落到水果上；
2. **手指动作**：把 `gripper∈[-1,1]`（+1 张开 / −1 闭合）映射成 D'Claw 的 9 维归一化目标。
   张开用 +0.2（指尖跨距 10.8 cm、占地 13 cm；−0.85 时占地 24 cm 会挡住来料），
   闭合到 +0.45（跨距 6.6 cm，可夹住 ~8 cm 水果）。

另外两点来自实测的机械约束：
- 基座抬高 0.30 m（腕到指尖 0.29 m，标准座高下静息姿态手指埋在传送带下方）；
- 合并 URDF 去掉腕/前臂 4 个 link 的碰撞体（长手指会与 Panda 前臂自碰撞，实测 1.7 kN）。

用法:
    python scripts/dexterous_sort5.py --kinds apple --episodes 1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

import fruitsort.sim.tasks  # noqa: F401
import fruitsort.sim.agents.panda_dclaw  # noqa: F401
import fruitsort.sim.agents.panda_allegro  # noqa: F401
from fruitsort.policies.scripted import HOME_EULER, PolicyConfig, ScriptedSorter

ROOT = Path(__file__).resolve().parent.parent
KINDS = ("apple", "strawberry", "banana", "pear")


class DClawSorter(ScriptedSorter):
    def __init__(self, env, *, tcp_dz: float = 0.186, hand_open: float = 0.20,
                 hand_close: float = 0.45, base_z: float = 0.0, **kw):
        super().__init__(env, **kw)
        self.base_z = float(base_z)   # pd_ee_pose 的位姿是基座系，抬座后要减掉
        self.tcp_dz = float(tcp_dz)
        self.hand_open = float(hand_open)
        self.hand_close = float(hand_close)

    def _action(self, target, gripper: float = 1.0, max_step: float | None = None):
        raw = np.asarray(target, dtype=np.float64).copy()
        raw[2] += self.tcp_dz                      # 指尖在虚拟 TCP 下方 0.186 m
        tcp = self._tcp()
        err = raw - tcp
        dist = float(np.linalg.norm(err))
        self.last_target = np.array([raw[0], raw[1], raw[2], gripper], dtype=np.float32)
        step = self.cfg.max_step if max_step is None else max_step
        tgt = raw if dist <= step else tcp + err / dist * step
        x, y, z = tgt
        # 世界系 → 基座系：基座在 (0, -0.5, base_z) 且绕 z 转 +90°
        cmd = np.array([y + 0.5, -x, z - self.base_z], dtype=np.float32)
        tilt = self.cfg.grasp_tilt
        eul = (HOME_EULER[0] - tilt, HOME_EULER[1], HOME_EULER[2] + self.cfg.grasp_yaw)
        t = (1.0 - float(gripper)) / 2.0           # +1 张开 → 0；−1 闭合 → 1
        hand = self.hand_open + t * (self.hand_close - self.hand_open)
        n_hand = int(self.env.action_space.shape[0]) - 6      # 6 维位姿 + N 维手指
        return np.array([*cmd, *eul, *([hand] * n_hand)], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="panda_allegro",
                    choices=["panda_allegro", "panda_dclaw", "panda_wristcam"])
    ap.add_argument("--kinds", nargs="+", default=list(KINDS))
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--belt", type=float, default=0.08)
    ap.add_argument("--spawn-x", type=float, default=-0.55)
    ap.add_argument("--spawn-spacing", type=float, default=0.45)
    ap.add_argument("--control-freq", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--base-z", type=float, default=0.0)
    ap.add_argument("--tcp-dz", type=float, default=0.0)
    ap.add_argument("--hand-open", type=float, default=0.0)
    ap.add_argument("--hand-close", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--verbose-states", action="store_true")
    ap.add_argument("--grasp-depth", type=float, default=-0.01,
                    help="相对水果中心的额外下探（负值=更深，让手指包住果身）")
    ap.add_argument("--close-steps", type=int, default=12)
    ap.add_argument("--grasp-tilt", type=float, default=0.0,
                    help="抓取倾角：0=俯视；~1.5=侧向（手掌竖直，水平捏取）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    # 更长的末端（腕到指尖 0.29 m）需要更高的安装座；任务本身不变
    from fruitsort.sim.tasks.fruit_sort import FruitSortEnv
    FruitSortEnv.robot_base = (0.0, -0.50, float(args.base_z))

    env = gym.make(
        "FruitSort-v1", obs_mode="state", control_mode="pd_ee_pose", sim_backend="cpu",
        robot_uids=args.robot, active_kinds=tuple(args.kinds),
        belt_speed_range=(args.belt, args.belt), spawn_x=args.spawn_x,
        spawn_spacing=args.spawn_spacing, instruction_kind=None,
        sim_config=dict(control_freq=args.control_freq, sim_freq=120))

    results = []
    t0 = time.time()
    for kind in args.kinds:
        for ep in range(args.episodes):
            env.reset(seed=args.seed + ep * 17)
            u = env.unwrapped
            cfg = PolicyConfig(grasp_depth=args.grasp_depth, close_steps=args.close_steps,
                               grasp_tilt=args.grasp_tilt)
            if args.robot == "panda_wristcam":      # 同台对照：平行夹爪 + 同一脚本/种子
                sorter = ScriptedSorter(env, only_kind=kind, cfg=cfg)
            else:
                sorter = DClawSorter(env, only_kind=kind, cfg=cfg, tcp_dz=args.tcp_dz,
                                     hand_open=args.hand_open, hand_close=args.hand_close,
                                     base_z=args.base_z)
            info, t, prev, t_ok = {}, 0, None, None
            for t in range(args.max_steps):
                act = sorter.act()
                if args.verbose_states and sorter.state != prev:
                    fp = u.fruit_actors[kind].pose.p[0].cpu().numpy()
                    tp = u.agent.tcp_pose.p[0].cpu().numpy()
                    print(f"   t={t:4d} {prev}->{sorter.state} 果={np.round(fp, 2).tolist()} "
                          f"tcp={np.round(tp, 2).tolist()}", flush=True)
                    prev = sorter.state
                obs, _, term, trunc, info = env.step(act)
                if t_ok is None:
                    try:
                        if bool(u._in_bin(kind)[0]):
                            t_ok = t + 1
                    except Exception:  # noqa: BLE001
                        pass
                if bool(term) or bool(trunc):
                    break
            # 按类判定（与 per_class 评测口径一致）：只看目标类是否进框，
            # 否则 instruction_kind=None 时要求四类全入框，单类失败会把整局算 0
            try:
                ok = bool(u._in_bin(kind)[0])
            except Exception:  # noqa: BLE001
                ok = bool(info.get("success", [False])[0]) if info else False
            results.append({"kind": kind, "ok": ok, "steps": t + 1,
                            "t_s": (t_ok or (t + 1)) / args.control_freq,
                            "state": sorter.state,
                            "dropped": bool(info.get("dropped", [False])[0]) if info else False})
            print(f"[{'✓' if ok else '✗'}] {kind:10s} ep{ep} 步 {t + 1} "
                  f"({(t + 1) / args.control_freq:.1f}s) 终态 {sorter.state}", flush=True)

    n, ok = len(results), sum(r["ok"] for r in results)
    succ = [r["t_s"] for r in results if r["ok"]]
    print(f"\n{args.robot} @{args.belt} m/s: {ok}/{n} = {ok / max(n, 1) * 100:.1f}%"
          + (f" | 成功单件均值 {np.mean(succ):.2f}s（首次入框）" if succ else ""))
    for k in args.kinds:
        sub = [r for r in results if r["kind"] == k]
        if sub:
            print(f"  {k:10s} {sum(r['ok'] for r in sub)}/{len(sub)}"
                  f"  单件 {np.mean([r['t_s'] for r in sub]):.1f}s"
                  f"  掉落 {sum(r['dropped'] for r in sub)}")
    out = Path(args.out) if args.out else ROOT / "runs" / f"{time.strftime('%Y-%m-%d-%H%M')}-dclaw-sort5"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"robot": args.robot, "belt": args.belt, "success": ok, "n": n,
         "results": results, "args": vars(args)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果写入 {out}/metrics.json | 用时 {(time.time() - t0) / 60:.1f} min")
    env.close()


if __name__ == "__main__":
    main()
