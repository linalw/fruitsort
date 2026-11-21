#!/usr/bin/env python
"""Panda + D'Claw 灵巧手：任务级脚本化分拣（传送带 + 4 类 + 入框判定）。

复用 `ScriptedSorter` 的状态机（WAIT→TRACK→DESCEND→CLOSE→CARRY→LOWER→RELEASE），
只把动作末尾的"夹爪 1 维"换成"9 维手指绝对位置"（-0.85 张开 / +0.85 收拢）。

用法:
    python scripts/dexterous_sort3.py --kinds apple --episodes 1 --belt 0.08
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

import fruitsort.sim.tasks  # noqa: F401
import fruitsort.sim.agents.panda_dclaw  # noqa: F401  (注册 panda_dclaw)
from fruitsort.policies.scripted import HOME_EULER, ScriptedSorter

ROOT = Path(__file__).resolve().parent.parent
KINDS = ("apple", "strawberry", "banana", "pear")
OPEN, CLOSE = -0.85, 0.85


class DClawSorter(ScriptedSorter):
    """把夹爪 1 维换成 D'Claw 的 9 维手指目标。"""

    def _action(self, target, gripper: float = 1.0):
        raw = np.asarray(target, dtype=np.float64)
        tcp = self._tcp()
        err = raw - tcp
        dist = float(np.linalg.norm(err))
        self.last_target = np.array([raw[0], raw[1], raw[2], gripper], dtype=np.float32)
        tgt = raw if dist <= self.cfg.max_step else tcp + err / dist * self.cfg.max_step
        x, y, z = tgt
        hand = CLOSE if gripper < 0 else OPEN
        # 虚拟 TCP 相对 panda_hand 有 -45° 偏航，这里补回来，保持"夹爪朝下"的姿态约定
        eul = (HOME_EULER[0], HOME_EULER[1], HOME_EULER[2] + 0.7853981633974483)
        return np.array([y + 0.5, -x, z, *eul, *([hand] * 9)], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=list(KINDS))
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--belt", type=float, default=0.08)
    ap.add_argument("--spawn-x", type=float, default=-0.55)
    ap.add_argument("--spawn-spacing", type=float, default=0.45)
    ap.add_argument("--control-freq", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    env = gym.make(
        "FruitSort-v1", obs_mode="state", control_mode="pd_ee_pose", sim_backend="cpu",
        robot_uids="panda_dclaw", active_kinds=tuple(args.kinds),
        belt_speed_range=(args.belt, args.belt), spawn_x=args.spawn_x,
        spawn_spacing=args.spawn_spacing, instruction_kind=None,
        sim_config=dict(control_freq=args.control_freq, sim_freq=120))

    results = []
    t0 = time.time()
    for kind in args.kinds:
        for ep in range(args.episodes):
            env.reset(seed=args.seed + ep * 17)
            u = env.unwrapped
            sorter = DClawSorter(env, only_kind=kind)
            info, t = {}, 0
            for t in range(args.max_steps):
                obs, _, term, trunc, info = env.step(sorter.act())
                if bool(term) or bool(trunc):
                    break
            ok = bool(info.get("success", [False])[0]) if info else False
            results.append({"kind": kind, "ok": ok, "steps": t + 1,
                            "t_s": (t + 1) / args.control_freq})
            print(f"[{'✓' if ok else '✗'}] {kind:10s} ep{ep} 步 {t+1} "
                  f"({(t + 1) / args.control_freq:.1f}s)", flush=True)

    n = len(results)
    ok = sum(r["ok"] for r in results)
    succ = [r["t_s"] for r in results if r["ok"]]
    print(f"\nPanda+D'Claw 脚本化分拣 @{args.belt} m/s: {ok}/{n} = {ok / max(n,1) * 100:.1f}%"
          f" | 单件 {np.mean(succ):.2f}s（n={len(succ)}）" if succ else f"\n{ok}/{n}")
    for k in args.kinds:
        sub = [r for r in results if r["kind"] == k]
        if sub:
            print(f"  {k:10s} {sum(r['ok'] for r in sub)}/{len(sub)}")
    out = Path(args.out) if args.out else ROOT / "runs" / f"{time.strftime('%Y-%m-%d-%H%M')}-dclaw-sort"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"robot": "panda_dclaw", "belt": args.belt, "results": results,
         "success": ok, "n": n}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"用时 {(time.time()-t0)/60:.1f} min")
    env.close()


if __name__ == "__main__":
    main()
