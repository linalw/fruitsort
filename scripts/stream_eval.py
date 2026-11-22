#!/usr/bin/env python
"""连续作业评测：传送带不停、间隔随机、尺寸随机，看策略能否正常干活。

与 `eval_act_pure.py` 的区别
---------------------------
- 环境换成 `FruitSortStream-v1`：每类 3 个 slot（共 12 件在飞行中），
  上料间隔 `U(gap_min, gap_max)` 米、尺寸 `U(scale_min, scale_max)`（±15%）；
- **目标选择层**：策略本身只会"抓 kind 这一类的最近一件"，谁来抓、什么时候换由这里决定
  （按"最快离开工作区"排序 + 抓取中不切换）；
- 指标按件统计：delivered / dropped / missed / throughput / 每类成功率。

用法:
    python scripts/stream_eval.py --ckpt runs/act-wide-4c/ckpt.pt --belt 0.08 --seconds 90
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
from fruitsort.learning.bc import build_obs
from fruitsort.policies.act_runner import ActPureRunner
from fruitsort.policies.scripted import HOME_EULER, ScriptedSorter

ROOT = Path(__file__).resolve().parent.parent
KINDS = ("apple", "strawberry", "banana", "pear")


class _Stub:
    """只给 build_obs 用的替身：提供 .pose.p / .linear_velocity 两个张量。"""

    def __init__(self, p, v):
        self.pose = type("P", (), {"p": torch.tensor([p], dtype=torch.float32)})
        self.linear_velocity = torch.tensor([v], dtype=torch.float32)


class StreamController:
    """在"多件在飞"的场景里给 ACT 选目标。

    约定：`u.fruit_actors[kind]` 指向**该类最近的一件**（保持训练时的观测结构），
    目标类的 `fruit_actors[kind]` 指向**当前目标件**（runner 的所有判据都读它）。
    """

    def __init__(self, env, runner: ActPureRunner, u, kinds=KINDS, grasp_x: float = -0.10,
                 only_kind: str | None = None, scripted: bool = False,
                 in_dist: bool = False, noise: float = 0.0):
        self.env, self.runner, self.u = env, runner, u
        self.scripted = scripted
        self.in_dist = in_dist
        self.noise = float(noise)
        self.sorter = None
        self.kinds = tuple(kinds)
        self.only_kind = only_kind
        self.grasp_x = float(grasp_x)
        self.belt = float(getattr(u, "belt_speed", [0.08])[0]) if hasattr(u, "belt_speed") else 0.08
        self.target = None          # item dict
        self.switches = 0
        self._repoint()

    # ---------------- 内部 ----------------
    def _live(self):
        items = []
        ks = (self.only_kind,) if self.only_kind else self.kinds
        for k in ks:
            for slot in self.u._slots[k]:
                if slot["live"]:
                    items.append(slot["item"])
        return items

    def _repoint(self):
        """fruit_actors[kind] = 目标件本身 / 其余类按训练布局重映射的替身。

        模型训练时四类固定在 x 间隔 0.45 m 的布局上；连续作业里布局是随机的，
        直接把真实位置喂进去会让"接近"相位输出错误增量（实测 TCP 冻结）。
        这里把非目标类摆到"训练布局"（目标 x ± 0.45·Δi），让状态观测回到分布内。
        """
        tcp = self.u.agent.tcp_pose.p[0].cpu().numpy()
        if self.in_dist and self.target is not None:
            tk = self.target["kind"]
            tp = self.target["actor"].pose.p[0].cpu().numpy()
            belt = float(self.u.belt_speed[0])
            for j, k in enumerate(self.kinds):
                if k == tk:
                    self.u.fruit_actors[k] = self.target["actor"]
                    continue
                dx = 0.45 * (self.kinds.index(tk) - j)
                self.u.fruit_actors[k] = _Stub(
                    [tp[0] + dx, tp[1], tp[2]], [belt, 0.0, 0.0])
            return
        for k in self.kinds:
            live = [s["item"] for s in self.u._slots[k] if s["live"]]
            if not live:
                # 该类当前没有在飞件：必须给一个"合理"的替身（刚上料的位姿），
                # 否则 fruit_actors[k] 仍指向停在场外 z=-5 的 actor → 观测分布外，
                # 任何学习策略都会冻结（这正是此前"连续作业 0–17%"的真正原因）
                spec = self.u.specs[k]
                self.u.fruit_actors[k] = _Stub(
                    [self.u.stream_x, 0.0, self.u.belt_top_z + spec.half_height + 0.002],
                    [self.belt, 0.0, 0.0])
                continue
            if self.target is not None and self.target["kind"] == k:
                self.u.fruit_actors[k] = self.target["actor"]
            else:
                nearest = min(live, key=lambda it: np.linalg.norm(
                    it["actor"].pose.p[0].cpu().numpy() - tcp))
                self.u.fruit_actors[k] = nearest["actor"]

    def _pick_target(self):
        """选**最接近理想拦截点**的一件（不是最紧急的）。

        实测：按"x 最大"选会一路追到工作区边缘（x>0.3，训练分布外），
        模型只会悬停不闭爪；按"离拦截点多近"选能留出下探时间。
        """
        tcp = self.u.agent.tcp_pose.p[0].cpu().numpy()
        urg, late = [], []
        for it in self._live():
            p = it["actor"].pose.p[0].cpu().numpy()
            if np.linalg.norm(p - tcp) < 0.05:   # 就在指尖上（可能正抓着）
                continue
            if -0.60 <= p[0] <= -0.05:           # 还来得及：优先最靠前的一件
                urg.append((p[0], it))
            elif -0.05 < p[0] <= 0.15:           # 已经比较靠后：兜底
                late.append((p[0], it))
        if urg:
            return max(urg, key=lambda t: t[0])[1]
        if late:
            return max(late, key=lambda t: t[0])[1]
        return None

    # ---------------- 主循环 ----------------
    def act(self, obs):
        self._repoint()
        # 目标失效（已被回收/送达/漏过）→ 换目标
        if self.target is not None and not any(
                self.target is s["item"] for s in self.u._slots[self.target["kind"]] if s["live"]):
            self.target = None
        if self.target is None:
            self.target = self._pick_target()
            if self.target is not None:
                self.runner.kind = self.target["kind"]
                self.runner.reset()
                self.switches += 1
                self.u.fruit_actors[self.target["kind"]] = self.target["actor"]
                if self.scripted:
                    self.sorter = ScriptedSorter(self.env, only_kind=self.target["kind"])
        if self.target is None:
            # 没活可干：回到待命位姿、张开手（基座系：x=y+0.5, y=-x, z=z）
            return np.array([0.5, 0.0, 0.55, *HOME_EULER, 1.0], dtype=np.float32)
        if self.scripted:
            act = self.sorter.act()
            if self.noise > 0:
                act = act.copy()
                act[:3] += np.random.normal(0, self.noise, 3).astype(np.float32)
            return act
        return self.runner.act(obs)


def _pack_images(obs):
    """与 record_demos.py 一致：128×128 JPEG（BGR）两路相机。"""
    sd = obs["sensor_data"]
    oh = sd["overhead"]["rgb"][0].cpu().numpy()
    oh = cv2.resize(oh, (128, 128), interpolation=cv2.INTER_AREA)
    wr = None
    if "hand_camera" in sd:
        wr = sd["hand_camera"]["rgb"][0].cpu().numpy()
        wr = cv2.resize(wr, (128, 128), interpolation=cv2.INTER_AREA)
    return (cv2.imencode(".jpg", oh[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes(),
            cv2.imencode(".jpg", wr[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
            if wr is not None else b"")


def _save_demo(out_dir: Path, idx: int, buf: dict, seed: int) -> None:
    if len(buf["obs"]) < 20:
        return
    payload = dict(obs=np.stack(buf["obs"]), action=np.stack(buf["act"]),
                   seed=seed, kinds=np.array(KINDS), instruction=np.array(buf["kind"]),
                   overhead_jpg=np.array(buf["oh"], dtype=object),
                   wrist_jpg=np.array(buf["wr"], dtype=object))
    np.savez_compressed(out_dir / f"ep_{idx:04d}.npz", **payload)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/act-wide-4c/ckpt.pt")
    ap.add_argument("--belt", type=float, default=0.08)
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--slots", type=int, default=3, help="每类候选件数（共 4×slots 件）")
    ap.add_argument("--spawn-x", type=float, default=-0.85, help="上料点 x（米）")
    ap.add_argument("--gap-min", type=float, default=0.60,
                    help="相邻两件的间距下限（米）")
    ap.add_argument("--gap-max", type=float, default=1.20, help="间距上限（米）")
    ap.add_argument("--scale-min", type=float, default=0.85)
    ap.add_argument("--scale-max", type=float, default=1.15)
    ap.add_argument("--control-freq", type=int, default=30,
                    help="与部署评测一致（act-wide-4c 的 87.5%/96.9% 都是 30 Hz）")
    ap.add_argument("--fb", type=float, default=0.5)
    ap.add_argument("--grip-ema", type=float, default=0.6)
    ap.add_argument("--carry-latch", action="store_true", default=True)
    ap.add_argument("--carry-max-step", type=float, default=0.02)
    ap.add_argument("--grasp-x", type=float, default=-0.10, help="理想拦截点 x（米）")
    ap.add_argument("--in-dist-obs", action="store_true",
                    help="把非目标类按训练布局重映射（让状态观测回到分布内）")
    ap.add_argument("--policy", default="act", choices=["act", "scripted"],
                    help="act=部署策略；scripted=规则专家（测场景上限）")
    ap.add_argument("--only-kind", default=None,
                    help="只抓这一类（诊断用：隔离'指令切换'的影响）")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="录制时在专家动作上加的高斯噪声（米）——专家主导 + 小扰动，"
                         "比收集模型失败状态安全得多")
    ap.add_argument("--record", default="", help="把专家在连续流里的成功片段录成训练数据（目录）")
    ap.add_argument("--seed", type=int, default=40000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    env = gym.make(
        "FruitSortStream-v1", obs_mode="rgbd", control_mode="pd_ee_pose", sim_backend="cpu",
        robot_uids="panda_wristcam", active_kinds=KINDS,
        belt_speed_range=(args.belt, args.belt), spawn_x=args.spawn_x,
        slots_per_kind=args.slots, gap_range=(args.gap_min, args.gap_max),
        scale_range=(args.scale_min, args.scale_max),
        always_stock=bool(args.record),
        sim_config=dict(control_freq=args.control_freq,
                        sim_freq=120 if args.control_freq == 30 else 100))
    runner = ActPureRunner(env, args.ckpt, "apple", fb=args.fb, grip_ema=args.grip_ema,
                           carry_latch=args.carry_latch, carry_max_step=args.carry_max_step)
    ctrl = StreamController(env, runner, env.unwrapped, grasp_x=args.grasp_x,
                            only_kind=args.only_kind, scripted=(args.policy == "scripted"),
                            in_dist=args.in_dist_obs, noise=args.noise)

    rec_dir = Path(args.record) if args.record else None
    if rec_dir is not None:
        rec_dir.mkdir(parents=True, exist_ok=True)
    n_saved = 0
    max_steps = int(args.seconds * args.control_freq)
    runs = []
    t0 = time.time()
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        u = env.unwrapped
        runner.reset()
        ctrl = StreamController(env, runner, u, grasp_x=args.grasp_x,
                                only_kind=args.only_kind, scripted=(args.policy == "scripted"),
                            in_dist=args.in_dist_obs, noise=args.noise)
        buf = {"obs": [], "act": [], "oh": [], "wr": [], "kind": None, "id": None}

        def flush():
            nonlocal n_saved, buf
            it = u.stream_stats["items"].get(buf["id"] or -1)
            if (rec_dir is not None and it is not None
                    and it.get("outcome") == "delivered"):
                _save_demo(rec_dir, n_saved, buf, args.seed + ep)
                n_saved += 1
            buf = {"obs": [], "act": [], "oh": [], "wr": [], "kind": None, "id": None}

        for t in range(max_steps):
            act = ctrl.act(obs)
            it = ctrl.target
            if it is None or it["id"] != buf["id"]:
                flush()
                if it is not None:
                    buf["id"], buf["kind"] = it["id"], it["kind"]
            # 只用"四类都在场"的帧录制：缺类时我们喂的是 ghost 观测，
            # 图像里没有这颗水果、状态里却有 → 图文不一致会污染训练数据
            all_live = all(any(sl["live"] for sl in u._slots[k]) for k in KINDS)
            if rec_dir is not None and not all_live:
                flush()          # 缺类 → 结束当前段，等四类齐全再开新段
                buf["id"] = None
            if buf["id"] is not None and ctrl.scripted and rec_dir is not None:
                buf["obs"].append(build_obs(u, buf["kind"]))
                # 专家的 last_target 是 4 维绝对目标（与 record_demos 一致）
                buf["act"].append(np.asarray(ctrl.sorter.last_target, dtype=np.float32))
                oh, wr = _pack_images(obs)
                buf["oh"].append(oh); buf["wr"].append(wr)
            obs, _, term, trunc, info = env.step(act)
            if bool(term) or bool(trunc):
                break
        flush()
        st = u.stream_stats
        resolved = st["delivered"] + st["dropped"] + st["missed"]
        row = dict(ep=ep, belt=args.belt, seconds=round(u._t, 1),
                   spawned=st["spawned"], delivered=st["delivered"],
                   dropped=st["dropped"], missed=st["missed"], picked=st["picked"],
                   switches=ctrl.switches,
                   success_rate=(st["delivered"] / resolved) if resolved else 0.0,
                   throughput_per_min=60.0 * st["delivered"] / max(u._t, 1e-6),
                   per_kind=st["per_kind"])
        runs.append(row)
        print(f"[ep{ep}] {row['seconds']:.0f}s 上料 {row['spawned']} | 送达 {row['delivered']} "
              f"掉落 {row['dropped']} 漏过 {row['missed']} | 抓取过 {row['picked']} | "
              f"成功率 {row['success_rate']*100:.1f}% | 吞吐 {row['throughput_per_min']:.2f} 件/min "
              f"| 换目标 {row['switches']}", flush=True)
        print("      每类: " + "  ".join(
            f"{k} {v['delivered']}/{v['spawned']}(drop {v['dropped']}) "
            for k, v in row["per_kind"].items()), flush=True)

    n = len(runs)
    agg = {k: sum(r[k] for r in runs) for k in
           ("spawned", "delivered", "dropped", "missed", "picked")}
    resolved = agg["delivered"] + agg["dropped"] + agg["missed"]
    print(f"\n== 连续作业 @{args.belt} m/s，间隔 {args.gap_min}–{args.gap_max} m，"
          f"尺寸 ±{int((args.scale_max-1)*100)}% ==")
    print(f"总上料 {agg['spawned']} | 送达 {agg['delivered']} | 掉落 {agg['dropped']} | "
          f"漏过 {agg['missed']} | 成功率 {agg['delivered']/max(resolved,1)*100:.1f}% | "
          f"吞吐 {np.mean([r['throughput_per_min'] for r in runs]):.2f} 件/min")
    for k in KINDS:
        d = sum(r["per_kind"][k]["delivered"] for r in runs)
        s = sum(r["per_kind"][k]["spawned"] for r in runs)
        print(f"  {k:10s} {d}/{s}" + (f" = {d/s*100:.0f}%" if s else ""))
    out = Path(args.out) if args.out else ROOT / "runs" / f"{time.strftime('%Y-%m-%d-%H%M')}-stream-eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"args": vars(args), "runs": runs, "aggregate": agg}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    if rec_dir is not None:
        print(f"录制 {n_saved} 条成功片段 → {rec_dir}")
    print(f"结果写入 {out}/metrics.json | 用时 {(time.time()-t0)/60:.1f} min")
    env.close()


if __name__ == "__main__":
    main()
