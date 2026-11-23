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
                 in_dist: bool = False, noise: float = 0.0, runners: dict | None = None,
                 single_runner=None, queue_thr: float = 0.30, engage_x: float = -0.45,
                 scripted_kinds=()):
        self.env, self.runner, self.u = env, runner, u
        self.scripted = scripted
        self.in_dist = in_dist
        self.noise = float(noise)
        self.default_runner = runner
        # 按类路由：不同类别用不同模型（banana 用未被流数据稀释的版本）
        self.runners = dict(runners or {})
        # 按**队列状态**切换的双策略：单件模式（queue<=thr）用 single_runner，
        # 队列模式用默认 runner。切换只在"选新目标"时发生，避免搬运途中换模型。
        self.single_runner = single_runner
        self.queue_thr = float(queue_thr)
        # 只在**可达窗口**内接管目标：水果还在 x<-0.45 时，Panda 从基座 (0,-0.5) 过去
        # 距离≈0.85 m（臂展极限），命令位姿不可达 → PD 卡住、TCP 冻结（docs 6.26.35）。
        self.engage_x = float(engage_x)
        # 混合系统：这些类别交给规则专家（模型在队列里做不好的长条类），
        # 其余类别仍由模型驱动。只在"选新目标"时决定，搬运途中不换。
        self.scripted_kinds = tuple(scripted_kinds)
        self.use_scripted = False
        self.sorter = None
        self.kinds = tuple(kinds)
        self.only_kind = only_kind
        self.grasp_x = float(grasp_x)
        self.belt = float(getattr(u, "belt_speed", [0.08])[0]) if hasattr(u, "belt_speed") else 0.08
        self.target = None          # item dict
        self.switches = 0
        self._repoint()

    # ---------------- 内部 ----------------
    def _queue_level(self) -> float:
        """队列标志：场上 |x|<0.6 且在带面的水果数 /4（与训练/推理的特征同源）。"""
        n = 0
        for k in self.kinds:
            a = self.u.fruit_actors.get(k)
            if a is None:
                continue
            arr = a.pose.p[0].cpu().numpy() if hasattr(a, "pose") else np.zeros(3)
            if abs(float(arr[0])) < 0.6 and 0.40 < float(arr[2]) < 0.60:
                n += 1
        return n / 4.0

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
            if self.engage_x <= p[0] <= -0.05:   # 已进入可达窗口：优先最靠前的一件
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
                q = self._queue_level()
                if q <= self.queue_thr and self.single_runner is not None:
                    # 单件模式：用"干净单件数据"训练的模型（apple 8/8、banana 7/8）
                    self.runner = self.single_runner
                elif self.target["kind"] in self.runners:
                    # 队列模式：优先按类专用模型（各类自带流上下文数据）
                    self.runner = self.runners[self.target["kind"]]
                self.runner.kind = self.target["kind"]
                self.runner.reset()
                self.switches += 1
                self.u.fruit_actors[self.target["kind"]] = self.target["actor"]
                self.use_scripted = (self.scripted
                                     or self.target["kind"] in self.scripted_kinds)
                if self.use_scripted:
                    self.sorter = ScriptedSorter(self.env, only_kind=self.target["kind"])
        if self.target is None:
            # 没活可干：回到待命位姿、张开手（基座系：x=y+0.5, y=-x, z=z）
            return np.array([0.5, 0.0, 0.55, *HOME_EULER, 1.0], dtype=np.float32)
        if self.use_scripted:
            act = self.sorter.act()
            if self.noise > 0:
                act = act.copy()
                act[:3] += np.random.normal(0, self.noise, 3).astype(np.float32)
            return act
        return self.runner.act(obs)


def parse_map(spec: str) -> dict:
    """'banana=-0.01,pear=0.015' -> {'banana': -0.01, 'pear': 0.015}"""
    out = {}
    for kv in (spec or "").split(","):
        kv = kv.strip()
        if not kv or "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        out[k.strip()] = float(v)
    return out


WIDTH_MAP = {"apple": 0.062, "strawberry": 0.034, "banana": 0.044, "pear": 0.068}


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
    ap.add_argument("--engage-x", type=float, default=-0.45,
                    help="只在 x>=该值时接管目标（更远的水果超出臂展，命令不可达会冻结）")
    ap.add_argument("--grasp-x", type=float, default=-0.10, help="理想拦截点 x（米）")
    ap.add_argument("--in-dist-obs", action="store_true",
                    help="把非目标类按训练布局重映射（让状态观测回到分布内）")
    ap.add_argument("--yaw-align", action="store_true",
                    help="验证性实验：用仿真真值 yaw 旋转手腕，使夹爪闭合方向垂直于水果长轴")
    ap.add_argument("--yaw-align-off", type=float, default=0.0,
                    help="长轴假设偏移（0 或 π/2），用于试两种本地轴")
    ap.add_argument("--grasp-verify", action="store_true",
                    help="只碰到水果顶端不算抓住（要求指面到达果身高度或已被抬起）")
    ap.add_argument("--intercept", type=float, default=0.0,
                    help="拦截式下探的垂直速度（m/s，0=关闭）：按'下探耗时×带速'预测落点")
    ap.add_argument("--close-win-xy", type=float, default=0.03)
    ap.add_argument("--close-win-z", type=float, default=0.06)
    ap.add_argument("--close-z-gate", type=float, default=0.0,
                    help="闭爪高度门（米）：离果心高于该值时禁止闭爪，改为下探（0=关闭）")
    ap.add_argument("--descend-commit", type=int, default=0,
                    help="下探承诺步数：模型一旦判定下探，就保持该相位 N 步，避免 0/1 抖动")
    ap.add_argument("--z-bias-map", default="",
                    help="按类抓取高度偏置，如 'pear=0.015,banana=-0.01'")
    ap.add_argument("--yaw-map", default="",
                    help="按类手腕偏航（rad），如 'banana=0.6'（让夹爪闭合方向对齐长轴）")
    ap.add_argument("--width-setpoint", action="store_true",
                    help="闭爪到按类的宽度设定点(LeTac 式)，而不是合到底")
    ap.add_argument("--stall-guard", type=float, default=0.0,
                    help="停滞守卫的最小接近步长（米，0=关闭）：指令位移≈0 且水果仍在窗口内时"
                         "强制朝水果迈这一步，用于打破\"接近相位塌陷→TCP 冻结\"")
    ap.add_argument("--single-ckpt", default="",
                    help="按队列状态切换的双策略：单件模式(queue<=--queue-thr)用该模型")
    ap.add_argument("--queue-thr", type=float, default=0.30)
    ap.add_argument("--ckpt-map", default="",
                    help="按类路由模型：'banana=runs/act-noisy/ckpt.pt'（逗号分隔多条）")
    ap.add_argument("--scripted-kinds", default="",
                    help="混合系统：这些类（逗号分隔）交给规则专家，其余由模型驱动，"
                         "如 'banana,pear'")
    ap.add_argument("--policy", default="act", choices=["act", "scripted"],
                    help="act=部署策略；scripted=规则专家（测场景上限）")
    ap.add_argument("--only-kind", default=None,
                    help="只抓这一类（诊断用：隔离'指令切换'的影响）")
    ap.add_argument("--allow-ghost-frames", action="store_true",
                    help="录制时允许'缺类'帧（部署端同样会喂 ghost 观测）。关闭时只录四类齐全帧，"
                         "产量会低一个数量级（实测 4/60s vs 20/60s）")
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
    def _mk(ckpt, kind):
        return ActPureRunner(env, ckpt, kind, fb=args.fb, grip_ema=args.grip_ema,
                             carry_latch=args.carry_latch,
                             carry_max_step=args.carry_max_step,
                             stall_guard=args.stall_guard,
                             descend_commit=args.descend_commit,
                             close_z_gate=args.close_z_gate,
                             grasp_verify=args.grasp_verify,
                             yaw_align=args.yaw_align,
                             yaw_align_off=args.yaw_align_off,
                             intercept=args.intercept,
                             close_win_xy=args.close_win_xy,
                             close_win_z=args.close_win_z,
                             grasp_z_bias=parse_map(args.z_bias_map),
                             yaw_map=parse_map(args.yaw_map),
                             width_map=(dict(WIDTH_MAP) if args.width_setpoint else None))

    runner = _mk(args.ckpt, "apple")
    runners = {}
    if args.ckpt_map:
        for kv in args.ckpt_map.split(","):
            _k, _path = kv.split("=", 1)
            runners[_k.strip()] = _mk(_path, _k.strip())
        print("按类路由:", list(runners), flush=True)
    single_runner = None
    if args.single_ckpt:
        single_runner = _mk(args.single_ckpt, "apple")
        print(f"队列切换: 单件模式={args.single_ckpt} (queue<={args.queue_thr})", flush=True)
    ctrl = StreamController(env, runner, env.unwrapped, grasp_x=args.grasp_x,
                            only_kind=args.only_kind, scripted=(args.policy == "scripted"),
                            in_dist=args.in_dist_obs, noise=args.noise, runners=runners,
                            single_runner=single_runner, queue_thr=args.queue_thr,
                            engage_x=args.engage_x,
                            scripted_kinds=[k.strip() for k in args.scripted_kinds.split(",") if k.strip()])

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
                                in_dist=args.in_dist_obs, noise=args.noise, runners=runners,
                                single_runner=single_runner, queue_thr=args.queue_thr,
                                engage_x=args.engage_x,
                                scripted_kinds=[k.strip() for k in args.scripted_kinds.split(",") if k.strip()])
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
            if rec_dir is not None and not all_live and not args.allow_ghost_frames:
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
