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
                 scripted_kinds=(), scripted_yaw: float = 0.0, objs: dict | None = None,
                 decline_yaw: float = 0.0, long_extent: float = 0.11,
                 kind_priority: dict | None = None):
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
        self.engaged_ids = set()     # 被本臂"接管过"的件（= 我们的机会）
        self.scripted_kinds = tuple(scripted_kinds)
        self.use_scripted = False
        # 队列上下文录制时的手腕偏航（每集随机一个，与 record_demos 的 --grasp-yaw-range 同义）：
        # 让"腕部朝向"成为训练分布内的变量，执行期才可能按水果长轴对准。
        self.scripted_yaw = float(scripted_yaw)
        # 感知模式：目标选择/队列标志都读这个字典（而不是仿真真值）
        self.objs = objs
        # 长条目标（世界长度 > long_extent）只在"长轴与夹爪闭合方向够垂直"时接管：
        # 平口夹爪沿世界 y 闭合，香蕉长轴若也接近 y，只能夹住两端 → 必失败。
        # 现场语义：不接这些件，让给旁边朝向更合适的臂（记账里算 declined）。
        self.decline_yaw = float(decline_yaw)
        self.long_extent = float(long_extent)
        # 按类优先（数字越小越先接）：短周期/高可靠的类先做，长条留到空档
        self.kind_priority = dict(kind_priority or {})
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
            a = (self.objs or self.u.fruit_actors).get(k)
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
                if self.objs is not None:
                    continue
                if self.objs is None:
                    self.u.fruit_actors[k] = self.target["actor"]
            else:
                nearest = min(live, key=lambda it: np.linalg.norm(
                    it["actor"].pose.p[0].cpu().numpy() - tcp))
                if self.objs is None:
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
            if self.objs is not None:
                # 感知模式下不要追"场外幽灵"：该类当前没有检测（替身停在 clip_x）时，
                # 说明还看不见它，接管只会让手臂朝 -1.1 m 走（不可达 → 冻结）
                est = self.objs.get(it["kind"])
                if est is None or float(est.pose.p[0, 0]) <= -1.05:
                    continue
                if self.decline_yaw > 0:
                    ext = float(getattr(est, "extent", 0.0) or 0.0)
                    yy = getattr(est, "yaw", None)
                    if ext > self.long_extent and yy is not None:
                        # 夹爪闭合方向≈世界 y：|sin(yaw)| 大 → 长轴贴着闭合方向 → 夹不住
                        if abs(float(np.sin(yy))) > self.decline_yaw:
                            self.declined_yaw = getattr(self, "declined_yaw", 0) + 1
                            continue
            if self.engage_x <= p[0] <= -0.05:   # 已进入可达窗口：优先最靠前的一件
                urg.append((p[0], it))
            elif -0.05 < p[0] <= 0.15:           # 已经比较靠后：兜底
                late.append((p[0], it))
        if urg:
            if self.kind_priority:
                return min(urg, key=lambda t: (self.kind_priority.get(t[1]["kind"], 9.0),
                                               -t[0]))[1]
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
            # 目标结束（送达/掉落/漏过）：清掉执行器状态，否则 phase 会停在"搬运"
            self.runner.reset()
        if self.target is None:
            self.target = self._pick_target()
            if self.target is not None:
                self.engaged_ids.add(self.target["id"])
                if "t_engage" not in self.target:
                    self.target["t_engage"] = float(self.u._t)
                q = self._queue_level()
                if q <= self.queue_thr and self.single_runner is not None:
                    # 单件模式：用"干净单件数据"训练的模型（apple 8/8、banana 7/8）
                    self.runner = self.single_runner
                elif self.target["kind"] in self.runners:
                    # 队列模式：优先按类专用模型（各类自带流上下文数据）
                    self.runner = self.runners[self.target["kind"]]
                self.runner.kind = self.target["kind"]
                self.runner.reset()
                # 切模型后必须重新注入感知替身（否则新 runner 这一步读到仿真真值）
                if self.objs is not None:
                    self.runner.set_actors(self.objs)
                self.switches += 1
                if self.objs is None:
                    self.u.fruit_actors[self.target["kind"]] = self.target["actor"]
                self.use_scripted = (self.scripted
                                     or self.target["kind"] in self.scripted_kinds)
                if self.use_scripted:
                    cfg_yaw = None
                    if self.scripted_yaw:
                        from fruitsort.policies.scripted import PolicyConfig
                        cfg_yaw = PolicyConfig(grasp_yaw=self.scripted_yaw)
                    self.sorter = ScriptedSorter(
                        self.env, cfg=cfg_yaw, only_kind=self.target["kind"])
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
                    help="与部署评测一致（act-wide-4c 的 87.5%%/96.9%% 都是 30 Hz）")
    ap.add_argument("--max-step-map", default="",
                    help="按类接近步长，如 'apple=0.12,strawberry=0.12,pear=0.10,banana=0.06'")
    ap.add_argument("--carry-step-map", default="",
                    help="按类搬运步长，如 'apple=0.04,pear=0.03,banana=0.02'")
    ap.add_argument("--max-step", type=float, default=0.06,
                    help="接近段每步最大位移（米/步）：调大可加速机械臂，代价是精度/稳定性")
    ap.add_argument("--fb", type=float, default=0.5)
    ap.add_argument("--grip-ema", type=float, default=0.6)
    ap.add_argument("--carry-latch", action="store_true", default=True)
    ap.add_argument("--carry-max-step", type=float, default=0.02)
    ap.add_argument("--engage-x", type=float, default=-0.45,
                    help="只在 x>=该值时接管目标（更远的水果超出臂展，命令不可达会冻结）")
    ap.add_argument("--grasp-x", type=float, default=-0.10, help="理想拦截点 x（米）")
    ap.add_argument("--in-dist-obs", action="store_true",
                    help="把非目标类按训练布局重映射（让状态观测回到分布内）")
    ap.add_argument("--grasp-yaw-range", type=float, nargs=2, default=None,
                    help="录制时每集随机手腕偏航（rad）：'队列上下文 + 随机腕部'合并成一批数据")
    ap.add_argument("--yaw-align", action="store_true",
                    help="验证性实验：用仿真真值 yaw 旋转手腕，使夹爪闭合方向垂直于水果长轴")
    ap.add_argument("--yaw-align-off", type=float, default=0.0,
                    help="长轴假设偏移（0 或 π/2），用于试两种本地轴")
    ap.add_argument("--yaw-align-extent", type=float, default=0.0,
                    help="只对世界长度 ≥ 该值的目标做腕部对准（0=全部；香蕉建议 0.12）")
    ap.add_argument("--close-commit", type=float, default=0.0,
                    help="闭爪承诺：离果心 z 低于该值且水平对齐时强制闭爪并冻结水平目标（0=关闭）")
    ap.add_argument("--close-commit-xy", type=float, default=0.030)
    ap.add_argument("--close-commit-map", default="",
                    help="按类闭爪承诺阈值，如 'apple=0.025,strawberry=0.025'")
    ap.add_argument("--grasp-proxy", default="auto", choices=["auto", "always", "never"],
                    help="抓取近似判据（离得近+闭爪）：auto=仅当查不到真实接触时用")
    ap.add_argument("--grasp-verify", action="store_true",
                    help="只碰到水果顶端不算抓住（要求指面到达果身高度或已被抬起）")
    ap.add_argument("--lead-long", type=float, default=0.0,
                    help="长条目标的拦截点前移（秒）。实测有害：长条会被推到尖端去抓（默认关）")
    ap.add_argument("--lead-long-extent", type=float, default=0.11)
    ap.add_argument("--lead-long-taper", type=float, default=0.15,
                    help="前移在距果心该距离内线性衰减到 0（0=不衰减）")
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
    ap.add_argument("--kind-priority", default="",
                    help="按类优先接件，如 'apple=0,strawberry=1,pear=2,banana=3'（空=只按 x）")
    ap.add_argument("--decline-yaw", type=float, default=0.64,
                    help="长条目标：|sin(长轴)| 超过该值就不接管（长轴贴住闭合方向夹不住；0=关）")
    ap.add_argument("--long-extent", type=float, default=0.11)
    ap.add_argument("--grasp-axis-offset", type=float, default=0.0,
                    help="抓取点沿长轴偏移（米，只作用于长条目标；感知 yaw 给方向）")
    ap.add_argument("--grasp-axis-extent", type=float, default=0.12)
    ap.add_argument("--near-hold", action=argparse.BooleanOptionalAction, default=True,
                    help="感知：丢检且夹爪压到目标上方时位置取 TCP（只对紧凑水果生效；默认开）")
    ap.add_argument("--perception", default="oracle", choices=["oracle", "vision", "rgbd", "fused"],
                    help="oracle=仿真真值；fused=俯视+侧视 RGB-D 感知（无真值）")
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
    ap.add_argument("--video", default="",
                    help="输出演示视频（mp4）。画面叠加：时间/带速/目标类别/送达·漏检·掉落/相位")
    ap.add_argument("--video-fps", type=int, default=0,
                    help="视频帧率，0 = 跟随 control-freq（与实时一致）")
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
        render_mode=("rgb_array" if args.video else None),
        sim_config=dict(control_freq=args.control_freq,
                        sim_freq=120 if args.control_freq == 30 else 100))
    src = None
    if args.perception == "fused":
        from fruitsort.perception.adapter import FusedSource
        src = FusedSource(env.unwrapped, dt=1.0 / args.control_freq,
                          near_hold=args.near_hold,
                          grasp_axis_offset=args.grasp_axis_offset,
                          grasp_axis_extent=args.grasp_axis_extent)
    elif args.perception == "rgbd":
        from fruitsort.perception.adapter import RGBDSource
        src = RGBDSource(env.unwrapped, dt=1.0 / args.control_freq,
                         near_hold=args.near_hold,
                         grasp_axis_offset=args.grasp_axis_offset,
                         grasp_axis_extent=args.grasp_axis_extent)
    elif args.perception == "vision":
        from fruitsort.perception.adapter import VisionSource
        src = VisionSource(env.unwrapped, dt=1.0 / args.control_freq)

    def _mk(ckpt, kind):
        return ActPureRunner(env, ckpt, kind, fb=args.fb, grip_ema=args.grip_ema,
                             max_step=args.max_step,
                             max_step_map=parse_map(args.max_step_map),
                             carry_step_map=parse_map(args.carry_step_map),
                             carry_latch=args.carry_latch,
                             carry_max_step=args.carry_max_step,
                             stall_guard=args.stall_guard,
                             descend_commit=args.descend_commit,
                             close_z_gate=args.close_z_gate,
                             lead_long=args.lead_long,
                             lead_long_extent=args.lead_long_extent,
                             lead_long_taper=args.lead_long_taper,
                             close_commit=args.close_commit,
                             close_commit_xy=args.close_commit_xy,
                             close_commit_map=parse_map(args.close_commit_map),
                             grasp_verify=args.grasp_verify,
                             grasp_proxy=args.grasp_proxy,
                             yaw_align=args.yaw_align,
                             yaw_align_off=args.yaw_align_off,
                             yaw_align_extent=args.yaw_align_extent,
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
    yaw_rng = np.random.default_rng(args.seed + 7)
    ep_yaw = 0.0
    ctrl = StreamController(env, runner, env.unwrapped, grasp_x=args.grasp_x,
                            objs=(None if src is None else {}),
                            only_kind=args.only_kind, scripted=(args.policy == "scripted"),
                            in_dist=args.in_dist_obs, noise=args.noise, runners=runners,
                            single_runner=single_runner, queue_thr=args.queue_thr,
                            engage_x=args.engage_x,
                            scripted_kinds=[k.strip() for k in args.scripted_kinds.split(",") if k.strip()],
                            decline_yaw=args.decline_yaw, long_extent=args.long_extent,
                            kind_priority=parse_map(args.kind_priority))

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
        if args.grasp_yaw_range is not None:
            ep_yaw = float(yaw_rng.uniform(*args.grasp_yaw_range))
        ctrl = StreamController(env, runner, u, grasp_x=args.grasp_x,
                                objs=(None if src is None else {}),
                                only_kind=args.only_kind, scripted=(args.policy == "scripted"),
                                in_dist=args.in_dist_obs, noise=args.noise, runners=runners,
                                single_runner=single_runner, queue_thr=args.queue_thr,
                                engage_x=args.engage_x,
                                scripted_kinds=[k.strip() for k in args.scripted_kinds.split(",") if k.strip()],
                                scripted_yaw=ep_yaw,
                                decline_yaw=args.decline_yaw, long_extent=args.long_extent,
                            kind_priority=parse_map(args.kind_priority))
        writer = None
        if args.video:
            _v = Path(args.video)
            _v.parent.mkdir(parents=True, exist_ok=True)
            _v = _v.with_suffix(".mp4")
        buf = {"obs": [], "act": [], "oh": [], "wr": [], "kind": None, "id": None}

        def flush():
            nonlocal n_saved, buf
            it = u.stream_stats["items"].get(buf["id"] or -1)
            if (rec_dir is not None and it is not None
                    and it.get("outcome") == "delivered"):
                _save_demo(rec_dir, n_saved, buf, args.seed + ep)
                n_saved += 1
            writer = None
        if args.video:
            _v = Path(args.video)
            _v.parent.mkdir(parents=True, exist_ok=True)
            _v = _v.with_suffix(".mp4")
        buf = {"obs": [], "act": [], "oh": [], "wr": [], "kind": None, "id": None}

        for t in range(max_steps):
            if src is not None:
                objs = src.update(obs, held_kind=(ctrl.runner.kind if ctrl.runner.grasping else None))
                ctrl.objs = objs
                ctrl.runner.set_actors(objs)
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
            if args.video:
                frame = env.render()
                arr = frame.cpu().numpy() if hasattr(frame, "cpu") else np.asarray(frame)
                arr = arr[0] if arr.ndim == 4 else arr
                img = np.ascontiguousarray(arr[..., ::-1].astype(np.uint8))   # RGB→BGR
                if writer is None:
                    h, w = img.shape[:2]
                    writer = cv2.VideoWriter(str(_v), cv2.VideoWriter_fourcc(*"mp4v"),
                                             float(args.video_fps or args.control_freq), (w, h))
                st_v = u.stream_stats
                tgt_v = ctrl.target["kind"] if ctrl.target is not None else "-"
                ph = ctrl.runner.phase_state if ctrl.runner is not None else -1
                lines = [f"t={u._t:5.1f}s   belt={args.belt:.2f} m/s   gap={args.gap_min}-{args.gap_max} m",
                         f"target={tgt_v}   phase={ph}   grip={'closed' if ctrl.runner.prev_grip < 0 else 'open'}",
                         f"delivered={st_v['delivered']}   missed={st_v['missed']}   dropped={st_v['dropped']}"
                         f"   spawned={st_v['spawned']}",
                         f"engaged={len(getattr(ctrl, 'engaged_ids', ()))}   "
                         f"handed_to_peers={max(0, st_v['missed'] - 0)}   "
                         f"step_mode={'per-class' if args.max_step_map else 'uniform'}"]
                hh, ww = img.shape[:2]
                fs = max(0.6, ww / 1920.0 * 1.0)
                lh = int(round(34 * fs))
                pad = int(round(10 * fs))
                overlay = img.copy()
                cv2.rectangle(overlay, (0, 0), (int(ww * 0.42), lh * len(lines) + 2 * pad),
                              (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
                for i, txt in enumerate(lines):
                    org = (pad, pad + lh * (i + 1) - int(8 * fs))
                    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs,
                                (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs,
                                (255, 255, 255), 2, cv2.LINE_AA)
                writer.write(img)
            if bool(term) or bool(trunc):
                break
        flush()
        if writer is not None:
            writer.release(); writer = None
        st = u.stream_stats
        # "机会内"口径：只算本臂接管过的件（其余漏检视为让给旁边的工作臂）
        items = st.get("items", {})
        eng = [it for it in items.values() if it.get("id") in getattr(ctrl, "engaged_ids", set())]
        eng_done = [it for it in eng if it.get("outcome") == "delivered"]
        eng_res = [it for it in eng if it.get("outcome") in ("delivered", "dropped", "missed")]
        declined = [it for it in items.values()
                    if it.get("outcome") == "missed" and it.get("id") not in getattr(ctrl, "engaged_ids", set())]
        handle = [it["t_end"] - it["t_engage"] for it in eng
                  if it.get("outcome") == "delivered" and it.get("t_end") and it.get("t_engage")]
        st["handle_s"] = float(np.mean(handle)) if handle else 0.0
        st["engaged"] = len(eng)
        st["engaged_by_kind"] = {}
        for k in KINDS:
            e_k = [it for it in eng if it.get("kind") == k]
            d_k = [it for it in e_k if it.get("outcome") == "delivered"]
            st["engaged_by_kind"][k] = (len(e_k), len(d_k))
        st["engaged_delivered"] = len(eng_done)
        st["declined"] = len(declined)
        resolved = st["delivered"] + st["dropped"] + st["missed"]
        row = dict(ep=ep, belt=args.belt, seconds=round(u._t, 1),
                   spawned=st["spawned"], delivered=st["delivered"],
                   dropped=st["dropped"], missed=st["missed"], picked=st["picked"],
                   switches=ctrl.switches,
                   success_rate=(st["delivered"] / resolved) if resolved else 0.0,
                   engaged=st["engaged"], engaged_delivered=st["engaged_delivered"],
                   engaged_success=(len(eng_done) / len(eng_res)) if eng_res else 0.0,
                   declined=st["declined"], handle_s=st["handle_s"],
                   throughput_per_min=60.0 * st["delivered"] / max(u._t, 1e-6),
                   per_kind=st["per_kind"])
        runs.append(row)
        print(f"[ep{ep}] {row['seconds']:.0f}s 上料 {row['spawned']} | 送达 {row['delivered']} "
              f"| 机会内 {row['engaged_delivered']}/{row['engaged']} "
              f"= {(row['engaged_delivered']/row['engaged']*100) if row['engaged'] else 0:.0f}% "
              f"| 让给同伴 {row['declined']} | 单件 {row['handle_s']:.1f}s "
              f"掉落 {row['dropped']} 漏过 {row['missed']} | 抓取过 {row['picked']} | "
              f"成功率 {row['success_rate']*100:.1f}% | 吞吐 {row['throughput_per_min']:.2f} 件/min "
              f"| 换目标 {row['switches']}", flush=True)
        print("      每类: " + "  ".join(
            f"{k} {v['delivered']}/{v['spawned']}(drop {v['dropped']}) "
            for k, v in row["per_kind"].items()), flush=True)
        ebk = st.get("engaged_by_kind", {})
        print("      机会内每类: " + "  ".join(
            f"{k} {ebk.get(k, (0, 0))[1]}/{ebk.get(k, (0, 0))[0]}" for k in KINDS), flush=True)

    n = len(runs)
    agg = {k: sum(r[k] for r in runs) for k in
           ("spawned", "delivered", "dropped", "missed", "picked")}
    resolved = agg["delivered"] + agg["dropped"] + agg["missed"]
    print(f"\n== 连续作业 @{args.belt} m/s，间隔 {args.gap_min}–{args.gap_max} m，"
          f"尺寸 ±{int((args.scale_max-1)*100)}% ==")
    print(f"总上料 {agg['spawned']} | 送达 {agg['delivered']} | 掉落 {agg['dropped']} | "
          f"漏过 {agg['missed']} | 成功率 {agg['delivered']/max(resolved,1)*100:.1f}% | "
          f"吞吐 {np.mean([r['throughput_per_min'] for r in runs]):.2f} 件/min")
    eng = sum(r["engaged"] for r in runs); engd = sum(r["engaged_delivered"] for r in runs)
    print(f"机会内成功率: {engd}/{eng} = {(engd / eng * 100) if eng else 0:.1f}% | "
          f"让给同伴（未接管即漏检）: {sum(r['declined'] for r in runs)} 件")
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
