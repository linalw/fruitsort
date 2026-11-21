"""FruitSortStream-v1：**连续作业**评测环境（传送带不停、间隔随机、尺寸随机）。

与 `FruitSort-v1` 的区别
------------------------
`FruitSort-v1` 每种水果只有一个 actor，一次只放一颗；成功的定义是"四类全部入框"。
连续作业需要：

1. **多件在飞行中**：每种水果一个候选池（默认 3 个 actor，共 12 件），
   上料、清空、再上料循环使用；
2. **随机间隔**：相邻两件的间距 `U(gap_range)`（米），换算成秒；
3. **随机尺寸**：每个 slot 在创建时抽一个 ±15% 的缩放（正常范围内的果径变化）；
4. **按件统计**：spawned / delivered / dropped / missed / picked，供吞吐与成功率计算。

关键约定
--------
- `self.pool[kind]` 是真实在用的 actor 列表；`self.fruit_actors[kind]` 保持"每种一个"的旧语义，
  由评测脚本指向"该类最近的一件"——这样 `build_obs` 与 `ActPureRunner` 不用改。
- 本环境只支持 `num_envs == 1`（评测用）。
"""
from __future__ import annotations

import numpy as np
import sapien
import torch

from mani_skill.utils.registration import register_env

from ..assets.fruits import build_fruit
from .fruit_sort import ALL_KINDS, FruitSortEnv


@register_env("FruitSortStream-v1", max_episode_steps=7200)
class FruitStreamEnv(FruitSortEnv):
    """连续上料的传送带分拣场景。"""

    def __init__(
        self,
        *args,
        slots_per_kind: int = 3,
        scale_range: tuple = (0.85, 1.15),
        gap_range: tuple = (0.60, 1.20),
        stream_x: float = -0.85,
        always_stock: bool = False,
        **kwargs,
    ):
        self.slots_per_kind = int(slots_per_kind)
        self.scale_range = tuple(scale_range)
        self.gap_range = tuple(gap_range)
        self.stream_x = float(stream_x)
        # True：某类空了立刻补一件（录制训练数据用：保证"四类都在场"，图文一致）
        self.always_stock = bool(always_stock)
        super().__init__(*args, **kwargs)
        if self.num_envs != 1:
            raise ValueError("FruitSortStream-v1 只支持 num_envs=1（评测场景）")

    # ------------------------------------------------------------------ #
    # 场景：在父类基础上给每类再建 (slots-1) 个不同尺寸的 actor
    # ------------------------------------------------------------------ #
    def _load_scene(self, options: dict):
        super()._load_scene(options)
        rng = np.random.default_rng(0)
        self.pool: dict[str, list] = {}
        self.slot_scale: dict[str, list[float]] = {}
        for kind in self.active_kinds:
            base = self.fruit_actors[kind]
            actors = [base]
            scales = [1.0]
            for i in range(1, self.slots_per_kind):
                s = float(rng.uniform(*self.scale_range))
                actors.append(build_fruit(self.scene, kind, f"fruit_{kind}_{i}",
                                          sapien.Pose(p=[0, 0, -5]), scale=s))
                scales.append(s)
            self.pool[kind] = actors
            self.slot_scale[kind] = scales
            self.fruit_actors[kind] = base          # 旧语义：代表件

    # ------------------------------------------------------------------ #
    # Episode：把全部 slot 收到台下，安排首次上料
    # ------------------------------------------------------------------ #
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # 评测侧可能把 fruit_actors[kind] 换成了"观测替身"（缺类时的 ghost），
        # reset 前必须先恢复成真实 actor，否则父类的 set_pose 会崩
        for kind in self.active_kinds:
            self.fruit_actors[kind] = self.pool[kind][0]
        super()._initialize_episode(env_idx, options)
        self._t = 0.0
        self._substeps = 0
        self.stream_stats = {
            "spawned": 0, "delivered": 0, "dropped": 0, "missed": 0, "picked": 0,
            "per_kind": {k: {"spawned": 0, "delivered": 0, "dropped": 0,
                             "missed": 0, "picked": 0} for k in self.active_kinds},
            "events": [],           # [(t, kind, event, item_id)]
            "items": {},            # item_id -> dict
        }
        self._item_seq = 0
        self._slots = {k: [dict(live=False, item=None) for _ in range(self.slots_per_kind)]
                       for k in self.active_kinds}
        belt = float(self.belt_speed[0])
        # 单一来料流：随机类别、随机间隔（更贴近"传送带上混着来"）
        self._next_spawn = 0.0    # 立刻开始补第一批（四类各一件）
        for kind in self.active_kinds:
            for a in self.pool[kind]:
                a.set_pose(sapien.Pose(p=[0.0, 0.0, -5.0]))
                a.set_linear_velocity(torch.zeros(1, 3))
                a.set_angular_velocity(torch.zeros(1, 3))
        # 开局按训练时布局铺 4 件（x 间隔 0.45），避免"某类为空"的分布外观测
        for i, kind in enumerate(self.active_kinds):
            slot = self._slots[kind][0]
            spec = self.specs[kind]
            scale = self.slot_scale[kind][0]
            x = self.stream_x - 0.45 * i
            y = float(self._episode_rng.uniform(-0.05, 0.05))
            yaw = float(self._episode_rng.uniform(-np.pi, np.pi))
            q = [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]
            actor = self.pool[kind][0]
            actor.set_pose(sapien.Pose(p=[x, y, self.belt_top_z + spec.half_height * scale + 0.002], q=q))
            actor.set_linear_velocity(torch.tensor([[belt, 0.0, 0.0]]))
            actor.set_angular_velocity(torch.zeros(1, 3))
            self._item_seq += 1
            item = dict(id=self._item_seq, kind=kind, actor=actor, scale=scale,
                        t_spawn=0.0, picked=False)
            slot.update(live=True, item=item)
            self.stream_stats["items"][item["id"]] = item
            self.stream_stats["spawned"] += 1
            self.stream_stats["per_kind"][kind]["spawned"] += 1
            self.stream_stats["events"].append((0.0, kind, "spawn", item["id"]))
        self._next_spawn = float(self._episode_rng.uniform(*self.gap_range) / max(belt, 1e-3))

    # ------------------------------------------------------------------ #
    # 每步：到点上料 + 驱动带面 + 回收/记账
    # ------------------------------------------------------------------ #
    def _before_simulation_step(self):
        self._substeps += 1
        dt = 1.0 / float(getattr(self, "sim_freq", 120))
        self._t = self._substeps * dt
        belt = float(self.belt_speed[0])

        # ---- 上料（单一来料流，随机类别）----
        if belt > 0 and self._t >= self._next_spawn:
            free_kinds = [k for k in self.active_kinds
                          if any(not s["live"] for s in self._slots[k])]
            if free_kinds:
                # 优先补齐"这一类当前一件都没有"的类别：策略是在"四类都在场"的观测下训练的，
                # 缺类会让 build_obs 出现 active=0 的分布外输入（实测策略完全不动）
                missing = [k for k in free_kinds
                           if not any(s["live"] for s in self._slots[k])]
                kind = str(self._episode_rng.choice(missing or free_kinds))
                slot = next(s for s in self._slots[kind] if not s["live"])
                self._spawn(kind, slot, belt)

        # ---- 带面驱动 + 回收 ----
        for kind in self.active_kinds:
            for slot in self._slots[kind]:
                if not slot["live"]:
                    continue
                actor = slot["item"]["actor"]
                p = actor.pose.p
                vel = actor.linear_velocity
                on_belt = (
                    (p[:, 2] > self.belt_top_z - 0.02)
                    & (p[:, 2] < self.belt_top_z + 0.06)
                    & (p[:, 0] < self.pick_zone[1] + 0.20)
                    & (p[:, 0] > self.conveyor_spec.x_min)
                    & (p[:, 1].abs() < self.conveyor_spec.width / 2)
                )
                grasped = self.agent.is_grasping(actor)
                if bool(grasped[0]):
                    slot["item"]["picked"] = True
                drive = on_belt & ~grasped
                if bool(drive.any()):
                    target = torch.zeros_like(vel)
                    target[:, 0] = belt + 0.03
                    actor.set_linear_velocity(torch.where(drive[:, None], target, vel))
                    actor.set_angular_velocity(torch.where(
                        drive[:, None], torch.zeros_like(vel), actor.angular_velocity))
                self._recycle_if_spent(kind, slot, belt)

    # ------------------------------------------------------------------ #
    def _spawn(self, kind: str, slot: dict, belt: float, reuse_item: bool = True):
        free_idx = self._slots[kind].index(slot)
        actor = self.pool[kind][free_idx]
        scale = self.slot_scale[kind][free_idx]
        spec = self.specs[kind]
        x = self.stream_x + float(self._episode_rng.uniform(0.0, 0.05))
        y = float(self._episode_rng.uniform(-0.05, 0.05))
        yaw = float(self._episode_rng.uniform(-np.pi, np.pi))
        z = self.belt_top_z + spec.half_height * scale + 0.002
        q = [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]
        actor.set_pose(sapien.Pose(p=[x, y, z], q=q))
        actor.set_linear_velocity(torch.tensor([[belt, 0.0, 0.0]]))
        actor.set_angular_velocity(torch.zeros(1, 3))
        self._item_seq += 1
        item = dict(id=self._item_seq, kind=kind, actor=actor, scale=scale,
                    t_spawn=self._t, picked=False)
        slot.update(live=True, item=item)
        self.stream_stats["items"][item["id"]] = item
        self.stream_stats["spawned"] += 1
        self.stream_stats["per_kind"][kind]["spawned"] += 1
        self.stream_stats["events"].append((self._t, kind, "spawn", item["id"]))
        if reuse_item:
            gap_m = float(self._episode_rng.uniform(*self.gap_range))
            self._next_spawn = self._t + gap_m / max(belt, 1e-3)

    def _recycle_if_spent(self, kind: str, slot: dict, belt: float):
        item = slot["item"]
        actor = item["actor"]
        p = actor.pose.p[0].cpu().numpy()
        b = self.bins[kind].bounds if self.bins else None
        in_bin_xy = (b is not None and b[0] <= p[0] <= b[1] and b[2] <= p[1] <= b[3])
        delivered = in_bin_xy and p[2] < 0.20
        dropped = (p[2] < 0.10 and not in_bin_xy) or abs(p[1]) > 1.2
        # 带面驱动只到 pick_zone[1]+0.20（0.45），之后水果会停在带上；
        # 所以"漏过"用 0.60 判定（已经离开机械臂可达区），否则 slot 永远不回收
        # 带面驱动只到 pick_zone[1]+0.20（0.45），之后水果停下；过了 0.45 且仍贴带面 = 漏过。
        # 搬运中的水果 z 更高，且通常仍被判为 grasped，不会被误判
        missed = (p[0] > 0.45 and not in_bin_xy and p[2] < self.belt_top_z + 0.08
                  and not bool(self.agent.is_grasping(actor)[0]))
        if not (delivered or dropped or missed):
            return
        ev = "delivered" if delivered else ("dropped" if dropped else "missed")
        self.stream_stats[ev] += 1
        self.stream_stats["per_kind"][kind][ev] += 1
        if item["picked"]:
            self.stream_stats["picked"] += 1
            self.stream_stats["per_kind"][kind]["picked"] += 1
        self.stream_stats["events"].append((self._t, kind, ev, item["id"]))
        item.update(t_end=self._t, outcome=ev)
        actor.set_pose(sapien.Pose(p=[0.0, 0.0, -5.0]))
        actor.set_linear_velocity(torch.zeros(1, 3))
        actor.set_angular_velocity(torch.zeros(1, 3))
        slot.update(live=False, item=None)
        # 注意：这里**不**立刻补件。曾经为了"四类都在场"在回收时立即补一件，
        # 结果 gap_range 完全失效（喂料率被传送时间锁死 ≈26 件/min），
        # 正确做法是按 gap 调度、上料时优先补"当前空缺的类别"。

    # ------------------------------------------------------------------ #
    # 观测/评估：连续作业不做"四类全入框"的布尔判定
    # ------------------------------------------------------------------ #
    def evaluate(self):
        zero = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return {
            "success": zero,
            "fail": zero,
            "dropped": zero,
            "belt_speed": self.belt_speed,
            "instruction": getattr(self, "instruction", [""])[: self.num_envs],
        }

    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs, device=self.device)
