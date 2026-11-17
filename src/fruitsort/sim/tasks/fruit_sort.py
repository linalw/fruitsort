"""FruitSort-v1：动态传送带水果分拣任务（ManiSkill3 / SAPIEN）。

场景
----
- 4m 长传送带，带面高度 0.78m，水果从 -x 端向 +x 端匀速运动
- Franka Panda 位于传送带一侧（y = -0.62），面向带面
- 左右各两个分拣框，按水果类别着色（苹果/草莓/香蕉/梨）

任务
----
把每个水果抓起并放入对应的框。全部入框才算成功；水果掉到地面或飞出带面算失败。

观测
----
- 视觉：overhead / side 两路 RGB-D（obs_mode="rgbd"）
- 状态：TCP 位姿、关节、各水果位姿与速度、带速（obs_mode="state" 时可用）
"""
from __future__ import annotations

from typing import Any, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from ..assets.fruits import build_fruit, load_specs, spawn_pose
from ..conveyor import ConveyorSpec, build_bins, build_conveyor

ALL_KINDS = ("apple", "strawberry", "banana", "pear")

DOC = """**任务**：动态传送带水果分拣。

水果以随机间隔从传送带一端进入，机器人需要：
1. 识别目标水果类别；
2. 预测其运动并在拦截点抓取；
3. 放入对应类别的分拣框。

**随机化**：水果初始 y 位置、朝向、带速 [0.08, 0.18] m/s、初始关节噪声。
**成功**：所有水果都进入正确分拣框。
**失败**：任一水果掉到地面。
"""


@register_env("FruitSort-v1", max_episode_steps=1200)
class FruitSortEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    belt_top_z = 0.45
    robot_base = (0.0, -0.50, 0.0)
    pick_zone = (-0.25, 0.25)

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        active_kinds: tuple[str, ...] = ALL_KINDS,
        with_bins: bool = True,
        belt_speed_range=(0.08, 0.18),
        spawn_x: float = -0.75,
        spawn_spacing: float = 0.45,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.active_kinds = tuple(active_kinds)
        self.with_bins = with_bins
        self.belt_speed_range = belt_speed_range
        self.spawn_x = spawn_x
        self.spawn_spacing = spawn_spacing
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ------------------------------------------------------------------ #
    # 场景
    # ------------------------------------------------------------------ #
    @property
    def _default_sensor_configs(self):
        return [
            CameraConfig(
                "overhead",
                sapien_utils.look_at([0.0, -0.05, 2.45], [0.0, -0.05, self.belt_top_z]),
                640, 480, 1.02, 0.01, 12,
            ),
            CameraConfig(
                "side",
                sapien_utils.look_at([1.35, -1.30, 1.15], [0.0, -0.10, self.belt_top_z]),
                640, 480, np.pi / 3.2, 0.01, 10,
            ),
        ]

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig(
            "render_camera",
            sapien_utils.look_at([1.45, -1.50, 1.25], [0.0, -0.15, self.belt_top_z]),
            1280, 720, np.pi / 3, 0.01, 10,
        )

    def _load_agent(self, options: dict):
        q = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
        super()._load_agent(options, sapien.Pose(p=list(self.robot_base), q=q))

    def _load_scene(self, options: dict):
        self.conveyor_spec = ConveyorSpec(top_z=self.belt_top_z)
        self.conveyor_actor = build_conveyor(self.scene, self.conveyor_spec)
        self.bins = build_bins(self.scene) if self.with_bins else {}

        # 地面
        ground_mat = sapien.physx.PhysxMaterial(0.9, 0.8, 0.0)
        gm = sapien.render.RenderMaterial()
        gm.base_color = [0.22, 0.24, 0.26, 1.0]
        gm.roughness = 0.95
        fb = self.scene.create_actor_builder()
        fb.set_initial_pose(sapien.Pose())
        fb.add_box_collision(half_size=(6, 6, 0.01), pose=sapien.Pose(p=[0, 0, -0.01]), material=ground_mat)
        fb.add_box_visual(half_size=(6, 6, 0.01), pose=sapien.Pose(p=[0, 0, -0.01]), material=gm)
        fb.build_static(name="ground")

        self.specs = load_specs()
        self.fruit_actors = {}
        for kind in self.active_kinds:
            actor = build_fruit(self.scene, kind, f"fruit_{kind}", sapien.Pose(p=[0, 0, -5]))
            actor.set_angular_damping(1.0)
            actor.set_linear_damping(0.05)
            self.fruit_actors[kind] = actor

        self.instruction = [self._instruction_text() for _ in range(self.num_envs)]

    def _instruction_text(self) -> str:
        names = "、".join(self.specs[k].label_zh for k in self.active_kinds)
        return f"把传送带上的{names}分别放入对应的分拣框。"

    # ------------------------------------------------------------------ #
    # Episode
    # ------------------------------------------------------------------ #
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.belt_speed = torch.empty(b, device=self.device).uniform_(*self.belt_speed_range)

            # 抬起的 home 位姿：避免从传送带下方出发
            qpos = np.array([0.0, -0.9, 0.0, -1.9, 0.0, 1.0, 0.8, 0.04, 0.04])
            qpos = np.tile(qpos, (b, 1))
            qpos[:, :7] += self._episode_rng.normal(0, self.robot_init_qpos_noise, (b, 7))
            self.agent.reset(qpos)
            self.agent.robot.set_pose(
                sapien.Pose(p=list(self.robot_base), q=[np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
            )

            for i, kind in enumerate(self.active_kinds):
                spec = self.specs[kind]
                x = self.spawn_x - i * self.spawn_spacing + torch.rand(b, device=self.device) * 0.05
                y = (torch.rand(b, device=self.device) * 2 - 1) * 0.05
                z = torch.full((b,), self.belt_top_z + spec.half_height + 0.002, device=self.device)
                yaw = (torch.rand(b, device=self.device) * 2 - 1) * np.pi
                qs = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw),
                                  torch.zeros_like(yaw), torch.sin(yaw / 2)], dim=1)
                pose = Pose.create_from_pq(torch.stack([x, y, z], dim=1), qs)
                actor = self.fruit_actors[kind]
                actor.set_pose(pose)
                actor.set_linear_velocity(torch.zeros(b, 3, device=self.device))
                actor.set_angular_velocity(torch.zeros(b, 3, device=self.device))

            self._done: dict[str, torch.Tensor] = {
                k: torch.zeros(b, dtype=torch.bool, device=self.device) for k in self.active_kinds
            }
            self._ever_grasped = {k: torch.zeros(b, dtype=torch.bool, device=self.device)
                                  for k in self.active_kinds}

    # ------------------------------------------------------------------ #
    # 传送带驱动
    # ------------------------------------------------------------------ #
    def _before_simulation_step(self):
        """传送带：低摩擦带面 + 直接给定带速（等效伺服皮带）。

        只约束带上的水果；被抓取或越过工作区后不再干预。
        """
        if getattr(self, "fruit_actors", None) is None:
            return
        if float(self.belt_speed[0]) <= 0:  # 标定/静态场景：不驱动
            return
        for kind, actor in self.fruit_actors.items():
            p = actor.pose.p
            vel = actor.linear_velocity
            on_belt = (
                (p[:, 2] > self.belt_top_z - 0.02)
                & (p[:, 2] < self.belt_top_z + 0.30)
                & (p[:, 0] < self.pick_zone[1] + 0.20)
                & (p[:, 0] > self.conveyor_spec.x_min)
                & (p[:, 1].abs() < self.conveyor_spec.width / 2)
            )
            grasped = self.agent.is_grasping(actor)
            drive = on_belt & ~grasped
            if not bool(drive.any()):
                continue
            target = torch.zeros_like(vel)
            target[:, 0] = self.belt_speed + 0.03  # 补偿带面摩擦造成的速度衰减
            actor.set_linear_velocity(torch.where(drive[:, None], target, vel))
            actor.set_angular_velocity(torch.where(
                drive[:, None], torch.zeros_like(vel), actor.angular_velocity))

    # ------------------------------------------------------------------ #
    # 观测 / 评估
    # ------------------------------------------------------------------ #
    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp_pose.raw_pose,
            belt_speed=self.belt_speed.unsqueeze(-1),
        )
        if "state" in self.obs_mode:
            obs["qpos"] = self.agent.robot.get_qpos()
            for kind, actor in self.fruit_actors.items():
                obs[f"{kind}_pose"] = actor.pose.raw_pose
                obs[f"{kind}_vel"] = actor.linear_velocity
        return obs

    def _in_bin(self, kind: str) -> torch.Tensor:
        """在框内、高度足够低且基本静止（避免下落途中误判成功）。"""
        actor = self.fruit_actors[kind]
        p = actor.pose.p
        v = actor.linear_velocity
        b = self.bins[kind].bounds
        inside_xy = (
            (p[:, 0] >= b[0]) & (p[:, 0] <= b[1])
            & (p[:, 1] >= b[2]) & (p[:, 1] <= b[3])
        )
        settled = (p[:, 2] <= 0.16) & (torch.linalg.norm(v, dim=1) < 0.10)
        return inside_xy & settled

    def _in_bin_xy(self, kind: str) -> torch.Tensor:
        p = self.fruit_actors[kind].pose.p
        b = self.bins[kind].bounds
        return (
            (p[:, 0] >= b[0]) & (p[:, 0] <= b[1])
            & (p[:, 1] >= b[2]) & (p[:, 1] <= b[3])
        )

    def evaluate(self):
        if not self.bins:
            zero = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return {"success": zero, "fail": zero, "dropped": zero,
                    "belt_speed": self.belt_speed, "instruction": self.instruction}
        in_bin = {k: self._in_bin(k) for k in self.active_kinds}
        success = torch.stack(list(in_bin.values()), dim=1).all(dim=1)
        dropped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for kind, actor in self.fruit_actors.items():
            z = actor.pose.p[:, 2]
            y = actor.pose.p[:, 1]
            on_floor = ((z < 0.10) | (y.abs() > 1.2)) & ~self._in_bin_xy(kind)
            dropped |= on_floor
        info = {
            "success": success,
            "fail": dropped & ~success,
            "dropped": dropped,
            "belt_speed": self.belt_speed,
            "instruction": self.instruction,
        }
        for kind, v in in_bin.items():
            info[f"in_bin_{kind}"] = v
        return info

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        if not self.bins:
            return torch.zeros(self.num_envs, device=self.device)
        reward = torch.zeros(self.num_envs, device=self.device)
        for kind, actor in self.fruit_actors.items():
            tcp = self.agent.tcp_pose.p
            dist = torch.linalg.norm(actor.pose.p - tcp, dim=1)
            reward += 1 - torch.tanh(3 * dist)
            reward += 1.0 * info[f"in_bin_{kind}"]
            reward += 0.5 * self.agent.is_grasping(actor)
        reward[info["success"]] = 8.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info) / 8.0


FruitSortEnv.__doc__ = DOC
