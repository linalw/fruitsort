"""Panda 臂 + D'Claw 三指灵巧手（自制裁剪 URDF，见 scripts/make_panda_dclaw.py）。

- 臂：沿用 panda 的 `pd_ee_pose`（绝对位姿），与现有 `ScriptedSorter` 兼容；
- 手：9 个关节（3 指 × 3），`pd_joint_pos` 归一化绝对位置（-1..1 映射到关节范围）；
- `tcp_pose` = D'Claw 的掌（`dc_base_link`），供任务观测/评估使用。
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.panda.panda import Panda

ROOT = Path(__file__).resolve().parents[3]
# 必须放在 ManiSkill 资产目录内，否则会被当成"需下载的远端资产"而弹交互提示
URDF = Path("/home/ubuntu/linalw/App/minconda3/envs/maniskill3/lib/python3.10/"
            "site-packages/mani_skill/assets/robots/panda/panda_dclaw.urdf")

HAND_JOINTS = [f"dc_joint_f{i}_{j}" for i in range(1, 4) for j in range(3)]


@register_agent()
class PandaDClaw(Panda):
    uid = "panda_dclaw"
    urdf_path = str(URDF)
    ee_link_name = "panda_tcp"   # 虚拟 TCP：与 panda 原 panda_hand_tcp 同位（复用现有执行器几何）
    # 合并模型（末端多了一整只三指手）惯量变化大，panda 原增益会导致关节爆炸，这里降增益
    arm_stiffness = 400.0
    arm_damping = 40.0
    arm_force_limit = 200.0
    urdf_config = dict()

    # 静息姿态必须"手在高处"：D'Claw 腕到指尖 0.28 m，用 panda 原静息姿态时
    # 指尖落在 z≈0.20（传送带体内），一开机就被几何体顶住、整臂跟不上指令。
    # 下面这组关节让 TCP 在基座上方 ~0.56 m（FK 实测），指尖 ~0.28 m，稳在带面上方。
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([0.66, 0.48, -0.51, -0.47, -0.03, 0.73, -1.79,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            pose=sapien.Pose(),
        )
    )

    @property
    def _controller_configs(self):
        cfgs = super()._controller_configs
        hand = PDJointPosControllerConfig(
            HAND_JOINTS,
            lower=None,
            upper=None,
            # 官方 D'Claw agent 用 100/10/20；用 400/40 时手关节会累积发散（实测到 53 rad）
            stiffness=100.0,
            damping=10.0,
            force_limit=20.0,
            normalize_action=True,
            use_delta=False,
        )
        for mode in list(cfgs.keys()):
            entry = cfgs[mode]
            if isinstance(entry, dict) and "gripper" in entry:
                cfgs[mode]["gripper"] = hand
        return deepcopy(cfgs)

    # ------------------------------------------------------------------ #
    def _after_init(self):
        try:
            self.robot.links_map["dc_base_link"]
        except Exception:  # noqa: BLE001
            pass

    @property
    def tcp_pose(self):
        return self.robot.links_map["panda_tcp"].pose

    def is_grasping(self, obj=None, min_force: float = 0.5, max_dist: float = 0.07, **kwargs):
        """必须"手指有接触力 **且** 手指离该物体足够近"。

        早期版本只看接触力，结果手指碰到传送带就返回 True，
        而任务里的传送带驱动用的是 `~is_grasping(actor)` → 整条带子被误停。
        """
        out = torch.zeros(self.robot.get_qpos().shape[0], dtype=torch.bool)
        try:
            tips = [l for l in self.robot.links if "head" in l.name]
            f = sum(t.get_net_contact_forces().norm(dim=-1) for t in tips)
            near = torch.ones_like(f, dtype=torch.bool)
            if obj is not None:
                tip_c = torch.stack([t.pose.p for t in tips], dim=0).mean(0)
                near = torch.linalg.norm(tip_c - obj.pose.p, dim=-1) < max_dist
            out = near & (f > min_force)
        except Exception:  # noqa: BLE001
            pass
        return out
