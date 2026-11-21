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
    urdf_config = dict()

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, np.pi * 3 / 4, np.pi / 4,
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
            stiffness=400.0,
            damping=40.0,
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

    def is_grasping(self, obj=None, min_force: float = 0.5, **kwargs):
        out = torch.zeros(self.robot.get_qpos().shape[0], dtype=torch.bool)
        try:
            tips = [l for l in self.robot.links if "head" in l.name]
            f = sum(float(t.get_net_contact_forces().abs().sum()) for t in tips)
            if f > min_force:
                out[:] = True
        except Exception:  # noqa: BLE001
            pass
        return out
