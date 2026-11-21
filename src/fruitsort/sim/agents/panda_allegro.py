"""Panda 臂 + Allegro Hand（16 DoF，论文中最常用的灵巧手）。

URDF 由 `scripts/make_panda_allegro.py` 生成：panda 去掉平行夹爪，
Allegro 加 `al_` 前缀后经 fixed joint 装到 `panda_link8`，绕 x 翻转 π 让手指朝下。
另含一个虚拟 `panda_tcp`（与 panda 原 `panda_hand_tcp` 同位），供 `pd_ee_pose` 使用。
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

HAND_JOINTS = [f"al_joint_{i}.0" for i in range(16)]
TIP_LINKS = ("al_link_15.0_tip", "al_link_3.0_tip", "al_link_7.0_tip", "al_link_11.0_tip")

# 必须放在 ManiSkill 资产目录内，否则会被当成"需下载的远端资产"而弹交互提示
URDF = Path("/home/ubuntu/linalw/App/minconda3/envs/maniskill3/lib/python3.10/"
            "site-packages/mani_skill/assets/robots/panda/panda_allegro.urdf")


@register_agent()
class PandaAllegro(Panda):
    uid = "panda_allegro"
    urdf_path = str(URDF)
    ee_link_name = "panda_tcp"
    urdf_config = dict()
    # 合并模型末端惯量变化，panda 原增益（1000/100）会让关节爆掉，降增益
    arm_stiffness = 400.0
    arm_damping = 40.0
    arm_force_limit = 200.0

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([0.66, 0.48, -0.51, -0.47, -0.03, 0.73, -1.79] + [0.0] * 16),
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
            # Allegro 官方 agent：4e2 / 1e1 / 5e1
            stiffness=400.0,
            damping=10.0,
            force_limit=50.0,
            normalize_action=True,
            use_delta=False,
        )
        for mode in list(cfgs.keys()):
            entry = cfgs[mode]
            if isinstance(entry, dict) and "gripper" in entry:
                cfgs[mode]["gripper"] = hand
        return deepcopy(cfgs)

    @property
    def tcp_pose(self):
        return self.robot.links_map["panda_tcp"].pose

    def is_grasping(self, obj=None, min_force: float = 0.5, max_dist: float = 0.07, **kwargs):
        out = torch.zeros(self.robot.get_qpos().shape[0], dtype=torch.bool)
        try:
            tips = [self.robot.links_map[n] for n in TIP_LINKS if n in self.robot.links_map]
            f = sum(t.get_net_contact_forces().norm(dim=-1) for t in tips)
            near = torch.ones_like(f, dtype=torch.bool)
            if obj is not None:
                tip_c = torch.stack([t.pose.p for t in tips], dim=0).mean(0)
                near = torch.linalg.norm(tip_c - obj.pose.p, dim=-1) < max_dist
            out = near & (f > min_force)
        except Exception:  # noqa: BLE001
            pass
        return out
