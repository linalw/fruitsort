"""传送带与分拣框的 SAPIEN 场景构建。

传送带本身是静态刚体；"带面运动"通过在 env 中给带上的水果施加线速度实现
（等效于摩擦驱动，且抓取后自动停止驱动）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sapien
import sapien.render


@dataclass
class ConveyorSpec:
    length: float = 5.0
    width: float = 0.40
    top_z: float = 0.45
    thickness: float = 0.05
    rail_height: float = 0.06
    rail_width: float = 0.02

    @property
    def x_min(self) -> float:
        return -self.length / 2

    @property
    def x_max(self) -> float:
        return self.length / 2

    @property
    def y_min(self) -> float:
        return -self.width / 2

    @property
    def y_max(self) -> float:
        return self.width / 2


@dataclass
class Bin:
    kind: str
    center: np.ndarray
    size: np.ndarray          # x, y, 高度
    bounds: np.ndarray        # [xmin, xmax, ymin, ymax, zmax]

    def contains(self, p) -> bool:
        return bool(
            self.bounds[0] <= p[0] <= self.bounds[1]
            and self.bounds[2] <= p[1] <= self.bounds[3]
            and p[2] <= self.bounds[4]
        )


BIN_LAYOUT = {
    "apple": (-0.46, -0.28),
    "strawberry": (-0.46, -0.60),
    "banana": (0.46, -0.28),
    "pear": (0.46, -0.60),
}

BIN_RIM_COLORS = {
    "apple": (0.75, 0.08, 0.10),
    "strawberry": (0.90, 0.20, 0.25),
    "banana": (0.95, 0.80, 0.20),
    "pear": (0.55, 0.70, 0.25),
}


def _render_material(scene, color, roughness=0.6, metallic=0.0) -> sapien.render.RenderMaterial:
    m = sapien.render.RenderMaterial()
    m.base_color = [*color, 1.0] if len(color) == 3 else color
    m.roughness = roughness
    m.metallic = metallic
    return m


def build_conveyor(scene, spec: ConveyorSpec = ConveyorSpec(), name: str = "conveyor") -> "Actor":  # noqa: F821
    """静态传送带：带面 + 侧导轨 + 支腿 + 条纹。"""
    belt_mat = sapien.physx.PhysxMaterial(0.04, 0.03, 0.0)  # 低摩擦带面 + 外部速度驱动
    frame_mat = sapien.physx.PhysxMaterial(0.6, 0.5, 0.0)
    belt_vis = _render_material(scene, (0.09, 0.09, 0.10), roughness=0.92)
    frame_vis = _render_material(scene, (0.55, 0.57, 0.60), roughness=0.35, metallic=0.85)
    stripe_vis = _render_material(scene, (0.16, 0.16, 0.17), roughness=0.9)

    L, W, T = spec.length, spec.width, spec.thickness
    z_top = spec.top_z
    b = scene.create_actor_builder()
    b.set_initial_pose(sapien.Pose())

    # 带面
    b.add_box_collision(
        half_size=(L / 2, W / 2, T / 2),
        pose=sapien.Pose(p=[0, 0, z_top - T / 2]),
        material=belt_mat,
    )
    b.add_box_visual(
        half_size=(L / 2, W / 2, T / 2),
        pose=sapien.Pose(p=[0, 0, z_top - T / 2]),
        material=belt_vis,
    )

    # 侧导轨
    for sy in (-1, 1):
        b.add_box_collision(
            half_size=(L / 2, spec.rail_width / 2, spec.rail_height / 2),
            pose=sapien.Pose(p=[0, sy * (W / 2 + spec.rail_width / 2), z_top + spec.rail_height / 2]),
            material=frame_mat,
        )
        b.add_box_visual(
            half_size=(L / 2, spec.rail_width / 2, spec.rail_height / 2),
            pose=sapien.Pose(p=[0, sy * (W / 2 + spec.rail_width / 2), z_top + spec.rail_height / 2]),
            material=frame_vis,
        )

    # 支腿
    leg_h = z_top - T
    for sx in (-1, 1):
        for sy in (-1, 1):
            p = [sx * (L / 2 - 0.25), sy * (W / 2 - 0.06), leg_h / 2]
            b.add_box_collision(half_size=(0.04, 0.04, leg_h / 2), pose=sapien.Pose(p=p), material=frame_mat)
            b.add_box_visual(half_size=(0.04, 0.04, leg_h / 2), pose=sapien.Pose(p=p), material=frame_vis)

    # 带面条纹（纯视觉，帮助观察带速方向）
    n_stripes = int(L / 0.12)
    for i in range(n_stripes):
        x = -L / 2 + 0.06 + i * 0.12
        b.add_box_visual(
            half_size=(0.012, W / 2 - 0.01, 0.0015),
            pose=sapien.Pose(p=[x, 0, z_top + 0.0015]),
            material=stripe_vis,
        )

    actor = b.build_static(name=name)
    return actor


def build_bin(scene, kind: str, center, size=(0.24, 0.24, 0.22), wall=0.008,
              name: str | None = None) -> Bin:
    """开口上方的分拣框，带彩色边沿。"""
    cx, cy = center
    sx, sy, sh = size
    mat = sapien.physx.PhysxMaterial(0.7, 0.6, 0.0)
    body_vis = _render_material(scene, (0.30, 0.31, 0.33), roughness=0.55, metallic=0.3)
    rim_vis = _render_material(scene, BIN_RIM_COLORS[kind], roughness=0.45)

    b = scene.create_actor_builder()
    b.set_initial_pose(sapien.Pose())
    # 底板
    b.add_box_collision(half_size=(sx / 2, sy / 2, 0.005), pose=sapien.Pose(p=[cx, cy, 0.005]), material=mat)
    b.add_box_visual(half_size=(sx / 2, sy / 2, 0.005), pose=sapien.Pose(p=[cx, cy, 0.005]), material=body_vis)
    # 四面墙
    walls = [
        ((sx / 2, wall / 2, sh / 2), (cx, cy - sy / 2, sh / 2)),
        ((sx / 2, wall / 2, sh / 2), (cx, cy + sy / 2, sh / 2)),
        ((wall / 2, sy / 2, sh / 2), (cx - sx / 2, cy, sh / 2)),
        ((wall / 2, sy / 2, sh / 2), (cx + sx / 2, cy, sh / 2)),
    ]
    for half, p in walls:
        b.add_box_collision(half_size=half, pose=sapien.Pose(p=list(p)), material=mat)
        b.add_box_visual(half_size=half, pose=sapien.Pose(p=list(p)), material=body_vis)
    # 顶部彩色边沿（纯视觉）
    for half, p in walls:
        h = [half[0] * 1.02, half[1] * 1.6, 0.006]
        pp = [p[0], p[1], sh - 0.004]
        b.add_box_visual(half_size=h, pose=sapien.Pose(p=pp), material=rim_vis)

    b.build_static(name=name or f"bin_{kind}")
    bounds = np.array([
        cx - sx / 2 + wall, cx + sx / 2 - wall,
        cy - sy / 2 + wall, cy + sy / 2 - wall,
        sh - 0.005,
    ])
    return Bin(kind=kind, center=np.array([cx, cy, 0.0]), size=np.array([sx, sy, sh]), bounds=bounds)


def build_bins(scene) -> dict[str, Bin]:
    return {kind: build_bin(scene, kind, c) for kind, c in BIN_LAYOUT.items()}
