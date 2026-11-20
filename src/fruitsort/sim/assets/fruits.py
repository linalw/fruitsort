"""水果物理/外观规格与 SAPIEN Actor 构建。

尺寸、质量、摩擦、颜色参考真实水果（见 docs/sim-assets.md）。
单位：米 / 千克。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import sapien
import sapien.render
import torch
import trimesh

from mani_skill.utils.structs.pose import Pose

from . import fruit_meshes


@dataclass
class FruitSpec:
    name: str
    mesh: Path
    mass: float            # kg
    color: tuple           # RGB, 0-1
    roughness: float
    metallic: float
    static_friction: float
    dynamic_friction: float
    restitution: float
    half_height: float = 0.0   # 生成后按网格实测
    density: float = 1000.0    # 由 mass / 凸包体积反推

    @property
    def label_zh(self) -> str:
        return {"apple": "苹果", "pear": "梨", "strawberry": "草莓", "banana": "香蕉"}[self.name]


FRUIT_COLORS = {
    # 参考 sRGB：富士苹果深红、黄冠梨黄绿、草莓鲜红、香蕉黄
    "apple": (0.62, 0.04, 0.06),
    "pear": (0.78, 0.72, 0.32),
    "strawberry": (0.80, 0.06, 0.08),
    "banana": (0.92, 0.76, 0.22),
}


def load_specs() -> dict[str, FruitSpec]:
    return _load_specs_cached()


@lru_cache(maxsize=1)
def _load_specs_cached() -> dict[str, FruitSpec]:
    paths = fruit_meshes.build_all()
    specs = {}
    raw = {
        "apple": dict(mass=0.150, roughness=0.32, metallic=0.0,
                      static_friction=0.70, dynamic_friction=0.60, restitution=0.03),
        "pear": dict(mass=0.160, roughness=0.40, metallic=0.0,
                     static_friction=1.00, dynamic_friction=0.90, restitution=0.03),
        "strawberry": dict(mass=0.018, roughness=0.45, metallic=0.0,
                           static_friction=0.60, dynamic_friction=0.50, restitution=0.03),
        "banana": dict(mass=0.110, roughness=0.55, metallic=0.0,
                       static_friction=1.00, dynamic_friction=0.90, restitution=0.03),
    }
    for kind, p in paths.items():
        mesh = trimesh.load(p, force="mesh")
        hull = mesh.convex_hull
        specs[kind] = FruitSpec(
            name=kind,
            mesh=p,
            color=FRUIT_COLORS[kind] + (1.0,),
            half_height=float(mesh.extents[2] / 2),
            density=float(raw[kind]["mass"] / max(hull.volume, 1e-9)),
            **raw[kind],
        )
    return specs


def build_fruit(
    scene,
    kind: str,
    name: str,
    pose: sapien.Pose,
    spec: FruitSpec | None = None,
) -> "Actor":  # noqa: F821
    """在场景中创建一个水果 actor（动态刚体）。"""
    spec = spec or load_specs()[kind]
    # FRICTION_SCALE：接触摩擦倍率，用来模拟"软指/胶垫"（1.0 = 原始材质）
    fs = float(globals().get("FRICTION_SCALE", 1.0))
    mat = sapien.physx.PhysxMaterial(
        spec.static_friction * fs, spec.dynamic_friction * fs, spec.restitution
    )
    rmat = sapien.render.RenderMaterial()
    rmat.base_color = spec.color
    rmat.roughness = spec.roughness
    rmat.metallic = spec.metallic

    builder = scene.create_actor_builder()
    builder.set_initial_pose(pose)
    builder.add_visual_from_file(str(spec.mesh), material=rmat)
    builder.add_convex_collision_from_file(
        str(spec.mesh), material=mat, density=spec.density
    )
    actor = builder.build(name=name)
    actor.set_pose(pose)
    return actor


def spawn_pose(x: float, y: float, z_surface: float, yaw: float,
               spec: FruitSpec) -> sapien.Pose:
    """水果底部贴住 z_surface。"""
    q = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    return sapien.Pose(p=[x, y, z_surface + spec.half_height + 0.001], q=q)
