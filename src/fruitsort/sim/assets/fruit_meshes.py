"""程序化生成水果网格（GLB），保证可复现、无外部下载。

所有网格以几何中心为原点、+Z 朝上，导出到 assets/meshes/。
碰撞体由 SAPIEN 用凸包近似（水果近似凸体，抓取仿真足够）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[4]
MESH_DIR = ROOT / "assets" / "meshes"


def _center(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """把包围盒中心移到原点。"""
    mesh.apply_translation(-mesh.bounding_box.centroid)
    return mesh


def make_apple() -> trimesh.Trimesh:
    """富士苹果：直径 ~77mm，高 ~70mm，顶部有浅凹。"""
    m = trimesh.creation.icosphere(subdivisions=4, radius=0.033)
    m.apply_scale([1.0, 1.0, 1.0])
    v = m.vertices.copy()
    d = np.linalg.norm(v[:, :2], axis=1)
    top = v[:, 2] > 0
    # 顶部凹陷
    v[top, 2] -= 0.006 * np.exp(-((d[top] / 0.011) ** 2))
    # 底部略微收窄，并压出小平面（避免球体在传送带上打滑滚动）
    bot = v[:, 2] < -0.02
    v[bot, 2] += 0.004 * np.exp(-((d[bot] / 0.014) ** 2))
    v[:, 2] = np.maximum(v[:, 2], -0.0315)
    m.vertices = v
    return _center(m)


def make_pear() -> trimesh.Trimesh:
    """圆梨：高 ~70mm，最大半径 ~30mm（更圆的回转体，避免在带面上翻倒侧滚）。"""
    profile = np.array([
        (0.000, 0.000), (0.014, 0.000), (0.024, 0.004), (0.029, 0.012),
        (0.030, 0.022), (0.029, 0.032), (0.026, 0.042), (0.021, 0.052),
        (0.015, 0.062), (0.010, 0.068), (0.005, 0.0695), (0.000, 0.070),
    ])
    m = trimesh.creation.revolve(profile, sections=64, cap=True)
    return _center(m)


def make_strawberry() -> trimesh.Trimesh:
    """草莓：高 ~42mm，最大直径 ~36mm，下尖上圆。"""
    profile = np.array([
        (0.000, 0.000), (0.008, 0.000), (0.013, 0.003), (0.0165, 0.008),
        (0.0185, 0.015), (0.0185, 0.023), (0.0170, 0.031), (0.0135, 0.037),
        (0.0080, 0.041), (0.000, 0.042),
    ])
    m = trimesh.creation.revolve(profile, sections=48, cap=True)
    # 表面轻微起伏，避免完美回转体
    v = m.vertices.copy()
    theta = np.arctan2(v[:, 1], v[:, 0])
    z = v[:, 2]
    bump = 0.0006 * np.sin(12 * theta) * np.clip(1 - np.abs(z) / 0.03, 0, 1)
    v[:, :2] *= (1 + bump)[:, None]
    m.vertices = v
    return _center(m)


def _tube(path: np.ndarray, radii: np.ndarray, sections: int = 20) -> trimesh.Trimesh:
    """沿路径扫掠圆截面（平行传输标架），支持逐点半径。"""
    n = len(path)
    tangents = np.gradient(path, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

    ref = np.array([0.0, 0.0, 1.0])
    normal = np.cross(tangents[0], ref)
    if np.linalg.norm(normal) < 1e-6:
        normal = np.cross(tangents[0], np.array([0.0, 1.0, 0.0]))
    normal /= np.linalg.norm(normal)

    angles = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    rings = []
    for i in range(n):
        if i > 0:
            v = np.cross(tangents[i - 1], tangents[i])
            s = np.linalg.norm(v)
            if s > 1e-8:
                c = float(np.dot(tangents[i - 1], tangents[i]))
                axis = v / s
                K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                              [-axis[1], axis[0], 0]])
                Rm = np.eye(3) + s * K + (1 - c) * (K @ K)
                normal = Rm @ normal
        binormal = np.cross(tangents[i], normal)
        rings.append(path[i] + radii[i] * (
            np.cos(angles)[:, None] * normal + np.sin(angles)[:, None] * binormal
        ))
    rings = np.stack(rings)

    verts = rings.reshape(-1, 3)
    faces = []
    for i in range(n - 1):
        for j in range(sections):
            a = i * sections + j
            b = i * sections + (j + 1) % sections
            c = (i + 1) * sections + (j + 1) % sections
            d = (i + 1) * sections + j
            faces += [[a, b, c], [a, c, d]]
    start_c = len(verts)
    verts = np.vstack([verts, path[0], path[-1]])
    for j in range(sections):
        b = (j + 1) % sections
        faces.append([start_c, b, j])
        a0 = (n - 1) * sections + j
        b0 = (n - 1) * sections + (j + 1) % sections
        faces.append([start_c + 1, a0, b0])
    return trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)


def make_banana() -> trimesh.Trimesh:
    """香蕉：弧长 ~180mm，截面直径 ~32mm，沿 X 方向弯曲。"""
    R = 0.115
    thetas = np.linspace(-0.78, 0.78, 40)
    path = np.stack([
        R * np.sin(thetas),
        R * (1 - np.cos(thetas)) * 0.55,
        np.zeros_like(thetas),
    ], axis=1)
    t = np.linspace(0, 1, len(path))
    radii = 0.016 * (0.55 + 0.45 * np.sin(np.pi * t) ** 0.35)
    m = _tube(path, radii, sections=20)
    return _center(m)


BUILDERS = {
    "apple": make_apple,
    "pear": make_pear,
    "strawberry": make_strawberry,
    "banana": make_banana,
}


def build_all(force: bool = False) -> dict[str, Path]:
    """生成全部水果 GLB，返回 {kind: path}。已存在则跳过（除非 force）。"""
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for kind, fn in BUILDERS.items():
        path = MESH_DIR / f"{kind}.glb"
        if force or not path.exists():
            mesh = fn()
            mesh.export(path)
        out[kind] = path
    return out


if __name__ == "__main__":
    for kind, p in build_all(force=True).items():
        m = trimesh.load(p, force="mesh")
        print(f"{kind:11s} {p.name:15s} 尺寸 {np.round(m.extents * 1000, 1)} mm  "
              f"体积 {m.volume * 1e6:.1f} cm³  水密={m.is_watertight}")
