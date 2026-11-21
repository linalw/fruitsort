#!/usr/bin/env python
"""把 Allegro 灵巧手合并到 Franka Panda 手腕上，生成 `assets/urdf/panda_allegro.urdf`。

做法
----
1. 读 panda_v2.urdf，截断在 `panda_hand_joint` 之前（去掉平行夹爪）；
2. 读 allegro_hand_right_glb.urdf，把所有 link/joint 名加 `al_` 前缀，mesh 路径改成绝对路径；
3. 加一个 fixed joint：`panda_link8 → al_base`（偏移可调）；
4. 写出合并后的 URDF。

用法:
    python scripts/make_panda_allegro.py [--z 0.06] [--pitch 0.0]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MS = Path("/home/ubuntu/linalw/App/minconda3/envs/maniskill3/lib/python3.10/site-packages/mani_skill")
PANDA = MS / "assets/robots/panda/panda_v2.urdf"
DCLAW = MS / "assets/robots/allegro/allegro_hand_right_glb.urdf"
OUT = ROOT / "assets/urdf/panda_allegro.urdf"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", type=float, default=0.02, help="腕到掌的 z 偏移 (m)")
    ap.add_argument("--pitch", type=float, default=3.141592653589793, help="绕 x 翻转，让手指朝下")
    args = ap.parse_args()

    panda = PANDA.read_text()
    # 1) 截断掉平行夹爪（从 panda_hand_joint 到文件末尾的 </robot> 之前全部丢弃）
    cut = panda.index('<joint name="panda_hand_joint"')
    panda = panda[:cut]

    claw = DCLAW.read_text()
    claw = re.sub(r"<\?xml[^>]*\?>", "", claw)     # 去掉内嵌的 XML 声明（否则合并后非法）
    # 2) mesh 绝对路径
    claw = claw.replace('filename="meshes/', f'filename="{DCLAW.parent}/meshes/')
    # 3) 前缀
    claw = re.sub(r'(<link\s+name=")([^"]+)"', r'\1al_\2"', claw)
    claw = re.sub(r'(<joint\s+name=")([^"]+)"', r'\1al_\2"', claw)
    claw = re.sub(r'(<parent\s+link=")([^"]+)"', r'\1al_\2"', claw)
    claw = re.sub(r'(<child\s+link=")([^"]+)"', r'\1al_\2"', claw)
    claw = re.sub(r"<robot[^>]*>", "", claw).replace("</robot>", "")

    merged = panda + f'''
  <!-- 虚拟 TCP：与 panda 原 panda_hand_tcp 完全同位，复用现有 ScriptedSorter 的几何约定 -->
  <link name="panda_tcp"/>
  <joint name="panda_tcp_joint" type="fixed">
    <parent link="panda_link8"/>
    <child link="panda_tcp"/>
    <origin rpy="0 0 -0.7853981633974483" xyz="0 0 0.1034"/>
  </joint>
  <!-- ===== Allegro dexterous hand mounted on panda_link8 ===== -->
  <joint name="panda_to_allegro" type="fixed">
    <parent link="panda_link8"/>
    <child link="al_base_link"/>
    <origin xyz="0 0 {args.z}" rpy="{args.pitch} 0 0"/>
  </joint>
''' + claw + "</robot>\n"
    # 只去掉腕部两节（Allegro 指长只有 0.15 m，一般不碰前臂；留 link5/link6 保真）
    for ln in ("panda_link7", "panda_link8"):
        merged = re.sub(r'(<link\s+name="' + ln + r'"[^>]*>)(.*?)(</link>)',
                        lambda m: m.group(1) + re.sub(r"<collision.*?</collision>", "",
                                                      m.group(2), flags=re.S) + m.group(3),
                        merged, flags=re.S)
    # 掌根 link 与腕部贴合，去掉它的碰撞体（只留外观）
    merged = re.sub(r'(<link\s+name="al_base_link"[^>]*>)(.*?)(</link>)',
                    lambda m: m.group(1) + re.sub(r"<collision.*?</collision>", "", m.group(2),
                                                  flags=re.S) + m.group(3),
                    merged, flags=re.S)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(merged)
    n_links = merged.count("<link ")
    n_joints = merged.count("<joint ")
    print(f"✓ 写出 {OUT}（{n_links} links / {n_joints} joints）")
    names = re.findall(r'<link\s+name="([^"]+)"', merged)
    print("  links:", names[-6:])
    jn = re.findall(r'<joint\s+name="([^"]+)"[^>]*type="(?!fixed)([^"]+)"', merged)
    print("  可动关节:", [n for n, t in jn][:20])


if __name__ == "__main__":
    main()
