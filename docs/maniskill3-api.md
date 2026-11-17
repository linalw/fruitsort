# ManiSkill3 3.0.1 API 速查（本机源码 + 实测）

> 适用环境：`/home/ubuntu/linalw/App/minconda3/envs/maniskill3`
> `mani_skill 3.0.1` / `sapien 3.0.3` / Python 3.10.21 / RTX 5090
> 本文所有 `mani_skill/...`、`sapien/...` 路径都相对于：
>
> ```text
> /home/ubuntu/linalw/App/minconda3/envs/maniskill3/lib/python3.10/site-packages/
> ```
>
> 行号以本机安装版本为准；升级后请重新核对。
> 标注「实测」的结论都在本机跑过，命令与结果见文末附录。

---

## 0. 30 秒确认环境

```bash
cd /media/ubuntu/66A27065A2703B9D/project/Fruitsort
source scripts/env.sh maniskill3

python - <<'PY'
import mani_skill, sapien, sys
print("mani_skill", mani_skill.__version__)
print("sapien", sapien.__version__)
print("python", sys.version.split()[0])
PY
```

本机输出：

```text
mani_skill 3.0.1
sapien 3.0.3
python 3.10.21
```

版本定义在 `mani_skill/version.py:1`。渲染系统是 SAPIEN 3.0（`mani_skill/render/version.py:1`），不是 3.1。

---

## 1. 自定义任务：BaseEnv + @register_env 最小骨架

### 1.1 可直接运行的最小任务

把下面保存为 `src/fruitsort/envs/fruit_sort.py`（或任意模块），直接 `python` 运行即可。

```python
# src/fruitsort/envs/fruit_sort.py
import numpy as np
import sapien
import torch
import gymnasium as gym

import mani_skill.envs  # 注册内置任务
import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig


@register_env("FruitSort-v0", max_episode_steps=100)
class FruitSortEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        # 可选；不写也默认是 SimConfig()
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        # 用于 obs_mode="rgb"/"rgbd"/"depth" 的相机
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[0, 0, 0.05])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        # 用于 render_mode="rgb_array" 的相机
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0, 0, 0.2])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        # Panda 底座放到桌边，避免和桌面/物体初始重叠
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # 只在创建/重配置时调用；这里只 build，不做每回合随机
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0.02)
        self.table_scene.build()

        self.apple = actors.build_sphere(
            self.scene,
            radius=0.03,
            color=[0.8, 0.1, 0.1, 1],
            name="apple",
            initial_pose=sapien.Pose(p=[0, 0, 0.03]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # 每个回合 reset 时调用；env_idx 是本次要重置的环境索引
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 0] = 0.05 + 0.10 * torch.rand(b, device=self.device)
            xyz[:, 1] = -0.10 + 0.20 * torch.rand(b, device=self.device)
            xyz[:, 2] = 0.03

            q = randomization.random_quaternions(
                b, lock_x=True, lock_y=True, device=self.device
            )
            self.apple.set_pose(Pose.create_from_pq(xyz, q))

    def _get_obs_extra(self, info: dict):
        return dict(
            tcp_pose=self.agent.tcp_pose.raw_pose,   # (N, 7), wxyz
            obj_pose=self.apple.pose.raw_pose,       # (N, 7), wxyz
        )

    def evaluate(self):
        # 必须返回 batched tensor，形状 (N,)
        return dict(
            success=self.apple.pose.p[:, 2] > 0.05,
            is_grasped=self.agent.is_grasping(self.apple),
        )

    def compute_dense_reward(self, obs, action: torch.Tensor, info: dict):
        # 必须返回 (N,)
        dist = torch.linalg.norm(self.apple.pose.p - self.agent.tcp_pose.p, axis=1)
        reward = 1 - torch.tanh(5 * dist)
        reward = reward + info["is_grasped"].float()
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        # 默认 reward_mode="normalized_dense" 会调用这个函数
        return self.compute_dense_reward(obs, action, info) / 2


if __name__ == "__main__":
    env = gym.make(
        "FruitSort-v0",
        num_envs=4,
        obs_mode="rgbd",
        render_mode="rgb_array",
        sim_backend="physx_cuda",
    )
    obs, info = env.reset(seed=0)
    action = torch.zeros(env.unwrapped.action_space.shape)
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(action)
    print("action", env.unwrapped.action_space.shape)
    print("rgb", obs["sensor_data"]["base_camera"]["rgb"].shape)
    print("depth", obs["sensor_data"]["base_camera"]["depth"].shape)
    print("reward", reward.shape, "render", env.render().shape)
    env.close()
```

本机实测（4 env GPU）：

```text
action (4, 8)
rgb torch.Size([4, 128, 128, 3])
depth torch.Size([4, 128, 128, 1])
reward torch.Size([4]) render torch.Size([4, 512, 512, 3])
```

对应源码：`mani_skill/envs/minimal_template.py:15-66` 是官方最小模板；
`@register_env` 定义在 `mani_skill/utils/registration.py:192-260`。
注意 `@register_env(..., **kwargs)` 的额外 kwargs 必须 JSON 可序列化（`:204-208`），
不要传类/函数/`sapien.Pose` 之类对象。

### 1.2 各钩子的职责与 batched 语义

| 钩子 | 何时调用 | 你要做什么 | 形状/批量语义 |
| --- | --- | --- | --- |
| `_load_scene(options)` | `_reconfigure()` 中，创建/重配置时，在 `scene._setup()` 之前 | 建场景、建 actor/articulation、建 scene builder | 不按 env 分；一次建好所有并行环境 |
| `_initialize_episode(env_idx, options)` | 每次 `reset()`，在清空速度、重置 controller 之前 | 设置物体初始 pose/qpos、随机化 | `env_idx` 是 `torch.Tensor`；只改这些 env；常用 `b = len(env_idx)` |
| `evaluate()` | 每次 `get_info()`，即 `reset()` 后和每个 `step()` 后 | 计算 success/fail 及可复用中间量 | 无参数；返回的每个值形状 `(num_envs,)`，总是全量 env |
| `_get_obs_extra(info)` | 生成 state / visual obs 时 | 返回任务相关观测 | 返回 dict，每个值第一维是 `num_envs` |
| `compute_dense_reward(obs, action, info)` | `reward_mode="dense"` 时 | 稠密奖励 | 返回 `(num_envs,)` |
| `compute_normalized_dense_reward(...)` | `reward_mode="normalized_dense"`（**默认**） | 归一化奖励 | 返回 `(num_envs,)` |

关键源码位置：

- `_load_scene` 由 `_reconfigure` 调用：`mani_skill/envs/sapien_env.py:737-746`；默认空实现 `:766-768`。
- `_initialize_episode` 调用点：`mani_skill/envs/sapien_env.py:946-951`；默认空实现 `:1018-1021`。
- `evaluate` 由 `get_info` 调用：`mani_skill/envs/sapien_env.py:1146-1156`；默认空实现 `:1134-1144`。
- `_get_obs_extra` 调用点：`mani_skill/envs/sapien_env.py:546-560`、`:627-634`。
- 奖励分发：`mani_skill/envs/sapien_env.py:648-670`；默认 `normalized_dense` 在 `:128`。
- `reset()` 中 `options["env_idx"]` 的处理：`mani_skill/envs/sapien_env.py:900-905`；`_reset_mask` 设置：`:923-926`。
- `step()` / `_step_action()`：`mani_skill/envs/sapien_env.py:1042-1071`、`:1073-1132`。

`evaluate()` 为什么总是全量：`get_info()` 不接收 `env_idx`，直接 `self.evaluate()`（`sapien_env.py:1155`）。部分 reset 时也是如此，所以 `evaluate()` 里不要假设只有 `len(env_idx)` 个环境。

`_get_obs_extra` 的典型写法（来自 `PickCube`）：只在 `if "state" in self.obs_mode` 时加入物体绝对 pose，视觉模式下尽量少放（`mani_skill/envs/tasks/tabletop/pick_cube.py:132-145`）。

奖励里可以用布尔 mask 直接改值：

```python
reward[info["success"]] = 5
```

见 `mani_skill/envs/tasks/tabletop/pick_cube.py:185`。

### 1.3 `env_idx` 的正确用法

```python
def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
    with torch.device(self.device):
        b = len(env_idx)
        self.table_scene.initialize(env_idx)

        xyz = torch.zeros((b, 3), device=self.device)
        xyz[:, 0] = ...
        q = randomization.random_quaternions(b, lock_x=True, lock_y=True,
                                             device=self.device)
        self.apple.set_pose(Pose.create_from_pq(xyz, q))
```

原因：`Actor.set_pose` 在 GPU sim 下只写 `scene._reset_mask == True` 的那些 env 的 body 数据（`mani_skill/utils/structs/actor.py:389-391`）。全量 reset 时 mask 全是 True；部分 reset 时只有 `env_idx` 为 True（`sapien_env.py:923-926`）。所以：

- `xyz` 的第一维必须是 `len(env_idx)`，不能无脑写 `self.num_envs`。
- 不要在这里用 `torch.rand(self.num_envs)` 再切片，容易和 `env_idx` 对不上。
- 部分 reset 不能和 `reconfigure=True` 同时用，源码会直接报错（`sapien_env.py:900-903`）。

实测：

```text
num_envs=4, reset(seed=[10,11], options={"env_idx": [0,2]})
env 0/2 改变，env 1/3 不变  -> OK
set_pose 传 (2,7) 而 mask 有 4 个 env
  RuntimeError: shape mismatch: value tensor of shape [2, 7]
  cannot be broadcast to indexing result of shape [4, 7]
```

---

## 2. 资产：primitive、mesh、质量/摩擦/颜色、body type

### 2.1 `actors.build_*` 的签名

全部在 `mani_skill/utils/building/actors/common.py`，并通过
`mani_skill/utils/building/actors/__init__.py` 导出。

```python
build_cube(
    scene, half_size, color, name,
    body_type="dynamic", add_collision=True,
    scene_idxs=None, initial_pose=None,
)

build_box(
    scene, half_sizes, color, name,
    body_type="dynamic", add_collision=True,
    scene_idxs=None, initial_pose=None,
)

build_cylinder(
    scene, radius, half_length, color, name,
    body_type="dynamic", add_collision=True,
    scene_idxs=None, initial_pose=None,
)

build_sphere(
    scene, radius, color, name,
    body_type="dynamic", add_collision=True,
    scene_idxs=None, initial_pose=None,
)
```

行号：`common.py:70-91`（cube）、`:94-115`（box）、`:118-142`（cylinder）、`:145-166`（sphere）。

`body_type` 通过 `_build_by_type` 分派（`common.py:47-66`）：

```python
builder.build(name)          # dynamic
builder.build_static(name)   # static
builder.build_kinematic(name)# kinematic
```

注意：`build_*` 只设置颜色，**不暴露摩擦/恢复系数/质量**。需要这些参数时自己 `scene.create_actor_builder()`。

### 2.2 自定义 mesh：实际方法名不是 `add_convex_shape`

本机 SAPIEN 3.0.3 的 `sapien.ActorBuilder` **没有**
`add_convex_shape` / `add_multiple_convex_shapes` / `add_mesh`。
实测 `dir(sapien.ActorBuilder)` 只有：

```text
add_box_collision, add_box_visual,
add_capsule_collision, add_capsule_visual,
add_convex_collision_from_file,
add_cylinder_collision, add_cylinder_visual,
add_multiple_convex_collisions_from_file,
add_nonconvex_collision_from_file,
add_plane_collision, add_plane_visual,
add_sphere_collision, add_sphere_visual,
add_visual_from_file,
build, build_entity, build_kinematic, build_physx_component,
build_render_component, build_static,
set_initial_pose, set_mass_and_inertia, set_name,
set_physx_body_type, set_scene
```

对应源码在 `sapien/wrapper/actor_builder.py`：

| 方法 | 作用 | 源码 |
| --- | --- | --- |
| `add_convex_collision_from_file(filename, pose=..., scale=(1,1,1), material=..., density=1000, ...)` | 把整个 mesh 当**一个凸包**做碰撞 | `:447-474` |
| `add_multiple_convex_collisions_from_file(..., decomposition="none" \| "coacd", decomposition_params={})` | 从文件加载**多个凸包**；`decomposition="coacd"` 会先做 CoACD 凸分解 | `:476-507` |
| `add_nonconvex_collision_from_file(...)` | 三角网格碰撞（非凸） | `:509-536` |
| `add_visual_from_file(filename, pose=..., scale=..., material=None, name="")` | **只加视觉**三角网格，不加碰撞 | `:649-671` |

区别：

- `add_convex_collision_from_file`：一个凸包。水果、方块、简单道具足够；凹形物体会被“包起来”。
- `add_multiple_convex_collisions_from_file`：一个文件里含多个凸包时用；或 `decomposition="coacd"` 自动分解。**本机没装 `coacd`**，实测会报
  `ModuleNotFoundError: No module named 'coacd'`（见附录）。
- `add_nonconvex_collision_from_file`：真正的三角网格，形状最准，但动态物体上更贵；本机 GPU 4 env 实测可跑。
- `add_visual_from_file`：只负责好看，**不产生碰撞**。必须另加 collision，否则物体会穿模。

ManiSkill 自己的用法可直接抄：

- 单个凸包 + 视觉：`mani_skill/envs/tasks/fmb/fmb.py:81-86`
- 多个凸包：`mani_skill/envs/tasks/fmb/fmb.py:90-95`
- 非凸 + 视觉：`mani_skill/envs/tasks/fmb/fmb.py:118-124`
- 从 `.obj`/`.glb` 加载：`mani_skill/envs/tasks/digital_twins/bridge_dataset_eval/base_env.py:246-259`

#### trimesh 导出 + 建 ActorBuilder

```python
import trimesh
import sapien

# 1) 生成并导出 mesh（glb 或 obj 都行）
mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.04)
mesh.export("/tmp/apple.glb")

# 2) 在 _load_scene 里 build
b = self.scene.create_actor_builder()

# 碰撞：一个凸包即可
b.add_convex_collision_from_file(
    "/tmp/apple.glb",
    density=500,  # kg/m^3
    material=sapien.physx.PhysxMaterial(
        static_friction=0.8,
        dynamic_friction=0.8,
        restitution=0.05,
    ),
)

# 视觉：单独加；material 可覆盖 mesh 自带材质
b.add_visual_from_file(
    "/tmp/apple.glb",
    material=sapien.render.RenderMaterial(
        base_color=[0.8, 0.1, 0.1, 1.0],
        roughness=0.4,
        metallic=0.0,
    ),
)

b.set_initial_pose(sapien.Pose(p=[0, 0, 0.1]))
self.apple = b.build("apple")   # dynamic
```

如果物体是凹的（比如香蕉、篮子），把 `add_convex_collision_from_file` 换成：

```python
b.add_multiple_convex_collisions_from_file("/tmp/banana.glb", density=500)
# 或者先安装 coacd 再：
# b.add_multiple_convex_collisions_from_file("/tmp/banana.glb",
#     decomposition="coacd", decomposition_params=dict())
```

如果形状不能近似成凸包、又想要精确碰撞：

```python
b.add_nonconvex_collision_from_file("/tmp/basket.glb", density=500)
```

### 2.3 质量、摩擦、恢复系数、颜色、roughness

**质量**有两种方式：

1. 每个 collision shape 给 `density`（默认 1000 kg/m³），SAPIEN 自动算质量和惯量：
   ```python
b.add_box_collision(half_size=[0.05, 0.05, 0.05], density=200)
```
2. 显式设置质量和惯量（**必须在 build 前**）：
   ```python
b.set_mass_and_inertia(
    mass=0.2,                       # kg
    cmass_local_pose=sapien.Pose(), # 质心局部系
    inertia=[2e-4, 2e-4, 2e-4],     # kg*m^2，对角
)
   ```
   源码：`sapien/wrapper/actor_builder.py:123-128`。
   只给 mass 不给 inertia 会走默认 0 惯量，不建议。

读质量：`actor.mass`，返回 CPU tensor，形状 `(N,)`（`mani_skill/utils/structs/base.py:272-274`）。

**不要**在 GPU 初始化后写 `actor.mass = x`：该 setter 带 `@before_gpu_init`
（`base.py:276-280`、`mani_skill/utils/structs/decorators.py:1-13`），实测会报
`can only be called when the GPU simulation has not been initialized yet`。

**摩擦/恢复系数**：每个 collision shape 单独给 `sapien.physx.PhysxMaterial`：

```python
mat = sapien.physx.PhysxMaterial(
    static_friction=1.0,
    dynamic_friction=1.0,
    restitution=0.0,
)
b.add_box_collision(half_size=[0.3, 0.2, 0.02], material=mat, density=1000)
```

构造参数顺序是 `(static_friction, dynamic_friction, restitution)`（本机实测
`PhysxMaterial(0.9, 0.9, 0.05)` 读回 `0.9 / 0.9 / 0.05`）。
也可以关键字传参，ManiSkill 内部就是这么用的（`mani_skill/agents/robots/panda/panda.py:20-23`）。

**颜色/roughness/metallic**：用渲染材质，不是物理材质：

```python
render_mat = sapien.render.RenderMaterial(
    base_color=[0.2, 0.8, 0.2, 1.0],  # RGBA, 0~1
    roughness=0.5,
    metallic=0.0,
)
b.add_box_visual(half_size=[0.3, 0.2, 0.02], material=render_mat)
```

`RenderMaterial` 支持 `base_color`、`roughness`、`metallic`，以及
`set_base_color`、`set_roughness`、`set_metallic` 等（本机 `dir` 验证）。
视觉材质和物理摩擦完全独立，别把 `RenderMaterial` 传给 collision。

### 2.4 dynamic / kinematic / static

```python
b = self.scene.create_actor_builder()
b.add_box_collision(half_size=[0.05, 0.05, 0.05], density=500)
b.add_box_visual(half_size=[0.05, 0.05, 0.05],
                 material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 1]))
b.set_initial_pose(sapien.Pose(p=[0, 0, 0.2]))

apple   = b.build("apple")           # dynamic：受重力/碰撞/力
belt    = b.build_kinematic("belt") # kinematic：由 pose 驱动，不受力
table   = b.build_static("table")   # static：完全不动
```

源码：`mani_skill/utils/building/actor_builder.py:166-176`（`build_dynamic` / `build_kinematic` / `build_static`），
底层体类型在 `sapien/wrapper/actor_builder.py:185-195`。

注意：GPU sim 下 **static actor 不能改 pose**。`Actor.pose` setter 会直接断言：
`Static objects cannot change poses in GPU sim after environment is loaded`
（`mani_skill/utils/structs/actor.py:369-372`）。需要动的东西用 dynamic/kinematic。

**必须设 `initial_pose`**。否则会打警告并把物体放在原点：

```text
WARNING - No initial pose set for actor builder of kin, setting to default pose ...
```

源码：`mani_skill/utils/building/actor_builder.py:215-219`。原点初始 pose 很容易和机器人/桌面重叠，
导致仿真变慢或直接弹飞（源码注释引用了 ManiSkill issue #421）。

读取/判断 body type：

```python
actor.px_body_type  # "dynamic" | "kinematic" | "static"
```

定义见 `mani_skill/utils/structs/actor.py:34`、`:74-89`。

### 2.5 每个并行环境用不同资产/颜色/摩擦

两种做法：

**做法 A：每个 env 单独 build，再 `Actor.merge`。** 这是官方推荐的做法，见
`mani_skill/envs/tasks/digital_twins/so100_arm/grasp_cube.py:229-259`。

```python
cubes = []
for i in range(self.num_envs):
    b = self.scene.create_actor_builder()
    b.add_box_collision(
        half_size=[0.03] * 3,
        material=sapien.physx.PhysxMaterial(
            static_friction=frictions[i],
            dynamic_friction=frictions[i],
            restitution=0.0,
        ),
        density=200,
    )
    b.add_box_visual(
        half_size=[0.03] * 3,
        material=sapien.render.RenderMaterial(base_color=colors[i]),
    )
    b.set_initial_pose(sapien.Pose(p=[0, 0, 0.03]))
    b.set_scene_idxs([i])          # 只在第 i 个子场景 build
    cubes.append(b.build(name=f"cube-{i}"))

self.cube = Actor.merge(cubes, name="cube")  # 合成一个 view
```

`set_scene_idxs`：`mani_skill/utils/building/actor_builder.py:38-48`；
`Actor.merge`：`mani_skill/utils/structs/actor.py:100-126`。

**做法 B：build 一次，之后改材质。** 适合只随机颜色/roughness：

```python
for link in self.agent.robot.links:
    for i, obj in enumerate(link._objs):
        comp = obj.entity.find_component_by_type(sapien.render.RenderBodyComponent)
        if comp is None:
            continue
        for shape in comp.render_shapes:
            for part in shape.parts:
                part.material.set_base_color([r, g, b, 1.0])
```

参考 `mani_skill/envs/tasks/digital_twins/so100_arm/grasp_cube.py:282-308`。

---

## 3. 传送带：让物体随带面运动

### 3.1 每个 step / substep 的调用链（3.0.1 没有 scene 级回调）

`BaseEnv.step(action)`：

```text
step(action)
└─ _step_action(action)
   ├─ agent.set_action(action)                 # 控制目标
   ├─ _before_control_step()                  # 每个 control step 一次
   ├─ for _ in range(sim_steps_per_control):  # 默认 100/20 = 5 次
   │    ├─ agent.before_simulation_step()     # controller 插值等
   │    ├─ _before_simulation_step()          # ★ 每个 physx step 前
   │    ├─ scene.step()                       # 只调用 px.step()
   │    └─ _after_simulation_step()           # ★ 每个 physx step 后
   ├─ _after_control_step()                   # 每个 control step 一次
   └─ scene._gpu_fetch_all()
```

源码：

- `step` -> `_step_action`：`mani_skill/envs/sapien_env.py:1042-1071`。
- substep 循环：`mani_skill/envs/sapien_env.py:1123-1128`。
- hook 定义：`mani_skill/envs/sapien_env.py:1158-1168`。
- `ManiSkillScene.step()` 只有一行 `self.px.step()`：`mani_skill/envs/scene.py:379-380`。
- 默认 `sim_freq=100`、`control_freq=20`：`mani_skill/utils/structs/types.py:84-87`；
  `sim_steps_per_control` 计算：`sapien_env.py:281-283`；
  `scene.timestep = 1/sim_freq`：`sapien_env.py:1227`。

**结论**：这个版本没有 `_simulate` / `simulate` 方法，也没有
`ManiSkillScene` 级回调注册。要每个物理步做事，就在 task 类里 override
`_before_simulation_step()` 或 `_after_simulation_step()`。
如果你自己直接调 `scene.px.step()`，这些 hook 不会执行。

### 3.2 `set_linear_velocity` / `set_pose` / `apply_force` 的 batched 用法

| API | 支持形状 | GPU 语义 | CPU 语义 | 源码 |
| --- | --- | --- | --- | --- |
| `actor.set_pose(p)` / `actor.pose = p` | `sapien.Pose`、`Pose`、`(7,)`、`(N,7)` | 写 `cuda_rigid_body_data[idx,:7]`，只写 `_reset_mask` 为 True 的 env | 逐个 `obj.pose = ...`；`(N,7)` 按顺序 | `actor.py:367-406` |
| `actor.set_linear_velocity(v)` | `(3,)` 或 `(N,3)` | 写 `cuda_rigid_body_data[idx,7:10]`，只写 reset mask 中的 env；CPU tensor 会自动搬到 device。**非 reset 期间写入后需 `scene.px.gpu_apply_rigid_dynamic_data()` 才生效** | 只支持单 env（只写 `_bodies[0]`） | `base.py:437-448` |
| `actor.set_angular_velocity(w)` | `(3,)` 或 `(N,3)` | 写 `cuda_rigid_body_data[idx,10:13]` | 单 env | `base.py:366-377` |
| `actor.apply_force(F)` | GPU：`(3,)` 或 `(N,3)`；CPU：`np.float32 (3,)` | 写 `cuda_rigid_body_force`，调 `gpu_apply_rigid_dynamic_force()`，作用在质心，**只作用一个 physx step** | 逐个 `body.add_force_at_point(...)`；传 torch 会 `TypeError` | `actor.py:316-327` |

几个关键细节：

- `Actor.pose` 返回 `Pose`，其中 `p=(N,3)`、`q=(N,4)`、`raw_pose=(N,7)`（`pose.py:31-38`）。
- `Pose.create_from_pq(p, q)` 会自动补 batch 维：`(3,) -> (1,3)`、`(4,) -> (1,4)`；`p` 和 `q` 一个 batch=1 时会自动 repeat 到另一个的 batch（`pose.py:88-119`）。
- `actor.set_pose(torch_tensor)` 在 GPU 下**不会**自动搬 device；传 CPU tensor 会报：
  ```text
  RuntimeError: Expected all tensors to be on the same device,
  but found at least two devices, cuda:0 and cpu!
  ```
  传 `Pose` 对象则会自动 `vectorize_pose(..., device=self.device)`（`actor.py:373-374`）。
  所以推荐写法：
  ```python
pose = Pose.create_from_pq(p_cuda, q_cuda)  # p/q 在 device 上
actor.set_pose(pose)
# 或者
actor.set_pose(Pose.create_from_pq(p, q, device=self.device))
  ```
- `set_pose` 的 batch 必须等于当前 reset mask 的数量。全量 reset 时等于 `num_envs`；
  部分 reset 时等于 `len(env_idx)`。实测错误：
  ```text
  RuntimeError: shape mismatch: value tensor of shape [2, 7]
  cannot be broadcast to indexing result of shape [4, 7]
  ```
- `set_linear_velocity` / `set_angular_velocity` 在 GPU 下也受 reset mask 影响；
  `apply_force` **不**受 reset mask 影响，会对所有 env 生效。
- `apply_force` 的 CPU 路径实测：
  ```text
  np.array([0,0,5], dtype=np.float32)  -> OK
  torch.tensor([[0.,0,5.]])            -> TypeError: add_force_at_point(): incompatible function arguments
  ```
  CPU sim 只能是 `num_envs=1`，所以 CPU 下直接传一个 `np.float32 (3,)` 即可。
- **GPU 下 `set_pose` / `set_linear_velocity` 的写 buffer 语义**：这两个 setter 只是写
  `cuda_rigid_body_data`，不会自动调用 `gpu_apply_rigid_dynamic_data()`。
  reset 期间 ManiSkill 会在 `_initialize_episode` 后统一 `_gpu_apply_all()`
  （`sapien_env.py:956-960`），所以 reset 里的 set_pose 没问题。
  如果你在 `_before_simulation_step` 里每步写 pose/velocity，就必须自己 apply；
  而 apply 会连 pose 一起上传，容易把物体 teleport 回去。**每 substep 驱动请优先用
  `apply_force`**，它走独立的 force buffer，不需要 `gpu_apply_rigid_dynamic_data()`。

接触检测：

```python
f = actor.get_net_contact_forces()          # (N, 3), 单位 N
f = self.scene.get_pairwise_contact_forces(a, b)  # (N, 3)
```

`base.py:120-126`、`scene.py:789-802`。

### 3.3 三种传送带实现（本机实测对比）

测试条件：带面 `half_size=[0.5,0.2,0.01]`，带速 0.2 m/s，物体
0.06 m 立方，Panda 底座在 `[-0.615,0,0]`，物体初始在 `x=0.5`，跑 100 个 control step（=1 s 仿真）。

| 方案 | 结果 | 结论 |
| --- | --- | --- |
| A. kinematic 带面，每个 substep `set_pose` 向前 teleport | 带面移动 1.0 m，物体 x 仍为 0.500，几乎不动 | **不要用**。teleport 不产生相对速度，PhysX 摩擦不驱动 |
| B. dynamic 带面 + `disable_gravity=True` + 每 substep `set_linear_velocity(0.2)` | 物体 y 速度冲到 47 m/s，直接飞出 | **不要用**。大质量 dynamic 带面和物体接触会炸 |
| C. static 带面 + 每 substep 施加 `mu*m*g` 的摩擦力 | 物体从 0.5 加速到 1.04 m，约 0.2 m/s，离开带面后落地 | **推荐（物理保真）** |
| D. 无摩擦 static 带面 + 在带面上时 `set_linear_velocity` 的 x 直接设为带速 | CPU：物体精确 0.2 m/s，50 step 走 0.463 m；GPU 下需要额外 apply，见下 | **CPU 可用；GPU 请用 C** |

#### 方案 C：按摩擦系数施力（推荐，物理上像“传送带”）

> 下面是同一个 `BaseEnv` 子类里的方法片段；imports 和类骨架同 §1.1。

```python
import numpy as np
import torch

class FruitSortEnv(BaseEnv):
    belt_speed = 0.2   # m/s
    belt_mu = 1.0      # 带面摩擦系数
    belt_x_min, belt_x_max = 0.1, 1.1
    belt_top_z = 0.02

    def _load_scene(self, options):
        # static 带面；摩擦系数只影响接触，不影响我们施加的力
        mat = sapien.physx.PhysxMaterial(
            static_friction=self.belt_mu,
            dynamic_friction=self.belt_mu,
            restitution=0.0,
        )
        b = self.scene.create_actor_builder()
        b.add_box_collision(half_size=[0.5, 0.2, 0.01], material=mat)
        b.add_box_visual(
            half_size=[0.5, 0.2, 0.01],
            material=sapien.render.RenderMaterial(base_color=[0.1, 0.1, 0.1, 1]),
        )
        b.set_initial_pose(sapien.Pose(p=[0.6, 0, 0.01]))
        self.belt = b.build_static("belt")

        self.fruit = actors.build_box(
            self.scene,
            half_sizes=[0.03, 0.03, 0.03],
            color=[1, 0, 0, 1],
            name="fruit",
            initial_pose=sapien.Pose(p=[0.4, 0, 0.05]),
        )

    def _before_simulation_step(self):
        # 每个 physx step 调用（默认 5 次 / env.step）
        p = self.fruit.pose.p
        v = self.fruit.linear_velocity
        m = self.fruit.mass.to(self.device)  # (N,)
        g = 9.81

        on_belt = (
            (p[:, 2] < self.belt_top_z + 0.05)
            & (p[:, 0] > self.belt_x_min)
            & (p[:, 0] < self.belt_x_max)
        )

        # 想让物体达到带速所需的加速度；再被 mu*m*g 上限截断
        rel = self.belt_speed - v[:, 0]
        acc = torch.clamp(rel, min=0.0) / self.scene.timestep
        fx = torch.minimum(m * acc, self.belt_mu * m * g)
        fx = torch.where(on_belt, fx, torch.zeros_like(fx))

        if self.gpu_sim_enabled:
            F = torch.zeros((self.num_envs, 3), device=self.device)
            F[:, 0] = fx
            self.fruit.apply_force(F)          # GPU: (N,3) torch
        else:
            # CPU sim 只能 num_envs=1；且必须用 np.float32 (3,)
            self.fruit.apply_force(
                np.array([float(fx[0]), 0.0, 0.0], dtype=np.float32)
            )
```

要点：

- 力必须每个 **substep** 施加；只每个 `env.step` 施加一次会小 5 倍。
- 上限 `mu*m*g` 是摩擦力的物理上限；不加会瞬间到带速，看起来不自然。
- 物体离开带面（`on_belt=False`）就不要再施力，否则会在空中加速。
- 更严谨的 `on_belt` 可以用 `self.scene.get_pairwise_contact_forces(self.belt, self.fruit)` 判接触。

本机实测（GPU 4 env，50 个 control step）：物体从 `x=0.4` 走到约 `0.98~1.01`，速度约 `0.2 m/s`。

#### 方案 D：无摩擦带面 + 直接速度覆盖（CPU 可用，GPU 不推荐）

> 片段，省略了与方案 C 相同的 builder/import 代码。

```python
def _load_scene(self, options):
    mat = sapien.physx.PhysxMaterial(0.0, 0.0, 0.0)  # ★ 必须无摩擦
    ...
    self.belt = b.build_static("belt")

def _before_simulation_step(self):
    p = self.fruit.pose.p
    v = self.fruit.linear_velocity.clone()
    on_belt = (
        (p[:, 2] < self.belt_top_z + 0.05)
        & (p[:, 0] > self.belt_x_min)
        & (p[:, 0] < self.belt_x_max)
    )
    v[on_belt, 0] = self.belt_speed
    self.fruit.set_linear_velocity(v)
```

实测（**CPU sim**）：0.2 m/s 精确，50 step 走 0.463 m。

**为什么带面必须无摩擦**：如果带面 `mu=2.0`，每个 substep 把物体 x 速度设成 0.2，
PhysX 的静摩擦会在同一个 step 内把它几乎完全抵消。实测同样代码只走了 0.02 m，
而不是 1.0 m。

**GPU sim 下不要直接用这个方案**：`Actor.set_linear_velocity` 只是写
`cuda_rigid_body_data` 这块 buffer；在 `_before_simulation_step` 里写完，
不会自动上传到 PhysX。必须再调
`self.scene.px.gpu_apply_rigid_dynamic_data()`，但该调用会把**整块 body data**
（包括 pose）一起上传，等于每个 substep 都把物体 teleport 回上一个 control step
的 pose。实测 20 个 control step 只走了 0.037 m，而不是 0.2 m/s × 1 s = 0.2 m。
GPU 下请用方案 C（`apply_force` 走独立的 force buffer，不受这个问题影响）。

#### 不推荐：kinematic `set_pose` 传送带

实测 kinematic 带面从 x=0 移动到 x=1.0，物体 x 一直是 0.5。原因是 teleport 只改 pose，
PhysX 不会把它当成有相对速度的运动表面，摩擦不产生驱动力；而且如果带面和物体初始重叠，
下一次 teleport 会造成巨大穿透，直接把物体弹飞。

如果确实要 kinematic 视觉带面，建议：

- 带面只做视觉/支撑，物体用方案 C/D 驱动；
- 或者不要依赖接触摩擦，改用 `scene.create_drive(...)`（`mani_skill/envs/scene.py:474-488`）做关节驱动。

---

## 4. 相机

### 4.1 `CameraConfig`

定义在 `mani_skill/sensors/camera.py:31-67`：

```python
@dataclass
class CameraConfig(BaseSensorConfig):
    uid: str
    pose: Pose
    width: int
    height: int
    fov: float = None          # 垂直 FOV，弧度；与 intrinsic 二选一
    near: float = 0.01
    far: float = 100
    intrinsic: Array = None    # 3x3；与 fov 二选一
    entity_uid: Optional[str] = None  # 挂载到某个 link 名字
    mount: Union[Actor, Link] = None  # 或直接挂到 Actor/Link
    shader_pack: Optional[str] = "minimal"
    shader_config: Optional[ShaderConfig] = None
```

`fov` 和 `intrinsic` 必须且只能给一个（`camera.py:155-158`）。
`fov` 传给 SAPIEN 的 `fovy`（`:168`）。

最小写法（和 `PickCube` 一样）：

```python
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
import numpy as np

pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[0, 0, 0.05])
cam = CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)
```

`look_at` 的相机约定是 `(forward, right, up) = (x, -y, z)`（`mani_skill/utils/sapien_utils.py:317-332`）。

挂载到机器人 link：

```python
CameraConfig(
    "wrist_camera",
    pose=sapien.Pose(p=[0.05, 0, 0.03], q=[0, 0.707, 0, 0.707]),
    width=128, height=128, fov=1.57,
    entity_uid="panda_hand",   # link 名字
)
```

`entity_uid` 的解析在 `camera.py:139-153`。

### 4.2 `obs_mode="rgbd"` 的输出结构

实测（`PickCube-v1`, N=2）：

```text
obs = {
    "agent": {
        "qpos": (N, 9) float32,
        "qvel": (N, 9) float32,
    },
    "extra": { ... },                       # 你的 _get_obs_extra
    "sensor_param": {
        "base_camera": {
            "extrinsic_cv": (N, 3, 4),
            "cam2world_gl": (N, 4, 4),
            "intrinsic_cv": (N, 3, 3),
        }
    },
    "sensor_data": {
        "base_camera": {
            "rgb":   (N, H, W, 3) uint8,
            "depth": (N, H, W, 1) int16,    # ★ 单位是毫米
        }
    },
}
```

代码路径：

- `get_obs` 分派：`mani_skill/envs/sapien_env.py:501-533`。
- 视觉 obs 组装：`sapien_env.py:627-634`。
- 相机取图 + 纹理变换：`mani_skill/sensors/camera.py:190-242`。
- 参数：`camera.py:248-253`；`get_sensor_params`：`sapien_env.py:571-576`。
- shader 的 depth 变换：`mani_skill/render/shaders.py:41-47`；`minimal` 定义 `:68-84`。

**深度 scale**：`depth` 是 `int16`，单位 **毫米**。转米：

```python
depth_m = obs["sensor_data"]["base_camera"]["depth"].float() / 1000.0
```

实测 `PickCube` 的深度范围 `272..9436`（mm），中位数 671 mm，符合相机到桌面的距离。
不要把它当 float 米，也不要用 `get_sensor_images()` 里的 depth 做几何——那个是归一化到
0..255 的 3 通道可视化图（`camera.py:256-293`）。

**内参**：`intrinsic_cv` 是 3×3，`fx=intrinsic[0,0]`、`fy=intrinsic[1,1]`、
`cx=intrinsic[0,2]`、`cy=intrinsic[1,2]`。反投影：

```python
z = depth_m                         # (N,H,W,1)
u = torch.arange(W, device=z.device).view(1, 1, W)
v = torch.arange(H, device=z.device).view(1, H, 1)
fx, fy = K[..., 0, 0, None, None], K[..., 1, 1, None, None]
cx, cy = K[..., 0, 2, None, None], K[..., 1, 2, None, None]
x = (u - cx) * z / fx
y = (v - cy) * z / fy
```

`extrinsic_cv` 是 OpenCV 约定（x 右、y 下、z 前）；`cam2world_gl` 是 OpenGL 约定。
两者都在 `sensor_param` 里，按需选。

### 4.3 按 env 取图

```python
obs, _ = env.reset(seed=0)

rgb_all = obs["sensor_data"]["base_camera"]["rgb"]     # (N,H,W,3)
rgb_i = rgb_all[i]                                     # 第 i 个 env

depth_all = obs["sensor_data"]["base_camera"]["depth"] # (N,H,W,1)
K_all = obs["sensor_param"]["base_camera"]["intrinsic_cv"]  # (N,3,3)

# 所有 env 的传感器可视化拼图（rgb/depth 都是 3 通道 uint8）
imgs = env.unwrapped.get_sensor_images()
rgb_vis = imgs["base_camera"]["rgb"]                   # (N,H,W,3)
```

`get_sensor_images`：`mani_skill/envs/sapien_env.py:567-569`，
`mani_skill/envs/scene.py:1128-1135`。

### 4.4 headless 渲染 + Vulkan ICD 警告

SAPIEN 是离屏 Vulkan 渲染，不需要 X11/显示器。本机启动时会出现：

```text
UserWarning: Failed to find Vulkan ICD file. This is probably due to an incorrect
or partial installation of the NVIDIA driver. SAPIEN will attempt to provide an ICD
file anyway but it may not work.
```

来源是 `sapien/_vulkan_tricks.py:66-83`：它只检查
`/usr/share/vulkan/icd.d/nvidia_icd.json`，而本机这个文件在
`/etc/vulkan/icd.d/nvidia_icd.json`，所以 warning 会误报。实测**即使有 warning 也能正常渲染**。

消除 warning 并显式指定 ICD：

```bash
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
```

验证 headless 渲染（本机实测通过）：

```bash
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
python - <<'PY'
import gymnasium as gym
import mani_skill

env = gym.make("PickCube-v1", num_envs=1,
               obs_mode="rgbd", render_mode="rgb_array")
obs, _ = env.reset(seed=0)
img = env.render()
depth = obs["sensor_data"]["base_camera"]["depth"]

print("can_render", env.unwrapped.scene.can_render())
print("rgb", img.shape, img.dtype, "mean", img.float().mean().item())
print("depth mm", depth.min().item(), depth.max().item())
PY
```

本机输出（有/无环境变量都能渲染；设置后无 warning）：

```text
can_render True
rgb torch.Size([1, 512, 512, 3]) torch.uint8 mean 94.58650207519531
depth mm 272 9436
```

如果 `mean` 接近 0 或 depth 全 0，说明渲染没起来。

**`render_backend=None` 有坑**：文档字符串说可以传 `None`，但本机实测会崩：

```text
TypeError: argument of type 'NoneType' is not iterable
  at mani_skill/envs/utils/system/backend.py:47
```

要关渲染请用字符串 `render_backend="none"`。关掉后：

- `scene.can_render()` 为 False；
- `_load_lighting` / `_setup_sensors` 被跳过（`sapien_env.py:744,748`）；
- 没有相机，`env.render()` 会报
  `RuntimeError: failed to get system: no system with name [render] is added to scene`。

所以需要 `rgbd` 时保持默认 `render_backend="gpu"`。

### 4.5 `render_mode="rgb_array"` 与 `env.render()`

- `render_mode` 必须是 `("human", "rgb_array", "sensors", "all")` 之一（`sapien_env.py:129-133`）。
- `env.render()` 按 `render_mode` 分派（`sapien_env.py:1416-1439`）：
  - `"human"` -> 打开 GUI viewer；
  - `"rgb_array"` -> `render_rgb_array()`，返回 **torch.Tensor** `(N,H,W,3)` uint8；
  - `"sensors"` -> 所有传感器图拼图；
  - `"all"` -> 人眼相机 + 传感器图拼图。
- 没设 `render_mode` 调 `env.render()` 会 `RuntimeError: render_mode is not set.`（`:1426-1427`）。
- `render_rgb_array()` 可以直接调，不要求 `render_mode`；它用的是
  `_default_human_render_camera_configs` 里的相机，不是 obs 相机（`sapien_env.py:1369-1386`）。
- 有多个 human render camera 时，`render_rgb_array()` 会把多张图拼起来（`:1380-1386`）；
  想只取一个用 `env.render_rgb_array(camera_name="render_camera")`。

保存视频时注意转 numpy：

```python
frame = env.render()[0].cpu().numpy()   # (H,W,3) uint8
```

---

## 5. 并行、GPU sim、可复现性

### 5.1 backend 与 `num_envs`

- `sim_backend` 接受：`"cpu"`、`"gpu"`、`"cuda"`、`"physx_cpu"`、`"physx_cuda"`，
  以及带显卡号的 `"physx_cuda:0"`（`mani_skill/envs/utils/system/backend.py:28-43`、`:52-70`）。
- `sim_backend="auto"`（默认）：`num_envs == 1` 用 CPU，`num_envs > 1` 用 GPU
  （`sapien_env.py:233-237`）。
- CPU backend 不允许 `num_envs > 1`：
  ```text
  RuntimeError: Cannot set the sim backend to 'cpu' and have multiple environments.
  ```
  （`sapien_env.py:248-251`）。要 CPU 多环境请用 gymnasium 的 `AsyncVectorEnv` 多进程。
- GPU PhysX 在进程内**只能 enable 一次**：`sapien_env.py:243-245`。实测先创建 CPU env、
  再创建 GPU env 会报：
  ```text
  RuntimeError: GPU PhysX can only be enabled once before any other code involving PhysX
  ```
  所以一个进程里只用一种 backend，或先建 GPU env。

### 5.2 并行场景布局

GPU sim 会为每个 env 建一个 sub-scene，按方形网格排布，间距 `SimConfig.spacing = 5` 米：

- `SimConfig.spacing`：`mani_skill/utils/structs/types.py:81-83`。
- 建场景/设 offset：`sapien_env.py:1186-1210`。
- 读取 offset：`env.scene.scene_offsets`（`scene.py:804-812`）。

实测 `num_envs=4` 的 offset：

```text
[[-5, -5, 0], [0, -5, 0], [-5, 0, 0], [0, 0, 0]]
```

物体如果很大/很快，把 `spacing` 调大，或通过
`sim_config=dict(spacing=10)` 传入（`sapien_env.py:205`）。

### 5.3 自定义 mesh 在 GPU sim 下是否支持

**支持**。本机实测（`physx_cuda`, `num_envs=4`，`num_envs=1` 时同样）以下都能 build 并 step：

```text
add_multiple_convex_collisions_from_file(".../board_1.glb") + add_visual_from_file(...)
add_convex_collision_from_file(".../board_1.glb") + add_visual_from_file(...)
add_nonconvex_collision_from_file(".../board_1.glb") + add_visual_from_file(...)
```

三者都能跑 4 env 的 `reset()` + `step()`。
**唯一注意**：`decomposition="coacd"` 需要额外安装 `coacd`；本机没装，实测报
`ModuleNotFoundError: No module named 'coacd'`。不要默认打开。

### 5.4 `reset(seed=...)` 的可复现性

源码设计（`sapien_env.py:880-892`）：

- 环境创建时 main seed 固定为 `2022 .. 2022+N-1`；
- `reset(seed=None)`：main seed 不变。注意 `_set_episode_rng(None)` 在
  `enhanced_determinism=False`（默认）时**不会更新** `_batched_episode_rng`
  （`sapien_env.py:999-1001`）。实际随机性来自 `_initialize_episode` 里未 fork 的
  全局 torch RNG（`:946-951`）：用 `torch.rand` 的任务每次 reset 会变，
  只用 `_batched_episode_rng` 的任务会重复同一回合。要每回合重新抽 episode seed，
  显式传 `seed=` 或开 `enhanced_determinism=True`；
- `reset(seed=int)`：main RNG 和 episode seed 都设为该 seed；
- `reset(seed=list[int])`：每个 env 单独 seed。

实测（GPU 4 env）：

```python
env.reset(seed=[0,1,2,3]); p = env.unwrapped.cube.pose.p.clone()
env.reset(seed=[0,1,2,3])
torch.allclose(p, env.unwrapped.cube.pose.p)   # True
```

```python
env.reset(seed=42); p = env.unwrapped.cube.pose.p.clone()
env.reset()
torch.allclose(p, env.unwrapped.cube.pose.p)   # False，未指定 seed 会换回合
```

```python
env.reset(seed=7);  p7 = env.unwrapped.cube.pose.p.clone()
env.reset(seed=7);  print(torch.allclose(p7, env.unwrapped.cube.pose.p))  # True
env.reset(seed=8);  print(torch.allclose(p7, env.unwrapped.cube.pose.p))  # False
```

细节：

- `num_envs > 1` 且只给一个 int seed 时，其余 env 的 seed 由
  `np.random.RandomState(seed)` 确定性派生（`sapien_env.py:999-1016`）。
- `_initialize_episode` 里的 `torch.rand` 只有在 seed 给定或
  `enhanced_determinism=True` 时才会被 `torch.manual_seed(self._episode_seed[0])`
  固定（`sapien_env.py:946-951`）。注意只用了 `seed[0]`，所以它是全局 torch 流，
  不是每 env 独立流。`enhanced_determinism=True` 还会让 `_set_episode_rng`
  在每次 reset 时都重新采样 episode seed（`:1001`）。
- 需要每 env 独立随机数时，用 `self._batched_episode_rng`：
  ```python
rng = self._batched_episode_rng[i]        # 第 i 个 env 的 numpy RandomState
color = rng.uniform(0, 1, size=3)
  ```
  源码/说明：`sapien_env.py:168-174`、`:1011-1016`；用法示例：
  `mani_skill/envs/tasks/digital_twins/so100_arm/grasp_cube.py:296-301`。

### 5.5 GPU 内存

如果出现 contact/patch buffer overflow，用 `sim_config` 调大：

```python
env = gym.make(
    "FruitSort-v0",
    num_envs=4096,
    sim_backend="physx_cuda",
    sim_config=dict(
        gpu_memory_config=dict(
            max_rigid_contact_count=2**20,
            max_rigid_patch_count=2**19,
            temp_buffer_capacity=2**25,
        )
    ),
)
```

默认值和对应报错见 `mani_skill/utils/structs/types.py:12-31`。

---

## 6. 关节 / 夹爪控制（Panda）

### 6.1 Panda 的基本信息

`mani_skill/agents/robots/panda/panda.py`：

- `uid = "panda"`，URDF `panda_v2.urdf`（`:16-19`）。
- 7 个 arm joint + 2 个 finger joint（`:53-65`）。
- `ee_link_name = "panda_hand_tcp"`（`:66`）。
- `tcp` 在 `_after_init` 里按名字取（`:233-235`）。

支持的 control mode（`:187-215`，按顺序）：

```text
pd_joint_delta_pos   # ★ 默认，第一个
pd_joint_pos
pd_ee_delta_pos
pd_ee_delta_pose
pd_ee_pose
pd_joint_target_delta_pos
pd_ee_target_delta_pos
pd_ee_target_delta_pose
pd_joint_vel
pd_joint_pos_vel
pd_joint_delta_pos_vel
```

默认 control mode 取 `supported_control_modes[0]`（`mani_skill/agents/base_agent.py:108-114`），
所以是 **`pd_joint_delta_pos`**，不是 `pd_ee_delta_pose`。实测：

```python
env = gym.make("PickCube-v1", num_envs=1)
env.unwrapped.agent.control_mode   # "pd_joint_delta_pos"
```

### 6.2 action 维度与含义

实测各 control mode 的 `single_action_space.shape`：

| control_mode | arm 维度 | 夹爪 | 总维度 | 说明 |
| --- | --- | --- | --- | --- |
| `pd_joint_delta_pos`（默认） | 7 | 1 | **8** | 7 个关节增量，范围 ±0.1 rad |
| `pd_joint_pos` | 7 | 1 | **8** | 7 个关节绝对目标，范围=关节限位 |
| `pd_ee_delta_pos` | 3 | 1 | **4** | 末端 xyz 增量，范围 ±0.1 m |
| `pd_ee_delta_pose` | 6 | 1 | **7** | xyz 增量 ±0.1 m + 欧拉角增量（XYZ）±0.1 rad |
| `pd_ee_pose` | 6 | 1 | **7** | 末端位姿绝对目标 |

- `pd_ee_delta_pose` 的 action space 构造：`mani_skill/agents/controllers/pd_ee_pose.py:205-222`；
  旋转部分按向量范数裁剪再乘 `rot_lower`：`:224-237`；
  实际解释为 **Euler XYZ 角**，不是轴角：`compute_target_pose` 里调用
  `euler_angles_to_matrix(delta_rot, "XYZ")`（`pd_ee_pose.py:239-243`）。
- `pd_joint_pos` 的 action space 直接取关节限位：`pd_joint_pos.py:21-41`。
- 所有 controller 默认 `normalize_action=True`，所以 action 是 `[-1, 1]`，
  内部再 clip+scale 到真实范围（`base_controller.py:125-138`）。
- `num_envs=1` 时 `env.action_space.shape == (d,)`；
  `num_envs>1` 时是 `(N, d)`。源码：`_step_action` 只在 `num_envs == 1` 时
  自动 `batch(action)`（`sapien_env.py:1107-1110`）。

Panda 的 `pd_ee_delta_pose` 配置（`panda.py:113-124`）：

```python
PDEEPoseControllerConfig(
    joint_names=arm_joint_names,
    pos_lower=-0.1, pos_upper=0.1,
    rot_lower=-0.1, rot_upper=0.1,
    stiffness=1e3, damping=1e2, force_limit=100,
    ee_link="panda_hand_tcp",
    urdf_path=...,
)
```

### 6.3 夹爪开合

Panda 夹爪是 **mimic controller**：

- 配置：`PDJointPosMimicControllerConfig`，`lower=-0.01`、`upper=0.04`，
  `mimic={"panda_finger_joint2": {"joint": "panda_finger_joint1"}}`
  （`panda.py:177-185`）。
- 动作只有 **1 维**，控制 `panda_finger_joint1`，另一根手指按 1:1 跟随
  （`PDJointPosMimicController`，`pd_joint_pos.py:129-258`）。

实测（`PickCube`, 10 个 step）：

```text
action[-1] = +1  -> finger qpos = [0.04, 0.04]   # 张开
action[-1] = -1  -> finger qpos = [0.00, 0.00]   # 闭合（关节下限 0）
```

所以约定是：**最后一维 +1 张开，-1 闭合**。夹爪目标下限是 -0.01（为了薄物体保持力），
但实际 qpos 到 0 就被限位挡住。

读夹爪关节：

```python
qpos = env.unwrapped.agent.robot.get_qpos()   # (N, 9)
gripper_qpos = qpos[:, -2:]                  # 两根手指
```

`get_qpos` / `get_qvel`：`mani_skill/utils/structs/articulation.py:644-648`。

### 6.4 读 TCP pose 与关节状态

```python
agent = env.unwrapped.agent

tcp_pose = agent.tcp_pose        # Pose, p=(N,3), q=(N,4) wxyz, raw_pose=(N,7)
tcp_pos  = agent.tcp_pos         # (N,3)
qpos     = agent.robot.get_qpos()  # (N,9)
qvel     = agent.robot.get_qvel()  # (N,9)
```

Panda 的 `tcp_pose` 定义：`mani_skill/agents/robots/panda/panda.py:271-277`。
通用写法（不依赖 Panda 属性）：

```python
from mani_skill.utils import sapien_utils
tcp_link = sapien_utils.get_obj_by_name(
    agent.robot.get_links(), "panda_hand_tcp"
)
tcp_pose = tcp_link.pose
```

抓取判断（Panda 专用）：

```python
is_grasped = agent.is_grasping(self.apple)   # (N,) bool
is_static  = agent.is_static(0.2)            # (N,) bool
```

`panda.py:237-269`。`is_grasping` 用两根手指的 pairwise contact force 和接触方向判断。

---

## 7. 域随机化

### 7.1 `mani_skill.envs.utils.randomization` 可用函数

这个模块实际是一个 package：`mani_skill/envs/utils/randomization/`，
`__init__.py` 导出 `common`、`pose`、`samplers`、`camera` 的内容。

可用 API：

```python
import mani_skill.envs.utils.randomization as randomization

# 1) 均匀分布（返回 torch.Tensor）
xyz = randomization.uniform(
    low=[0.05, -0.1, 0.03],
    high=[0.15, 0.1, 0.03],
    size=(b, 3),
    device=self.device,
)
```

源码：`randomization/common.py:9-22`。

```python
# 2) 随机四元数（wxyz），可锁定某些欧拉轴
q = randomization.random_quaternions(
    n=b,
    device=self.device,
    lock_x=True, lock_y=True, lock_z=False,
    bounds=(0, 2 * np.pi),
)
```

源码：`randomization/pose.py:13-36`。返回顺序是 **wxyz**，可直接给
`Pose.create_from_pq`。PickCube 的用法：`pick_cube.py:119-120`。

```python
# 3) 带最小间距的均匀放置采样器
sampler = randomization.UniformPlacementSampler(
    bounds=([0.05, -0.2], [0.3, 0.2]),
    batch_size=b,
    device=self.device,
)
xy = sampler.sample(radius=0.05, max_trials=100, append=True)
```

源码：`randomization/samplers.py:13-96`。

```python
# 4) 相机位置/朝向随机化
eyes = randomization.camera.make_camera_rectangular_prism(
    n=b, scale=[0.1, 0.1, 0.1], center=[0.3, 0, 0.6], theta=0,
    device=self.device,
)
pose = randomization.camera.noised_look_at(
    eye=eyes, target=[0, 0, 0.05],
    look_at_noise=1e-2, view_axis_rot_noise=2e-1,
    device=self.device,
)
```

源码：`randomization/camera.py:12-30`、`:32-68`。

注意：`randomization` 里**没有**专门的灯光/材质随机化函数，需要自己写。

### 7.2 材质/颜色随机化

**建资产时按 env 随机**（推荐）：每个 env 用不同 `RenderMaterial` / `PhysxMaterial`，
`set_scene_idxs([i])` 后 `Actor.merge`。完整例子见
`mani_skill/envs/tasks/digital_twins/so100_arm/grasp_cube.py:229-259`。

```python
colors = np.random.rand(self.num_envs, 4)
colors[:, 3] = 1.0
frictions = np.random.uniform(0.3, 1.2, size=self.num_envs)

cubes = []
for i in range(self.num_envs):
    b = self.scene.create_actor_builder()
    b.add_box_collision(
        half_size=[0.03] * 3,
        density=200,
        material=sapien.physx.PhysxMaterial(
            static_friction=frictions[i],
            dynamic_friction=frictions[i],
            restitution=0.0,
        ),
    )
    b.add_box_visual(
        half_size=[0.03] * 3,
        material=sapien.render.RenderMaterial(base_color=colors[i]),
    )
    b.set_initial_pose(sapien.Pose(p=[0, 0, 0.03]))
    b.set_scene_idxs([i])
    cubes.append(b.build(name=f"cube-{i}"))

self.cube = Actor.merge(cubes, name="cube")
```

**建好后改颜色/roughness**：遍历 `RenderBodyComponent.render_shapes -> shape.parts ->
part.material` 并调用 `set_base_color` / `set_roughness` / `set_metallic`。
参考 `grasp_cube.py:282-308`，本机验证 `part.material.set_base_color([r,g,b,1])` 可用。

```python
for i, obj in enumerate(self.cube._objs):
    comp = obj.entity.find_component_by_type(sapien.render.RenderBodyComponent)
    for shape in comp.render_shapes:
        for part in shape.parts:
            part.material.set_base_color(colors[i].tolist())
            part.material.set_roughness(float(roughness[i]))
```

### 7.3 灯光随机化

可用 API（`mani_skill/envs/scene.py`）：

```python
scene.set_ambient_light([0.3, 0.3, 0.3])                    # :575-576
scene.add_directional_light(                                # :612-654
    direction=[1, 1, -1],
    color=[1, 1, 1],
    shadow=False,
    position=[0, 0, 0],
    shadow_scale=10.0,
    shadow_near=-10.0,
    shadow_far=10.0,
    shadow_map_size=2048,
    scene_idxs=None,       # None = 所有 sub-scene
)
scene.add_point_light(                                      # :578-610
    position=[0, 0, 2], color=[1, 1, 1],
    shadow=False, shadow_near=0.1, shadow_far=10.0,
    shadow_map_size=2048, scene_idxs=None,
)
scene.add_spot_light(                                       # :656-695
    position=[0, 0, 2], direction=[0, 0, -1],
    inner_fov=0.5, outer_fov=1.0, color=[1, 1, 1],
    shadow=False, shadow_near=0.1, shadow_far=10.0,
    shadow_map_size=2048, scene_idxs=None,
)
```

默认灯光在 `BaseEnv._load_lighting`：ambient 0.3 + 两个 directional light
（`sapien_env.py:845-853`）。要自定义就 override `_load_lighting`。

**按 env 加灯**：

```python
def _load_lighting(self, options):
    self.scene.set_ambient_light([0.2, 0.2, 0.2])
    for i in range(self.num_envs):
        self.scene.add_directional_light(
            direction=[1, 1, -1],
            color=[1, 1, 1],
            shadow=self.enable_shadow,
            scene_idxs=[i],
        )
```

注意 `add_*_light` 的返回值只保留最后一个 light（point light 返回 `light`，
directional/spot 返回 `None`）。要后续改颜色，自己从 sub-scene 里找组件：

```python
self._dir_lights = []
for sc in self.scene.sub_scenes:
    for ent in sc.entities:
        c = ent.find_component_by_type(
            sapien.render.RenderDirectionalLightComponent
        )
        if c is not None:
            self._dir_lights.append(c)

# 每回合随机（本机验证可以直接赋值）
def _initialize_episode(self, env_idx, options):
    for light in self._dir_lights:
        light.color = [0.8 + 0.2 * np.random.rand(),
                       0.8 + 0.2 * np.random.rand(),
                       0.8 + 0.2 * np.random.rand()]
```

本机实测：`PickCube-v1` 默认有 4 个 directional light（2 个/子场景），
`light.color = [0.5,0.4,0.3]` 可以正常读写。

不要在每个 episode 里反复 `add_*_light`，会不断向 scene 加 entity，造成泄漏。
正确做法是 `_load_scene`/`_load_lighting` 里加一次，之后只改 color/pose。

---

## 8. 已知坑（按踩坑概率排序）

1. **四元数顺序是 wxyz**。`sapien.Pose` 默认 `q=[1,0,0,0]`，`Pose.raw_pose` 是
   `[x,y,z,qw,qx,qy,qz]`。见 `pose.py:64`、`:107-108`、`:118`。
   和 Isaac Lab / LeRobot 的 `xyzw` 转换：
   ```python
q_xyzw = q_wxyz[..., [1, 2, 3, 0]]
q_wxyz = q_xyzw[..., [3, 0, 1, 2]]
   ```
   不要直接把 Isaac Lab 的 quaternion 塞进 ManiSkill。

2. **GPU sim 下 `set_pose` 的 raw tensor 必须已经在 `env.device` 上，且 batch 等于
   当前 reset mask 的 env 数**。传 CPU tensor：
   ```text
   RuntimeError: Expected all tensors to be on the same device,
   but found at least two devices, cuda:0 and cpu!
   ```
   batch 不对：
   ```text
   RuntimeError: shape mismatch: value tensor of shape [2, 7]
   cannot be broadcast to indexing result of shape [4, 7]
   ```
   用 `Pose.create_from_pq(..., device=self.device)` 最稳。
   另外：**非 reset 期间**在 GPU 下改完 pose/velocity，需要手动
   `self.scene.px.gpu_apply_rigid_dynamic_data()` 才会同步到 PhysX；
   reset 期间 ManiSkill 会自动 `_gpu_apply_all()`（`sapien_env.py:956-960`）。
   但每 substep 调用它会连 pose 一起上传，驱动传送带请用 `apply_force`。

3. **默认 `control_mode` 是 `pd_joint_delta_pos`（8 维），默认 `reward_mode` 是
   `normalized_dense`**。如果你按 `pd_ee_delta_pose`（7 维）写策略，要显式传
   `control_mode="pd_ee_delta_pose"`；如果只实现 `compute_dense_reward` 而不实现
   `compute_normalized_dense_reward`，默认 reward 会抛 `NotImplementedError`
   （实测）。

4. **`Actor.apply_force` 的 CPU 路径只接受 `np.float32 (3,)`**，传 torch 会
   `TypeError: add_force_at_point(): incompatible function arguments`（实测）。
   GPU 路径接受 torch `(N,3)` 或 `(3,)`。CPU sim 只能 `num_envs=1`，所以按
   backend 分支即可。

5. **`actor.mass = ...` 和 `set_locked_motion_axes(...)` 只能在 GPU 初始化前调用**。
   两者都带 `@before_gpu_init`（`base.py:276-280`、`:459-461`）。在 GPU env 创建后调用
   会报：
   ```text
   AssertionError ... can only be called when the GPU simulation has not been initialized yet
   ```
   质量用 `builder.set_mass_and_inertia(...)`；锁定轴在 `_load_scene` 里 build 完立刻设。

6. **`gym.make()` 返回的是 `TimeLimitWrapper`，不是 `BaseEnv`**（`registration.py:127-170`）。
   `env.single_action_space`、`env.agent`、`env.scene` 在 wrapper 上可能没有，用
   `env.unwrapped`。实测 `env.single_action_space` 直接 `AttributeError`。

7. **`num_envs=1` 的张量形状**：action space 是 `(8,)`（不是 `(1,8)`），但
   obs/reward/terminated/info 全是 `(1, ...)`。`num_envs>1` 时 action space 才是 `(N,8)`。
   写 replay buffer 时别把 action 的 batch 维搞错。源码：`sapien_env.py:1107-1110`。

8. **`env.step` 的 action 形状在 `num_envs>1` 时必须是 `(N,d)`**。传 `(8,)` 不会被自动
   broadcast，会：
   ```text
   AssertionError: Received action of shape torch.Size([8]) but expected shape (4, 8)
   ```
   实测。`num_envs=1` 时才允许 `(8,)`。

9. **传送带不要用 kinematic `set_pose` teleport 来驱动摩擦**。CPU 实测带面移动 1 m，
   物体完全不动；初始重叠还会把物体弹飞。GPU 下 `set_pose` 还额外需要
   `gpu_apply_rigid_dynamic_data()`，更不该用。用 §3.3 的方案 C（每 substep 按
   `mu*m*g` 施力，CPU/GPU 都实测可用）；方案 D 只在 CPU 稳定。

10. **高摩擦带面会吃掉速度覆盖**。如果带面 `mu=2.0`，每个 substep 把物体速度设成
    0.2，PhysX 静摩擦会在同一 step 把它几乎抵消：CPU 实测 50 个 control step 只走
    0.02 m。方案 D 必须把带面材质设成 `PhysxMaterial(0, 0, 0)`，而且只建议在 CPU
    下用；GPU 下 `set_linear_velocity` 还需要整块 `gpu_apply_rigid_dynamic_data()`，
    会连 pose 一起回传，实测只走 0.037 m/s。

11. **深度是 int16 毫米，不是 float 米**。`obs["sensor_data"][cam]["depth"] / 1000`
    才是米（`shaders.py:41-47`、`:68-84`）。`get_sensor_images()` 返回的 depth 是
    0..255 的可视化图，不能用于几何。

12. **`render_backend=None` 会崩**（`backend.py:47` 对 None 做 `":" in backend`）。
    关渲染用 `render_backend="none"`。关掉后没有相机，`render()` 也会失败。

13. **Vulkan ICD warning**：本机 ICD 在 `/etc/vulkan/icd.d/nvidia_icd.json`，
    SAPIEN 只查 `/usr/share/vulkan/icd.d/nvidia_icd.json`。设置
    `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json` 可消除；不设置也能渲染，
    但最好在 CI/headless 里显式设置。

14. **`decomposition="coacd"` 需要额外安装 `coacd`**。本机没装，实测
    `ModuleNotFoundError`。不装就不要开。

15. **GPU PhysX 一个进程只能 enable 一次**。先建 CPU env 再建 GPU env 会
    `RuntimeError: GPU PhysX can only be enabled once...`（实测）。

16. **visual mesh 不加碰撞就会穿模**。`add_visual_from_file` 只加渲染
    （`sapien/wrapper/actor_builder.py:649-671`）；必须另外调用
    `add_convex_collision_from_file` / `add_multiple_convex_collisions_from_file` /
    `add_nonconvex_collision_from_file`。

17. **GPU sim 下 static actor 不能 `set_pose`**。会报
    `AssertionError: Static objects cannot change poses in GPU sim after environment is loaded`
    （`actor.py:369-372`）。会动的传送带/平台用 kinematic 或 dynamic。

18. **所有 actor 都要设 `initial_pose`**。否则 warning + 原点初始 pose，容易和机器人
    底座/桌面重叠，导致仿真变慢或物体弹飞（`actor_builder.py:215-219`）。

19. **`obs_mode="state"` 返回的是扁平 tensor，不是 dict**；`"state_dict"` 才是 dict。
    源码：`sapien_env.py:521-524`、`:535-544`。实测：
    ```text
    state      -> torch.Size([4, 42])
    state_dict -> {"agent": {"qpos": (4,9), "qvel": (4,9)}, ...}
    ```

20. **`evaluate()` 里不要写 `if info["success"]:`**。它是 `(N,)` bool tensor，
    Python `if` 会报 "Boolean value of Tensor with more than one value is ambiguous"。
    用 `info["success"].any()` / `.all()` / `torch.where`。

21. **`_initialize_episode` 里用 `torch.rand(self.num_envs)` 在部分 reset 时是错的**。
    `set_pose` 只写 `env_idx` 对应的 env，但你的随机数 batch 会对不上。
    永远用 `b = len(env_idx)`。

22. **`evaluate()` 在 reset 后也会被调用**，此时 `_initialize_episode` 刚执行完，
    物体 pose 已经是初始状态；不要在 `evaluate()` 里做有副作用的状态修改。

23. **`sim_backend="cpu"` 不能 `num_envs>1`**。要 CPU 并行用
    `gymnasium.AsyncVectorEnv` 多进程（`sapien_env.py:248-251`）。

---

## 附录：本机实测记录

### A.1 版本与 backend

```text
mani_skill 3.0.1
sapien 3.0.3
python 3.10.21
render system 3.0
```

`PickCube-v1`, `num_envs=4`, `sim_backend="physx_cuda"`：

```text
sim_backend physx_cuda
gpu_sim True
device cuda
action_space (4, 8)
obs (state) torch.Size([4, 42])
scene_offsets [[-5,-5,0],[0,-5,0],[-5,0,0],[0,0,0]]
```

### A.2 rgbd 输出

```text
rgb    torch.Size([4, 128, 128, 3]) torch.uint8
depth  torch.Size([4, 128, 128, 1]) torch.int16
intrinsic_cv (4, 3, 3)
extrinsic_cv (4, 3, 4)
cam2world_gl (4, 4, 4)
render torch.Size([4, 512, 512, 3]) torch.uint8
```

`PickCube-v1`, N=1, headless：

```text
rgb mean 94.58650207519531
depth min/max 272 / 9436 (mm)
```

### A.3 set_pose 的 device / batch

```text
set_pose(Pose.create_from_pq(torch.tensor([[0.,0,0.05]]), ...))  # CPU p/q
  -> OK（Pose 对象会被搬到 device）

cube.set_pose(torch.tensor([0.,0,0.05,1,0,0,0]))                  # CPU raw (7,)
  -> RuntimeError: Expected all tensors to be on the same device ...

cube.set_pose(torch.zeros(2,7,device="cuda"))                    # 4 env
  -> RuntimeError: shape mismatch: value tensor of shape [2, 7]
     cannot be broadcast to indexing result of shape [4, 7]
```

### A.4 控制模式维度

```text
pd_joint_delta_pos (default)  (4, 8)   single (8,)
pd_joint_pos                  (4, 8)   single (8,)
pd_ee_delta_pos               (4, 4)   single (4,)
pd_ee_delta_pose              (4, 7)   single (7,)
pd_ee_pose                    (4, 7)   single (7,)
```

夹爪（10 step）：

```text
action[-1]=+1 -> finger qpos [0.04, 0.04]
action[-1]=-1 -> finger qpos [0.00, 0.00]
```

### A.5 传送带三种实现

条件：带面 `0.5 x 0.2 x 0.01`，带速 0.2 m/s，100 control step（1 s）。

```text
kinematic + set_pose teleport:
  带面相对位移: 0 -> 1.0 m
  fruit x: 0.500 -> 0.500        # 完全不驱动

dynamic + disable_gravity + set_linear_velocity:
  fruit y velocity -> 47 m/s     # 炸

static + 每 substep 施加 mu*m*g:
  CPU 1 env / GPU 4 env 都 OK
  fruit x: 0.4 -> ~0.98-1.01     # 约 0.2 m/s，正常

无摩擦 static + set_linear_velocity x:
  CPU 1 env: fruit x 0.5 -> 0.963 (50 step)  # 精确 0.2 m/s
  GPU 4 env（不 apply）: fruit x 不动
  GPU 4 env（每 substep apply）: 20 step 只走 0.037 m  # 不要用
```

### A.6 自定义 mesh / GPU

```text
add_multiple_convex_collisions_from_file(glb) + add_visual_from_file(glb)
  GPU 4 env reset+step: OK

add_convex_collision_from_file(glb) + add_visual_from_file(glb)
  GPU 4 env reset+step: OK

add_nonconvex_collision_from_file(glb) + add_visual_from_file(glb)
  GPU 4 env reset+step: OK

add_multiple_convex_collisions_from_file(..., decomposition="coacd")
  ModuleNotFoundError: No module named 'coacd'
```

### A.7 其它实测报错原文

```text
env.single_action_space  (gym.make 返回 TimeLimitWrapper)
  AttributeError: 'TimeLimitWrapper' object has no attribute 'single_action_space'

num_envs=4, env.step(np.zeros(8))
  AssertionError: Received action of shape torch.Size([8]) but expected shape (4, 8)

render_backend=None
  TypeError: argument of type 'NoneType' is not iterable
  at mani_skill/envs/utils/system/backend.py:47

先 CPU env 后 GPU env
  RuntimeError: GPU PhysX can only be enabled once before any other code involving PhysX

GPU env 创建后 actor.mass = 1.0
  AssertionError: ... can only be called when the GPU simulation has not been initialized yet
```

---

## 附：常用路径速查

| 内容 | 路径 |
| --- | --- |
| BaseEnv | `mani_skill/envs/sapien_env.py:45` |
| 最小模板 | `mani_skill/envs/minimal_template.py:15-66` |
| 注册 | `mani_skill/utils/registration.py:192-260` |
| ActorBuilder | `mani_skill/utils/building/actor_builder.py:21` |
| SAPIEN ActorBuilder 实现 | `sapien/wrapper/actor_builder.py:102` |
| primitive builders | `mani_skill/utils/building/actors/common.py:70-166` |
| Actor 批量操作 | `mani_skill/utils/structs/actor.py:316-406` |
| 刚体速度/质量 | `mani_skill/utils/structs/base.py:272-448` |
| Pose | `mani_skill/utils/structs/pose.py:31` |
| CameraConfig / Camera | `mani_skill/sensors/camera.py:31-253` |
| shader / depth | `mani_skill/render/shaders.py:41-84` |
| 场景 / 灯光 | `mani_skill/envs/scene.py:379-695` |
| SimConfig | `mani_skill/utils/structs/types.py:78-92` |
| backend 解析 | `mani_skill/envs/utils/system/backend.py:28-70` |
| Panda | `mani_skill/agents/robots/panda/panda.py:16-277` |
| 控制器 | `mani_skill/agents/controllers/` |
| 随机化 | `mani_skill/envs/utils/randomization/` |
| 参考任务 | `mani_skill/envs/tasks/tabletop/pick_cube.py:33-191` |
| 每 env 材质随机化 | `mani_skill/envs/tasks/digital_twins/so100_arm/grasp_cube.py:229-308` |
