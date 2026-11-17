"""确定性夹爪时序状态机（gripper finite state machine）。

给 20 Hz 单臂 Panda 使用：每一步喂入当前观测，输出**世界坐标系**下的
TCP 目标位置、夹爪指令与阶段名。它替代学习策略里抖动的夹爪 logit，
并把"何时闭爪 / 何时释放"变成可调、可测、可复现的几何规则。

阶段流转::

    APPROACH → ALIGN → DESCEND → CLOSE → VERIFY → LIFT → CARRY
             → PLACE → RELEASE → DONE

- APPROACH / ALIGN：预瞄拦截点上方悬停，粗对齐 → 精对齐；
- DESCEND：下探到水果中心（可加 ``grasp_offset``）；
- CLOSE：闭爪并保持 ``close_frames`` 帧；
- VERIFY：``is_grasping`` 或指间距 < ``grasp_gap_thresh`` 视为抓住，
  否则重试（最多 ``max_retries`` 次）；
- LIFT / CARRY / PLACE：抬起 → 搬到框上方 → 下放到放置高度；
- RELEASE：开爪并保持 ``release_frames`` 帧；
- DONE：结束（成功或放弃）。

可靠性约定：

- 每个阶段有独立超时（``timeouts``，单位秒，按 ``dt`` 累计）。
- 任一阶段超时 → 回到 APPROACH，``retries += 1``；
  ``retries > max_retries`` → 放弃（``done=True``，``aborted=True``）。
- 纯函数式接口：``step(state)`` 内部只保存阶段与计数器，不依赖仿真。

坐标与高度约定：

- ``target`` 是**世界坐标系**的 TCP 位置，与 ``scripted.py`` 保持一致；
- ``approach_height`` / ``lift_height`` 是**相对水果**的高度增量；
- ``carry_height`` / ``place_height`` 是**绝对世界 z**（默认值对齐本仓库仿真：
  传送带顶面 z≈0.45、框沿 z≈0.50、搬运高度 z≈0.58）。

注意：模块不做单步限速，调用方仍需把 ``target`` 限幅到控制器可达范围
（见 ``docs/learning.md`` 与 ``scripts/eval_vision_policy.py`` 的 6 cm 限速）。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
APPROACH = "APPROACH"
ALIGN = "ALIGN"
DESCEND = "DESCEND"
CLOSE = "CLOSE"
VERIFY = "VERIFY"
LIFT = "LIFT"
CARRY = "CARRY"
PLACE = "PLACE"
RELEASE = "RELEASE"
DONE = "DONE"

PHASES: tuple[str, ...] = (
    APPROACH,
    ALIGN,
    DESCEND,
    CLOSE,
    VERIFY,
    LIFT,
    CARRY,
    PLACE,
    RELEASE,
    DONE,
)

GRIP_CLOSE = -1.0
GRIP_OPEN = 1.0

# 每阶段默认超时（秒）。可在 GripperFSMConfig.timeouts 里按需覆盖。
DEFAULT_TIMEOUTS: dict[str, float] = {
    APPROACH: 4.0,
    ALIGN: 1.5,
    DESCEND: 2.0,
    CLOSE: 1.0,
    VERIFY: 1.0,
    LIFT: 2.0,
    CARRY: 4.0,
    PLACE: 2.0,
    RELEASE: 1.0,
    DONE: float("inf"),
}

_MISSING = object()


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class GripperState:
    """单步观测（纯 numpy，不依赖仿真）。

    ``tcp_z`` 若给出则以它为准（便于和只暴露标量的感知模块对接），
    否则取 ``tcp_pos[2]``。
    """

    tcp_pos: np.ndarray
    fruit_pos: np.ndarray
    fruit_vel: np.ndarray
    fingers_gap: float
    is_grasping: bool
    bin_xy: np.ndarray
    tcp_z: float | None = None
    dt: float = 0.05


class GripperOutput(NamedTuple):
    """``step`` 的返回值，可直接解包为 ``target, grip, phase, done``。"""

    target: np.ndarray
    grip: float
    phase: str
    done: bool


@dataclass
class GripperFSMConfig:
    """状态机全部可调参数。"""

    # 预瞄：pred = fruit + vel * lead（x/y 两个方向都外推）
    lead: float = 0.12

    # APPROACH / ALIGN
    approach_height: float = 0.12  # 抓取点上方悬停高度（相对水果）
    approach_tol: float = 0.04     # APPROACH 到位判据（xyz）
    align_tol: float = 0.015       # ALIGN 水平到位判据
    align_z_tol: float = 0.012     # ALIGN 高度到位判据

    # DESCEND
    grasp_offset: float = 0.0      # 抓取目标相对水果中心的 z 偏移
    descend_tol: float = 0.012     # 下探到位判据（z）
    grasp_xy_tol: float = 0.02     # 下探时水平容差

    # CLOSE / VERIFY
    close_frames: int = 5          # 闭爪保持帧数
    verify_frames: int = 3         # VERIFY 判定窗口（帧）
    grasp_gap_thresh: float = 0.04 # 指间距小于该值视为抓住
    min_grasp_gap: float = 0.0     # >0 时可排除"夹空"（完全闭合）的假阳性

    # LIFT / CARRY / PLACE
    lift_height: float = 0.12      # 相对抓取点抬升高度
    lift_tol: float = 0.012
    carry_height: float = 0.58     # 搬运高度（绝对世界 z）
    carry_z_tol: float = 0.02
    place_xy_tol: float = 0.02
    place_height: float = 0.50     # 放置高度（绝对世界 z）
    place_z_tol: float = 0.012

    # RELEASE / 重试
    release_frames: int = 4        # 开爪保持帧数
    max_retries: int = 2           # 超过后放弃（done=True）

    # 阶段超时（秒），只需给出想覆盖的键，未给出的用 DEFAULT_TIMEOUTS
    timeouts: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.lead < 0:
            raise ValueError(f"lead 必须 >= 0，得到 {self.lead}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries 必须 >= 0，得到 {self.max_retries}")
        for name in ("close_frames", "verify_frames", "release_frames"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} 必须 >= 1，得到 {getattr(self, name)}")
        for name in (
            "approach_height", "approach_tol", "align_tol", "align_z_tol",
            "descend_tol", "grasp_xy_tol", "grasp_gap_thresh", "lift_height",
            "lift_tol", "carry_height", "carry_z_tol", "place_xy_tol",
            "place_height", "place_z_tol",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} 必须 >= 0，得到 {getattr(self, name)}")
        if not 0.0 <= self.min_grasp_gap < self.grasp_gap_thresh:
            raise ValueError("需要 0 <= min_grasp_gap < grasp_gap_thresh")
        for phase, timeout in self.timeouts.items():
            if phase not in PHASES:
                raise ValueError(f"未知阶段 {phase!r}，可选 {PHASES}")
            if timeout <= 0:
                raise ValueError(f"{phase} 超时必须 > 0，得到 {timeout}")


# --------------------------------------------------------------------------- #
# 输入转换
# --------------------------------------------------------------------------- #
def _get(state: object, name: str, default: object = _MISSING) -> object:
    if isinstance(state, Mapping):
        if name in state:
            return state[name]
    elif hasattr(state, name):
        return getattr(state, name)
    if default is _MISSING:
        raise KeyError(f"GripperState 缺少字段 {name!r}")
    return default


def _vec(state: object, name: str, size: int) -> np.ndarray:
    value = np.asarray(_get(state, name), dtype=np.float64).reshape(-1)
    if value.size != size:
        raise ValueError(f"{name} 需要 {size} 维，得到 shape={value.shape}")
    return value


def _coerce_state(state: GripperState | Mapping[str, object] | object) -> GripperState:
    """把 dataclass / dict / 任意带属性的对象统一成 GripperState。"""
    if isinstance(state, GripperState):
        return state
    tcp_z = _get(state, "tcp_z", None)
    dt = float(_get(state, "dt", 0.05))
    if dt <= 0:
        raise ValueError(f"dt 必须 > 0，得到 {dt}")
    return GripperState(
        tcp_pos=_vec(state, "tcp_pos", 3),
        fruit_pos=_vec(state, "fruit_pos", 3),
        fruit_vel=_vec(state, "fruit_vel", 3),
        fingers_gap=float(_get(state, "fingers_gap")),
        is_grasping=bool(_get(state, "is_grasping")),
        bin_xy=_vec(state, "bin_xy", 2),
        tcp_z=None if tcp_z is None else float(tcp_z),
        dt=dt,
    )


# --------------------------------------------------------------------------- #
# 状态机
# --------------------------------------------------------------------------- #
class GripperFSM:
    """确定性夹爪时序状态机；``step`` 每次推进一拍（默认 20 Hz）。"""

    def __init__(self, cfg: GripperFSMConfig | None = None) -> None:
        self.cfg = cfg or GripperFSMConfig()
        self._done = False
        self._aborted = False
        self.retries = 0
        self.steps = 0
        self.last_failure: str | None = None
        self._last_target: np.ndarray | None = None
        self._lift_z: float | None = None
        self._carry_z: float | None = None
        self._enter(APPROACH)

    # ---------------------------- 只读属性 ---------------------------- #
    @property
    def phase(self) -> str:
        return self._phase

    @property
    def done(self) -> bool:
        return self._done

    @property
    def aborted(self) -> bool:
        """是否因超时/重试耗尽而放弃。"""
        return self._aborted

    @property
    def succeeded(self) -> bool:
        return self._done and not self._aborted

    @property
    def phase_steps(self) -> int:
        return self._phase_steps

    @property
    def phase_time(self) -> float:
        return self._phase_time

    # ---------------------------- 生命周期 ---------------------------- #
    def reset(self) -> None:
        """复位到 APPROACH，清空所有计数与锁存值。"""
        self._done = False
        self._aborted = False
        self.retries = 0
        self.steps = 0
        self.last_failure = None
        self._last_target = None
        self._lift_z = None
        self._carry_z = None
        self._enter(APPROACH)

    def _enter(self, phase: str) -> None:
        self._phase = phase
        self._phase_steps = 0
        self._phase_time = 0.0

    def _timeout(self, phase: str) -> float:
        return float(self.cfg.timeouts.get(phase, DEFAULT_TIMEOUTS.get(phase, float("inf"))))

    # ---------------------------- 主入口 ------------------------------ #
    def step(self, state: GripperState | Mapping[str, object] | object) -> GripperOutput:
        """推进一拍。

        参数
        ----
        state:
            至少包含 ``tcp_pos(3,)``、``fruit_pos(3,)``、``fruit_vel(3,)``、
            ``fingers_gap``、``is_grasping``、``bin_xy(2,)``；可选
            ``tcp_z`` 与 ``dt``（默认 0.05）。

        返回
        ----
        GripperOutput(target, grip, phase, done)
            ``target`` 为世界系 TCP 目标 (3,)，``grip`` 为 -1（闭）/ +1（开）。
        """
        obs = _coerce_state(state)
        self.steps += 1

        if self._done:
            return GripperOutput(self._hold(obs, self._last_target), GRIP_OPEN, DONE, True)

        self._phase_time += obs.dt
        if self._phase_time > self._timeout(self._phase) + 1e-9:
            self._retry_or_abort(obs, f"timeout:{self._phase}")

        if not self._done:
            self._advance(obs)

        if self._done:
            return GripperOutput(self._hold(obs, self._last_target), GRIP_OPEN, DONE, True)

        target, grip = self._command(obs)
        self._last_target = target
        self._phase_steps += 1
        return GripperOutput(target, grip, self._phase, False)

    # ---------------------------- 几何工具 ---------------------------- #
    @staticmethod
    def _tcp_z(obs: GripperState) -> float:
        return float(obs.tcp_pos[2] if obs.tcp_z is None else obs.tcp_z)

    def _pred_xy(self, obs: GripperState) -> np.ndarray:
        lead = self.cfg.lead
        return np.array(
            [
                obs.fruit_pos[0] + obs.fruit_vel[0] * lead,
                obs.fruit_pos[1] + obs.fruit_vel[1] * lead,
            ],
            dtype=np.float64,
        )

    def _pregrasp(self, obs: GripperState) -> np.ndarray:
        xy = self._pred_xy(obs)
        return np.array(
            [xy[0], xy[1], obs.fruit_pos[2] + self.cfg.approach_height],
            dtype=np.float64,
        )

    def _grasp_point(self, obs: GripperState) -> np.ndarray:
        xy = self._pred_xy(obs)
        return np.array(
            [xy[0], xy[1], obs.fruit_pos[2] + self.cfg.grasp_offset],
            dtype=np.float64,
        )

    @staticmethod
    def _xy_err(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _near_xy(self, tcp: np.ndarray, target: np.ndarray, tol: float) -> bool:
        return self._xy_err(tcp, target) <= tol

    def _grasped(self, obs: GripperState) -> bool:
        gap = float(obs.fingers_gap)
        gap_hit = self.cfg.min_grasp_gap <= gap < self.cfg.grasp_gap_thresh
        return bool(obs.is_grasping) or gap_hit

    def _hold(self, obs: GripperState, fallback: np.ndarray | None = None) -> np.ndarray:
        if fallback is not None:
            return np.array(fallback, dtype=np.float64)
        return np.array(
            [obs.tcp_pos[0], obs.tcp_pos[1], self._tcp_z(obs)], dtype=np.float64
        )

    # ---------------------------- 阶段推进 ---------------------------- #
    def _advance(self, obs: GripperState) -> None:
        cfg = self.cfg
        tcp = obs.tcp_pos
        tcp_z = self._tcp_z(obs)
        phase = self._phase

        if phase == APPROACH:
            pre = self._pregrasp(obs)
            if (
                self._near_xy(tcp, pre, cfg.approach_tol)
                and abs(tcp_z - pre[2]) <= cfg.approach_tol
            ):
                self._enter(ALIGN)

        elif phase == ALIGN:
            pre = self._pregrasp(obs)
            if (
                self._near_xy(tcp, pre, cfg.align_tol)
                and abs(tcp_z - pre[2]) <= cfg.align_z_tol
            ):
                self._enter(DESCEND)

        elif phase == DESCEND:
            grasp = self._grasp_point(obs)
            if (
                self._near_xy(tcp, grasp, cfg.grasp_xy_tol)
                and tcp_z <= grasp[2] + cfg.descend_tol
            ):
                self._enter(CLOSE)

        elif phase == CLOSE:
            if self._phase_steps >= cfg.close_frames:
                self._enter(VERIFY)

        elif phase == VERIFY:
            if self._grasped(obs):
                # 以"实际抓住时的高度"为基准抬升，避免水果在传送带上移动导致目标追逐
                self._lift_z = tcp_z + cfg.lift_height
                self._carry_z = max(cfg.carry_height, self._lift_z)
                self._enter(LIFT)
            elif self._phase_steps >= cfg.verify_frames:
                self._retry_or_abort(obs, "grasp_missed")

        elif phase == LIFT:
            lift_z = self._lift_z if self._lift_z is not None else tcp_z + cfg.lift_height
            # 边抬边向料框方向移动：某些抓取点（如 x≈-0.36）纯垂直抬升会顶到臂展极限
            if tcp_z >= lift_z - cfg.lift_tol or self._phase_steps >= 15:
                self._enter(CARRY)

        elif phase == CARRY:
            carry_z = self._carry_z if self._carry_z is not None else cfg.carry_height
            if (
                self._near_xy(tcp, obs.bin_xy, cfg.place_xy_tol)
                and tcp_z >= carry_z - cfg.carry_z_tol
            ):
                self._enter(PLACE)

        elif phase == PLACE:
            if (
                self._near_xy(tcp, obs.bin_xy, cfg.place_xy_tol)
                and tcp_z <= cfg.place_height + cfg.place_z_tol
            ):
                self._enter(RELEASE)

        elif phase == RELEASE:
            if self._phase_steps >= cfg.release_frames:
                self._finish()

    def _retry_or_abort(self, obs: GripperState, reason: str) -> None:
        """超时或抓空：重试一次 APPROACH，超过上限则放弃。"""
        self.retries += 1
        self.last_failure = reason
        if self.retries > self.cfg.max_retries:
            self._aborted = True
            self._done = True
            self._last_target = self._hold(obs)
            self._enter(DONE)
        else:
            self._enter(APPROACH)

    def _finish(self) -> None:
        """放置完成。"""
        self._done = True
        self._aborted = False
        self._enter(DONE)

    # ---------------------------- 指令计算 ---------------------------- #
    def _command(self, obs: GripperState) -> tuple[np.ndarray, float]:
        cfg = self.cfg
        phase = self._phase
        tcp = obs.tcp_pos
        tcp_z = self._tcp_z(obs)

        if phase in (APPROACH, ALIGN):
            return self._pregrasp(obs), GRIP_OPEN

        if phase == DESCEND:
            return self._grasp_point(obs), GRIP_OPEN

        if phase in (CLOSE, VERIFY):
            # 原地闭爪/验证，避免继续下压把水果推走
            return self._hold(obs), GRIP_CLOSE

        if phase == LIFT:
            lift_z = self._lift_z if self._lift_z is not None else tcp_z + cfg.lift_height
            vx = tcp[0] + 0.4 * (obs.bin_xy[0] - tcp[0])
            vy = tcp[1] + 0.4 * (obs.bin_xy[1] - tcp[1])
            return np.array([vx, vy, lift_z], dtype=np.float64), GRIP_CLOSE

        if phase == CARRY:
            carry_z = self._carry_z if self._carry_z is not None else cfg.carry_height
            return np.array(
                [obs.bin_xy[0], obs.bin_xy[1], carry_z], dtype=np.float64
            ), GRIP_CLOSE

        if phase == PLACE:
            return np.array(
                [obs.bin_xy[0], obs.bin_xy[1], cfg.place_height], dtype=np.float64
            ), GRIP_CLOSE

        if phase == RELEASE:
            return np.array(
                [obs.bin_xy[0], obs.bin_xy[1], cfg.place_height], dtype=np.float64
            ), GRIP_OPEN

        # DONE
        return self._hold(obs, self._last_target), GRIP_OPEN
