"""``fruitsort.policies.gripper_fsm`` 的单元测试。

既可用 pytest 运行::

    conda run -n maniskill3 python -m pytest tests/test_gripper_fsm.py -q

也可以直接跑（无 pytest 依赖）::

    python tests/test_gripper_fsm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fruitsort.policies.gripper_fsm import (  # noqa: E402
    ALIGN,
    APPROACH,
    CARRY,
    CLOSE,
    DESCEND,
    DONE,
    LIFT,
    PHASES,
    PLACE,
    RELEASE,
    VERIFY,
    GRIP_CLOSE,
    GRIP_OPEN,
    GripperFSM,
    GripperFSMConfig,
    GripperState,
)

# 测试用几何：与仓库仿真一致的量级（传送带顶面 z≈0.45）
FRUIT_START = np.array([0.0, 0.0, 0.47])
BIN_XY = np.array([0.50, 0.40])
TCP_START = np.array([0.0, 0.0, 0.72])
DT = 0.05


def _state(tcp, fruit, *, vel=None, gap=0.08, grasping=False, tcp_z=None, dt=DT):
    return GripperState(
        tcp_pos=np.asarray(tcp, dtype=float),
        fruit_pos=np.asarray(fruit, dtype=float),
        fruit_vel=np.zeros(3) if vel is None else np.asarray(vel, dtype=float),
        fingers_gap=gap,
        is_grasping=grasping,
        bin_xy=BIN_XY.copy(),
        tcp_z=tcp_z,
        dt=dt,
    )


def simulate(
    fsm: GripperFSM,
    *,
    tcp_start=TCP_START,
    fruit_start=FRUIT_START,
    fruit_vel=(0.0, 0.0, 0.0),
    grasp_from_attempt: int = 1,
    speed: float = 0.6,
    gap_open: float = 0.08,
    max_steps: int = 2000,
):
    """极简运动学假手臂：tcp 朝 target 匀速移动，处于 CLOSE/VERIFY 时闭爪。

    返回 (trace, tcp, fruit, fsm)，trace 为 [(phase, grip, target), ...]。
    """
    tcp = np.asarray(tcp_start, dtype=float).copy()
    fruit = np.asarray(fruit_start, dtype=float).copy()
    vel = np.asarray(fruit_vel, dtype=float)
    gap = gap_open
    grasped = False
    attempt = 0
    prev_phase = None
    trace = []

    for _ in range(max_steps):
        out = fsm.step(_state(tcp, fruit, vel=vel, gap=gap, grasping=grasped))
        trace.append((out.phase, out.grip, out.target.copy()))

        if out.phase == CLOSE and prev_phase != CLOSE:
            attempt += 1
        prev_phase = out.phase

        # 位置伺服：单拍最多走 speed*dt
        err = out.target - tcp
        dist = float(np.linalg.norm(err))
        step = speed * DT
        tcp = tcp + err / dist * step if dist > step and dist > 0 else out.target.copy()

        # 夹爪开合
        if out.grip < 0:
            gap = max(0.0, gap - 0.04)
        else:
            gap = min(gap_open, gap + 0.04)

        # 接触 = 闭爪 + 到位 + 该次尝试允许成功
        near = float(np.linalg.norm(tcp - fruit)) < 0.06
        if out.grip < 0 and near and attempt >= grasp_from_attempt:
            grasped = True
        elif out.grip > 0:
            grasped = False

        if grasped:  # 抓住后水果随夹爪走
            fruit = tcp.copy()
        else:
            fruit = fruit + vel * DT

        if out.done:
            break

    return trace, tcp, fruit, fsm


def test_full_success_trajectory():
    """APPROACH → … → DONE 全链路成功，且各阶段顺序正确。"""
    fsm = GripperFSM()
    trace, _tcp, _fruit, _ = simulate(fsm)
    phases = [p for p, _, _ in trace]

    assert trace[-1][0] == DONE
    assert fsm.succeeded and fsm.done and not fsm.aborted
    assert fsm.retries == 0

    # 每个阶段至少出现一次，且第一次出现顺序严格递增
    for phase in PHASES:
        assert phase in phases, f"缺少阶段 {phase}: {phases}"
    first_idx = [phases.index(phase) for phase in PHASES]
    assert first_idx == sorted(first_idx), f"阶段乱序: {phases}"

    # 夹爪指令极性
    closed = {CLOSE, VERIFY, LIFT, CARRY, PLACE}
    opened = {APPROACH, ALIGN, DESCEND, RELEASE, DONE}
    for phase, grip, _ in trace:
        if phase in closed:
            assert grip == GRIP_CLOSE, (phase, grip)
        elif phase in opened:
            assert grip == GRIP_OPEN, (phase, grip)

    # 抓取发生在水果附近，且最终停在框上方
    assert _fruit[0] > FRUIT_START[0] + 0.1
    assert abs(_fruit[0] - BIN_XY[0]) < 0.1 and abs(_fruit[1] - BIN_XY[1]) < 0.1


def test_missed_grasp_retries_from_approach():
    """VERIFY 抓空一次 → 回到 APPROACH，重试后成功。"""
    # min_grasp_gap>0：把"夹空完全闭合"排除出成功判据，更接近真实夹爪
    cfg = GripperFSMConfig(min_grasp_gap=0.01)
    fsm = GripperFSM(cfg)
    trace, _tcp, _fruit, _ = simulate(fsm, grasp_from_attempt=2)
    phases = [p for p, _, _ in trace]

    assert fsm.retries == 1
    assert fsm.succeeded and not fsm.aborted
    assert phases[-1] == DONE

    # 第一次 VERIFY 之后必须重新出现 APPROACH
    first_verify = phases.index(VERIFY)
    assert APPROACH in phases[first_verify + 1:], phases
    # 第二次 VERIFY 之后进入 LIFT
    second_verify = phases.index(VERIFY, first_verify + 1)
    assert LIFT in phases[second_verify + 1:], phases


def test_timeout_aborts_after_max_retries():
    """一直够不到水果 → 每阶段超时 → 重试耗尽后 done=True。"""
    cfg = GripperFSMConfig(max_retries=1, timeouts={APPROACH: 0.1})
    fsm = GripperFSM(cfg)

    tcp = np.array([1.0, 1.0, 1.0])
    fruit = np.array([-2.0, -2.0, 0.47])
    out = None
    for _ in range(100):
        out = fsm.step(_state(tcp, fruit))
        if out.done:
            break

    assert out is not None and out.done
    assert out.phase == DONE and fsm.phase == DONE
    assert fsm.aborted and not fsm.succeeded
    assert fsm.retries == 2  # 第一次超时重试，第二次超时放弃
    assert fsm.last_failure == f"timeout:{APPROACH}"


def test_lead_prediction_shifts_target():
    """fruit_vel 非零时，APPROACH 目标 x 必须大于 fruit_x。"""
    cfg = GripperFSMConfig(lead=0.15)
    fsm = GripperFSM(cfg)
    fruit = np.array([0.10, 0.20, 0.47])
    vel = np.array([0.30, 0.0, 0.0])

    target, grip, phase, done = fsm.step(_state(TCP_START, fruit, vel=vel))

    assert phase == APPROACH and grip == GRIP_OPEN and not done
    assert target[0] > fruit[0]
    assert abs(target[0] - (fruit[0] + 0.30 * 0.15)) < 1e-9
    assert abs(target[1] - fruit[1]) < 1e-9
    assert abs(target[2] - (fruit[2] + cfg.approach_height)) < 1e-9

    # lead=0 时退化为不预瞄
    fsm0 = GripperFSM(GripperFSMConfig(lead=0.0))
    target0, _, _, _ = fsm0.step(_state(TCP_START, fruit, vel=vel))
    assert abs(target0[0] - fruit[0]) < 1e-9


def test_mapping_input_and_reset():
    """支持 dict 输入（便于接入脚本），且 reset 可重新开始。"""
    fsm = GripperFSM()
    obs = {
        "tcp_pos": TCP_START.copy(),
        "tcp_z": 0.72,
        "fruit_pos": np.array([0.13, -0.02, 0.47]),
        "fruit_vel": np.array([0.05, 0.0, 0.0]),
        "fingers_gap": 0.08,
        "is_grasping": False,
        "bin_xy": BIN_XY.copy(),
        "dt": DT,
    }
    target, grip, phase, done = fsm.step(obs)
    assert phase == APPROACH and grip == GRIP_OPEN and not done
    assert target.shape == (3,)

    fsm.reset()
    assert fsm.phase == APPROACH
    assert fsm.retries == 0 and fsm.steps == 0 and not fsm.done


def test_invalid_config_rejected():
    """非法参数尽早报错，而不是在闭环里静默跑飞。"""
    for factory in (
        lambda: GripperFSMConfig(close_frames=0),
        lambda: GripperFSMConfig(max_retries=-1),
        lambda: GripperFSMConfig(lead=-0.1),
        lambda: GripperFSMConfig(min_grasp_gap=0.05, grasp_gap_thresh=0.04),
        lambda: GripperFSMConfig(timeouts={"nope": 1.0}),
    ):
        try:
            factory()
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("应当拒绝非法配置")


def _main() -> int:
    tests = [
        test_full_success_trajectory,
        test_missed_grasp_retries_from_approach,
        test_timeout_aborts_after_max_retries,
        test_lead_prediction_shifts_target,
        test_mapping_input_and_reset,
        test_invalid_config_rejected,
    ]
    for fn in tests:
        fn()
        print(f"[ok] {fn.__name__}")
    print(f"\n{len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
