"""纯 ACT 策略的闭环执行器（无脚本执行器/任务 FSM）。

被 `scripts/eval_act_pure.py` 与 `scripts/eval_disturbance.py --policy act` 共用，
保证「扰动评测」和「成功率评测」跑的是同一套部署逻辑。

部署配置（v1.4 达标配置）
------------------------
- 增量动作：模型输出相对当前 TCP 的位移（与训练标签一致）
- 残差反馈 `fb`：`target = tcp + Δ_θ + fb·(fruit − tcp)`（默认只作用 xy）
- 相位：模型自己的 3 类相位头（k=0）反馈回下一步状态；`descend_commit` 给「下探」加承诺
- 抓取/搬运判据：手指接触，或「距离 <4.5cm 且上一步闭爪」
- 去抖/恢复：夹爪 logit 平滑、搬运段锁爪、抓空 12 步自动重试

这些都是**模型自身决策的迟滞**或安全界限，不含任务级 FSM（何时抓、抓哪个、放哪由策略输出）。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from ..learning.bc import Normalizer, build_obs, build_tactile
from ..models.act_policy import ActConfig, ActPolicy
from .scripted import HOME_EULER

INSTR_ID = {"all": 0, "apple": 1, "strawberry": 2, "banana": 3, "pear": 4}


def prep(x, size):
    if x.shape[0] != size:
        x = cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
    return (x.astype(np.float32) / 255.0).transpose(2, 0, 1)


def finger_normal_force(u) -> float:
    """两指法向力均值（N）——真机上就是指端 F/T/触觉读数。

    这里刻意**不读物体状态**：只用左右指的接触力与它们的连线方向，
    因此可以原样搬到真机（换一只带 F/T 的夹爪即可）。
    """
    try:
        L = u.agent.robot.links_map["panda_leftfinger"]
        R = u.agent.robot.links_map["panda_rightfinger"]
        FL = L.get_net_contact_forces()[0].cpu().numpy()
        FR = R.get_net_contact_forces()[0].cpu().numpy()
        pL = L.pose.p[0].cpu().numpy()
        pR = R.pose.p[0].cpu().numpy()
    except Exception:  # noqa: BLE001
        return 0.0
    ax = pL - pR
    n = float(np.linalg.norm(ax))
    ax = ax / n if n > 1e-6 else np.array([0.0, 1.0, 0.0])
    return 0.5 * (abs(float(np.dot(FL, ax))) + abs(float(np.dot(FR, -ax))))


class ActPureRunner:
    def __init__(self, env, ckpt_path, kind: str, device: str = "cpu", *,
                 fb: float = 0.0, fb_z: bool = False, grip_ema: float = 0.0,
                 carry_latch: bool = False, descend_commit: int = 0,
                 max_step: float = 0.06, lead: float = 0.0, z_gate: bool = False,
                 phase_fb: bool = True, hold_latch: bool = False,
                 grip_hyst: float = 0.0, contact_force_th: float = 0.2,
                 release_gate: float = 0.0, release_wait: int = 120,
                 grasp_retry: bool = True, chunk_exec: int = 1,
                 carry_max_step: float = 0.0, slip_guard: bool = False,
                 slip_thr: float = 0.05, slip_slow_step: float = 0.012,
                 slip_hold: int = 6,
                 carry_assist: bool = False, carry_assist_min: float = 0.0,
                 regrasp: bool = False, regrasp_thr: float = 0.6,
                 regrasp_depth: float = 0.02, drop_ckpt: str = "",
                 grasp_gate: bool = False, gate_min_force: float = 2.0,
                 gate_asym: float = 0.5, gate_slip: float = 0.02,
                 gate_delay: int = 5, gate_retries: int = 2,
                 gate_bias: float = 0.005, close_force_th: float = 0.0,
                 close_force_hold: float = 0.0,
                 side_servo: bool = False, side_step: float = 0.002,
                 width_map: dict | None = None, width_delta: float = 0.002,
                 width_g: float | None = None, close_ramp: float = 0.0,
                 grasp_z_bias: dict | None = None, stall_guard: float = 0.0,
                 stall_eps: float = 0.006, yaw_map: dict | None = None,
                 close_z_gate: float = 0.0, close_z_gate_map: dict | None = None,
                 close_z_off_map: dict | None = None,
                 side_grasp_kinds: tuple = (), side_offset: float = 0.09,
                 side_carry: str = "scripted",
                 intercept: float = 0.0, zg_hold_chunks: int = 0,
                 yaw_align_extent: float = 0.0, yaw_align_period: int = 10,
                 yaw_align_gain: float = 0.35, yaw_align_tol: float = 0.05,
                 yaw_align_prerotate: bool = True, yaw_align_hold_dist: float = 0.18,
                 yaw_align_ready: float = 0.17, yaw_align_pre_max: int = 70,
                 yaw_align_rot_frac: float = 0.0, yaw_align_fast: bool = False,
                 yaw_align_hold: bool = True, yaw_hold_fade: float = 0.12,
                 suction: bool = False, suction_radius: float = 0.10,
                 suction_carry: str = "scripted",
                 yaw_hold_frac: float = 1.0, yaw_hold_cap: float = 0.0,
                 close_commit: float = 0.0, close_commit_xy: float = 0.030,
                 close_commit_steps: int = 8, close_commit_map: dict | None = None,
                 lead_long: float = 0.0, lead_long_extent: float = 0.11,
                 lead_long_taper: float = 0.15, grasp_proxy: str = "always",
                 close_win_xy: float = 0.03, close_win_z: float = 0.06,
                 grasp_verify: bool = False, grasp_verify_xy: float = 0.0,
                 wrist_checker=None, wrist_tol: float = 0.02,
                 yaw_align: bool = False,
                 yaw_align_off: float = 0.0,
                 max_step_map: dict | None = None,
                 carry_step_map: dict | None = None):
        ck = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
        self.cfg = ActConfig(**ck["cfg"])
        self.model = ActPolicy(self.cfg).to(device)
        self.model.load_state_dict(ck["model"]); self.model.eval()
        self.ns = Normalizer.from_state(ck["norm_state"])
        self.na = Normalizer.from_state(ck["norm_act"])
        self.delta_actions = bool(ck.get("delta_actions", False))
        self.use_phase = self.cfg.obs_dim > 49
        self.tactile_token = int(getattr(self.cfg, "tactile_dim", 0) or 0)
        self.use_force = bool(getattr(self.cfg, "use_force_head", False))
        # 布局：[49 base][9 tactile][3 phase]。触觉与否由 obs_dim 决定
        self.tactile_dim = 9 if (self.cfg.obs_dim - (3 if self.use_phase else 0)) >= 58 else 0
        _extra = self.cfg.obs_dim - (3 if self.use_phase else 0) - 49
        self.use_queue = _extra in (1, 5)
        self.use_geom = _extra in (4, 5)

        self.env, self.u, self.device = env, env.unwrapped, device
        # 头部图像：全局（相位/上下文）+ 目标 ROI（精细定位）双路；ROI 变换与录制端一致
        self.use_roi = bool(getattr(self.cfg, "use_roi", False))
        try:
            from ..perception.roi import RoiCrop
            self.roi = RoiCrop(width_m=0.50, out_size=128) if self.use_roi else None
        except Exception:  # noqa: BLE001
            self.roi = None
        # 观测/判据用的"物体源"：默认仿真真值；--perception vision 时换成感知替身
        self.actors = None
        self.kind = kind
        self.fb, self.fb_z, self.grip_ema = fb, fb_z, grip_ema
        self.carry_latch, self.descend_commit = carry_latch, descend_commit
        self.max_step, self.lead, self.z_gate = max_step, lead, z_gate
        # 按类限速：长条/易滑类（banana/pear）需要更小的步长，其余可以更快
        self.max_step_map = max_step_map or {}
        self.carry_step_map = carry_step_map or {}
        self.phase_fb, self.hold_latch, self.grasp_retry = phase_fb, hold_latch, grasp_retry
        # 夹爪指令迟滞：模型在"闭不闭"之间输出抖动（logit 在 0 附近），EMA 之后仍会
        # 每 3–5 步翻一次符号 → 指面刚合上又张开，永远夹不住。>0 时用双阈值锁存。
        self.grip_hyst = float(grip_hyst)
        self.contact_force_th = float(contact_force_th)
        # 放料门：模型会在**离料框 15–27 cm** 时就开爪（实测 trace），果子直接掉地上。
        # 料框位置是环境夹具（事先已知，不是物体真值），所以可以在部署侧硬拦：
        # 未进入框上方 release_gate 米内 → 不放开 + 直接把目标点设成框心。
        self.release_gate = float(release_gate)
        self.release_wait = int(release_wait)
        self._rel_wait = 0
        self.chunk_exec = chunk_exec
        # 搬运段单独限速：接近段要快要准，搬运段过冲会把水果带到框外
        self.carry_max_step = carry_max_step
        # 触觉补救：搬运中检测到滑移（物体相对 TCP 有速度且有夹持力）→ 立即降速搬运
        self.slip_guard = slip_guard
        self.slip_thr = slip_thr
        self.slip_slow_step = slip_slow_step
        # 自适应限速的"拖尾"：rho 掉回阈值以下后再保持 N 步慢速，避免在阈值附近抖动
        # （一步快一步慢会把果子晃松，正好是要避免的）
        self.slip_hold = int(slip_hold)
        # 搬运段"保相位"辅助：**不动相位机、不改模型的速度指令**，只把目标点方向
        # 旋转到"朝向料框中心"。carry_assist_min>0 时再给一个步长下限（相当于提速）。
        self.carry_assist = bool(carry_assist)
        self.carry_assist_min = float(carry_assist_min)
        self._slip_left = 0
        # 触觉预测驱动「闭爪前重抓」：滑移头认为这一抓会滑 → 先开爪、下沉 2cm、再闭
        self.regrasp = regrasp
        self.regrasp_thr = regrasp_thr
        self.regrasp_depth = regrasp_depth
        # 独立的掉落预测器（轻量 MLP，只看触觉历史 + 几何；主策略不参与）
        # ---- 抓取验证门 + 有偏重试（论文①：抬升后判"这一抓是否成立"）----
        self.grasp_gate = grasp_gate
        self.gate_min_force = gate_min_force
        self.gate_asym = gate_asym
        self.gate_slip = gate_slip
        self.gate_delay = gate_delay
        self.gate_retries = gate_retries
        self.gate_bias = gate_bias
        # ---- 闭合到力阈（论文③：力→位置，小果不被挤飞、大果夹紧）----
        self.close_force_th = close_force_th
        # 力控闭合：闭爪过程中指端 Fn 到 close_force_hold（N）就**冻结**当前指令并锁存。
        # 标定表：g=+0.4 → 开口 58.7 mm、Fn=4.1 N 是最稳的夹持点；
        # 位置式"宽度设定点"的可夹窗口只有 58–60 mm（再小就把果子挤飞），力控没有这个问题。
        self.close_force_hold = float(close_force_hold)
        self._fhold_grip: float | None = None
        # ---- 单侧接触微伺服（论文②）：闭爪过程中只有一指受力 → 朝弱侧平移再闭 ----
        self.side_servo = side_servo
        self.side_step = side_step
        # ---- 宽度设定点（LeTac 式位置等价力控）：闭合到 w*(类) − δ，而不是合到底 ----
        # 实测空载映射：gap ≈ 0.030 + 0.050·g（g∈[-0.5,1]），反解 g = (w-0.030)/0.050
        self.width_map = width_map or {}
        self.width_delta = width_delta
        self.width_g = width_g
        # ---- 渐进闭爪：不一步跳到 -1，而是每步 -close_ramp，避免闭爪瞬间把水果弹飞 ----
        self.close_ramp = close_ramp
        # ---- 按类的抓取高度偏置（执行层工具偏置）：梨需要 +1.5cm 才能夹住果身 ----
        self.grasp_z_bias = grasp_z_bias or {}
        self.drop_net = None
        self.drop_prob = 0.0
        if drop_ckpt:
            ck = torch.load(Path(drop_ckpt), map_location="cpu", weights_only=False)
            h = int(ck.get("hist", 5)); t = int(ck.get("tac", 9))
            net = torch.nn.Sequential(
                torch.nn.Linear(h * t + 3, 128), torch.nn.GELU(),
                torch.nn.Linear(128, 128), torch.nn.GELU(), torch.nn.Linear(128, 1))
            net.load_state_dict({k.replace("net.", ""): v for k, v in ck["model"].items()})
            net.eval()
            self.drop_net = net
            self.drop_mu = np.asarray(ck["mean"], dtype=np.float32)
            self.drop_sd = np.maximum(np.asarray(ck["std"], dtype=np.float32), 1e-6)
            self.drop_hist = h
        # 搬运段单独限速：接近段要快要准，搬运段过冲会把水果带到框外
        self.carry_max_step = carry_max_step
        # 触觉补救：搬运中检测到滑移（物体相对 TCP 有速度且有夹持力）→ 立即降速搬运
        self.slip_guard = slip_guard
        self.slip_thr = slip_thr
        self.slip_slow_step = slip_slow_step
        self.slip_hold = int(slip_hold)
        # 触觉预测驱动「闭爪前重抓」：滑移头认为这一抓会滑 → 先开爪、下沉 2cm、再闭
        self.regrasp = regrasp
        self.regrasp_thr = regrasp_thr
        self.regrasp_depth = regrasp_depth
        # 独立的掉落预测器（轻量 MLP，只看触觉历史 + 几何；主策略不参与）
        # ---- 抓取验证门 + 有偏重试（论文①：抬升后判"这一抓是否成立"）----
        self.grasp_gate = grasp_gate
        self.gate_min_force = gate_min_force
        self.gate_asym = gate_asym
        self.gate_slip = gate_slip
        self.gate_delay = gate_delay
        self.gate_retries = gate_retries
        self.gate_bias = gate_bias
        # ---- 闭合到力阈（论文③：力→位置，小果不被挤飞、大果夹紧）----
        self.close_force_th = close_force_th
        # ---- 单侧接触微伺服（论文②）：闭爪过程中只有一指受力 → 朝弱侧平移再闭 ----
        self.side_servo = side_servo
        self.side_step = side_step
        # ---- 宽度设定点（LeTac 式位置等价力控）：闭合到 w*(类) − δ，而不是合到底 ----
        # 实测空载映射：gap ≈ 0.030 + 0.050·g（g∈[-0.5,1]），反解 g = (w-0.030)/0.050
        self.width_map = width_map or {}
        self.width_delta = width_delta
        self.width_g = width_g
        # ---- 渐进闭爪：不一步跳到 -1，而是每步 -close_ramp，避免闭爪瞬间把水果弹飞 ----
        self.close_ramp = close_ramp
        # ---- 按类的抓取高度偏置（执行层工具偏置）：梨需要 +1.5cm 才能夹住果身 ----
        self.grasp_z_bias = grasp_z_bias or {}
        self.drop_net = None
        self.drop_prob = 0.0
        if drop_ckpt:
            ck = torch.load(Path(drop_ckpt), map_location="cpu", weights_only=False)
            h = int(ck.get("hist", 5)); t = int(ck.get("tac", 9))
            net = torch.nn.Sequential(
                torch.nn.Linear(h * t + 3, 128), torch.nn.GELU(),
                torch.nn.Linear(128, 128), torch.nn.GELU(), torch.nn.Linear(128, 1))
            net.load_state_dict({k.replace("net.", ""): v for k, v in ck["model"].items()})
            net.eval()
            self.drop_net = net
            self.drop_mu = np.asarray(ck["mean"], dtype=np.float32)
            self.drop_sd = np.maximum(np.asarray(ck["std"], dtype=np.float32), 1e-6)
            self.drop_hist = h
        # ---- 停滞守卫（stall guard）：队列里"接近相位塌陷"会让模型输出 ≈0 增量，
        # 与残差反馈饱和值抵消 → TCP 逐帧冻结、水果被带走（docs 6.26.35 的 trace）。
        # 检测到"指令位移≈0 且水果仍在接近窗口内、还离得远"→ 沿"朝水果"的方向
        # 用最小接近步长强制重规划（纯执行层，不改模型权重）。
        self.stall_guard = float(stall_guard)
        self.stall_eps = float(stall_eps)
        # 按类的手腕偏航（绕竖直轴）：用于让夹爪闭合方向对齐长条水果的长轴
        self.yaw_map = yaw_map or {}
        # 闭爪高度门：离水果还很高就闭爪 = 夹空气（实测队列里发生在 z 高 9 cm 处）。
        # >0 时：若"想闭爪"但 TCP 比果心高超过该值，则改为张开并下探到果心上方 2 cm。
        self.close_z_gate = float(close_z_gate)
        # 按类闭爪高度门：果心以上 gate 米内才允许闭爪（小果不能在高处闭）
        self.close_z_gate_map = dict(close_z_gate_map or {})
        # 闭爪高度门触发时"下探到哪"：果心 + 该类偏移（默认 +2cm；小果可给 0/−5mm）
        self.close_z_off_map = dict(close_z_off_map or {})
        # ---- 小果的"侧向进入"抓取（结构改动，不依赖模型的垂直下探）----
        # 平行夹爪从正上方抓小果只能压到果顶（实测草莓 0/24）。改成：先到果子
        # **下游 +x 侧** 9 cm、与果心同高，张爪；再沿 -x 滑入两指之间；就位后闭爪。
        # 全程只用感知几何（无真值）。
        self.side_grasp_kinds = tuple(side_grasp_kinds or ())
        self.side_offset = float(side_offset)
        # 侧抓之后的搬运也交给 FSM（模型没在"果心高度侧抓"这个状态上训练过，搬运会甩掉）
        self.side_carry = str(side_carry)
        self._side_grip = 1.0
        # ---- 闭爪承诺：到果身高度且水平对齐 → 强制闭爪 + 冻结水平目标 ----
        # 动机（v1.62 诊断）：手臂悬到果子上方时俯视被自己挡住，估计值按带速外推，
        # 而果实在指间已经不再跟带面同速 → 估计**系统性超前 ~2 cm**。策略于是每步
        # "再挪 2 cm"，闭爪条件永远差一点，表现为 descend≈700 步、全程悬停。
        self.close_commit = float(close_commit)
        self.close_commit_xy = float(close_commit_xy)
        self.close_commit_map = dict(close_commit_map or {})
        # ---- 长条目标的"拦截点前移"（按带速，单位：秒）----
        # 追一个 0.08 m/s 的目标时，反馈项本身有滞后；长条目标（香蕉 0.17 m）
        # 又被尺寸闸排除在 near-hold 之外，横向/纵向偏移都没人补 → 每步都差一点。
        # 这里按感知到的世界长度设闸，只对长条目标把瞄准点沿带速前移 lead_long 秒。
        # 抓取近似判据：auto=只在查不到真实接触时用 / always=总是用 / never=从不用
        self.grasp_proxy = str(grasp_proxy)
        self.lead_long = float(lead_long)
        self.lead_long_extent = float(lead_long_extent)
        # 前移只在"还离得远"时生效，靠近后线性衰减到 0：
        # 恒定前移会把长条目标的抓取点推到最后 2 cm（实测单发 4/6 → 0/6）
        self.lead_long_taper = float(lead_long_taper)
        self.close_commit_steps = int(close_commit_steps)
        self._commit_left = 0
        # ---- 拦截式下探（intercept）----
        # 队列里水果一直在走，"按当前位置下探"会永远追不上（trace: x 对齐后不下降/空中闭爪）。
        # intercept>0 时：模型一旦在空中想闭爪，就按"下探耗时 × 带速"预测落点，
        # 一次下探到预测点；只有 |x_err|<close_win_xy 且离果心 <close_win_z 才真正闭爪。
        self.intercept = float(intercept)
        # 闭爪高度门触发后，把竖直目标**钉住**若干 chunk（--zg-hold-chunks）。
        # 实测：phase 1 的 90%+ 时间不是"降不下来"，而是在果心上方 ~4 cm **来回震荡** ——
        # 模型在空中闭爪 → 高度门压回并重设目标 → 下一个 chunk 模型又把 z 抬回去。
        # 钉住竖直目标就能打断这个极限环（水平仍由模型决定）。
        self.zg_hold_chunks = int(zg_hold_chunks)
        self._zg_left = 0
        self._zg_z = None
        # 诊断：每次进入 phase 1 时的高度差（TCP − 果心），判断下探是不是竖直为主
        self.p1_dz_in = []
        self._prev_phase = 0
        self.close_win_xy = float(close_win_xy)
        self.close_win_z = float(close_win_z)
        # 抓取确认：只"碰到顶端"不算抓住（队列里会误判成 phase 2、随后水果滑走）
        self.grasp_verify = bool(grasp_verify)
        # 腕部抓取确认：遮挡期（头部看不到）用腕部掩码判断"果子还在两指之间"。
        # 由控制器注入 `callable(kind, tcp, obs) -> 距离(米) | None`。
        self.wrist_checker = wrist_checker
        self.wrist_tol = float(wrist_tol)
        self.wrist_ok_total = 0        # 腕部确认救回的次数（不随 reset 清零）
        # 附加水平对齐窗口（米；0=关闭）。默认关闭：实测 4–6 cm 的感知误差会把真抓取否掉
        self.grasp_verify_xy = float(grasp_verify_xy)
        # ---- 手腕对准（验证性实验，用仿真真值 yaw）----
        # 让"两指连线（闭合方向）"垂直于水果长轴：实测 δ 旋转 → 指面轴角度变化 ≈ −δ。
        # yaw_align_off 用来试两种假设（长轴沿本地 x / 沿本地 y，差 π/2）。
        self.yaw_align = bool(yaw_align)
        self.yaw_align_off = float(yaw_align_off)
        # 只对"够长"的目标对准（圆形水果主轴没定义，感知 yaw 误差 24–28°）
        self.yaw_align_extent = float(yaw_align_extent)
        # 闭环对准：每 period 步重算一次（腕部转动到位要 ~10-20 步，太密会追滞后 → 振荡）
        self.yaw_align_period = max(1, int(yaw_align_period))
        self.yaw_align_gain = float(yaw_align_gain)
        self.yaw_align_tol = float(yaw_align_tol)
        self._yaw_t = 0
        self.yaw_err_deg = 0.0
        self._pre_steps = 0
        # 两段式：离目标还远（> hold_dist）且姿态残差 > ready 时，**只转不走**
        # （v1.65 证据：pd_ee_pose 在平移阶段 170 步不执行姿态指令 → 对准永远来不及）
        self.yaw_align_prerotate = bool(yaw_align_prerotate)
        self.yaw_align_hold_dist = float(yaw_align_hold_dist)
        self.yaw_align_ready = float(yaw_align_ready)
        self.yaw_align_pre_max = int(yaw_align_pre_max)
        # 预转时是否允许**同时**靠近：0=原地只转（v1.66 行为），>0=按该比例下发放平移
        # （0.35 表示靠近速度降到 35%，避免"先停 2 s 转腕"吃掉单件时间）
        self.yaw_align_rot_frac = float(yaw_align_rot_frac)
        # 快速预转（残差大时 period=1/gain=0.9）：收敛快 3 倍，但抓取瞬间仍带角速度，
        # 薄弯果身会被甩掉（v1.67 实测 rho 14→40、单发 2/6→1/6）→ 默认关闭
        self.yaw_align_fast = bool(yaw_align_fast)
        # 目标不在"对准范围"（苹果/草莓）时**保持**当前腕部偏航，不要回 HOME：
        # 否则空档期预先转好的朝向会在接它们时被清零，香蕉轮到时就白攒了。
        # yw-* 数据本来就把 ±0.9 rad 的腕部偏航覆盖到所有类，属于分布内动作。
        self.yaw_align_hold = bool(yaw_align_hold)
        # hold 的"淡出距离"：非长条件在距果心 fade 米内把保持的偏航线性收到 0，
        # 让苹果/草莓的**最后一段闭合**仍用训练里最常见的 HOME 朝向
        self.yaw_hold_fade = float(yaw_hold_fade)
        # ---- 吸盘（软）末端：不靠夹爪包住，靠"贴近即吸" ----
        # 复用 eval_policy_suction.py 的语义：模型说"吸"(grip<0) 且 TCP 距果心 <r → 附着，
        # 之后水果跟着 TCP；模型说"放"且在目标框上方 → 落到框底（env 判 in_bin=delivered）。
        self.suction = bool(suction)
        self.suction_radius = float(suction_radius)
        # 吸附后的搬运：scripted=直接奔向目标框再释放（task 级 FSM，吸盘常见做法）；
        # policy=沿用策略输出的搬运段（实测流内会少走 10–16 cm，进不了框）
        self.suction_carry = str(suction_carry)
        self.attached = False
        self.attach_off = np.zeros(3)
        self._side_grip = 1.0
        self._hold_steps = 0
        self._hold_tcp = np.zeros(3)
        self._release_to = None
        self._belt_put = None
        # 非长条件"保持"的量：可整体缩放（frac）并限幅（cap，rad）。
        # 目的：让空档期预转**部分**活到香蕉那一拍，同时苹果/草莓尽量留在训练主分布（yaw≈0）
        self.yaw_hold_frac = float(yaw_hold_frac)
        self.yaw_hold_cap = float(yaw_hold_cap)
        self._pre_steps = 0
        self._yaw_cmd = 0.0
        self._yaw_target = None
        self._yaw_t = 0
        self.yaw_err_deg = 0.0
        self._pre_steps = 0
        self.attached = False
        self.attach_off = np.zeros(3)
        self._release_to = None
        self._belt_put = None
        self._hold_steps = 0
        self._hold_tcp = np.zeros(3)
        self._hold_steps = 0
        self._hold_tcp = np.zeros(3)
        self._descend_target = None
        self._descend_steps = 0
        self._stall_tcp = np.zeros(3)
        self._commit_left = 0
        self._stall_run = 0
        self.reset()

    # ------------------------------------------------------------------ #
    def set_actors(self, actors: dict | None):
        """注入物体源（感知替身）。传 None 恢复仿真真值。"""
        self.actors = actors

    @property
    def _objs(self) -> dict:
        return self.actors if self.actors is not None else self.u.fruit_actors

    def _close_cmd(self, kind: str) -> float:
        """闭爪时的指令：有宽度设定点就按目标开口反解（**可能为正**），否则合到底。"""
        if self.width_map and kind in self.width_map:
            w = float(self.width_map[kind]) - float(self.width_delta)
            return float(np.clip((w - 0.030) / 0.050, -1.0, 1.0))
        return -1.0

    def _grasp_proxy(self, actor, tcp) -> bool:
        """感知模式下的"抓住"判据（无成对接触力可用）：闭合 + 水平对齐 + 指面到果身高度。"""
        try:
            p = actor.pose.p[0].cpu().numpy().astype(np.float64)
            belt = float(getattr(self.u, "belt_top_z", 0.45))
            near_xy = float(np.linalg.norm(p[:2] - tcp[:2])) < 0.05
            at_pad = abs(float(tcp[2] - p[2])) < 0.05
            lifted = float(p[2]) > belt + 0.02
            return bool(self.prev_closed and near_xy and (at_pad or lifted))
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------ #
    def reset(self):
        self.oh, self.wr, self.st, self.tac = [], [], [], []
        self.roi_buf = []
        self.phase_state = 0
        self.phase_cand = 0
        self.same = 0
        self.commit = 0
        self.grasp_lost = 0
        self.closed_steps = 0
        self.grip_ema_val = None
        self._fhold_grip = None
        self.prev_grip = 1.0
        # 显式的"开/合"状态：grip 的**数值**可能为正（宽度设定点），不能再靠符号判断
        self.closed = False
        self.prev_closed = False
        self.open_streak = 0
        self.slip_slow = False
        self._slip_left = 0
        self._zg_left = 0
        self._zg_z = None
        self.force_target = None
        self.tac_hist = []
        self.gate_timer = 0
        self.gate_retries_done = 0
        self.retry_bias = np.zeros(3)
        self.gate_failed_once = False
        self.regrasp_done = False
        self.slip_prob = None
        self.cur_grip = 1.0
        self._yaw_cmd = 0.0
        self._yaw_target = None
        self._yaw_t = 0
        self.yaw_err_deg = 0.0
        self._pre_steps = 0
        self.attached = False
        self.attach_off = np.zeros(3)
        self._release_to = None
        self._belt_put = None
        self._hold_steps = 0
        self._hold_tcp = np.zeros(3)
        self._hold_steps = 0
        self._hold_tcp = np.zeros(3)
        self._descend_target = None
        self._descend_steps = 0
        self._stall_run = 0
        self._stall_tcp = np.zeros(3)
        self._commit_left = 0
        self.queue = []
        self.queue_logits = None
        self.stats = {"close": 0, "descend": 0, "carry": 0, "steps": 0,
                      "slip": 0, "slip_first": -1, "regrasp": 0, "slip_prob_max": 0.0,
                      "force_target": 0.0, "fn_peak": 0.0, "drop_prob_max": 0.0,
                      "gate_fail": 0, "retries": 0, "fL": 0.0, "fR": 0.0,
                      "side_push": 0, "grip_w": 0.0}
        self.grasping = False

    # ------------------------------------------------------------------ #
    def act(self, obs) -> np.ndarray:
        u, cfg = self.u, self.cfg
        kind = self.kind
        actor = self._objs.get(kind)
        tcp = u.agent.tcp_pose.p[0].cpu().numpy().astype(np.float64)
        grasping = False
        if actor is not None:
            fpos_now = actor.pose.p[0].cpu().numpy().astype(np.float64)
            fd = float(np.linalg.norm(fpos_now - tcp))
            if self.suction:
                # ---- 吸盘状态机（无接触力、无朝向要求）----
                if self.attached:
                    # 卡死守卫（v1.73）："**到目标框的距离** 连续 30 步没有变短" → 放回带面重试。
                    # 旧版判据是"TCP 15 步几乎不动"，但实测卡死是**缓慢移动**（手臂在框外
                    # 来回蹭），静止判据永远不触发 → 每局丢 1 件。
                    try:
                        from fruitsort.sim.conveyor import BIN_LAYOUT
                        _bx, _by = BIN_LAYOUT[kind]
                    except Exception:  # noqa: BLE001
                        _bx, _by = 0.0, -0.3
                    _bd = float(np.hypot(tcp[0] - _bx, tcp[1] - _by))
                    self._att_steps = getattr(self, "_att_steps", 0) + 1
                    if _bd < getattr(self, "_bin_best", 9.9) - 0.010:
                        self._bin_best = _bd
                        self._hold_steps = 0
                    else:
                        self._hold_steps += 1
                    # 30 步(1 s)没有缩短，或附着总时长超过 4 s → 放回并把位置挪回**好抓区**
                    if self._hold_steps >= 30 or self._att_steps >= 120:
                        # 放回"好抓区"（x<=-0.30，带面中线 y=0）：放到 x≈0 会进手臂折叠死区。
                        # 实际 teleport 交给控制器（它持有 target['actor']，避免拿错同类件）
                        belt_z = float(getattr(u, "belt_top_z", 0.45)) + 0.03
                        _px = min(float(np.clip(tcp[0], -0.70, 0.30)), -0.30)
                        self._belt_put = (float(_px), 0.0, float(belt_z))
                        self.attached = False
                        self._hold_steps = 0
                        self._bin_best = 9.9
                        self.stats["suction_retry"] = self.stats.get("suction_retry", 0) + 1
                if self.attached:
                    # 已吸附：水果跟着 TCP（用上一步的 TCP，1 步滞后可忽略）
                    p = np.clip(tcp + self.attach_off, -2.5, 2.5)
                    try:
                        from mani_skill.utils.structs.pose import Pose
                        _a = u.fruit_actors.get(kind)
                        if _a is not None:
                            _a.set_pose(Pose.create_from_pq(
                                torch.tensor([p], dtype=torch.float32)))
                            _a.set_linear_velocity(torch.zeros(1, 3))
                            _a.set_angular_velocity(torch.zeros(1, 3))
                    except Exception:  # noqa: BLE001
                        pass
                # 吸盘物理上"接触即吸"：贴到 5 cm 内直接吸上（不再依赖模型的闭爪时机，
                # 那正是平口夹爪失败的地方）；模型只负责"什么时候放"
                if (not self.attached and float(fpos_now[2]) > 0.30
                        and (fd < 0.05 or (self.prev_closed and fd < self.suction_radius))):
                    self.attached = True
                    # 附着偏移取"果心相对 TCP"，但**至少往下一段**，避免吸住的果身与
                    # 指面/机身在物理上重叠（重叠会把手臂顶起来，实测 z 从 0.82 被顶到 0.92）
                    off = fpos_now - tcp
                    if off[2] > -0.09:
                        off = np.array([off[0], off[1], -0.09], dtype=np.float64)
                    self.attach_off = off
                    self.stats["suction_on"] = self.stats.get("suction_on", 0) + 1
                # 到框上方就"吹气释放"（吸盘典型行为）：不再等模型给"放"的信号——
                # 实测模型在吸盘语义下常忘了开，导致抱着果子跑满 900 步
                if self.attached:
                    try:
                        from fruitsort.sim.conveyor import BIN_LAYOUT
                        bx, by = BIN_LAYOUT[kind]
                    except Exception:  # noqa: BLE001
                        bx, by = 0.0, -0.3
                    # 释放判据改成"到框心的半径"（v1.74）：右侧框（banana/pear, x=+0.46）
                    # 在 Panda 的伸展极限上，实测手臂只能到 11–13 cm，用 ±7 cm 的方框
                    # 永远不触发（trace: bd 11.0/19.9cm 反复被守卫放回）。
                    if float(np.hypot(float(tcp[0]) - bx, float(tcp[1]) - by)) < 0.13 \
                            and float(tcp[2]) < 0.62:
                        # 由控制器用 **target['actor']** 落到框底（runner 里用
                        # u.fruit_actors 可能拿到别的同类件）
                        self._release_to = (float(bx), float(by))
                        self.attached = False
                        self.stats["suction_off"] = self.stats.get("suction_off", 0) + 1
                grasping = bool(self.attached)
            # 手指接触，或「贴得很近且上一步确实闭爪」
            # 抓取判定用**指端接触力**（现实中对应指端 F/T / 触觉），
            # 与"物体状态来自感知还是真值"无关；感知模式下 actor 是替身，
            # 因此这里显式对真实物体做接触查询（物体身份由跟踪提供）。
            # 感知模式（actors 已注入）下**不读** `u.fruit_actors`，也不做真实接触查询：
            # 真机上没有 `is_grasping`，只有指端力/触觉（走 build_tactile）+ 几何判据。
            if self.actors is not None:
                # 感知模式：没有 `is_grasping`（那是仿真真值），改用**指端法向力**
                # ——真机上对应 F/T 或触觉，是最直接的可迁移信号。
                _real = None
                grabbing = bool(finger_normal_force(u) > self.contact_force_th)
            else:
                _real = u.fruit_actors.get(kind)
                if _real is not None and not hasattr(_real, "_bodies"):
                    _real = None
                grabbing = bool(u.agent.is_grasping(_real)[0]) if _real is not None else False
            # 近似判据（"离得近 + 上一步闭爪"）只在**查不到真实接触**时才用：
            # 感知模式下估计误差 3–6 cm 时，它会把"在果实上方 10 cm 闭爪"误判成抓住，
            # 之后一路搬运空气、果实从指间滑走（连续流 banana 0/4 的直接原因）。
            use_proxy = {"always": True, "never": False}.get(
                self.grasp_proxy, not (self.actors is not None and _real is not None))
            if self.suction:
                use_proxy = False
            if self.grasp_retry and use_proxy:
                grasping = grabbing or (fd < 0.045 and self.prev_closed)
            else:
                grasping = grabbing
            if self.suction:
                # 吸盘没有指端接触：以"是否吸附"为准（上面的接触/近似判据会把它覆盖掉）
                grasping = bool(self.attached)
            fz = float(actor.pose.p[0, 2])
            if grasping and self.grasp_verify and not self.suction:
                # "碰到上表面"不算抓住：要求指面到达果身高度（|dz|<3 cm）或已被抬起。
                # 注意：感知模式下 object 的位姿可能已被"跟随夹爪"平滑，确认要用 raw_p，
                # 否则判据会自证（snap 之后永远满足）。
                raw = getattr(actor, "raw_p", None)
                fz_chk = float(raw[2]) if raw is not None else fz
                belt_z = float(getattr(u, "belt_top_z", 0.45))
                lifted = fz_chk > belt_z + 0.05
                at_pad = abs(float(tcp[2]) - fz_chk) < 0.030
                # 还要"水平也对上"：感知估计差 3–6 cm 时，仅凭高度会把
                # "在果子旁边闭爪"当成抓住 → 之后一路搬运空气（v1.75 流内 trace）
                near_xy = True
                if raw is not None and self.grasp_verify_xy > 0:
                    near_xy = float(np.hypot(float(tcp[0]) - float(raw[0]),
                                             float(tcp[1]) - float(raw[1]))) < self.grasp_verify_xy
                if not (lifted or (at_pad and near_xy)):
                    # 几何判据不过 → 再问一次腕部相机（只在遮挡期有效）：
                    # 果子落在两指连线中点 wrist_tol 内，就认为"确实夹在两指之间"。
                    ok_wrist = False
                    if self.wrist_checker is not None:
                        try:
                            _d = self.wrist_checker(kind, tcp, obs)
                        except Exception:  # noqa: BLE001
                            _d = None
                        if _d is not None and float(_d) <= self.wrist_tol:
                            ok_wrist = True
                            self.wrist_ok_total += 1
                    if not ok_wrist:
                        grasping = False
            if grasping:
                self.phase_state, self.grasp_lost = 2, 0
            elif self.phase_state == 2:
                self.grasp_lost += 1
                if fz < 0.52 or self.grasp_lost > 20:
                    self.phase_state, self.same, self.phase_cand = 0, 0, 0
        self.grasping = grasping

        sd = obs["sensor_data"]
        _oh_rgb = sd["overhead"]["rgb"][0].cpu().numpy()
        self.oh.append(prep(_oh_rgb, cfg.img_size))          # 全局（相位/上下文）
        if self.roi is not None and actor is not None:
            _xy = actor.pose.p[0].cpu().numpy()[:2]
            self.roi_buf.append(prep(self.roi.crop(_oh_rgb, _xy), cfg.img_size))
        else:
            self.roi_buf.append(self.oh[-1])
        self.wr.append(prep(sd["hand_camera"]["rgb"][0].cpu().numpy(), cfg.img_size))
        sb = build_obs(u, kind, tactile=bool(self.tactile_dim), actors=self.actors)
        if self.use_phase:
            onehot = np.zeros(3, dtype=np.float32)
            onehot[self.phase_state if self.phase_fb else 0] = 1.0
            sb = np.concatenate([sb, onehot])
        # 队列条件（与训练同源）：场上 |x|<0.6 且在带面的水果个数 /4
        if self.use_queue:
            qx = sb[[17 + 8 * k for k in range(4)]]
            qz = sb[[19 + 8 * k for k in range(4)]]
            sb = np.concatenate([sb, np.array(
                [(float(((np.abs(qx) < 0.6) & (qz > 0.40) & (qz < 0.60)).sum()) / 4.0)],
                dtype=np.float32)])
        if self.use_geom:
            j = INSTR_ID.get(kind, 1) - 1
            fp = sb[17 + 8 * j: 17 + 8 * j + 3]
            d = (sb[0:3] - fp).astype(np.float32)
            sb = np.concatenate([sb, d, np.array([np.linalg.norm(d)], dtype=np.float32)])
        self.st.append(sb)
        if self.tactile_token:
            self.tac.append(build_tactile(u, kind))
        for h in (self.oh, self.wr, self.roi_buf, self.st, self.tac) if self.tactile_token else (self.oh, self.wr, self.roi_buf, self.st):
            if len(h) > cfg.hist:
                h.pop(0)
        while len(self.oh) < cfg.hist:
            self.oh.insert(0, self.oh[0])
            self.wr.insert(0, self.wr[0])
            self.roi_buf.insert(0, self.roi_buf[0])
            self.st.insert(0, self.st[0])
            if self.tactile_token:
                self.tac.insert(0, self.tac[0] if self.tac else np.zeros(self.tactile_token, np.float32))

        if self.queue:
            p4 = self.queue.pop(0)
            logits = self.queue_logits
        else:
            tac_t = (torch.from_numpy(np.stack(self.tac)[None]).float().to(self.device)
                     if self.tactile_token else None)
            with torch.no_grad():
                if cfg.n_phase:
                    act, _, _, logits = self.model(
                        torch.from_numpy(np.stack(self.oh)[None]).float().to(self.device),
                        torch.from_numpy(np.stack(self.wr)[None]).float().to(self.device),
                        torch.from_numpy(self.ns.obs(np.stack(self.st).astype(np.float32))[None]).float().to(self.device),
                        torch.tensor([INSTR_ID[kind]], device=self.device),
                        use_prior=True, phase_out=True, tactile=tac_t,
                        imgs_roi=(torch.from_numpy(np.stack(self.roi_buf)[None]).float().to(self.device)
                                  if self.use_roi else None))
                else:
                    act, _, _ = self.model(
                        torch.from_numpy(np.stack(self.oh)[None]).float().to(self.device),
                        torch.from_numpy(np.stack(self.wr)[None]).float().to(self.device),
                        torch.from_numpy(self.ns.obs(np.stack(self.st).astype(np.float32))[None]).float().to(self.device),
                        torch.tensor([INSTR_ID[kind]], device=self.device),
                        use_prior=True, tactile=tac_t,
                        imgs_roi=(torch.from_numpy(np.stack(self.roi_buf)[None]).float().to(self.device)
                                  if self.use_roi else None))
                    logits = None
            acts_np = act[0].cpu().numpy()
            if self.chunk_exec > 1 and self.delta_actions:
                self.queue = [a for a in acts_np[: self.chunk_exec]]
                self.queue_logits = logits
            p4 = acts_np[0]

        # 滑移预测（辅助头）：sigmoid(logit)
        self.slip_prob = None
        if getattr(self.model, "last_slip", None) is not None:
            self.slip_prob = float(torch.sigmoid(self.model.last_slip).mean().item())
            self.stats["slip_prob_max"] = max(self.stats["slip_prob_max"], self.slip_prob)

        tgt = self.na.unact(p4[:3])
        if self.delta_actions:
            tgt = tcp + np.clip(tgt, -self.max_step, self.max_step)
        else:
            tgt = np.clip(tgt, -2, 2)

        # 残差反馈（IBVS 式）：策略给前馈，反馈项收残余误差
        if self.fb and kind in self._objs and not grasping:
            fpos = self._objs[kind].pose.p[0].cpu().numpy().astype(np.float64)
            corr = np.clip(fpos - tcp, -0.15, 0.15)
            if not self.fb_z:
                corr[2] = 0.0
            tgt = tgt + self.fb * corr
        if np.any(self.retry_bias) and self.phase_state != 2:
            tgt = tgt + self.retry_bias
        if self.lead and kind in self._objs:
            tgt[0] += self.lead * float(u.belt_speed[0])
        if (self.lead_long > 0 and actor is not None and self.phase_state != 2
                and float(getattr(actor, "extent", 0.0) or 0.0) > self.lead_long_extent):
            # 长条目标：瞄准点沿带速前移（补偿反馈滞后），靠近后衰减到 0 保住抓取点
            fp = actor.pose.p[0].cpu().numpy().astype(np.float64)
            dist = float(np.hypot(tcp[0] - fp[0], tcp[1] - fp[1]))
            w = 1.0 if self.lead_long_taper <= 0 else min(1.0, dist / self.lead_long_taper)
            tgt[0] += w * self.lead_long * float(u.belt_speed[0])
            self.stats["lead_long"] = self.stats.get("lead_long", 0) + 1
        if self.z_gate and kind in self._objs:
            fz = self._objs[kind].pose.p[0, 2].item()
            tgt[2] = float(np.clip(tgt[2], fz - 0.02, 0.90))

        glogit = float(p4[3])
        if self.grip_ema > 0:
            a = self.grip_ema
            self.grip_ema_val = glogit if self.grip_ema_val is None else a * self.grip_ema_val + (1 - a) * glogit
            glogit = self.grip_ema_val
        if self.grip_hyst > 0:
            # 双阈值：|logit| < h 时维持上一步，避免每几帧翻一次 → 指面抖动夹不住
            if glogit > self.grip_hyst:
                closed = True
            elif glogit < -self.grip_hyst:
                closed = False
            else:
                closed = bool(self.prev_closed)
            self.stats["grip_hold"] = self.stats.get("grip_hold", 0) + int(abs(glogit) < self.grip_hyst)
        else:
            closed = bool(glogit > 0)
        grip = self._close_cmd(kind) if closed else 1.0
        # ---- 力控闭合（--close-force-hold）：Fn 到位就冻结指令并锁存 ----
        if self.close_force_hold > 0 and closed and not grasping:
            if self._fhold_grip is None:
                _fn = finger_normal_force(u)
                if _fn >= self.close_force_hold:
                    self._fhold_grip = float(grip)
                    self.stats["force_latch"] = self.stats.get("force_latch", 0) + 1
            if self._fhold_grip is not None:
                grip = self._fhold_grip
        elif not closed:
            self._fhold_grip = None          # 张开就解锁

        # ---- 力作为动作通道：第 5 维 = 目标夹持力（0..1 → 0..40N），用触觉 Fn 伺服 ----
        self.force_target = None
        self.tac_hist = []
        self.gate_timer = 0
        self.gate_retries_done = 0
        self.retry_bias = np.zeros(3)
        self.gate_failed_once = False
        if self.use_force and p4.shape[-1] > 4:
            self.force_target = float(np.clip(p4[4], 0.0, 1.0)) * 40.0
            self.stats["force_target"] = self.force_target
            if glogit > 0:                       # 只在"想闭爪"时做力伺服
                fn = float(build_tactile(u, kind)[0])
                self.stats["fn_peak"] = max(self.stats.get("fn_peak", 0.0), fn)
                closed = True
                if fn < self.force_target - 3.0:
                    grip = -1.0                  # 力不够 → 继续闭紧
                elif fn > self.force_target + 3.0:
                    grip = -0.6                  # 力超了 → 略松（不彻底张开，避免掉果）
                else:
                    grip = -0.8                  # 命中目标力 → 维持
        # ---- 单侧接触微伺服：闭爪时只有一指受力说明水果偏了 → 朝弱侧平移 ----
        if self.side_servo and self.phase_state == 1 and closed and actor is not None:
            try:
                L = u.agent.robot.links_map["panda_leftfinger"]
                R = u.agent.robot.links_map["panda_rightfinger"]
                FL = L.get_net_contact_forces()[0].cpu().numpy()
                FR = R.get_net_contact_forces()[0].cpu().numpy()
                pL = L.pose.p[0].cpu().numpy(); pR = R.pose.p[0].cpu().numpy()
                axis = pL - pR; axis = axis / max(np.linalg.norm(axis), 1e-6)
                fL = abs(float(np.dot(FL, axis))); fR = abs(float(np.dot(FR, -axis)))
                if min(fL, fR) < 0.5 and max(fL, fR) > 2.0:
                    side = 1.0 if fL < fR else -1.0     # 朝没碰到的一侧挪
                    tgt = tgt + axis * self.side_step * side
                    self.stats["side_push"] += 1
            except Exception:  # noqa: BLE001
                pass

        # ---- 闭合到力阈（力→位置）：超过阈值就停止继续闭合，避免小果被挤飞 ----
        if self.close_force_th > 0 and glogit > 0 and not grasping:
            try:
                fn_now = float(build_tactile(u, kind)[0])
            except Exception:  # noqa: BLE001
                fn_now = 0.0
            if fn_now > self.close_force_th:
                grip = -0.3
        if self.carry_latch and grasping:
            self.open_streak = self.open_streak + 1 if glogit < 0 else 0
            if self.open_streak < 3:
                grip = -1.0
        else:
            self.open_streak = 0
        # ---- 放料门：只在框上方开爪（超时后放弃守卫，避免抱着果子跑整局）----
        if self.release_gate > 0 and grasping and not closed:
            try:
                from fruitsort.sim.conveyor import BIN_LAYOUT as _BLr
                _bxr, _byr = _BLr[kind]
            except Exception:  # noqa: BLE001
                _bxr, _byr = 0.0, -0.3
            # 判据用**果实**的位置而不是 TCP：搬运中果实会比夹爪滞后 10–15 cm（滑移），
            # 实测 seed 60015 的掉落就是"TCP 距框 9 cm 放爪、果实还在 21 cm 外" → 掉在框后面。
            _px, _py = float(tcp[0]), float(tcp[1])
            if actor is not None:
                try:
                    _fp = actor.pose.p[0].cpu().numpy()
                    _px, _py = float(_fp[0]), float(_fp[1])
                except Exception:  # noqa: BLE001
                    pass
            _db = float(np.hypot(_px - _bxr, _py - _byr))
            if _db > self.release_gate and self._rel_wait < self.release_wait:
                grip = self._close_cmd(kind)
                closed = True
                tgt = np.array([float(_bxr), float(_byr),
                                max(float(tcp[2]), 0.52)], dtype=np.float64)
                self._rel_wait += 1
                self.stats["rel_gate"] = self.stats.get("rel_gate", 0) + 1
            else:
                self._rel_wait = 0
        # 自适应限速（迟滞）：上一帧判定滑移后，再保持 slip_hold 步慢速，然后恢复原速。
        # 没有迟滞时，rho 在阈值附近抖动会让搬运"一步快一步慢"，反而把果子晃松。
        if self._slip_left > 0:
            self._slip_left -= 1
            self.slip_slow = True
        else:
            self.slip_slow = False
        self.force_target = None
        self.tac_hist = []
        self.gate_timer = 0
        self.gate_retries_done = 0
        self.retry_bias = np.zeros(3)
        self.gate_failed_once = False
        self.regrasp_done = False
        self.slip_prob = None
        self.cur_grip = 1.0
        # ---- 渐进闭爪（闭合速度限幅）----
        # 注意：给了 side_grasp_kinds 时，ramp 只作用于这些类（否则会把苹果的闭爪也拖慢，
        # 实测全局 ramp 会把 apple 从 50% 打到 0%）
        if self.close_ramp > 0 and (not self.side_grasp_kinds
                                    or kind in self.side_grasp_kinds):
            if glogit > 0:                                  # 想闭爪：单向渐进，不回头
                self.cur_grip = max(-1.0, self.cur_grip - self.close_ramp)
            elif self.phase_state == 0:                     # 回到接近段才重置为张开
                self.cur_grip = 1.0
            grip = self.cur_grip
        # ---- 按类抓取高度偏置（下探/闭爪段）----
        if self.grasp_z_bias and kind in self.grasp_z_bias and self.phase_state in (1, 2):
            tgt[2] = tgt[2] + float(self.grasp_z_bias[kind])
        # ---- 宽度设定点（LeTac 式位置等价力控）：闭合到 w*(类)−δ，而不是合到底 ----
        # 实测空载映射 gap ≈ 0.030 + 0.050·g，反解 g = (w−0.030)/0.050
        # 宽度设定点只在"**想闭爪**"时生效。
        # 原来的条件是 `glogit>0 or phase==2`——搬运段无条件套用，于是**放料时也被覆盖**，
        # 果子永远放不下来（实测 --width-setpoint 0/4，而全闭合 3/4）。
        if self.width_map and kind in self.width_map and closed:
            if self.width_g is not None:
                grip = float(np.clip(self.width_g, -1.0, 1.0))
            else:
                w_target = self.width_map[kind] - self.width_delta
                grip = float(np.clip((w_target - 0.030) / 0.050, -1.0, 0.7))
            self.stats["grip_w"] = grip
        self.stats["close"] += int(closed)
        self.stats["descend"] += int(self.phase_state == 1)
        self.stats["carry"] += int(self.phase_state == 2)
        self.stats["steps"] += 1

        # 相位反馈：模型自己的预测（k=0），未真正抓住时不允许自认搬运
        if logits is not None and self.phase_state != 2:
            p_cls = int(logits[0, 0].argmax())
            if p_cls == 2:
                p_cls = 1
            if p_cls == 1:
                self.phase_cand, self.same = 1, 1
                self.commit = self.descend_commit
                self.phase_state = 1
            elif self.commit > 0:
                self.commit -= 1
            else:
                if p_cls == self.phase_cand:
                    self.same += 1
                else:
                    self.phase_cand, self.same = p_cls, 0
                if self.same >= 1:
                    self.phase_state = self.phase_cand

        # ---- 独立掉落预测器（触觉历史 5 帧 + 几何）----
        if self.drop_net is not None and actor is not None:
            fpos_now = actor.pose.p[0].cpu().numpy().astype(np.float64)
            d_obj = float(np.linalg.norm(fpos_now - tcp))
            held_now = float(d_obj < 0.08)
            self.tac_hist.append(build_tactile(u, kind))
            while len(self.tac_hist) < self.drop_hist:
                self.tac_hist.insert(0, self.tac_hist[0])
            while len(self.tac_hist) > self.drop_hist:
                self.tac_hist.pop(0)
            feat = np.concatenate([np.stack(self.tac_hist).reshape(-1),
                                   np.array([d_obj, held_now, float(u.belt_speed[0])],
                                            dtype=np.float32)]).astype(np.float32)
            with torch.no_grad():
                z = torch.from_numpy(((feat - self.drop_mu) / self.drop_sd)[None])
                self.drop_prob = float(torch.sigmoid(self.drop_net(z)).item())
            self.stats["drop_prob_max"] = max(self.stats.get("drop_prob_max", 0.0), self.drop_prob)

        # ---- 抓取验证门（论文①）：进入搬运段 gate_delay 步后判"是否真的抓住" ----
        if self.grasp_gate and self.phase_state == 2 and actor is not None:
            self.gate_timer += 1
            if self.gate_timer == self.gate_delay:
                try:
                    L = u.agent.robot.links_map["panda_leftfinger"]
                    R = u.agent.robot.links_map["panda_rightfinger"]
                    FL = L.get_net_contact_forces()[0].cpu().numpy()
                    FR = R.get_net_contact_forces()[0].cpu().numpy()
                    pL = L.pose.p[0].cpu().numpy(); pR = R.pose.p[0].cpu().numpy()
                    axis = pL - pR; axis = axis / max(np.linalg.norm(axis), 1e-6)
                    fL = abs(float(np.dot(FL, axis))); fR = abs(float(np.dot(FR, -axis)))
                    sym = abs(fL - fR) / max(fL + fR, 1e-6)
                    tac_now = build_tactile(u, kind)
                    v_rel = float(tac_now[5])
                    self.stats["fL"], self.stats["fR"] = fL, fR
                    ok = (min(fL, fR) > self.gate_min_force and sym < self.gate_asym
                          and v_rel < self.gate_slip)
                    if not ok:
                        self.stats["gate_fail"] += 1
                        if self.gate_retries_done < self.gate_retries:
                            # 沿"偏弱的一侧"补一个横向偏移，下次下探用
                            side = 1.0 if fL < fR else -1.0
                            self.retry_bias = axis * self.gate_bias * side
                            self.gate_retries_done += 1
                            self.stats["retries"] += 1
                            # 复位到接近段，交给策略重新下探
                            self.phase_state, self.same, self.phase_cand = 0, 0, 0
                            self.gate_timer = 0
                            self.regrasp_done = False
                except Exception:  # noqa: BLE001
                    pass

        # ---- 触觉预测驱动的闭爪前重抓 ----
        if (self.regrasp and not grasping and self.phase_state == 1 and closed
                and not self.regrasp_done
                and max(self.slip_prob or 0.0, self.drop_prob) > self.regrasp_thr):
            grip = 1.0                      # 先张开
            tgt[2] = tgt[2] - self.regrasp_depth   # 再下沉 2cm
            self.regrasp_done = True
            self.stats["regrasp"] += 1
            self.phase_state, self.same, self.phase_cand, self.commit = 1, 1, 1, 12

        # ---- 拦截式下探：预测水果在下探耗时后的位置，一次下到那里并在到位瞬间闭爪 ----
        if (self.intercept > 0 and not grasping and self.phase_state != 2
                and actor is not None):
            fpos_i = actor.pose.p[0].cpu().numpy().astype(np.float64)
            fv_i = actor.linear_velocity[0].cpu().numpy().astype(np.float64)
            dz_i = float(tcp[2] - fpos_i[2])
            if self._descend_target is None and closed and dz_i > 0.05:
                self._descend_target = True
                self._descend_steps = 0
                self.stats["intercept"] = self.stats.get("intercept", 0) + 1
            if self._descend_target is not None:
                self._descend_steps += 1
                t_desc = max(dz_i, 0.0) / self.intercept
                ix = float(fpos_i[0] + fv_i[0] * t_desc)
                tgt = np.array([ix, float(fpos_i[1]), float(fpos_i[2]) + 0.01])
                aligned = (abs(float(tcp[0] - fpos_i[0])) < self.close_win_xy
                           and float(np.linalg.norm(tcp - fpos_i)) < self.close_win_z)
                grip = -1.0 if aligned else 1.0
                self.phase_state = 1
                if self._descend_steps > 45 or dz_i < 0.012:
                    self._descend_target = None

        # ---- 闭爪高度门：禁止在空中闭爪，改为继续下探 ----
        _czg = float(self.close_z_gate_map.get(kind, self.close_z_gate))
        if (_czg > 0 and closed and not grasping
                and self.phase_state != 2 and actor is not None):
            fpos_g = actor.pose.p[0].cpu().numpy().astype(np.float64)
            if float(tcp[2] - fpos_g[2]) > _czg:
                grip = 1.0
                closed = False
                _off = float(self.close_z_off_map.get(kind, 0.02))
                tgt = np.array([fpos_g[0], fpos_g[1], fpos_g[2] + _off])
                self.phase_state, self.same, self.phase_cand, self.commit = 1, 1, 1, 8
                self.stats["zgate"] = self.stats.get("zgate", 0) + 1
                if self.zg_hold_chunks > 0:
                    self._zg_left = self.zg_hold_chunks
                    self._zg_z = float(fpos_g[2]) + 0.004
        # 抓空复位（对应专家 close_steps=12）
        if closed and not grasping:
            self.closed_steps += 1
            if self.closed_steps > 12:
                grip = 1.0
                closed = False
                self.closed_steps = 0
                if self.phase_state == 1:
                    self.phase_state, self.same, self.phase_cand, self.commit = 0, 0, 0, 0
        else:
            self.closed_steps = 0
        if self.hold_latch and closed and not grasping:
            tgt = tcp.copy()

        # ---- 触觉补救：滑移检测 → 搬运段降速 ----
        # 判据用**摩擦利用率** rho = |Ft|/Fn（tac[2]，经典摩擦锥判据），而不是相对速度：
        # 感知模式下 `tac[5]`（v_slip）是用"物体估计速度 − TCP 速度"算的，而物体估计
        # 在夹住后被钉在 TCP 上（速度为 0）→ v_slip ≡ TCP 速度，永远"在滑"，
        # 等于把搬运段无条件下限速（那不是滑移检测）。rho 只依赖力传感器，真机同样可用。
        if self.slip_guard and actor is not None:
            tac = build_tactile(u, kind)
            slipping = float(tac[0]) > 3.0 and float(tac[2]) > self.slip_thr
            if slipping:
                self.stats["slip"] += 1
                if self.stats["slip_first"] < 0:
                    self.stats["slip_first"] = self.stats["steps"]
                if self.phase_state == 2:
                    self.slip_slow = True
                    self._slip_left = self.slip_hold

        # ---- 停滞守卫：连续多步"实际位移≈0"且水果仍在接近窗口 → 朝水果迈最小步 ----
        # 现象（docs 6.26.35）：模型接近增量为 0、残差反馈被 ±0.15 饱和成常数 →
        # 两者抵消，TCP 逐帧完全不变，水果自己走掉。
        # 注意：判据用**实际 TCP 位移**而不是指令位移——指令可能非零但被抵消/被 PD 吃掉。
        if (self.stall_guard > 0 and not grasping and self.phase_state != 2
                and grip > 0 and actor is not None):
            fpos_s = actor.pose.p[0].cpu().numpy().astype(np.float64)
            d_xy = fpos_s[:2] - tcp[:2]
            dist = float(np.linalg.norm(d_xy))
            in_window = (-0.70 <= float(fpos_s[0]) <= 0.25) and 0.40 < float(fpos_s[2]) < 0.60
            moved = float(np.linalg.norm(tcp[:2] - self._stall_tcp[:2]))
            if in_window and dist > 0.05 and moved < self.stall_eps:
                self._stall_run += 1
            else:
                self._stall_run = 0
            if self._stall_run >= 3:
                step = min(self.stall_guard, dist)
                tgt = tcp.copy()
                tgt[:2] += d_xy / max(dist, 1e-6) * step
                # 同时抬一点：冻结常发生在"臂已到伸展极限"（水果偏 y、x 又远），
                # 纯水平指令不可达；先回收一点高度才能继续前进
                tgt[2] = tcp[2] + 0.025
                self.stats["stall_fix"] = self.stats.get("stall_fix", 0) + 1
                self._stall_run = 0
        self._stall_tcp = np.array(tcp, dtype=np.float64, copy=True)

        # ---- 侧向进入（只对 side_grasp_kinds）----
        if (self.side_grasp_kinds and kind in self.side_grasp_kinds
                and not grasping and self.phase_state in (0, 1) and actor is not None):
            fp_s = actor.pose.p[0].cpu().numpy().astype(np.float64)
            wp_x = fp_s[0] + self.side_offset
            if float(tcp[0]) > wp_x + 0.02:
                # 1) 先到"下游 + 果心高度"的预备位（张爪）
                tgt = np.array([wp_x, fp_s[1], fp_s[2] + 0.010], dtype=np.float64)
                grip = 1.0
                self.stats["side_grasp"] = self.stats.get("side_grasp", 0) + 1
            elif float(np.hypot(tcp[0] - fp_s[0], tcp[1] - fp_s[1])) > 0.022:
                # 2) 沿 -x 滑入两指之间（张爪，保持果心高度）
                tgt = np.array([fp_s[0], fp_s[1], fp_s[2] + 0.005], dtype=np.float64)
                grip = 1.0
                self.stats["side_grasp"] = self.stats.get("side_grasp", 0) + 1
            else:
                # 3) 就位 → 闭爪。小果不能合到底（开口 3 cm < 草莓 4 cm，会挤出去）：
                #    用"位置等价力控"——闭到按类的宽度设定点，再按 close_ramp 渐进
                _w = self.width_map.get(kind) if self.width_map else None
                if _w is not None:
                    _wt = float(_w) - self.width_delta
                    _g = float(np.clip((_wt - 0.030) / 0.050, -1.0, 0.7))
                elif self.width_g is not None:
                    _g = float(np.clip(self.width_g, -1.0, 0.7))
                else:
                    _g = -1.0
                if self.close_ramp > 0:
                    self._side_grip = max(_g, self._side_grip - self.close_ramp)
                else:
                    self._side_grip = _g
                grip = self._side_grip
                self.stats["side_close"] = self.stats.get("side_close", 0) + 1

        if (self.side_carry == "scripted" and kind in self.side_grasp_kinds and grasping):
            try:
                from fruitsort.sim.conveyor import BIN_LAYOUT as _BL2
                _bx2, _by2 = _BL2[kind]
            except Exception:  # noqa: BLE001
                _bx2, _by2 = 0.0, -0.3
            _over = (abs(float(tcp[0]) - _bx2) < 0.06 and abs(float(tcp[1]) - _by2) < 0.06)
            if _over and float(tcp[2]) < 0.58:
                grip = 1.0                    # 到框上方 → 开爪放下
                tgt = np.array([float(_bx2), float(_by2), 0.50], dtype=np.float64)
            else:
                grip = -1.0
                tgt = np.array([float(_bx2), float(_by2), 0.60], dtype=np.float64)
            self.stats["side_carry"] = self.stats.get("side_carry", 0) + 1

        if self.suction and self.suction_carry == "scripted" and self.attached:
            # 吸住之后用最简单的"送到框正上方"：不再依赖策略的搬运段
            try:
                from fruitsort.sim.conveyor import BIN_LAYOUT
                bx, by = BIN_LAYOUT[kind]
            except Exception:  # noqa: BLE001
                bx, by = 0.0, -0.3
            tgt = np.array([float(bx), float(by), 0.55], dtype=np.float64)
            grip = -1.0
            self.stats["suction_carry"] = self.stats.get("suction_carry", 0) + 1

        # ---- 搬运段保相位辅助：只改方向，不改相位/速度 ----
        if self.carry_assist and self.phase_state == 2 and grasping and actor is not None:
            try:
                from fruitsort.sim.conveyor import BIN_LAYOUT as _BLa
                _bx, _by = _BLa[kind]
            except Exception:  # noqa: BLE001
                _bx, _by = 0.0, -0.30
            _dv = np.array([float(_bx) - float(tcp[0]), float(_by) - float(tcp[1])])
            _dist = float(np.linalg.norm(_dv))
            if _dist > 0.04:
                _dir = _dv / _dist
                _step = float(np.linalg.norm((tgt - tcp)[:2]))
                _step = max(_step, self.carry_assist_min)
                _step = min(_step, max(self.carry_assist_min, 0.04))
                tgt = np.array([float(tcp[0]) + _dir[0] * _step,
                                float(tcp[1]) + _dir[1] * _step, float(tgt[2])])
                self.stats["carry_assist"] = self.stats.get("carry_assist", 0) + 1

        err = tgt - tcp
        d = float(np.linalg.norm(err))
        step = float(self.max_step_map.get(kind, self.max_step))
        if self.phase_state == 2:
            _c = float(self.carry_step_map.get(kind, self.carry_max_step))
            if _c > 0:
                step = _c
        if getattr(self, "slip_slow", False) and self.phase_state == 2:
            step = min(step, self.slip_slow_step)
        if d > step:
            tgt = tcp + err / d * step
        if actor is not None:
            fpos = actor.pose.p[0].cpu().numpy().astype(np.float64)
            if not grasping:
                tgt[2] = max(tgt[2], float(fpos[2]) - 0.02)
                if float(tcp[0] - fpos[0]) < -0.08 and self.phase_state == 1:
                    self.phase_state, self.same, self.phase_cand = 0, 0, 0

        # ---- 闭爪承诺（见上面构造参数处的动机）----
        _cc = float(self.close_commit_map.get(kind, self.close_commit))
        if _cc > 0 and not grasping and actor is not None:
            fpos_c = actor.pose.p[0].cpu().numpy().astype(np.float64)
            dz_c = float(tcp[2] - fpos_c[2])
            dxy_c = float(np.hypot(tcp[0] - fpos_c[0], tcp[1] - fpos_c[1]))
            if self.phase_state == 1 and -0.005 <= dz_c <= _cc \
                    and dxy_c <= self.close_commit_xy:
                self._commit_left = self.close_commit_steps
                self.stats["close_commit"] = self.stats.get("close_commit", 0) + 1
            if self._commit_left > 0:
                # 冻结水平（不再追会"往前滑"的估计），只往下压到果心附近
                tgt[0], tgt[1] = float(tcp[0]), float(tcp[1])
                tgt[2] = (float(tcp[2]) - 0.006
                          if float(tcp[2]) > float(fpos_c[2]) - 0.005 else float(tcp[2]))
                grip = -1.0
                self._commit_left -= 1
        # 诊断：phase 1 入口高度差（每个 chunk 采样一次即可，入口值不变）
        if self.phase_state == 1 and self._prev_phase != 1 and actor is not None:
            try:
                _fz0 = float(actor.pose.p[0, 2].cpu().numpy())
                self.p1_dz_in.append(float(tcp[2]) - _fz0)
            except Exception:  # noqa: BLE001
                pass
        self._prev_phase = self.phase_state

        # ---- 高度门钉子：打断"闭爪被压回 → 下一 chunk 又抬起来"的极限环 ----
        if self._zg_left > 0 and self._zg_z is not None:
            if self.phase_state == 2 or grasping or closed:
                self._zg_left = 0                     # 已进入下一段/已闭爪 → 解除
            else:
                tgt[2] = min(float(tgt[2]), float(self._zg_z))
                self._zg_left -= 1
                self.stats["zg_hold"] = self.stats.get("zg_hold", 0) + 1

        # 诊断用（不影响控制）：本步模型输出/目标点/状态
        self.last_dbg = dict(p4=np.array(p4, dtype=np.float64), tgt=np.array(tgt, dtype=np.float64),
                             tcp=np.array(tcp, dtype=np.float64), phase=int(self.phase_state),
                             grasp=bool(grasping), grip=float(grip),
                             logits=(None if logits is None else logits[0].detach().cpu().numpy().copy()),
                             sb=np.array(sb, dtype=np.float64))
        self.prev_grip = grip
        self.prev_closed = closed
        self.closed = closed
        yaw = float(self.yaw_map.get(kind, 0.0))
        _ya_ext = float(getattr(actor, "extent", 0.0) or 0.0) if actor is not None else 0.0
        if (self.yaw_align and actor is not None
                and _ya_ext >= self.yaw_align_extent):
            # ---- 闭环对准（每 period 步重算 mod-π 指面轴误差）----
            _fast = (self.yaw_align_fast
                     and self.yaw_err_deg > float(np.degrees(self.yaw_align_ready)))
            _period = 1 if _fast else self.yaw_align_period
            _gain = 0.9 if _fast else self.yaw_align_gain
            self._yaw_t += 1
            _recalc = (self._yaw_target is None or (self._yaw_t % _period == 0))
            if not grasping and self.phase_state in (0, 1) and _recalc:
                try:
                    yy = getattr(actor, "yaw", None)
                    if yy is not None:
                        phi = float(yy)
                    else:
                        q = actor.pose.q[0].cpu().numpy()
                        w, xq, yq, zq = float(q[0]), float(q[1]), float(q[2]), float(q[3])
                        phi = float(np.arctan2(2.0 * (w * zq + xq * yq),
                                               1.0 - 2.0 * (yq * yq + zq * zq)))
                    pL = u.agent.robot.links_map["panda_leftfinger"].pose.p[0].cpu().numpy()
                    pR = u.agent.robot.links_map["panda_rightfinger"].pose.p[0].cpu().numpy()
                    d = pL - pR
                    psi = float(np.arctan2(d[1], d[0]))
                    want = phi + np.pi / 2.0 + self.yaw_align_off
                    # 轴向角（mod π）：ψ/want 只定义到 π，用 mod 2π 会朝反方向转（v1.64 实测）
                    d = psi - want
                    d = float(np.arctan2(np.sin(2.0 * d), np.cos(2.0 * d)) / 2.0)
                    self.yaw_err_deg = float(np.degrees(abs(d)))
                    if abs(d) > self.yaw_align_tol:
                        delta = _gain * d
                        self._yaw_target = float(np.clip(self._yaw_cmd + delta, -1.2, 1.2))
                except Exception:  # noqa: BLE001
                    self._yaw_target = 0.0
            # 抓住之后保持这个朝向（v1.64：搬运瞬间回正会把果子甩掉）
            # 注意：刚切换目标时 _yaw_target 可能是 None（该步在搬运/下探中没重算）
            self._yaw_cmd = 0.0 if self._yaw_target is None else float(self._yaw_target)
            yaw += self._yaw_cmd
            self.stats["yaw_align_deg"] = float(np.degrees(self._yaw_cmd))
            # ---- 两段式：先原地对准再进场 ----
            if (self.yaw_align_prerotate and not grasping and self.phase_state == 0
                    and self._pre_steps < self.yaw_align_pre_max):
                dist = float(np.hypot(tcp[0] - actor.pose.p[0, 0],
                                      tcp[1] - actor.pose.p[0, 1]))
                if (dist > self.yaw_align_hold_dist
                        and self.yaw_err_deg > np.degrees(self.yaw_align_ready)):
                    if self.yaw_align_rot_frac > 0:
                        tgt = np.array(tcp + (tgt - tcp) * self.yaw_align_rot_frac,
                                       dtype=np.float64)
                    else:
                        tgt = np.array(tcp, dtype=np.float64)  # 只下发姿态
                    grip = 1.0                                 # 对准期间保持张开
                    self._pre_steps += 1
                    self.stats["pre_rot"] = self.stats.get("pre_rot", 0) + 1
        elif (self.yaw_align and self.yaw_align_hold
              and abs(self._yaw_cmd or 0.0) > 1e-6):
            # 目标不在对准范围（苹果/草莓/梨）：**保持**已转好的腕部偏航
            # （否则空档期攒的预转会被清零，香蕉轮到时就白攒了）；
            # 但临近闭合时线性淡出到 HOME，保证最后一个下探/闭爪仍在训练主分布上。
            w = 1.0
            if actor is not None and self.yaw_hold_fade > 0:
                dist = float(np.hypot(tcp[0] - actor.pose.p[0, 0],
                                      tcp[1] - actor.pose.p[0, 1]))
                w = min(1.0, dist / self.yaw_hold_fade)
            hold = float(self._yaw_cmd or 0.0) * self.yaw_hold_frac
            if self.yaw_hold_cap > 0:
                hold = float(np.clip(hold, -self.yaw_hold_cap, self.yaw_hold_cap))
            yaw += hold * w
            self.stats["yaw_hold_deg"] = float(np.degrees(hold * w))
        return np.array([tgt[1] + 0.5, -tgt[0], tgt[2],
                         HOME_EULER[0], HOME_EULER[1], HOME_EULER[2] + yaw, grip],
                        dtype=np.float32)
