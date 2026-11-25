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


class ActPureRunner:
    def __init__(self, env, ckpt_path, kind: str, device: str = "cpu", *,
                 fb: float = 0.0, fb_z: bool = False, grip_ema: float = 0.0,
                 carry_latch: bool = False, descend_commit: int = 0,
                 max_step: float = 0.06, lead: float = 0.0, z_gate: bool = False,
                 phase_fb: bool = True, hold_latch: bool = False,
                 grasp_retry: bool = True, chunk_exec: int = 1,
                 carry_max_step: float = 0.0, slip_guard: bool = False,
                 slip_thr: float = 0.05, slip_slow_step: float = 0.012,
                 regrasp: bool = False, regrasp_thr: float = 0.6,
                 regrasp_depth: float = 0.02, drop_ckpt: str = "",
                 grasp_gate: bool = False, gate_min_force: float = 2.0,
                 gate_asym: float = 0.5, gate_slip: float = 0.02,
                 gate_delay: int = 5, gate_retries: int = 2,
                 gate_bias: float = 0.005, close_force_th: float = 0.0,
                 side_servo: bool = False, side_step: float = 0.002,
                 width_map: dict | None = None, width_delta: float = 0.002,
                 width_g: float | None = None, close_ramp: float = 0.0,
                 grasp_z_bias: dict | None = None, stall_guard: float = 0.0,
                 stall_eps: float = 0.006, yaw_map: dict | None = None,
                 close_z_gate: float = 0.0, intercept: float = 0.0,
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
                 grasp_verify: bool = False, yaw_align: bool = False,
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
        self.chunk_exec = chunk_exec
        # 搬运段单独限速：接近段要快要准，搬运段过冲会把水果带到框外
        self.carry_max_step = carry_max_step
        # 触觉补救：搬运中检测到滑移（物体相对 TCP 有速度且有夹持力）→ 立即降速搬运
        self.slip_guard = slip_guard
        self.slip_thr = slip_thr
        self.slip_slow_step = slip_slow_step
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
        # 搬运段单独限速：接近段要快要准，搬运段过冲会把水果带到框外
        self.carry_max_step = carry_max_step
        # 触觉补救：搬运中检测到滑移（物体相对 TCP 有速度且有夹持力）→ 立即降速搬运
        self.slip_guard = slip_guard
        self.slip_thr = slip_thr
        self.slip_slow_step = slip_slow_step
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
        self.close_win_xy = float(close_win_xy)
        self.close_win_z = float(close_win_z)
        # 抓取确认：只"碰到顶端"不算抓住（队列里会误判成 phase 2、随后水果滑走）
        self.grasp_verify = bool(grasp_verify)
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

    def _grasp_proxy(self, actor, tcp) -> bool:
        """感知模式下的"抓住"判据（无成对接触力可用）：闭合 + 水平对齐 + 指面到果身高度。"""
        try:
            p = actor.pose.p[0].cpu().numpy().astype(np.float64)
            belt = float(getattr(self.u, "belt_top_z", 0.45))
            near_xy = float(np.linalg.norm(p[:2] - tcp[:2])) < 0.05
            at_pad = abs(float(tcp[2] - p[2])) < 0.05
            lifted = float(p[2]) > belt + 0.02
            return bool(self.prev_grip < 0 and near_xy and (at_pad or lifted))
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------ #
    def reset(self):
        self.oh, self.wr, self.st, self.tac = [], [], [], []
        self.phase_state = 0
        self.phase_cand = 0
        self.same = 0
        self.commit = 0
        self.grasp_lost = 0
        self.closed_steps = 0
        self.grip_ema_val = None
        self.prev_grip = 1.0
        self.open_streak = 0
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
                        and (fd < 0.05 or (self.prev_grip < 0 and fd < self.suction_radius))):
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
            _real = u.fruit_actors.get(kind)
            if _real is not None and not hasattr(_real, "_bodies"):
                _real = None            # 该槽位被替身占用（感知模式）：跳过接触查询
            grabbing = bool(u.agent.is_grasping(_real)[0]) if _real is not None else False
            # 近似判据（"离得近 + 上一步闭爪"）只在**查不到真实接触**时才用：
            # 感知模式下估计误差 3–6 cm 时，它会把"在果实上方 10 cm 闭爪"误判成抓住，
            # 之后一路搬运空气、果实从指间滑走（连续流 banana 0/4 的直接原因）。
            use_proxy = {"always": True, "never": False}.get(
                self.grasp_proxy, not (self.actors is not None and _real is not None))
            if self.suction:
                use_proxy = False
            if self.grasp_retry and use_proxy:
                grasping = grabbing or (fd < 0.045 and self.prev_grip < 0)
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
                if raw is not None:
                    near_xy = float(np.hypot(float(tcp[0]) - float(raw[0]),
                                             float(tcp[1]) - float(raw[1]))) < 0.060
                if not (lifted or (at_pad and near_xy)):
                    grasping = False
            if grasping:
                self.phase_state, self.grasp_lost = 2, 0
            elif self.phase_state == 2:
                self.grasp_lost += 1
                if fz < 0.52 or self.grasp_lost > 20:
                    self.phase_state, self.same, self.phase_cand = 0, 0, 0
        self.grasping = grasping

        sd = obs["sensor_data"]
        self.oh.append(prep(sd["overhead"]["rgb"][0].cpu().numpy(), cfg.img_size))
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
        for h in (self.oh, self.wr, self.st, self.tac) if self.tactile_token else (self.oh, self.wr, self.st):
            if len(h) > cfg.hist:
                h.pop(0)
        while len(self.oh) < cfg.hist:
            self.oh.insert(0, self.oh[0])
            self.wr.insert(0, self.wr[0])
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
                        use_prior=True, phase_out=True, tactile=tac_t)
                else:
                    act, _, _ = self.model(
                        torch.from_numpy(np.stack(self.oh)[None]).float().to(self.device),
                        torch.from_numpy(np.stack(self.wr)[None]).float().to(self.device),
                        torch.from_numpy(self.ns.obs(np.stack(self.st).astype(np.float32))[None]).float().to(self.device),
                        torch.tensor([INSTR_ID[kind]], device=self.device),
                        use_prior=True, tactile=tac_t)
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
        grip = -1.0 if glogit > 0 else 1.0
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
                if fn < self.force_target - 3.0:
                    grip = -1.0                  # 力不够 → 继续闭紧
                elif fn > self.force_target + 3.0:
                    grip = -0.6                  # 力超了 → 略松（不彻底张开，避免掉果）
                else:
                    grip = -0.8                  # 命中目标力 → 维持
        # ---- 单侧接触微伺服：闭爪时只有一指受力说明水果偏了 → 朝弱侧平移 ----
        if self.side_servo and self.phase_state == 1 and grip < 0 and actor is not None:
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
        if self.close_ramp > 0:
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
        if self.width_map and kind in self.width_map and (glogit > 0 or self.phase_state == 2):
            if self.width_g is not None:
                grip = float(np.clip(self.width_g, -1.0, 1.0))
            else:
                w_target = self.width_map[kind] - self.width_delta
                grip = float(np.clip((w_target - 0.030) / 0.050, -1.0, 0.7))
            self.stats["grip_w"] = grip
        self.stats["close"] += int(grip < 0)
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
        if (self.regrasp and not grasping and self.phase_state == 1 and grip < 0
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
            if self._descend_target is None and grip < 0 and dz_i > 0.05:
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
        if (self.close_z_gate > 0 and grip < 0 and not grasping
                and self.phase_state != 2 and actor is not None):
            fpos_g = actor.pose.p[0].cpu().numpy().astype(np.float64)
            if float(tcp[2] - fpos_g[2]) > self.close_z_gate:
                grip = 1.0
                tgt = np.array([fpos_g[0], fpos_g[1], fpos_g[2] + 0.02])
                self.phase_state, self.same, self.phase_cand, self.commit = 1, 1, 1, 8
                self.stats["zgate"] = self.stats.get("zgate", 0) + 1
        # 抓空复位（对应专家 close_steps=12）
        if grip < 0 and not grasping:
            self.closed_steps += 1
            if self.closed_steps > 12:
                grip = 1.0
                self.closed_steps = 0
                if self.phase_state == 1:
                    self.phase_state, self.same, self.phase_cand, self.commit = 0, 0, 0, 0
        else:
            self.closed_steps = 0
        if self.hold_latch and grip < 0 and not grasping:
            tgt = tcp.copy()

        # ---- 触觉补救：滑移检测 → 降速 ----
        if self.slip_guard and actor is not None:
            tac = build_tactile(u, kind)
            slipping = float(tac[0]) > 3.0 and float(tac[5]) > self.slip_thr
            if slipping:
                self.stats["slip"] += 1
                if self.stats["slip_first"] < 0:
                    self.stats["slip_first"] = self.stats["steps"]
                if self.phase_state == 2:
                    self.slip_slow = True

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
        # 诊断用（不影响控制）：本步模型输出/目标点/状态
        self.last_dbg = dict(p4=np.array(p4, dtype=np.float64), tgt=np.array(tgt, dtype=np.float64),
                             tcp=np.array(tcp, dtype=np.float64), phase=int(self.phase_state),
                             grasp=bool(grasping), grip=float(grip),
                             logits=(None if logits is None else logits[0].detach().cpu().numpy().copy()),
                             sb=np.array(sb, dtype=np.float64))
        self.prev_grip = grip
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
