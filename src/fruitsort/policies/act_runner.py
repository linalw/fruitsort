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
                 slip_thr: float = 0.05, slip_slow_step: float = 0.012):
        ck = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
        self.cfg = ActConfig(**ck["cfg"])
        self.model = ActPolicy(self.cfg).to(device)
        self.model.load_state_dict(ck["model"]); self.model.eval()
        self.ns = Normalizer.from_state(ck["norm_state"])
        self.na = Normalizer.from_state(ck["norm_act"])
        self.delta_actions = bool(ck.get("delta_actions", False))
        self.use_phase = self.cfg.obs_dim > 49
        # 布局：[49 base][9 tactile][3 phase]。触觉与否由 obs_dim 决定
        self.tactile_dim = 9 if (self.cfg.obs_dim - (3 if self.use_phase else 0)) >= 58 else 0

        self.env, self.u, self.device = env, env.unwrapped, device
        self.kind = kind
        self.fb, self.fb_z, self.grip_ema = fb, fb_z, grip_ema
        self.carry_latch, self.descend_commit = carry_latch, descend_commit
        self.max_step, self.lead, self.z_gate = max_step, lead, z_gate
        self.phase_fb, self.hold_latch, self.grasp_retry = phase_fb, hold_latch, grasp_retry
        self.chunk_exec = chunk_exec
        # 搬运段单独限速：接近段要快要准，搬运段过冲会把水果带到框外
        self.carry_max_step = carry_max_step
        # 触觉补救：搬运中检测到滑移（物体相对 TCP 有速度且有夹持力）→ 立即降速搬运
        self.slip_guard = slip_guard
        self.slip_thr = slip_thr
        self.slip_slow_step = slip_slow_step
        # 搬运段单独限速：接近段要快要准，搬运段过冲会把水果带到框外
        self.carry_max_step = carry_max_step
        # 触觉补救：搬运中检测到滑移（物体相对 TCP 有速度且有夹持力）→ 立即降速搬运
        self.slip_guard = slip_guard
        self.slip_thr = slip_thr
        self.slip_slow_step = slip_slow_step
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self):
        self.oh, self.wr, self.st = [], [], []
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
        self.queue = []
        self.queue_logits = None
        self.stats = {"close": 0, "descend": 0, "carry": 0, "steps": 0,
                      "slip": 0, "slip_first": -1}
        self.grasping = False

    # ------------------------------------------------------------------ #
    def act(self, obs) -> np.ndarray:
        u, cfg = self.u, self.cfg
        kind = self.kind
        actor = u.fruit_actors.get(kind)
        tcp = u.agent.tcp_pose.p[0].cpu().numpy().astype(np.float64)
        grasping = False
        if actor is not None:
            fpos_now = actor.pose.p[0].cpu().numpy().astype(np.float64)
            fd = float(np.linalg.norm(fpos_now - tcp))
            # 手指接触，或「贴得很近且上一步确实闭爪」
            grabbing = bool(u.agent.is_grasping(actor)[0])
            if self.grasp_retry:
                grasping = grabbing or (fd < 0.045 and self.prev_grip < 0)
            else:
                grasping = grabbing
            fz = float(actor.pose.p[0, 2])
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
        sb = build_obs(u, kind, tactile=bool(self.tactile_dim))
        if self.use_phase:
            onehot = np.zeros(3, dtype=np.float32)
            onehot[self.phase_state if self.phase_fb else 0] = 1.0
            sb = np.concatenate([sb, onehot])
        self.st.append(sb)
        for h in (self.oh, self.wr, self.st):
            if len(h) > cfg.hist:
                h.pop(0)
        while len(self.oh) < cfg.hist:
            self.oh.insert(0, self.oh[0])
            self.wr.insert(0, self.wr[0])
            self.st.insert(0, self.st[0])

        if self.queue:
            p4 = self.queue.pop(0)
            logits = self.queue_logits
        else:
            with torch.no_grad():
                if cfg.n_phase:
                    act, _, _, logits = self.model(
                        torch.from_numpy(np.stack(self.oh)[None]).float().to(self.device),
                        torch.from_numpy(np.stack(self.wr)[None]).float().to(self.device),
                        torch.from_numpy(self.ns.obs(np.stack(self.st).astype(np.float32))[None]).float().to(self.device),
                        torch.tensor([INSTR_ID[kind]], device=self.device),
                        use_prior=True, phase_out=True)
                else:
                    act, _, _ = self.model(
                        torch.from_numpy(np.stack(self.oh)[None]).float().to(self.device),
                        torch.from_numpy(np.stack(self.wr)[None]).float().to(self.device),
                        torch.from_numpy(self.ns.obs(np.stack(self.st).astype(np.float32))[None]).float().to(self.device),
                        torch.tensor([INSTR_ID[kind]], device=self.device),
                        use_prior=True)
                    logits = None
            acts_np = act[0].cpu().numpy()
            if self.chunk_exec > 1 and self.delta_actions:
                self.queue = [a for a in acts_np[: self.chunk_exec]]
                self.queue_logits = logits
            p4 = acts_np[0]

        tgt = self.na.unact(p4[:3])
        if self.delta_actions:
            tgt = tcp + np.clip(tgt, -self.max_step, self.max_step)
        else:
            tgt = np.clip(tgt, -2, 2)

        # 残差反馈（IBVS 式）：策略给前馈，反馈项收残余误差
        if self.fb and kind in u.fruit_actors and not grasping:
            fpos = u.fruit_actors[kind].pose.p[0].cpu().numpy().astype(np.float64)
            corr = np.clip(fpos - tcp, -0.15, 0.15)
            if not self.fb_z:
                corr[2] = 0.0
            tgt = tgt + self.fb * corr
        if self.lead and kind in u.fruit_actors:
            tgt[0] += self.lead * float(u.belt_speed[0])
        if self.z_gate and kind in u.fruit_actors:
            fz = u.fruit_actors[kind].pose.p[0, 2].item()
            tgt[2] = float(np.clip(tgt[2], fz - 0.02, 0.90))

        glogit = float(p4[3])
        if self.grip_ema > 0:
            a = self.grip_ema
            self.grip_ema_val = glogit if self.grip_ema_val is None else a * self.grip_ema_val + (1 - a) * glogit
            glogit = self.grip_ema_val
        grip = -1.0 if glogit > 0 else 1.0
        if self.carry_latch and grasping:
            self.open_streak = self.open_streak + 1 if glogit < 0 else 0
            if self.open_streak < 3:
                grip = -1.0
        else:
            self.open_streak = 0
        self.slip_slow = False
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

        err = tgt - tcp
        d = float(np.linalg.norm(err))
        step = self.max_step
        if self.carry_max_step > 0 and self.phase_state == 2:
            step = self.carry_max_step
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

        self.prev_grip = grip
        return np.array([tgt[1] + 0.5, -tgt[0], tgt[2], *HOME_EULER, grip], dtype=np.float32)
