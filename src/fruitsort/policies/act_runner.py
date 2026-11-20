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
                 side_servo: bool = False, side_step: float = 0.002):
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
        self.reset()

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
        self.queue = []
        self.queue_logits = None
        self.stats = {"close": 0, "descend": 0, "carry": 0, "steps": 0,
                      "slip": 0, "slip_first": -1, "regrasp": 0, "slip_prob_max": 0.0,
                      "force_target": 0.0, "fn_peak": 0.0, "drop_prob_max": 0.0,
                      "gate_fail": 0, "retries": 0, "fL": 0.0, "fR": 0.0,
                      "side_push": 0}
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
        if self.fb and kind in u.fruit_actors and not grasping:
            fpos = u.fruit_actors[kind].pose.p[0].cpu().numpy().astype(np.float64)
            corr = np.clip(fpos - tcp, -0.15, 0.15)
            if not self.fb_z:
                corr[2] = 0.0
            tgt = tgt + self.fb * corr
        if np.any(self.retry_bias) and self.phase_state != 2:
            tgt = tgt + self.retry_bias
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
