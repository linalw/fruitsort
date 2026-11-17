"""Teacher 策略：模型选择 + 近距伺服 + 几何夹爪/搬运（v0.6 的 5/5 配置）。

既用于评测，也用于**自蒸馏**：它访问到的状态就是纯模型需要覆盖的分布。
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from ..learning.bc import build_obs
from ..models.vision_policy import MultiViewTemporalPolicy, PolicyConfig
from ..sim.conveyor import BIN_LAYOUT
from .scripted import HOME_EULER

MAX_STEP = 0.06
INSTR_ID = {"all": 0, "apple": 1, "strawberry": 2, "banana": 3, "pear": 4}
ID_KIND = {0: "apple", 1: "strawberry", 2: "banana", 3: "pear"}


def _prep(x, size):
    if x.shape[0] != size:
        x = cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
    return (x.astype(np.float32) / 255.0).transpose(2, 0, 1)


class HybridTeacher:
    def __init__(self, env, ckpt_path, instruction: str, device="cpu",
                 near: float = 0.45, lead: float = 0.10, max_step: float = 0.06):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.cfg = PolicyConfig(**ck["cfg"])
        self.model = MultiViewTemporalPolicy(self.cfg).to(device)
        self.model.load_state_dict(ck["model"]); self.model.eval()
        self.ns = Normalizer.from_state(ck["norm_state"])
        self.na = Normalizer.from_state(ck["norm_act"])
        self.env, self.u, self.device = env, env.unwrapped, device
        self.instruction = instruction
        self.near, self.lead, self.max_step = near, lead, max_step
        self.reset()

    def reset(self):
        self.oh, self.wr, self.st = [], [], []
        self.carrying = None
        self.via_done = False
        self.carry_steps = 0
        self.close_streak = 0
        self.selected = None

    # ------------------------------------------------------------------ #
    def _select_kind(self, obs):
        sd = obs["sensor_data"]
        oh = _prep(sd["overhead"]["rgb"][0].cpu().numpy(), self.cfg.img_size)
        wr = _prep(sd["hand_camera"]["rgb"][0].cpu().numpy(), self.cfg.img_size)
        st = build_obs(self.u)
        with torch.no_grad():
            _, _, sel = self.model(
                torch.from_numpy(np.stack([oh] * self.cfg.hist)[None]).float().to(self.device),
                torch.from_numpy(np.stack([wr] * self.cfg.hist)[None]).float().to(self.device),
                torch.from_numpy(self.ns.obs(np.stack([st] * self.cfg.hist).astype(np.float32))[None]).float().to(self.device),
                torch.tensor([INSTR_ID[self.instruction]], device=self.device))
        kind = ID_KIND.get(int(sel.argmax(-1).item()), "apple")
        if kind not in self.u.fruit_actors:
            kind = self.instruction if self.instruction in self.u.fruit_actors else list(self.u.fruit_actors)[0]
        return kind

    def act(self, obs):
        """返回 (7 维动作指令, 4 维标签 [x,y,z,grip])。"""
        if self.selected is None:
            self.selected = self._select_kind(obs)
        kind = self.selected

        sd = obs["sensor_data"]
        oh = _prep(sd["overhead"]["rgb"][0].cpu().numpy(), self.cfg.img_size)
        wr = _prep(sd["hand_camera"]["rgb"][0].cpu().numpy(), self.cfg.img_size)
        st = build_obs(self.u)
        self.oh.append(oh); self.wr.append(wr); self.st.append(st)
        for h in (self.oh, self.wr, self.st):
            if len(h) > self.cfg.hist:
                h.pop(0)
        while len(self.oh) < self.cfg.hist:
            self.oh.insert(0, self.oh[0]); self.wr.insert(0, self.wr[0]); self.st.insert(0, self.st[0])

        tcp = self.u.agent.tcp_pose.p[0].cpu().numpy().astype(np.float64)
        actor = self.u.fruit_actors[kind]
        fp = actor.pose.p[0].cpu().numpy().astype(np.float64)
        fv = actor.linear_velocity[0].cpu().numpy().astype(np.float64)
        in_zone = bool(self.u.pick_zone[0] <= fp[0] <= self.u.pick_zone[1])
        dist = float(np.linalg.norm(fp - tcp))
        grasped = bool(self.u.agent.is_grasping(actor)[0])

        if self.carrying is None and self.close_streak >= 1 and (grasped or (self.close_streak >= 4 and dist < 0.05)):
            self.carrying, self.via_done, self.carry_steps = kind, False, 0

        tgt, grip = None, 1.0
        if self.carrying is not None:
            self.carry_steps += 1
            bx, by = BIN_LAYOUT[self.carrying]
            if dist > 0.10 or self.carry_steps > 200:
                self.carrying = None
            elif self.carry_steps <= 5:
                # 抓后先垂直慢抬（1.5cm/步），避免刚咬合就以 6cm/步 斜向甩飞
                tgt, grip = np.array([tcp[0], tcp[1], tcp[2] + 0.015]), -1.0
            elif not self.via_done and (abs(tcp[0]) > 0.06 or abs(tcp[1] + 0.34) > 0.06):
                tgt, grip = np.array([0.0, -0.34, 0.60]), -1.0
            else:
                self.via_done = True
                over = abs(tcp[0] - bx) < 0.04 and abs(tcp[1] - by) < 0.04
                if over and tcp[2] < 0.53:
                    tgt, grip = np.array([bx, by, 0.60]), 1.0
                else:
                    tgt, grip = np.array([bx, by, 0.60 if not over else 0.48]), -1.0
        if tgt is None and dist <= self.near and fp[2] > 0.3:
            tgt = fp + fv * self.lead
            grip = -1.0 if (dist < 0.05 and in_zone) else 1.0
        if tgt is None:
            with torch.no_grad():
                pred, _, _ = self.model(
                    torch.from_numpy(np.stack(self.oh)[None]).float().to(self.device),
                    torch.from_numpy(np.stack(self.wr)[None]).float().to(self.device),
                    torch.from_numpy(self.ns.obs(np.stack(self.st).astype(np.float32))[None]).float().to(self.device),
                    torch.tensor([INSTR_ID[self.instruction]], device=self.device))
            tgt = np.clip(self.na.unact(pred[0, 0, :3].cpu().numpy()), -2, 2)
            grip = 1.0

        self.close_streak = self.close_streak + 1 if grip < 0 else 0

        label = np.array([tgt[0], tgt[1], tgt[2], grip], dtype=np.float32)
        err = np.asarray(tgt) - tcp
        d = float(np.linalg.norm(err))
        if d > self.max_step:
            tgt = tcp + err / d * self.max_step
        cmd = np.array([tgt[1] + 0.5, -tgt[0], tgt[2], *HOME_EULER, grip], dtype=np.float32)
        return cmd, label


from ..learning.bc import Normalizer  # noqa: E402  (避免循环导入放到末尾)
