"""FruitSort 多视角时序策略（v0.5 架构）。

把论文里的有效机制拼成一个可训练模型：

- 多视角输入（固定第三人称 + 腕部）        ← StreamPI / FASTER / TEMPO
- 帧差作为廉价运动通道                     ← DynamicWAM / Reflex 的时序线索
- 历史 T 帧 + 因果注意力                   ← StreamPI 的原子时间单元
- 动作分块头（chunk）                      ← ACT / FASTER / DiscreteRTC
- 未来状态预测辅助头                       ← AHEAD / DynamicWAM / TrajWAM
- 类别指令 embedding                       ← VLA 的最小语言接口（后续换文本编码器）
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PolicyConfig:
    obs_dim: int = 49
    n_kinds: int = 5
    hist: int = 4
    chunk: int = 8
    future_steps: int = 3
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    img_size: int = 96
    use_wrist: bool = True


class ConvEncoder(nn.Module):
    """轻量 CNN：RGB + 帧差 4 通道 → 128 维。"""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 24, 5, stride=2, padding=2), nn.BatchNorm2d(24), nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.Conv2d(96, out_dim, 3, stride=2, padding=1), nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


class MultiViewTemporalPolicy(nn.Module):
    """输入：多视角图像历史 + 状态历史 + 任务 id；输出：动作块 + 未来状态。"""

    def __init__(self, cfg: PolicyConfig = PolicyConfig()):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        n_views = 2 if cfg.use_wrist else 1
        self.n_tokens = cfg.hist * (n_views + 1) + 1  # 每帧: 图像视角 + 状态，另加 CLS

        self.view_enc = ConvEncoder(d)
        self.state_enc = nn.Sequential(nn.Linear(cfg.obs_dim, d), nn.GELU(), nn.Linear(d, d))
        # 状态捷径：当前帧状态直接残差到动作头，保证精度（图像/历史提供上下文，状态提供精确几何）
        self.state_short = nn.Sequential(nn.Linear(cfg.obs_dim, d), nn.GELU(), nn.Linear(d, d))
        self.kind_emb = nn.Embedding(cfg.n_kinds, d)
        # 指针式选择头：4 个水果 token × 指令 → 注意力权重 → 选中水果位置
        self.fruit_proj = nn.Linear(6, 32)
        self.instr_proj = nn.Linear(d, 32)
        self.sel_proj = nn.Linear(3, d)
        self.slot_emb = nn.Parameter(torch.zeros(4, 32))  # 槽位身份，让指令能直接指向某一类
        self.view_type = nn.Parameter(torch.zeros(3, d))       # overhead / wrist / state
        self.time_pos = nn.Parameter(torch.zeros(cfg.hist, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))

        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads, dim_feedforward=4 * d,
            dropout=cfg.dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(d)

        self.act_head = nn.Linear(d, cfg.chunk * 3)
        self.grip_head = nn.Linear(d, cfg.chunk)  # 夹爪单独一个头，避免被 xyz 回归淹没
        self.future_head = nn.Linear(d, cfg.future_steps * 4 * 6)  # 固定 4 类水果
        self._init_weights()

    def _init_weights(self):
        for p in (self.view_type, self.time_pos, self.cls):
            nn.init.normal_(p, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _img_tokens(self, imgs: torch.Tensor, prev: torch.Tensor | None) -> torch.Tensor:
        """imgs: (B,T,3,H,W) → (B,T,D)，第 4 通道为帧差。"""
        B, T, C, H, W = imgs.shape
        x = imgs.reshape(B * T, C, H, W)
        if prev is None:
            diff = torch.zeros_like(x[:, :1])
        else:
            p = prev.reshape(B * T, C, H, W)
            diff = (x - p).abs().mean(dim=1, keepdim=True)
        return self.view_enc(torch.cat([x, diff], dim=1)).reshape(B, T, -1)

    def forward(self, imgs_oh: torch.Tensor, imgs_wr: torch.Tensor | None,
                states: torch.Tensor, kind_ids: torch.Tensor):
        """
        imgs_oh: (B,T,3,H,W)  float [0,1]
        imgs_wr: (B,T,3,H,W) or None
        states:  (B,T,49)
        kind_ids:(B,) long
        """
        B, T = states.shape[:2]
        tok = []
        # 上一帧用于帧差
        prev_oh = torch.cat([imgs_oh[:, :1], imgs_oh[:, :-1]], dim=1)
        tok.append(self._img_tokens(imgs_oh, prev_oh) + self.view_type[0] + self.time_pos)
        if imgs_wr is not None and self.cfg.use_wrist:
            prev_wr = torch.cat([imgs_wr[:, :1], imgs_wr[:, :-1]], dim=1)
            tok.append(self._img_tokens(imgs_wr, prev_wr) + self.view_type[1] + self.time_pos)
        tok.append(self.state_enc(states) + self.view_type[2] + self.time_pos)

        seq = torch.stack(tok, dim=2).reshape(B, T * len(tok), -1)  # (B, T*(V+1), D)
        cls = (self.cls + self.kind_emb(kind_ids)[:, None, :]).expand(B, 1, -1)
        x = torch.cat([cls, seq], dim=1)

        # 因果掩码：CLS 可见全部；时间 t 只能看到 <= t
        n = x.shape[1]
        mask = torch.zeros(n, n, dtype=torch.bool, device=x.device)
        per_t = len(tok)
        for i in range(1, n):
            ti = (i - 1) // per_t
            for j in range(1, n):
                if (j - 1) // per_t > ti:
                    mask[i, j] = True
        x = self.transformer(x, mask=mask)
        z = self.norm(x[:, 0]) + self.state_short(states[:, -1])

        # 指针选择：只用状态里的 4 个水果 + 指令，保证"按指令选水果"的归纳偏置
        feats = states[:, -1, 17:].view(B, 4, 8)[:, :, :6]
        q = self.instr_proj(self.kind_emb(kind_ids))              # (B,32)
        scores = ((self.fruit_proj(feats) + self.slot_emb) * q[:, None, :]).sum(-1)  # (B,4)
        attn = torch.softmax(scores, dim=-1)
        sel_pos = (attn[..., None] * feats[:, :, :3]).sum(dim=1)   # (B,3)
        z = z + self.sel_proj(sel_pos)
        sel_logits = scores

        act_xyz = self.act_head(z).view(B, self.cfg.chunk, 3)
        grip = self.grip_head(z).view(B, self.cfg.chunk, 1)
        act = torch.cat([act_xyz, grip], dim=-1)
        future = self.future_head(z).view(B, self.cfg.future_steps, 4 * 6)
        return act, future, sel_logits
