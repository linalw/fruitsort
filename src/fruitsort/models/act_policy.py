"""ACT-lite：动作分块 Transformer 策略（比 v0.5 的 MLP 容量/时序建模更强）。

与 `vision_policy.py` 的差别
---------------------------
v0.5 是「CLS + 因果注意力 + 单个线性头回归 K×4」：一个 token 要同时决定 8 步动作，
时序推理能力受限于 2 层 / d=96，而且动作维度之间没有归纳偏置。

ACT（Action Chunking with Transformers, 2304.13705）的结构是：

    多视角图像 + 状态 → 编码器（Transformer，读全部历史）
                              ↓ (memory)
    K 个可学习 query token → 解码器（self-attn + cross-attn 到 memory）→ K×4

每个 query 只负责一个时间步，解码器可以逐步细化，等价于学会了「从同一历史里
展开一整段轨迹」。再加 CVAE 隐变量 z（训练用 posterior 编码专家动作块，推理用
先验 z=0）来建模示教的多模态/噪声，这是 ACT 抑制复现误差抖动的关键部件。

本文件只做 **推理所需的最小实现**，不改动现有模型与脚本。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .vision_policy import ConvEncoder


@dataclass
class ActConfig:
    obs_dim: int = 49
    n_kinds: int = 5
    hist: int = 8          # 历史帧数 T
    chunk: int = 16        # 动作块长度 K
    d_model: int = 256
    n_heads: int = 8
    enc_layers: int = 4
    dec_layers: int = 4
    ffn_mult: int = 3
    dropout: float = 0.1
    img_size: int = 96
    use_wrist: bool = True
    use_cvae: bool = True
    latent_dim: int = 32
    n_phase: int = 0        # >0：相位辅助头类别数（0 = 关闭，兼容旧 ckpt）
    n_slip: int = 0         # >0：滑移/掉落辅助头
    tactile_dim: int = 0    # >0：触觉独立 token（不再拼进 state）
    use_force_head: bool = False   # 动作第 5 维 = 目标夹持力
    conv_norm: str = "gn"   # "gn": 把复用来的 ConvEncoder 的 BN 换成 GroupNorm


def bn_to_groupnorm(module: nn.Module, groups: int = 16) -> nn.Module:
    """把 BatchNorm2d 原地替换成 GroupNorm。

    闭环评测走 `model.eval()`（用 running stats），而小数据上 running stats
    与 batch stats 的偏差会让 CNN 特征整体漂移；GroupNorm 在 train/eval 下行为
    一致，等价于消除一个只在部署时才出现的分布偏移。
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            g = min(groups, child.num_features)
            while child.num_features % g:
                g -= 1
            setattr(module, name, nn.GroupNorm(g, child.num_features))
        else:
            bn_to_groupnorm(child, groups)
    return module


class ActPolicy(nn.Module):
    """ACT-lite：RGB(overhead+wrist)+帧差+状态+指令 → K 步世界目标位姿。"""

    def __init__(self, cfg: ActConfig = ActConfig()):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        n_views = 2 if cfg.use_wrist else 1
        self.n_views = n_views

        # ---------------- 观测编码 ----------------
        # 两个视角共享同一个 CNN（参数更少，视角间一致性更好）
        self.view_enc = ConvEncoder(d)
        if cfg.conv_norm == "gn":
            self.view_enc = bn_to_groupnorm(self.view_enc)
        self.state_enc = nn.Sequential(nn.Linear(cfg.obs_dim, d), nn.GELU(),
                                       nn.Linear(d, d))
        # 触觉独立 stem：9 维 → d，作为**每帧第 4 种 token**（文献建议触点单独编码）
        self.tactile_enc = (nn.Sequential(nn.Linear(cfg.tactile_dim, d), nn.GELU(),
                                          nn.Linear(d, d))
                            if cfg.tactile_dim > 0 else None)
        # 当前帧状态捷径：图像编码器分辨率有限，精确几何靠状态直连
        self.state_short = nn.Sequential(nn.Linear(cfg.obs_dim, d), nn.GELU(),
                                         nn.Linear(d, d))
        self.kind_emb = nn.Embedding(cfg.n_kinds, d)
        # 槽位：oh / wrist / state [/ tactile]。有触觉 token 时才多一槽（否则旧 ckpt 加载会 shape 不匹配）
        self.view_type = nn.Parameter(torch.zeros(n_views + (2 if cfg.tactile_dim else 1), d))
        self.time_pos = nn.Parameter(torch.zeros(cfg.hist, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.in_norm = nn.LayerNorm(d)

        # ---------------- 时序编码器 ----------------
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads, dim_feedforward=cfg.ffn_mult * d,
            dropout=cfg.dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.enc_layers,
                                             enable_nested_tensor=False)
        self.enc_norm = nn.LayerNorm(d)

        # ---------------- 解码器（K 个可学习 query）----------------
        self.query_pos = nn.Parameter(torch.zeros(cfg.chunk, d))
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=cfg.n_heads, dim_feedforward=cfg.ffn_mult * d,
            dropout=cfg.dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.dec_layers)
        self.dec_norm = nn.LayerNorm(d)

        # ---------------- CVAE 隐变量 ----------------
        # 训练：z ~ q(z | CLS, 专家动作块)；推理：z = 0（先验均值）
        self.use_cvae = cfg.use_cvae
        self.act_dim = 5 if cfg.use_force_head else 4
        if cfg.use_cvae:
            self.post_enc = nn.Sequential(
                nn.Linear(d + cfg.chunk * self.act_dim, d), nn.GELU(),
                nn.Linear(d, 2 * cfg.latent_dim),
            )
            self.z_proj = nn.Linear(cfg.latent_dim, d)

        # ---------------- 输出头 ----------------
        self.act_head = nn.Linear(d, 5 if cfg.use_force_head else 4)  # Δxyz + 夹爪 logit [+ 目标力]
        # 相位辅助头：每个 query 预测该步属于 接近/下探/搬运，推理端用 k=0 的
        # argmax 反馈回状态（相位由策略自己决定，不是脚本 FSM）
        self.phase_head = nn.Linear(d, cfg.n_phase) if cfg.n_phase else None
        self.slip_head = nn.Linear(d, 1) if cfg.n_slip else None
        self.last_slip: torch.Tensor | None = None
        self._init_weights()

    # ------------------------------------------------------------------ #
    def _init_weights(self):
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.view_type, std=0.02)
        nn.init.normal_(self.time_pos, std=0.02)
        nn.init.normal_(self.query_pos, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    def _img_tokens(self, imgs: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        """imgs: (B,T,3,H,W) → (B,T,D)；第 4 通道 = 与上一帧的绝对差（廉价运动线索）。"""
        B, T, C, H, W = imgs.shape
        x = imgs.reshape(B * T, C, H, W)
        p = prev.reshape(B * T, C, H, W)
        diff = (x - p).abs().mean(dim=1, keepdim=True)
        return self.view_enc(torch.cat([x, diff], dim=1)).reshape(B, T, -1)

    def encode(self, imgs_oh, imgs_wr, states, kind_ids, tactile=None):
        """历史观测 → (B, 1+T*(V+1), D) memory 与 CLS 特征。"""
        B, T = states.shape[:2]
        prev_oh = torch.cat([imgs_oh[:, :1], imgs_oh[:, :-1]], dim=1)
        tok = [self._img_tokens(imgs_oh, prev_oh) + self.view_type[0] + self.time_pos]
        if imgs_wr is not None and self.cfg.use_wrist:
            prev_wr = torch.cat([imgs_wr[:, :1], imgs_wr[:, :-1]], dim=1)
            tok.append(self._img_tokens(imgs_wr, prev_wr) + self.view_type[1] + self.time_pos)
        tok.append(self.state_enc(states)
                   + self.view_type[-2 if self.tactile_enc is not None else -1] + self.time_pos)
        if self.tactile_enc is not None and tactile is not None:
            tok.append(self.tactile_enc(tactile) + self.view_type[-1] + self.time_pos)

        seq = torch.stack(tok, dim=2).reshape(B, T * len(tok), -1)
        cls = self.cls + self.kind_emb(kind_ids)[:, None, :]
        x = torch.cat([cls, seq], dim=1)
        # 因果：时间 t 的 token 只能看到 <= t 的历史（防未来信息泄漏，训练/推理一致）；CLS 可见全部
        n_per_t = len(tok)
        order = torch.cat([torch.full((1,), 10 ** 6, device=x.device),
                           torch.arange(seq.shape[1], device=x.device) // n_per_t])
        mask = order[None, :] > order[:, None]          # (S,S) bool, True = 屏蔽
        x = self.encoder(self.in_norm(x), mask=mask)
        return self.enc_norm(x)

    def decode(self, memory, states, z=None):
        B = memory.shape[0]
        q = self.query_pos[None].expand(B, -1, -1)
        if z is not None:
            q = q + self.z_proj(z)[:, None, :]
        h = self.decoder(q, memory)
        h = self.dec_norm(h)
        # 与 ActPolicy.encode 的 CLS 捷径：当前帧状态直连每个 query
        base = h + self.state_short(states[:, -1])[:, None, :]
        if self.slip_head is not None:
            self.last_slip = self.slip_head(h[:, 0]).squeeze(-1)   # (B,) logit
        if self.phase_head is None:
            return self.act_head(base), None
        return self.act_head(base), self.phase_head(h)

    def forward(self, imgs_oh, imgs_wr, states, kind_ids, act_target=None,
                use_prior: bool = False, phase_out: bool = False, tactile=None):
        """
        imgs_oh: (B,T,3,H,W) float[0,1]；imgs_wr 同上或 None
        states:  (B,T,obs_dim)；kind_ids: (B,)
        act_target: (B,K,4) 归一化动作，训练时用于 CVAE posterior
        use_prior:  True → 强制 z=0（推理路径）
        phase_out:  True → 额外返回每步相位 logits (B,K,n_phase)

        返回 (act (B,K,4), mu, logvar[, phase_logits])。
        """
        memory = self.encode(imgs_oh, imgs_wr, states, kind_ids, tactile=tactile)
        mu = logvar = None
        z = None
        if self.use_cvae:
            if act_target is not None and not use_prior:
                feat = torch.cat([memory[:, 0], act_target.flatten(1)], dim=-1)
                mu, logvar = self.post_enc(feat).chunk(2, dim=-1)
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
            else:
                z = memory.new_zeros(memory.shape[0], self.cfg.latent_dim)
        act, logits = self.decode(memory, states, z)
        if phase_out:
            return act, mu, logvar, logits
        return act, mu, logvar

    @torch.no_grad()
    def act(self, imgs_oh, imgs_wr, states, kind_ids):
        """推理便捷入口：先验 z=0，返回 (B,K,4)。"""
        self.eval()
        return self.forward(imgs_oh, imgs_wr, states, kind_ids, use_prior=True)[0]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    cfg = ActConfig()
    m = ActPolicy(cfg)
    B, T = 2, cfg.hist
    oh = torch.rand(B, T, 3, cfg.img_size, cfg.img_size)
    st = torch.randn(B, T, cfg.obs_dim)
    k = torch.zeros(B, dtype=torch.long)
    out, mu, lv = m(oh, oh, st, k, act_target=torch.randn(B, cfg.chunk, 4))
    print("act", tuple(out.shape), "mu", None if mu is None else tuple(mu.shape))
    print(f"params {count_params(m)/1e6:.3f}M")
