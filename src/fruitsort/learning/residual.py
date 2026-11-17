"""学习式残差控制器（Res-Refiner，DSDyn 风格）。

动机
----
多视角策略 `models/vision_policy.py` 能给出"大致朝着目标"的世界坐标目标，但在
近距抓取、夹爪时序、确定性搬运这些阶段会漂移：纯模型闭环（不加任何几何伺服 /
脚本执行器）0/5。与其重训 40 万参数的视觉策略，不如学一个**有界残差**：

    tgt_world = model_tgt + Δxyz        |Δxyz| ≤ max_pos (默认 0.03 m)
    logit     = model_logit + Δgrip     |Δgrip| ≤ max_grip (默认 3.0)

- 输入 56 维：``[49 维状态, 模型输出的目标 xyz(3), 指令 one-hot(4)]``
- 隐层 64 维 MLP → 输出 Δxyz(3) + Δgrip(1)，``tanh`` 硬限幅
- 参数量 ~8k，单步推理远小于 1 ms

监督来自 `policies/hybrid_teacher.py::HybridTeacher`（模型选择 + 近距伺服 +
几何夹爪 + 确定性搬运，四类同场 95%），在**纯模型自己的轨迹**上逐步打标
（DAgger）：

    标签_xyz  = teacher_tgt - model_tgt
    标签_grip = teacher 的夹爪 0/1（1 = 闭合）

关键性质：Δ 被 tanh 限幅，即使网络输出爆炸，策略最多"偏 3 cm"，不会把整条
轨迹带飞 —— 这是残差式精修相对直接微调大模型的安全边界。
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

ALL_KINDS = ("apple", "strawberry", "banana", "pear")


@dataclass
class ResidualConfig:
    obs_dim: int = 49
    n_kinds: int = 4
    hidden: int = 64
    max_pos: float = 0.03     # Δxyz 限幅（m）
    max_grip: float = 3.0     # Δgrip logit 限幅


class ResidualRefiner(nn.Module):
    """(状态, 模型目标, 指令) → (有界 Δxyz, 有界 Δgrip)。"""

    def __init__(self, cfg: ResidualConfig = ResidualConfig()):
        super().__init__()
        self.cfg = cfg
        d_in = cfg.obs_dim + 3 + cfg.n_kinds
        self.net = nn.Sequential(
            nn.Linear(d_in, cfg.hidden), nn.GELU(),
            nn.Linear(cfg.hidden, cfg.hidden), nn.GELU(),
            nn.Linear(cfg.hidden, 4),
        )
        # 输入标准化（状态里 qpos 与 xyz 量级差很多，必须标准化）
        self.register_buffer("mu", torch.zeros(d_in))
        self.register_buffer("sd", torch.ones(d_in))
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)  # 近似零初始化 → 起步是恒等映射
            self.net[-1].bias.zero_()

    # ------------------------------------------------------------------ #
    @property
    def in_dim(self) -> int:
        return self.cfg.obs_dim + 3 + self.cfg.n_kinds

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def fit_normalizer(self, feats: np.ndarray) -> None:
        feats = np.asarray(feats, dtype=np.float32)
        with torch.no_grad():
            self.mu.copy_(torch.from_numpy(feats.mean(0)))
            self.sd.copy_(torch.from_numpy(feats.std(0) + 1e-6))

    def forward(self, state: torch.Tensor, model_tgt: torch.Tensor,
                kind_oh: torch.Tensor):
        """state (B,49) 已按策略的 norm_state 标准化；model_tgt (B,3)。"""
        f = torch.cat([state, model_tgt, kind_oh], dim=-1)
        f = (f - self.mu) / self.sd
        out = self.net(f)
        dxyz = self.cfg.max_pos * torch.tanh(out[..., :3])
        dgrip = self.cfg.max_grip * torch.tanh(out[..., 3])
        return dxyz, dgrip

    # ------------------------------------------------------------------ #
    def apply(self, state: np.ndarray, model_tgt: np.ndarray, model_logit: float,
              kind_id: int) -> tuple[np.ndarray, float, np.ndarray, float]:
        """单步 numpy 推理（闭环用）。返回 (tgt, logit, dxyz, dgrip)。"""
        with torch.no_grad():
            st = torch.from_numpy(np.asarray(state, dtype=np.float32)[None])
            tg = torch.from_numpy(np.asarray(model_tgt, dtype=np.float32)[None])
            oh = torch.zeros(1, self.cfg.n_kinds)
            oh[0, min(int(kind_id), self.cfg.n_kinds - 1)] = 1.0
            dev = self.mu.device
            dxyz, dgrip = self(st.to(dev), tg.to(dev), oh.to(dev))
        return (np.asarray(model_tgt, dtype=np.float64) + dxyz[0].cpu().numpy().astype(np.float64),
                float(model_logit) + float(dgrip[0].item()),
                dxyz[0].cpu().numpy(), float(dgrip[0].item()))


# ---------------------------------------------------------------------- #
# 归一化修复
# ---------------------------------------------------------------------- #
def safe_normalizer(ns, floor: float = 1e-3):
    """修掉状态归一化里的退化维度（**不改现有代码**，只在本地拷贝上打补丁）。

    训练集里恒定的观测维度 std == 1e-6（Normalizer 的 eps），推理时只要该维度
    的值与训练集均值不同，就会被放大到 1e6 量级。`runs/vpolicy-4c-v3` 的
    `pear_active`（索引 47）就是这种维度：四个数据集的 active_kinds 都含梨 →
    恒为 1.0，而评测只用 apple+strawberry → 值 0.0 → 归一化后 -1e6 → 策略输出
    饱和成常数 [2,-2,-2]、夹爪 logit ≈ 700，整条闭环 0/5。

    这里把退化的维度置成恒等（mean=0, std=1），即"忽略该维度"。
    """
    out = copy.copy(ns)
    bad = np.asarray(ns.obs_std) < floor
    out.obs_mean = np.where(bad, 0.0, np.asarray(ns.obs_mean))
    out.obs_std = np.where(bad, 1.0, np.asarray(ns.obs_std))
    out.degenerate_dims = np.flatnonzero(bad).tolist()
    return out


# ---------------------------------------------------------------------- #
# 存取 / 特征拼接
# ---------------------------------------------------------------------- #
def featurize(states: np.ndarray, model_tgt: np.ndarray, kind_oh: np.ndarray) -> np.ndarray:
    """(N,49)+(N,3)+(N,4) → (N,56)，与 ResidualRefiner.forward 的拼接顺序一致。"""
    return np.concatenate(
        [np.asarray(states, np.float32), np.asarray(model_tgt, np.float32),
         np.asarray(kind_oh, np.float32)], axis=-1)


def onehot(kind_id: int, n: int = 4) -> np.ndarray:
    z = np.zeros(n, dtype=np.float32)
    z[min(int(kind_id), n - 1)] = 1.0
    return z


def save_residual(path, model: ResidualRefiner, meta: dict) -> None:
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": asdict(model.cfg), "model": model.state_dict(), **meta}, p)


def load_residual(path, device="cpu") -> tuple[ResidualRefiner, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = ResidualRefiner(ResidualConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck
