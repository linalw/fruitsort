"""行为克隆（BC）基线：MLP + 动作分块。

数据由 `scripts/record_demos.py` 用脚本专家采集；观测/动作定义见该脚本。
这一层刻意做得简单，用来验证「数据 → 训练 → 闭环评测」链路，
后续可平移到 ACT / Diffusion Policy / VLA。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ALL = ("apple", "strawberry", "banana", "pear")
OBS_DIM = 49
ACT_DIM = 4


class Normalizer:
    def __init__(self, obs: np.ndarray, act: np.ndarray, eps: float = 1e-6):
        self.obs_mean = obs.mean(0)
        self.obs_std = obs.std(0) + eps
        self.act_mean = act.mean(0)
        self.act_std = act.std(0) + eps

    def obs(self, x):
        return (x - self.obs_mean) / self.obs_std

    def act(self, x):
        return (x - self.act_mean) / self.act_std

    def unact(self, x):
        return x * self.act_std + self.act_mean

    def state_dict(self):
        return {"obs_mean": self.obs_mean, "obs_std": self.obs_std,
                "act_mean": self.act_mean, "act_std": self.act_std}

    @classmethod
    def from_state(cls, d):
        obj = cls.__new__(cls)
        obj.obs_mean = d["obs_mean"]; obj.obs_std = d["obs_std"]
        obj.act_mean = d["act_mean"]; obj.act_std = d["act_std"]
        return obj


class ChunkBC(nn.Module):
    """历史观测 -> K 步动作块。推理时每步重规划，取第 0 步（receding horizon）。"""

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 chunk: int = 8, hidden: int = 512, hist: int = 5):
        super().__init__()
        self.chunk, self.act_dim, self.hist = chunk, act_dim, hist
        self.net = nn.Sequential(
            nn.Linear(obs_dim * hist, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, chunk * act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, self.chunk, self.act_dim)


def build_obs(u) -> np.ndarray:
    """与 record_demos.py 一致的观测向量（49 维）。"""
    tcp = u.agent.tcp_pose.raw_pose[0].cpu().numpy()
    qpos = u.agent.robot.get_qpos()[0].cpu().numpy()
    belt = np.array([float(u.belt_speed[0])], dtype=np.float32)
    feats = []
    for k in ALL:
        if k in u.fruit_actors:
            a = u.fruit_actors[k]
            p = a.pose.p[0].cpu().numpy()
            v = a.linear_velocity[0].cpu().numpy()
            held = float(np.linalg.norm(p - tcp[:3]) < 0.08)
            active = 1.0
        else:
            p = np.zeros(3); v = np.zeros(3); held = 0.0; active = 0.0
        feats.append(np.concatenate([p, v, [active, held]]))
    return np.concatenate([tcp, qpos, belt, *feats]).astype(np.float32)


@dataclass
class EpisodeData:
    obs: np.ndarray
    action: np.ndarray
    path: Path


def load_episodes(data_dir: Path) -> list[EpisodeData]:
    eps = []
    for f in sorted(Path(data_dir).glob("ep_*.npz")):
        d = np.load(f)
        eps.append(EpisodeData(obs=d["obs"], action=d["action"], path=f))
    if not eps:
        raise FileNotFoundError(f"{data_dir} 下没有 ep_*.npz")
    return eps


def build_windows(episodes, chunk: int, hist: int = 5, stride: int = 1):
    obs, act = [], []
    for ep in episodes:
        T = len(ep.obs)
        padded = np.concatenate([ep.action, np.repeat(ep.action[-1:], chunk, axis=0)], axis=0)
        for t in range(0, T - 1, stride):
            idx = [max(0, t - hist + 1 + i) for i in range(hist)]
            obs.append(ep.obs[idx].reshape(-1))
            act.append(padded[t:t + chunk])
    return np.array(obs, dtype=np.float32), np.array(act, dtype=np.float32)
