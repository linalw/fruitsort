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
    def __init__(self, obs: np.ndarray, act: np.ndarray, eps: float = 1e-3):
        # std floor=1e-3：训练集里恒定的维度（如 pear_active）不再放大成 1e6
        self.obs_mean = obs.mean(0)
        self.obs_std = np.maximum(obs.std(0), eps)
        self.act_mean = act.mean(0)
        self.act_std = np.maximum(act.std(0), eps)

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
        obj.obs_mean = d["obs_mean"]
        obj.obs_std = np.maximum(d["obs_std"], 1e-3)
        obj.act_mean = d["act_mean"]
        obj.act_std = np.maximum(d["act_std"], 1e-3)
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


TACTILE_DIM = 9


def build_tactile(u, kind: str | None = None) -> np.ndarray:
    """合成触觉（9 维，无量纲优先）：左右指接触力 + 摩擦利用率 + 开口 + 滑移。

    物理含义（对应文献里抓取稳定性/滑移检测的标准量）：
      0 Fn        法向力均值 (N)            —— 抓取力
      1 Fasym     左右指力不平衡度 ||F_L|-|F_R||/Fn —— 偏载/翻转前兆
      2 rho       摩擦利用率 |F_t|/Fn      —— 接近摩擦锥极限即滑移前兆（阈值 ~0.6）
      3 w         夹爪开口 (m)             —— 挤压程度
      4 delta     挤压比 clip((0.08-w)/0.08) —— 形变代理
      5 v_slip    物体与 TCP 的相对速度    —— 滑移速度
      6 slip      是否已明显滑移 (>0.03 m/s)
      7/8         左右指是否接触
    """
    try:
        L_link = u.agent.robot.links_map["panda_leftfinger"]
        R_link = u.agent.robot.links_map["panda_rightfinger"]
        FL = L_link.get_net_contact_forces()[0].cpu().numpy()
        FR = R_link.get_net_contact_forces()[0].cpu().numpy()
        pL = L_link.pose.p[0].cpu().numpy()
        pR = R_link.pose.p[0].cpu().numpy()
    except Exception:  # noqa: BLE001
        return np.zeros(TACTILE_DIM, dtype=np.float32)
    axis = pL - pR
    n = float(np.linalg.norm(axis))
    axis = axis / n if n > 1e-6 else np.array([0.0, 1.0, 0.0])
    fL_n = abs(float(np.dot(FL, axis)))
    fR_n = abs(float(np.dot(FR, -axis)))
    fL_t = float(np.linalg.norm(FL - np.dot(FL, axis) * axis))
    fR_t = float(np.linalg.norm(FR - np.dot(FR, -axis) * (-axis)))
    Fn = 0.5 * (fL_n + fR_n)
    Ft = 0.5 * (fL_t + fR_t)
    Fasym = abs(fL_n - fR_n) / (Fn + 1e-6)
    rho = Ft / (Fn + 1e-6)
    q = u.agent.robot.get_qpos()[0].cpu().numpy()
    w = float(q[7] + q[8])
    delta = float(np.clip((0.08 - w) / 0.08, 0.0, 1.0))
    a = u.fruit_actors.get(kind) if kind else None
    if a is not None:
        v_obj = a.linear_velocity[0].cpu().numpy()
        try:
            v_tcp = u.agent.robot.links_map["panda_hand_tcp"].linear_velocity[0].cpu().numpy()
        except Exception:  # noqa: BLE001
            v_tcp = np.zeros(3, dtype=np.float32)
        v_slip = float(np.linalg.norm(v_obj - v_tcp))
    else:
        v_slip = 0.0
    touch_L = float(np.linalg.norm(FL) > 0.5)
    touch_R = float(np.linalg.norm(FR) > 0.5)
    return np.array([Fn, Fasym, rho, w, delta, v_slip, float(v_slip > 0.03),
                     touch_L, touch_R], dtype=np.float32)


def scale_tactile(t: np.ndarray) -> np.ndarray:
    """把原始触觉无量纲化到 O(1)（文献建议：不要直接把牛顿值喂网络）。

    0 Fn/40N   1 Fasym   2 rho/2（Fn<1N 时置 0，避免自由运动期爆表）
    3 w/0.08   4 delta    5 v_slip/1.0   6 slip flag   7/8 接触
    """
    o = np.array(t, dtype=np.float32, copy=True)
    o[0] = np.clip(o[0] / 40.0, 0.0, 2.0)
    o[1] = np.clip(o[1], 0.0, 2.0)
    o[2] = 0.0 if t[0] < 1.0 else np.clip(o[2], 0.0, 2.0) / 2.0
    o[3] = np.clip(o[3] / 0.08, 0.0, 2.0)
    o[4] = np.clip(o[4], 0.0, 1.0)
    o[5] = np.clip(o[5], 0.0, 1.0)
    return o


def build_obs(u, kind: str | None = None, tactile: bool = False) -> np.ndarray:
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
    base = np.concatenate([tcp, qpos, belt, *feats]).astype(np.float32)
    if not tactile:
        return base
    return np.concatenate([base, scale_tactile(build_tactile(u, kind))]).astype(np.float32)


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
