"""图像示教数据集：加载 record_demos.py --images 生成的 npz。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ALL = ("apple", "strawberry", "banana", "pear")
IMG = 96


def _decode(buf: bytes) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    if arr.shape[0] != IMG:
        arr = cv2.resize(arr, (IMG, IMG), interpolation=cv2.INTER_AREA)
    return arr


BIN_XY = {"apple": (-0.46, -0.28), "strawberry": (-0.46, -0.60),
          "banana": (0.46, -0.28), "pear": (0.46, -0.60)}


def episode_phase(obs: np.ndarray, act: np.ndarray, instruction: str) -> np.ndarray:
    """从示教动作里还原 3 类相位标签（0=接近悬停, 1=下探/闭爪, 2=已抓起搬运）。

    为什么需要它
    ------------
    纯回归把「悬停 p.z+0.09」「下探到果心」「搬去料框」三个分支平均掉了：
    模型既不知道自己在哪一段，输出就落在中间解（实测 0.25 m/s 下悬停 10 cm 不下降，
    直到水果已经过夹爪才闭爪）。把相位显式喂进去 + 让策略自己预测相位，
    等于把 FSM 交给策略而不是脚本。

    标签来源（teacher-forced，全部可从示教数据算出）：
    - 目标 xy 落在料框范围内 → 2（一旦出现就锁存，等价于推理端「抓起后不退回接近」）
    - 目标 z 贴近水果中心（|Δz|<0.06）→ 1（下探/贴地闭爪段）
    - 其余 → 0
    """
    if instruction in ALL:
        j = ALL.index(instruction)
    else:
        j = int(np.argmax([obs[:, 17 + 8 * k + 2].max() for k in range(4)]))
    fz = obs[:, 17 + 8 * j + 2]
    bx, by = BIN_XY.get(instruction, (0.0, 0.0))
    to_bin = (np.abs(act[:, 0] - bx) < 0.20) & (np.abs(act[:, 1] - by) < 0.20)
    descend = np.abs(act[:, 2] - fz) < 0.06
    latched = np.maximum.accumulate(to_bin.astype(bool))
    return np.where(latched, 2, np.where(descend, 1, 0)).astype(np.int64)


class ImageDemoDataset(Dataset):
    def __init__(self, data_dir, hist: int = 5, chunk: int = 8, future: int = 4, stride: int = 1, delay_prob: float = 0.0, delta_actions: bool = False, use_phase: bool = False, delta_clip: float = 0.06, descend_boost: int = 1, carry_boost: int = 1):
        self.hist, self.chunk, self.future = hist, chunk, future
        self.delay_prob = delay_prob
        self.delta_actions = delta_actions
        self.use_phase = use_phase
        self.delta_clip = delta_clip
        self.descend_boost = max(1, int(descend_boost))
        self.carry_boost = max(1, int(carry_boost))
        dirs = [Path(d) for d in (data_dir if isinstance(data_dir, (list, tuple)) else [data_dir])]
        files = []
        for d in dirs:
            files += sorted(d.glob("ep_*.npz"))
        self.episodes = []
        for f in files:
            d = np.load(f, allow_pickle=True)
            if "overhead_jpg" not in d:
                continue
            instr = str(d["instruction"]) if "instruction" in d else "all"
            obs = d["obs"].astype(np.float32)
            ep = {
                "instruction": instr,
                "instruction_id": {"all": 0, "apple": 1, "strawberry": 2, "banana": 3, "pear": 4}.get(instr, 0),
                "obs": obs,
                "act": d["action"].astype(np.float32),
                "oh": np.stack([_decode(b) for b in d["overhead_jpg"]]),
                "wr": np.stack([_decode(b) for b in d["wrist_jpg"]]) if "wrist_jpg" in d else None,
            }
            if use_phase:
                ep["phase"] = episode_phase(obs, ep["act"], instr)
            self.episodes.append(ep)
        if not self.episodes:
            raise FileNotFoundError(f"{data_dir} 下没有带图像的 ep_*.npz")
        self.index = []
        for ei, ep in enumerate(self.episodes):
            T = len(ep["obs"])
            for t in range(self.hist - 1, T - 1, stride):
                self.index.append((ei, t))
                # 下探/闭爪窗只占一小部分帧，却是闭环成败的关键：
                # 过采样让「何时下探、何时闭爪」不被「悬停跟随」淹没
                if use_phase and self.descend_boost > 1 and t < T and ep["phase"][t] == 1:
                    self.index.extend([(ei, t)] * (self.descend_boost - 1))
                # 料框附近的减速窗同样稀少，却决定「停在框上方还是冲过去」
                if self.carry_boost > 1 and t < T:
                    bx, by = BIN_XY.get(ep["instruction"], (9.0, 9.0))
                    a = ep["act"][t]
                    if abs(a[0] - bx) < 0.12 and abs(a[1] - by) < 0.12:
                        self.index.extend([(ei, t)] * (self.carry_boost - 1))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ei, t = self.index[i]
        ep = self.episodes[ei]
        H = self.hist
        K = self.chunk
        idx = np.arange(t - H + 1, t + 1)

        oh = ep["oh"][idx].astype(np.float32) / 255.0
        wr = ep["wr"][idx].astype(np.float32) / 255.0 if ep["wr"] is not None else oh
        states = ep["obs"][idx].copy()
        phase_chunk = None
        if self.use_phase:
            ph = ep["phase"][idx]
            states = np.concatenate(
                [states, np.eye(3, dtype=np.float32)[ph]], axis=1)
            pc = ep["phase"][t:t + K].copy()
            if len(pc) < K:
                pc = np.concatenate([pc, np.repeat(pc[-1:], K - len(pc))])
            phase_chunk = pc
        # 延迟增强：随机把最后一帧替换成上一帧，模拟 1 帧推理/感知延迟
        if self.delay_prob > 0 and np.random.rand() < self.delay_prob:
            if len(oh) > 1:
                oh[-1] = oh[-2]
                states[-1] = states[-2]

        # 动作块（末尾重复）
        acts = ep["act"][t:t + K].copy()
        if len(acts) < K:
            acts = np.concatenate([acts, np.repeat(acts[-1:], K - len(acts), axis=0)])
        acts[:, 3] = (acts[:, 3] < 0).astype(np.float32)  # 夹爪：闭=1 / 开=0
        if self.delta_actions:
            # 相对当前 TCP 的增量 = **实际执行的位移**（与执行端同样裁剪到 ±max_step）。
            # 用未裁剪的原始目标差会让每一步都饱和成 6cm（收敛不了），
            # 而执行位移在到位后自然衰减到 0，闭环因此有负反馈。
            # 每个 chunk 步用各自的 TCP（t+k），而不是统一起点。
            tcp_future = ep["obs"][t:t + K, :3].copy()
            if len(tcp_future) == 0:
                tcp_future = ep["obs"][t, :3][None].copy()
            if len(tcp_future) < K:
                tcp_future = np.concatenate(
                    [tcp_future, np.repeat(tcp_future[-1:], K - len(tcp_future), axis=0)])
            acts[:, :3] = np.clip(acts[:, :3] - tcp_future,
                                  -self.delta_clip, self.delta_clip)

        # 未来 H 步的 4 类 [pos(3)+vel(3)]
        F = self.future
        fut = np.zeros((F, 24), dtype=np.float32)
        for k in range(F):
            tt = min(t + 1 + k, len(ep["obs"]) - 1)
            o = ep["obs"][tt]
            for j in range(4):
                base = 17 + 8 * j
                fut[k, j * 6:(j + 1) * 6] = o[base:base + 6]

        out = {
            "oh": torch.from_numpy(oh.transpose(0, 3, 1, 2)),   # (T,3,H,W)
            "wr": torch.from_numpy(wr.transpose(0, 3, 1, 2)),
            "state": torch.from_numpy(states),
            "act": torch.from_numpy(acts),
            "future": torch.from_numpy(fut),
            "kind": torch.tensor(ep["instruction_id"], dtype=torch.long),
        }
        if phase_chunk is not None:
            out["phase_chunk"] = torch.from_numpy(phase_chunk.astype(np.int64))
        return out
