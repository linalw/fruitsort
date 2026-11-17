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


class ImageDemoDataset(Dataset):
    def __init__(self, data_dir, hist: int = 5, chunk: int = 8, future: int = 4, stride: int = 1, delay_prob: float = 0.0, delta_actions: bool = False):
        self.hist, self.chunk, self.future = hist, chunk, future
        self.delay_prob = delay_prob
        self.delta_actions = delta_actions
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
            self.episodes.append({
                "instruction": instr,
                "instruction_id": {"all": 0, "apple": 1, "strawberry": 2, "banana": 3, "pear": 4}.get(instr, 0),
                "obs": d["obs"].astype(np.float32),
                "act": d["action"].astype(np.float32),
                "oh": np.stack([_decode(b) for b in d["overhead_jpg"]]),
                "wr": np.stack([_decode(b) for b in d["wrist_jpg"]]) if "wrist_jpg" in d else None,
            })
        if not self.episodes:
            raise FileNotFoundError(f"{data_dir} 下没有带图像的 ep_*.npz")
        self.index = []
        for ei, ep in enumerate(self.episodes):
            T = len(ep["obs"])
            for t in range(self.hist - 1, T - 1, stride):
                self.index.append((ei, t))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ei, t = self.index[i]
        ep = self.episodes[ei]
        H = self.hist
        idx = np.arange(t - H + 1, t + 1)

        oh = ep["oh"][idx].astype(np.float32) / 255.0
        wr = ep["wr"][idx].astype(np.float32) / 255.0 if ep["wr"] is not None else oh
        states = ep["obs"][idx].copy()
        # 延迟增强：随机把最后一帧替换成上一帧，模拟 1 帧推理/感知延迟
        if self.delay_prob > 0 and np.random.rand() < self.delay_prob:
            if len(oh) > 1:
                oh[-1] = oh[-2]
                states[-1] = states[-2]

        # 动作块（末尾重复）
        K = self.chunk
        acts = ep["act"][t:t + K].copy()
        if len(acts) < K:
            acts = np.concatenate([acts, np.repeat(acts[-1:], K - len(acts), axis=0)])
        acts[:, 3] = (acts[:, 3] < 0).astype(np.float32)  # 夹爪：闭=1 / 开=0
        if self.delta_actions:
            # 相对当前 TCP 的增量（消除"目标≈当前位姿"的悬停不动点）
            acts[:, :3] = acts[:, :3] - ep["obs"][t, :3][None, :]

        # 未来 H 步的 4 类 [pos(3)+vel(3)]
        F = self.future
        fut = np.zeros((F, 24), dtype=np.float32)
        for k in range(F):
            tt = min(t + 1 + k, len(ep["obs"]) - 1)
            o = ep["obs"][tt]
            for j in range(4):
                base = 17 + 8 * j
                fut[k, j * 6:(j + 1) * 6] = o[base:base + 6]

        return {
            "oh": torch.from_numpy(oh.transpose(0, 3, 1, 2)),   # (T,3,H,W)
            "wr": torch.from_numpy(wr.transpose(0, 3, 1, 2)),
            "state": torch.from_numpy(states),
            "act": torch.from_numpy(acts),
            "future": torch.from_numpy(fut),
            "kind": torch.tensor(ep["instruction_id"], dtype=torch.long),
        }
