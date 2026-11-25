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
    def __init__(self, data_dir, hist: int = 5, chunk: int = 8, future: int = 4, stride: int = 1, delay_prob: float = 0.0, delta_actions: bool = False, use_phase: bool = False, use_geom: bool = False, view_clip_x: float = -9.0, obj_noise: float = 0.0, vel_noise: float = 0.0, drop_prob: float = 0.0, delta_clip: float = 0.06, descend_boost: int = 1, carry_boost: int = 1, boost_kind: str = "", kind_boost: int = 1, drop_tactile: bool = False,
                 aux_dirs=(), split_tactile: bool = False, force_label: bool = False):
        self.hist, self.chunk, self.future = hist, chunk, future
        self.delay_prob = delay_prob
        self.delta_actions = delta_actions
        self.use_phase = use_phase
        self.use_geom = use_geom
        # 相机视野裁剪：x < view_clip_x 的水果在观测里替换成"刚上料"占位
        # （与 perception/adapter.py 的未见处理完全一致）——否则真机看不到远处水果，
        # 观测会与训练不一致（实测导致策略不去拦截）。
        self.view_clip_x = float(view_clip_x)
        # 感知噪声增强：位置/速度加高斯噪声 + 偶发丢检（沿用上一帧位置），
        # 让策略习惯"1–3 cm 定位误差 / 偶尔看不到目标"的真机感知条件。
        self.obj_noise = float(obj_noise)
        self.vel_noise = float(vel_noise)
        self.drop_prob = float(drop_prob)
        self.delta_clip = delta_clip
        self.descend_boost = max(1, int(descend_boost))
        self.carry_boost = max(1, int(carry_boost))
        self.boost_kind = boost_kind
        self.drop_tactile = drop_tactile
        self.aux_dirs = {Path(d).resolve() for d in aux_dirs}
        self.split_tactile = split_tactile
        self.force_label = force_label
        self.kind_boost = max(1, int(kind_boost))
        dirs = [Path(d) for d in (data_dir if isinstance(data_dir, (list, tuple)) else [data_dir])]
        files = []
        for d in dirs:
            files += sorted(d.glob("ep_*.npz"))
        self.episodes = []
        for f in files:
            d = np.load(f, allow_pickle=True)
            if "overhead_jpg" not in d:
                continue
            aux_only = f.parent.resolve() in self.aux_dirs
            instr = str(d["instruction"]) if "instruction" in d else "all"
            obs = d["obs"].astype(np.float32)
            ep = {
                "instruction": instr,
                "instruction_id": {"all": 0, "apple": 1, "strawberry": 2, "banana": 3, "pear": 4}.get(instr, 0),
                "obs": obs,
                "act": d["action"].astype(np.float32),
                "oh": np.stack([_decode(b) for b in d["overhead_jpg"]]),
                "wr": np.stack([_decode(b) for b in d["wrist_jpg"]]) if "wrist_jpg" in d else None,
                "roi": (np.stack([_decode(b) for b in d["roi_jpg"]])
                        if ("roi_jpg" in d and len(d["roi_jpg"]) > 0) else None),
                "aux_only": bool(aux_only),
            }
            if use_phase:
                ep["phase"] = episode_phase(obs, ep["act"], instr)
            _roi = ep.get("roi")
            if _roi is not None and len(_roi) != len(obs):
                _n = len(obs)
                if len(_roi) > _n:
                    _roi = _roi[: _n]
                else:
                    _roi = np.concatenate(
                        [_roi, np.repeat(_roi[-1:], _n - len(_roi), axis=0)], axis=0)
                ep["roi"] = _roi
            if obs.shape[1] >= 58:
                # 触觉布局：[49 base][9 tactile]；index 54=v_slip, 55/56=左右指接触
                # ---------------- 掉落起始标签（替代滑移标签）----------------
                # 依据"物体离开夹爪"而不是"有相对速度"：滑移≠失败（草莓成功搬运时中位滑移
                # 就有 12.4cm/s）。标签 = 当前贴着 TCP(d<6cm) 且 5~20 帧内 d 会超过 10cm。
                j = ALL.index(instr) if instr in ALL else 0
                d = np.linalg.norm(obs[:, 17 + 8 * j:17 + 8 * j + 3] - obs[:, :3], axis=1)
                now = ((d < 0.06) & (obs[:, 17 + 8 * j + 7] > 0.5)).astype(np.float32)
                ep["slip_now"] = now
                # 监督标签 = **即将滑移的起始**（当前没滑，但 5~20 帧内会滑）。
                # 关键：如果标签用 t+1..t+10 的 max、而 v_slip 又在观测里，模型直接读数就能拿到
                # 标签（AUROC 会退化成 1.0，见 docs 6.16.2）。改成"起始事件 + 5 帧盲区"后，
                # 模型必须从触觉历史/图像里推断"这一抓会不会滑"。
                T = len(now)
                fut = np.zeros(T, dtype=np.float32)
                for i in range(T):
                    lo, hi = i + 5, min(i + 21, T)
                    fut[i] = float((d[lo:hi] > 0.10).any()) if hi > lo else 0.0
                ep["drop_onset"] = (fut * now).astype(np.float32)
                ep["slip_onset"] = ep["drop_onset"]

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
                # 弱类过采样：某类样本太少时（如梨），把它的窗口重复 N 次
                if (self.kind_boost > 1 and self.boost_kind
                        and ep["instruction"] == self.boost_kind):
                    self.index.extend([(ei, t)] * (self.kind_boost - 1))
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
        tactile = None
        if states.shape[1] >= 58:
            if self.split_tactile:
                tactile = states[:, 49:58].copy()
                states = np.concatenate([states[:, :49], states[:, 58:]], axis=1)
        if self.drop_tactile and states.shape[1] >= 58:
            # 消融：丢掉 49:58 的触觉列（同一份数据 → 干净的 A/B）
            states = np.concatenate([states[:, :49], states[:, 58:]], axis=1)
        phase_chunk = None
        if self.use_phase:
            ph = ep["phase"][idx]
            states = np.concatenate(
                [states, np.eye(3, dtype=np.float32)[ph]], axis=1)
            pc = ep["phase"][t:t + K].copy()
            if len(pc) < K:
                pc = np.concatenate([pc, np.repeat(pc[-1:], K - len(pc))])
            phase_chunk = pc
        # 队列条件（**放在 phase 之后**，与 runner 一致）：场上 |x|<0.6 且在带面的水果数 /4
        _b = states[:, :49] if states.shape[1] >= 49 else states
        _qx = _b[:, [17 + 8 * k for k in range(4)]]
        _qz = _b[:, [19 + 8 * k for k in range(4)]]
        _q = (((np.abs(_qx) < 0.6) & (_qz > 0.40) & (_qz < 0.60)).sum(1, keepdims=True)
              / 4.0).astype(np.float32)
        states = np.concatenate([states, _q], axis=1)
        if (self.obj_noise > 0 or self.vel_noise > 0 or self.drop_prob > 0):
            Tn = states.shape[0]
            for _k in range(4):
                _o = 17 + 8 * _k
                if self.obj_noise > 0:
                    states[:, _o:_o + 2] += np.random.normal(0, self.obj_noise,
                                                             (Tn, 2)).astype(np.float32)
                    states[:, _o + 2] += np.random.normal(0, self.obj_noise * 0.5,
                                                          Tn).astype(np.float32)
                if self.vel_noise > 0:
                    states[:, _o + 3:_o + 5] += np.random.normal(0, self.vel_noise,
                                                                 (Tn, 2)).astype(np.float32)
                if self.drop_prob > 0:
                    d = np.random.rand(Tn) < self.drop_prob
                    if bool(d.any()):
                        src = np.concatenate([states[:1, _o:_o + 6],
                                              states[:-1, _o:_o + 6]], axis=0)
                        states[d, _o:_o + 6] = src[d]
        if self.view_clip_x > -8.0:
            _belt = states[:, 16].copy()
            for _k in range(4):
                _o = 17 + 8 * _k
                _far = states[:, _o] < self.view_clip_x
                if not bool(_far.any()):
                    continue
                states[_far, _o] = self.view_clip_x
                states[_far, _o + 1] = 0.0
                states[_far, _o + 3] = _belt[_far]      # vx = 带速
                states[_far, _o + 4] = 0.0
                states[_far, _o + 5] = 0.0
        # 抓取几何（显式给"TCP 相对目标水果"的偏差）：模型原本要自己从
        # 两个绝对位置里减出来，回归头很难学到"差 6 cm 该继续下沉"这种量。
        if self.use_geom:
            _tcp = states[:, 0:3]
            _instr = ep["instruction"]
            j = ALL.index(_instr) if _instr in ALL else 0
            _fp = states[:, 17 + 8 * j: 17 + 8 * j + 3]
            _d = (_tcp - _fp).astype(np.float32)
            _norm = np.linalg.norm(_d, axis=1, keepdims=True).astype(np.float32)
            states = np.concatenate([states, _d, _norm], axis=1)
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
        if self.force_label and ep["obs"].shape[1] >= 58:
            # 第 5 维 = 目标夹持力（示教里实测 Fn / 40N → [0,1]），chunk 内前向填充
            f = ep["obs"][t:t + K, 49] / 40.0
            if len(f) < K:
                f = np.concatenate([f, np.repeat(f[-1:], K - len(f))])
            acts = np.concatenate([acts, np.clip(f, 0, 1)[:, None]], axis=1)
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
        if tactile is not None:
            out["tactile"] = torch.from_numpy(tactile.astype(np.float32))
        if ep.get("roi") is not None:
            roi = ep["roi"][idx].astype(np.float32) / 255.0
            out["roi"] = torch.from_numpy(roi.transpose(0, 3, 1, 2))
        out["aux_only"] = torch.tensor(float(ep.get("aux_only", False)))
        if "slip_onset" in ep:
            out["slip"] = torch.tensor(float(ep["slip_onset"][t]))
        if phase_chunk is not None:
            out["phase_chunk"] = torch.from_numpy(phase_chunk.astype(np.int64))
        return out
