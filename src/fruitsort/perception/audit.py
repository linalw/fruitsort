"""真值审计：统计"谁在读仿真真值"。

用法（在真实评测流程里挂上，不改行为）::

    FRUITSORT_AUDIT_GT=1 python scripts/stream_eval.py --perception rgbdv2 ...

结束时打印按"调用方文件"聚合的次数，并标出哪些属于**模型/策略侧**
（`src/fruitsort/{policies,perception,learning}`）——那些才是"部署到真机就没有"的东西；
`src/fruitsort/sim` 的调用是环境自身（物理、回收、评测记账），真机上由相机与产线代替。

监视三个入口：
* `u.fruit_actors`    —— 每类水果的真实 actor（含位姿/速度）
* `u._slots`          —— 产线的槽位表（谁在飞、谁已送达）
* `u.agent.is_grasping` —— 真实接触判定（真机上对应力/触觉，但不是"真值"意义上的物体状态）
"""
from __future__ import annotations

import traceback
from collections import Counter

_STATE: dict = {"on": False, "hits": Counter(), "detail": Counter(), "depth": 0}


class _CountingDict(dict):
    def __init__(self, data, tag: str):
        super().__init__(data)
        self._tag = tag

    def _note(self):
        if _STATE["depth"] > 0:
            return
        _STATE["depth"] += 1
        try:
            stack = traceback.extract_stack(limit=10)[:-1]
        finally:
            _STATE["depth"] -= 1
        for fr in reversed(stack):
            fn = fr.filename
            if fn.endswith("audit.py"):
                continue
            short = fn.split("project/Fruitsort/")[-1]
            _STATE["hits"][short] += 1
            _STATE["detail"][f"{short}:{fr.lineno}:{self._tag}"] += 1
            return

    def __getitem__(self, k):
        self._note()
        return super().__getitem__(k)

    def get(self, k, default=None):
        self._note()
        return super().get(k, default)

    def items(self):
        self._note()
        return super().items()

    def values(self):
        self._note()
        return super().values()

    def keys(self):
        self._note()
        return super().keys()

    def __iter__(self):
        self._note()
        return super().__iter__()


def install(u) -> None:
    if _STATE["on"]:
        return
    _STATE["on"] = True
    u.fruit_actors = _CountingDict(dict(u.fruit_actors), "fruit_actors")
    u._slots = _CountingDict(dict(u._slots), "_slots")
    orig = u.agent.is_grasping

    def _wrapped(*a, **kw):
        if _STATE["depth"] == 0:
            _STATE["depth"] += 1
            try:
                stack = traceback.extract_stack(limit=10)[:-1]
            finally:
                _STATE["depth"] -= 1
            for fr in reversed(stack):
                fn = fr.filename
                if fn.endswith("audit.py"):
                    continue
                short = fn.split("project/Fruitsort/")[-1]
                _STATE["hits"][short] += 1
                _STATE["detail"][f"{short}:{fr.lineno}:is_grasping"] += 1
                break
        return orig(*a, **kw)

    u.agent.is_grasping = _wrapped


def report(model_side_prefixes=("src/fruitsort/policies", "src/fruitsort/perception",
                                "src/fruitsort/learning", "scripts/stream_eval.py")) -> None:
    if not _STATE["on"]:
        return
    hits = _STATE["hits"]
    total = sum(hits.values())
    model = {k: v for k, v in hits.items() if k.startswith(model_side_prefixes)}
    sim = {k: v for k, v in hits.items() if k not in model}
    print("\n[GT 审计] 真值入口被读取次数（按调用方文件）:")
    print(f"  模型/策略侧合计 : {sum(model.values()):6d}")
    for k, v in sorted(model.items(), key=lambda kv: -kv[1]):
        print(f"      {k:44s} {v:6d}")
    print(f"  环境(sim)侧合计 : {sum(sim.values()):6d}  ← 物理/回收/记账，真机上由产线代替")
    for k, v in sorted(sim.items(), key=lambda kv: -kv[1])[:6]:
        print(f"      {k:44s} {v:6d}")
    print(f"  总计 {total}")
    if model:
        print("  模型侧明细（前 12）:")
        for k, v in sorted(_STATE["detail"].items(), key=lambda kv: -kv[1]):
            if k.startswith(tuple(model_side_prefixes)):
                print(f"      {k:64s} {v:6d}")
    else:
        print("  ✓ 模型/策略侧没有读取任何真值入口")
