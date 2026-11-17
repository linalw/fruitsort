"""评测指标：成功率置信区间与 episode 汇总。纯标准库。"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval，小样本比正态近似可靠。返回 (lo, hi)。"""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def summarize(episodes: Iterable[dict]) -> dict:
    """episodes: [{"success": bool, "duration_s": float, "failure": str|None}, ...]"""
    eps = list(episodes)
    n = len(eps)
    if n == 0:
        return {"n_episodes": 0, "success_rate": 0.0}
    k = sum(1 for e in eps if e.get("success"))
    durs = [float(e["duration_s"]) for e in eps if e.get("duration_s") is not None]
    failures = Counter(e.get("failure") for e in eps if not e.get("success") and e.get("failure"))
    lo, hi = wilson_ci(k, n)
    return {
        "n_episodes": n,
        "n_success": k,
        "success_rate": round(k / n, 4),
        "success_rate_ci95": [round(lo, 4), round(hi, 4)],
        "avg_duration_s": round(sum(durs) / len(durs), 3) if durs else None,
        "failure_counts": dict(failures),
    }


def format_summary(s: dict) -> str:
    lo, hi = s.get("success_rate_ci95", [0, 0])
    line = (
        f"{s['n_success']}/{s['n_episodes']} = {s['success_rate'] * 100:.1f}% "
        f"(95% CI {lo * 100:.1f}–{hi * 100:.1f}%)"
    )
    if s.get("avg_duration_s") is not None:
        line += f", 平均 {s['avg_duration_s']:.2f}s/次"
    if s.get("failure_counts"):
        line += "\n失败分布: " + ", ".join(f"{k}={v}" for k, v in s["failure_counts"].items())
    return line
