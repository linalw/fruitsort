#!/usr/bin/env python
"""解析 `stream_eval.py --trace-close` 的输出，统计"闭爪瞬间的落点高度/水平误差"。

用法:
    python scripts/analyze_close.py trace_gate.txt [trace_base.txt ...]

每个文件按 `[close]` 行统计 |dz|（TCP 相对果心的高度，正=在果子上方）与 |dxy|。
判据：**闭爪落在果心 ±1.5 cm 内**算"高度对上了"；否则算"提前/滞后闭爪"。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

LINE = re.compile(r"\[close\] t=\s*(?P<t>[-\d.]+)\s+(?P<kind>\w+)\s+ph=(?P<ph>\d+)\s+"
                  r"fruit_x=(?P<fx>[-\d.]+)\s+d=\(\s*(?P<dx>[+\-\d.]+),\s*(?P<dy>[+\-\d.]+),\s*(?P<dz>[+\-\d.]+)\)cm")


def parse(path: Path) -> list[dict]:
    out = []
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = LINE.search(ln)
        if m:
            out.append(dict(t=float(m["t"]), kind=m["kind"], ph=int(m["ph"]),
                            fx=float(m["fx"]),
                            dx=float(m["dx"]), dy=float(m["dy"]), dz=float(m["dz"])))
    return out


def report(name: str, rows: list[dict]) -> None:
    if not rows:
        print(f"{name:28s} 无 close 事件")
        return
    dz = np.array([r["dz"] for r in rows])
    dxy = np.array([np.hypot(r["dx"], r["dy"]) for r in rows])
    ok = np.abs(dz) <= 1.5
    # 每个事件之间至少隔 1 步；0.05 s/步
    print(f"{name:28s} n={len(rows):3d}  |dz| 中位 {np.median(np.abs(dz)):5.2f} cm "
          f"P90 {np.percentile(np.abs(dz), 90):5.2f}  |dxy| 中位 {np.median(dxy):4.2f} cm  "
          f"落在 ±1.5cm {ok.mean()*100:5.1f}%  (dz>0 = 在果子上方闭爪)")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    for a in sys.argv[1:]:
        report(Path(a).name, parse(Path(a)))


if __name__ == "__main__":
    main()
