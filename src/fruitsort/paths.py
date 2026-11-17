"""统一的路径解析：仓库、数据、权重、实验目录。

大数据实际落在 /media/ubuntu/66A27065A2703B9D/data/fruitsort（仓库内 data/ 是软链）。
"""
from __future__ import annotations

import os
from pathlib import Path

_ENV_ROOT = os.environ.get("FRUITSORT_ROOT")
ROOT = Path(_ENV_ROOT).resolve() if _ENV_ROOT else Path(__file__).resolve().parents[2]


def data_dir(sub: str = "") -> Path:
    """data/datasets、data/checkpoints 等。自动创建。"""
    p = ROOT / "data" / sub if sub else ROOT / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p


def ckpt_dir(name: str = "") -> Path:
    return data_dir(f"checkpoints/{name}" if name else "checkpoints")


def runs_dir() -> Path:
    p = ROOT / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def latest_run() -> Path | None:
    runs = [p for p in runs_dir().iterdir() if p.is_dir() and (p / "meta.json").exists()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def new_run_dir(name: str, force: bool = False) -> Path:
    """轻量版：只建目录，完整脚手架用 scripts/new_run.py。"""
    import time

    p = runs_dir() / f"{time.strftime('%Y-%m-%d')}-{name}"
    if p.exists() and not force:
        raise FileExistsError(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
