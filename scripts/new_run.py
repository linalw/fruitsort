#!/usr/bin/env python3
"""新建一次实验目录，并登记到 runs/registry.csv。

用法:
    python scripts/new_run.py --name baseline-dp --tag baseline
    python scripts/new_run.py --name vla-ft -c configs/base.yaml -c configs/local.yaml
    python scripts/new_run.py --name smoke --note "先跑通链路"

生成:
    runs/<YYYY-MM-DD>-<slug>/
        meta.json     机器/环境/git 快照
        config.yaml   合并后的配置（后者覆盖前者）
        cmd.txt       启动命令
        metrics.json  空指标（JSON）
        notes.md      实验记录模板
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
REGISTRY = RUNS / "registry.csv"


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", s)
    return re.sub(r"-{2,}", "-", s).strip("-") or "run"


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def read_simple_yaml(path: Path) -> dict:
    """极简 YAML（仅 key: value / 一层嵌套），避免强依赖 PyYAML。"""
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        data: dict = {}
        cur = data
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"[new_run] 解析 {path} 失败: {e}") from e
    else:
        return data

    # 无 PyYAML 时的退化实现
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, _, val = line.strip().partition(":")
        if not _:
            continue
        if indent == 0 and val.strip() == "":
            cur = data.setdefault(key.strip(), {})
        else:
            v = val.strip().strip("'\"")
            if v.lower() in ("true", "false"):
                cur[key.strip()] = v.lower() == "true"
            else:
                try:
                    cur[key.strip()] = float(v) if "." in v else int(v)
                except ValueError:
                    cur[key.strip()] = v
        if indent == 0:
            cur = data
    return data


def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def dump_yaml(data: dict) -> str:
    try:
        import yaml  # type: ignore

        return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    except ImportError:
        def walk(d: dict, ind: int = 0) -> list[str]:
            lines = []
            pad = "  " * ind
            for k, v in d.items():
                if isinstance(v, dict):
                    lines.append(f"{pad}{k}:")
                    lines += walk(v, ind + 1)
                else:
                    lines.append(f"{pad}{k}: {v!r}")
            return lines

        return "\n".join(walk(data)) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="实验简述，如 baseline-dp")
    ap.add_argument("-c", "--config", action="append", default=[], help="配置 yaml，可多次，后者覆盖前者")
    ap.add_argument("--tag", default="", help="标签，逗号分隔")
    ap.add_argument("--note", default="", help="一句话备注")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="同名同日则覆盖")
    args = ap.parse_args()

    day = time.strftime("%Y-%m-%d")
    run_id = f"{day}-{slugify(args.name)}"
    run_dir = RUNS / run_id
    if run_dir.exists() and not args.force:
        raise SystemExit(f"[new_run] 已存在: {run_dir.relative_to(ROOT)}（加 --force 覆盖）")
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg: dict = {}
    for c in args.config:
        p = Path(c)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise SystemExit(f"[new_run] 配置不存在: {c}")
        cfg = deep_merge(cfg, read_simple_yaml(p) or {})
    cfg.setdefault("run", {})
    cfg["run"].update({"id": run_id, "name": args.name, "seed": args.seed,
                       "tags": [t.strip() for t in args.tag.split(",") if t.strip()],
                       "note": args.note, "created": time.strftime("%Y-%m-%dT%H:%M:%S")})
    (run_dir / "config.yaml").write_text(dump_yaml(cfg), encoding="utf-8")

    meta = {
        "run_id": run_id,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cwd": os.getcwd(),
        "argv": sys.argv,
        "host": socket.gethostname(),
        "user": os.environ.get("USER", ""),
        "python": sys.version.split()[0],
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "git_commit": sh(["git", "rev-parse", "HEAD"]),
        "git_branch": sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(sh(["git", "status", "--porcelain"])),
        "gpu": sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                   "--format=csv,noheader"]),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "cmd.txt").write_text(
        " ".join(shlex.quote(x) for x in sys.argv) + "\n# cwd: " + os.getcwd() + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "notes.md").write_text(
        f"""# {run_id}

## 假设 / 目的

{args.note or "（待填）"}

## 配置

来源: {' '.join(args.config) if args.config else '（无）'}

## 运行

```bash
# 在这里补上真正的训练/评测命令（用 cmd.txt 里的 python 入口）
```

## 观察

- 待填

## 结论 / 下一步

- 待填
""",
        encoding="utf-8",
    )
    (run_dir / "artifacts").mkdir(exist_ok=True)

    new = not REGISTRY.exists()
    with open(REGISTRY, "a", encoding="utf-8") as f:
        if new:
            f.write("run_id,created,tags,config,metrics,note\n")
        f.write(f"{run_id},{meta['created']},\"{args.tag}\",{args.config[-1] if args.config else ''},,"
                f"\"{args.note}\"\n")

    print(f"✓ 创建实验: {run_dir.relative_to(ROOT)}")
    print(f"  下一步: 编辑 {run_dir.relative_to(ROOT)}/notes.md 记录假设，跑完更新 metrics.json")
    print(f"  登记表: {REGISTRY.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
