#!/usr/bin/env python3
"""arXiv 日报：抓取 cs.RO 最新论文，按具身智能/分拣关键词过滤、打分、去重。

只用标准库，任何 conda 环境（甚至 base）都能跑。

用法:
    python scripts/arxiv_digest.py                 # 最近 3 天，写入 notes/digests/<日期>.md
    python scripts/arxiv_digest.py --days 7 --max 300
    python scripts/arxiv_digest.py --query 'abs:"fruit" AND cat:cs.RO'
    python scripts/arxiv_digest.py --dry-run       # 只打印不写文件
    python scripts/arxiv_digest.py --show-seen     # 连之前看过的也输出
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIGEST_DIR = ROOT / "notes" / "digests"
STATE_FILE = DIGEST_DIR / ".seen.json"
API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}

# 与研究主题相关的关键词：命中越多，排序越靠前
DEFAULT_KEYWORDS = [
    "fruit", "crop", "harvest", "agricultur", "sorting", "bin picking",
    "grasp", "pick-and-place", "dexterous", "manipulation", "robot arm",
    "vision-language-action", "vla", "imitation learning", "diffusion policy",
    "sim-to-real", "sim2real", "tactile", "embodied", "world model",
]
# 越具体的词权重越高
WEIGHTS = {
    "fruit": 5, "crop": 4, "harvest": 4, "agricultur": 4, "sorting": 3,
    "bin picking": 3, "vision-language-action": 3, "vla": 2, "grasp": 2,
    "diffusion policy": 2, "imitation learning": 2, "sim-to-real": 2,
    "sim2real": 2, "tactile": 2, "embodied": 1, "manipulation": 1,
}

DEFAULT_QUERY = (
    'cat:cs.RO AND (abs:"grasp" OR abs:"manipulation" OR abs:"imitation learning" '
    'OR abs:"vision-language-action" OR abs:"pick-and-place" OR abs:"fruit")'
)


def fetch(query: str, max_results: int, sort: str = "submittedDate", retries: int = 4) -> bytes:
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": sort,
        "sortOrder": "descending",
    }
    url = f"{API}?{urllib.parse.urlencode(params)}"
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "fruitsort-digest/1.0"})
            with urllib.request.urlopen(req, timeout=45) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504):
                wait = 10 * (2**attempt)
                print(f"[digest] HTTP {e.code}，{wait}s 后重试（{attempt + 1}/{retries}）", file=sys.stderr)
                time.sleep(wait)
                continue
            raise
        except Exception as e:  # noqa: BLE001
            last = e
            wait = 5 * (attempt + 1)
            print(f"[digest] 网络错误 {e}，{wait}s 后重试", file=sys.stderr)
            time.sleep(wait)
    raise SystemExit(
        f"[digest] 抓取失败: {last}\n"
        "提示: 出口 IP 可能被 arXiv 限流。稍后再试，或用 webfetch/browser 打开 "
        "https://arxiv.org/list/cs.RO/recent"
    )


def text(el, tag: str) -> str:
    # 注意: 必须带 a: 前缀，ElementTree 不会把默认命名空间套到无前缀路径上
    node = el.find(f"a:{tag}", NS)
    return re.sub(r"\s+", " ", (node.text or "")).strip() if node is not None else ""


def parse(xml_bytes: bytes) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    out = []
    for e in root.findall("a:entry", NS):
        raw_id = text(e, "id")
        m = re.search(r"abs/([^v]+)(v\d+)?$", raw_id)
        if not m:
            continue
        cats = [c.get("term") for c in e.findall("a:category", NS) if c.get("term")]
        out.append({
            "id": m.group(1),
            "version": (m.group(2) or "").lstrip("v"),
            "title": text(e, "title"),
            "abstract": text(e, "summary"),
            "published": text(e, "published")[:10],
            "updated": text(e, "updated")[:10],
            "authors": [text(a, "name") for a in e.findall("a:author", NS)],
            "categories": cats,
            "pdf": f"https://arxiv.org/pdf/{m.group(1)}",
            "abs": f"https://arxiv.org/abs/{m.group(1)}",
        })
    return out


def score(paper: dict, keywords: list[str]) -> tuple[int, list[str]]:
    hay = (paper["title"] + " " + paper["abstract"]).lower()
    hits = [k for k in keywords if k.lower() in hay]
    s = sum(WEIGHTS.get(k.lower(), 1) for k in hits)
    if any(k in paper["title"].lower() for k in keywords):
        s += 2  # 标题命中更重要
    return s, hits


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default=DEFAULT_QUERY, help="arXiv search_query")
    ap.add_argument("--days", type=int, default=3, help="只看最近 N 天提交/更新（默认 3）")
    ap.add_argument("--max", type=int, default=200, help="最多抓取条数（默认 200）")
    ap.add_argument("--min-score", type=int, default=2, help="最低相关度（默认 2）")
    ap.add_argument("--keywords", default="", help="逗号分隔，覆盖默认关键词")
    ap.add_argument("--limit", type=int, default=25, help="日报最多列出的篇数")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--show-seen", action="store_true")
    ap.add_argument("--out", default="", help="输出 md 路径")
    args = ap.parse_args()

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()] or DEFAULT_KEYWORDS
    print(f"[digest] 查询: {args.query}", file=sys.stderr)
    papers = parse(fetch(args.query, args.max))

    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=args.days)
    # 本机时钟可能超前，或 arXiv 索引滞后：以最新论文日期为基准更稳
    dates = [p["published"] for p in papers if p["published"]]
    if dates:
        newest = datetime.strptime(max(dates), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if (cutoff_dt - newest).days > 7:
            cutoff_dt = newest - timedelta(days=args.days)
            print(f"[digest] 时钟/索引滞后，基准日改为 {newest.date()}", file=sys.stderr)
    cutoff = cutoff_dt.strftime("%Y-%m-%d")
    state = load_state()

    picked, seen_n, old_n = [], 0, 0
    for p in papers:
        if p["published"] < cutoff and p["updated"] < cutoff:
            old_n += 1
            continue
        s, hits = score(p, keywords)
        if s < args.min_score:
            continue
        p["score"], p["hits"] = s, hits
        if p["id"] in state and not args.show_seen:
            seen_n += 1
            continue
        picked.append(p)

    picked.sort(key=lambda p: (-p["score"], p["published"]), reverse=False)
    picked = picked[: args.limit]

    today = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"# arXiv 日报 {today}",
        "",
        f"- 抓取: {len(papers)} 篇（最近 {args.days} 天窗口内 {len(papers) - old_n} 篇）",
        f"- 相关: {len([p for p in picked])} 篇列出；已看过跳过 {seen_n} 篇",
        f"- 查询: `{args.query}`",
        "",
    ]
    if not picked:
        lines.append("_今天没有新论文。_")
    for p in picked:
        au = ", ".join(p["authors"][:4]) + (" 等" if len(p["authors"]) > 4 else "")
        lines += [
            f"## [{p['title']}]({p['abs']})",
            "",
            f"- **score {p['score']}** · {p['published']} · `{p['id']}`",
            f"- 作者: {au}",
            f"- 分类: {', '.join(p['categories'][:5])}",
            f"- 命中: {', '.join(p['hits'][:8])}",
            "",
            f"> {p['abstract'][:600]}{'…' if len(p['abstract']) > 600 else ''}",
            "",
        ]
    body = "\n".join(lines)

    if args.dry_run:
        print(body)
        return

    DIGEST_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else DIGEST_DIR / f"{today}.md"
    if out.exists() and not args.show_seen:
        out = DIGEST_DIR / f"{today}-{datetime.now().strftime('%H%M')}.md"
    out.write_text(body, encoding="utf-8")

    for p in picked:
        state.setdefault(p["id"], {"first_seen": today, "title": p["title"]})
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"✓ {len(picked)} 篇 → {out.relative_to(ROOT)}")
    for p in picked[:8]:
        print(f"  [{p['score']:>2}] {p['id']}  {p['title'][:78]}")


if __name__ == "__main__":
    main()
