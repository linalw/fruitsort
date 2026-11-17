PY ?= python3
NAME ?= exp
ENV ?= maniskill3
CFG ?= configs/base.yaml

.PHONY: help doctor digest run demo bench check clean tree

help:
	@echo "make doctor            环境体检 -> docs/env-report.md"
	@echo "make digest            抓 arXiv 新论文 -> notes/digests/"
	@echo "make run NAME=xxx      新建实验目录（自动继承 $(CFG)）"
	@echo "make demo              跑一段苹果+草莓演示（不录像）"
	@echo "make bench             跑 pair 基准（10 episodes）"
	@echo "make check             语法检查 scripts/ 与 src/"
	@echo "make tree              查看目录结构"
	@echo "环境: source scripts/env.sh [$(ENV)]"

doctor:
	bash scripts/doctor.sh --write

digest:
	$(PY) scripts/arxiv_digest.py

run:
	$(PY) scripts/new_run.py --name $(NAME) -c $(CFG)

demo:
	$(PY) scripts/demo.py --kinds apple strawberry --belt-speed 0.06

bench:
	$(PY) scripts/benchmark.py --scenario pair --episodes 10 --belt-speed 0.06

check:
	$(PY) -m compileall -q scripts src && echo "✓ 语法检查通过"

tree:
	@find . -maxdepth 2 -not -path './.git*' -not -path '*/__pycache__*' | sort

clean:
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
