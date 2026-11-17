---
name: embodied-lit-review
description: 具身智能/VLA 领域的文献调研与精读流程。用 arxiv MCP 检索、下载、按节精读、做引文网络分析，用 deepwiki 读懂开源仓库，把结论沉淀成结构化笔记。适用于"找相关工作""跟踪新论文""读懂一篇论文/一个开源项目""写 related work"等任务。
---

# 具身智能文献调研

原则：**先检索、再筛选、后精读**。不要一上来就下载 PDF 全文塞进上下文。

## 1. 检索（arxiv MCP）

```text
arxiv.search_papers(query="vision language action model", max_results=20,
                    categories=["cs.RO","cs.CV","cs.LG"], sort_by="relevance")
arxiv.get_abstract(paper_id="2406.09246")      # 先看摘要，便宜
arxiv.watch_topic(topic="VLA manipulation", categories=["cs.RO"])  # 订阅日报
arxiv.check_alerts()                            # 查新论文
```

- 检索词用英文；同义词分别查一轮（VLA / vision-language-action、embodied agent、robot foundation model、diffusion policy、world model）。
- 时间敏感的调研加 `sort_by="submittedDate"`，并限定最近 6–12 个月。
- 拿到候选后，用 `get_abstract` 批量过一遍，只留 5–15 篇真正相关。

## 2. 精读（省 token 的关键）

```text
arxiv.download_paper(paper_id="...")            # 下载 + 建索引
arxiv.get_paper_outline(paper_id="...")         # 先看结构
arxiv.list_paper_latex_sections(paper_id="...") # 有哪些节
arxiv.get_paper_latex_section(paper_id="...", section="Method")
arxiv.read_paper(paper_id="...")                # 全文，最后才用
arxiv.semantic_search(query="...", ...)         # 在自己的论文库里语义检索
arxiv.search_paper_text(query="...", ...)
```

- 默认顺序：abstract → outline → 需要的 section（Method / Experiments）→ 只在必要时读全文。
- 公式和实现细节都在 LaTeX 里，比 PDF 文本可靠，优先用 `get_paper_latex_section`。

## 3. 建立领域地图

```text
arxiv.citation_graph(paper_id="...")    # 谁引用了它 / 它引用了谁
arxiv.reindex(clear_existing=false)      # 论文多了重建语义索引
arxiv.export_citations(...)              # 导出 bibtex
```

- 找 2–3 篇 survey 或高引奠基论文当种子，向两边扩（被引 = 后续工作，引用 = 前置工作）。
- 记录每个方法的 **(输入模态, 动作表示, 训练数据规模, 评测环境, 宣称增益)** 五元组，这是 related work 表格的骨架。

## 4. 读懂开源实现（deepwiki）

```text
deepwiki.read_wiki_structure(repoName="huggingface/lerobot")
deepwiki.read_wiki_contents(repoName="Physical-Intelligence/openpi")
deepwiki.ask_wiki_question(repoName="...", question="...")
```

- 适用：LeRobot、OpenPI、IsaacLab、ManiSkill、RoboTwin 这类结构复杂的仓库，比翻文件快。
- 结论要落到"我要改哪个文件/哪个函数"的层面，并顺手用 context7 查库 API。
- 需要看真实源码时再用 `clonedeps` skill 或直接 clone。

## 5. 数据集与开源权重（huggingface MCP）

论文看完要找"用什么数据训、权重在哪下"：

```text
huggingface.hub_repo_search(query="open x-embodiment", repo_types=["dataset"], sort="downloads")
huggingface.hub_repo_search(query="vla", repo_types=["model"], filters=["robotics"], sort="trendingScore")
huggingface.hub_repo_details(repo_ids=[...])       # 看 README / 文件 / license
huggingface.hf_fs(...)                              # 列目录、读文件、看 daily papers
```

- 找数据先按 `task_categories:robotics` / `Robotics` 标签过滤，再看下载量和最后更新时间。
- 注意 license 与 gated 标记；gated 仓库需要登录（`/mcps` 里给 huggingface 授权）。
- 记录到笔记时写清 repo id + revision（commit hash），否则复现不了。

## 6. 笔记规范

`notes/papers/<arxiv-id>-<slug>.md`：

```markdown
# <Title>
- arXiv: <id> · <venue/year> · <project page>
- 一句话: ...
- 问题: ...
- 方法: ...（关键公式/模块）
- 数据与评测: ...
- 结论与局限: ...
- 和我们的关系: 可复用 / 对比 baseline / 仅背景
```

`notes/related-work.md` 维护一张对比表，列 = 上面五元组 + 笔记链接。

## 反模式

- ❌ 不下载就直接编造论文内容。
- ❌ 一次读十篇全文；应当先摘要筛选。
- ❌ 只记录结论不记录评测设置——具身领域结论高度依赖环境和数据。
