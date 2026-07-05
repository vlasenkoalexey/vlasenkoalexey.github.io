---
date: 2026-07-04
url:
title: "wikify repo — turning codebases into grounded LLM wikis"
related_post: 2026-07-04-wikify-turning-codebases-into-grounded-llm-wikis
post_url: https://vlasenkoalexey.github.io/2026/07/wikify-turning-codebases-into-grounded-llm-wikis/
---

Your agent doesn't need to retrieve fragments about your code. It needs a map of it.

The TPU auto-optimization loop from my earlier posts runs on an LLM wiki instead of RAG: knowledge is compiled once at ingest into plain, cross-linked markdown, and every query after that is a cheap read. No vector database, no black box — a git repo anyone can clone, diff, and point any agent at.

But the most valuable "documents" in an optimization project are codebases — the model, the framework, the reference implementations. And naive ingestion fails on code: LLM summaries drift, and AST-based tools only see that a name appears, not which symbol it refers to.

wikify-repo fixes both:

- SCIP indexing — the real compiler/type-checker resolves every symbol, cross-file reference, and call path
- The LLM annotates only the most central ~20% of the code that explains ~80% of it; everything else gets a deterministic catalog page, so nothing is dropped
- A citation linter as a hard build gate — every claim must cite a compiler-resolved symbol, or the page doesn't ship
- Output is plain markdown in your own repo — shareable, diffable, readable by Claude Code, Codex, or Antigravity with nothing installed

Fitting test: I pointed it at the alternative code-mapping tools themselves — the comparison wiki cites their actual implementations.

Full write-up, comparison table, and the template to start your own: https://vlasenkoalexey.github.io/2026/07/wikify-turning-codebases-into-grounded-llm-wikis/

The tool: https://github.com/vlasenkoalexey/wikify-repo
Demo / template: https://github.com/vlasenkoalexey/wikify-repo-demo

Fourth post in the series:
- Part 1 — https://vlasenkoalexey.github.io/2026/05/tpu-model-performance-auto-optimization/
- Part 2 — https://vlasenkoalexey.github.io/2026/06/making-karpathy-autoresearch-production-ready/
- Part 3 — https://vlasenkoalexey.github.io/2026/06/making-tpu-auto-optimization-work-with-other-agents/
