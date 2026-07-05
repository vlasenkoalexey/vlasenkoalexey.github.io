---
date: 2026-07-04
url:
title: "wikify repo — turning codebases into grounded LLM wikis"
related_post: 2026-07-04-wikify-turning-codebases-into-grounded-llm-wikis
post_url: https://vlasenkoalexey.github.io/2026/07/wikify-turning-codebases-into-grounded-llm-wikis/
---

Your agent doesn't need to grep your code. It needs a map of it.

Out of the box an LLM knows your codebases as fuzzy training-data memories of some older version. Giving the agent a checkout helps, but grep returns text matches, not understanding — and every session re-discovers the same code structure from scratch. Ingesting the codebase into an LLM wiki — an annotated map the agent reads before touching the source — is what unlocks the real capabilities:

- Efficient navigation: overview page → concept page → exact file and line. Minimal context, no directory archaeology.
- Grounded internals answers: "how does chunked cross-entropy work here?" answered from pages that cite real symbols, one hop from pinned source.
- Cross-repo stack traces: follow a crash or a suspicious HLO op end-to-end — TorchTitan → PyTorch → TorchTPU → XLA → driver.
- Extracting optimizations from reference implementations: kernels and sharding tricks in MaxText become named, citable concepts to compare your model against.
- Migration between codebases: HuggingFace PyTorch → JAX, or MaxText → TorchTitan, becomes a mapping between two grounded descriptions.

Getting there took three attempts.

First the naive method: just tell the agent to "ingest this codebase" like an article. Better than nothing — but shallow, wasteful, and the summaries drift the moment the code changes.

Then off-the-shelf tools (graphify, understand-anything). Two dealbreakers: they store results in proprietary black-box formats you can't share as a git repo, and they parse code with AST — syntactic only, so they see that a name appears, not which symbol it refers to.

So I built wikify-repo. Why it's better:

- SCIP instead of AST — the real compiler/type-checker resolves every symbol, cross-file reference, and call path
- LLM annotates only the ~20% most central code that explains ~80% of the repo; the rest gets deterministic catalog pages, so nothing is dropped
- A citation linter as a hard build gate — every claim must cite a compiler-resolved symbol or the page doesn't ship
- Output is plain markdown in your own repo — shareable, diffable, readable by Claude Code, Codex, or Antigravity with nothing installed

Fitting test: I pointed it at the alternative code-mapping tools themselves — the comparison wiki cites their actual implementations.

Full write-up: https://vlasenkoalexey.github.io/2026/07/wikify-turning-codebases-into-grounded-llm-wikis/

The tool: https://github.com/vlasenkoalexey/wikify-repo
Demo / template: https://github.com/vlasenkoalexey/wikify-repo-demo

Fourth post in the series:
- Part 1 — https://vlasenkoalexey.github.io/2026/05/tpu-model-performance-auto-optimization/
- Part 2 — https://vlasenkoalexey.github.io/2026/06/making-karpathy-autoresearch-production-ready/
- Part 3 — https://vlasenkoalexey.github.io/2026/06/making-tpu-auto-optimization-work-with-other-agents/
