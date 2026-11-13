---
date: 2026-07-31
url:
title: "Teaching the auto-optimization loop to write Pallas kernels"
related_post: 2026-07-31-teaching-the-autoresearch-loop-to-write-pallas-kernels
post_url: https://vlasenkoalexey.github.io/2026/07/teaching-the-autoresearch-loop-to-write-pallas-kernels/
---

Kernel optimization is now part of my TPU auto-optimization project: the same autoresearch loop that tunes models now writes and optimizes hand-written Pallas kernels — unattended, from a single prompt, on Claude Code, Codex and Antigravity.

Same loop, one level down: hypothesis → code change → measurement → verdict → updated priors. A kernel experiment is one Python module and one TPU chip, so it runs locally on a Cloud TPU VM.

Three things worth calling out — the first two I hadn't planned for.

**Recursive self-improvement, narrow but real.** Kernel experiments are cheap enough to run two loops. The inner one hill-climbs on kernel code. When it finishes, an outer loop reads its session log and asks a different question: what made this hill climb slower than it needed to be, and what knowledge was missing from the wiki? It edits the process and knowledge files — all just markdown — and restarts. The model isn't improving; the weights never change. Everything around it is. A large part of what the kernel lane runs on today was written by the loop, not by me.

**A process-auditor sub-agent.** LLM workers follow a written process unevenly — they skip steps, drift, sometimes claim work they didn't do. You can't fix that with more instructions, because "audit yourself every iteration" lives in the exact document the agent is already failing to follow. But when I babysat runs and pasted corrections in by hand, the agent applied every one: the failure was never "can't follow corrections", it was "won't remember to run the check". So mechanize only the unreliable half — a cheap sub-agent, armed once at launch by the harness scheduler, that recomputes compliance from artifacts on disk and hands back paste-ready corrections. Coverage went from roughly 1-in-8 iterations to 100%, enough to make otherwise unreliable models usable autonomously.

**Verification is adversarial by construction.** The evaluator is the attack surface: a loop selecting over many candidates eventually selects for holes in its own grader. So authoring and grading are separated — whoever writes a candidate never produces the evidence its verdict cites. Every claimed speedup is independently re-measured, parity checked against a high-precision oracle, and a firing audit proves the kernel actually ran. A number without a receipt doesn't count.

On a public Pallas benchmark suite results are competitive with MaxKernel, a purpose-built kernel-generation system — every kernel authored cold, existing kernel libraries excluded.

Try it yourself: https://github.com/vlasenkoalexey/tpu_performance_autoresearch_wiki#starting-kernel-optimization-experiment
MaxKernel: https://github.com/AI-Hypercomputer/accelerator-agents/tree/main/MaxKernel

Full write-up: https://vlasenkoalexey.github.io/2026/07/teaching-the-autoresearch-loop-to-write-pallas-kernels/
Earlier posts in the series: https://vlasenkoalexey.github.io/tags/autoresearch/
