---
date: 2026-06-26
url:
title: "Making TPU Model Performance Auto-optimization work with other Agents"
related_post: 2026-06-26-making-tpu-auto-optimization-work-with-other-agents
post_url: https://vlasenkoalexey.github.io/2026/06/making-tpu-auto-optimization-work-with-other-agents/
---

Original version of TPU auto-optimization loop only ran reliably on Claude Code, and this was a big blocker since I couldn't officially use it to optimize models in Google.

Now it runs on Antigravity using Gemini, and also on Codex, and most likely on any other capable LLM.

Same wiki. Same skills. Same sub-agents. Whether the harness is Claude Code, Codex, or Antigravity, the loop iterates unattended — it hypothesizes, edits real model code, launches real TPU workloads, analyzes real xprof traces, files a verdict, moves on.

The change wasn't the models. It was the shape of the loop.

Version 1 (Part 1 of the series) was the original autoresearch loop — one monolithic prompt: "read the wiki, decide on a hypothesis, edit the model, build the container, submit the workload, poll it, analyze the profile, write up the verdict." It worked, but only when I was babysitting it.

Version 2 (Part 2) added external guardrails — /loop re-injection, stop hooks, retrospectives, per-experiment forked repos — that let the loop run unattended for hours.

Version 3 formalizes the same process as a chain of specialized components:

- Skills — slash-command prompts that stay in master's context (decisions like "what to try next")
- Sub-agents — isolated context windows for long-running work (workload polling, profile analysis)
- Sub-wikis — each skill and agent loads only its own scoped topic index, not the whole knowledge base

The result is a framework that's genuinely LLM-agnostic. Different agents still have different quirks and different strengths — but the loop itself doesn't care which one is driving. Portability is the point.

Full write-up with the expanded loop diagram, the abstract skill/sub-agent pattern, and concrete code links: https://vlasenkoalexey.github.io/2026/06/making-tpu-auto-optimization-work-with-other-agents/

Third post in the series:
- Part 1 — https://vlasenkoalexey.github.io/2026/05/tpu-model-performance-auto-optimization/
- Part 2 — https://vlasenkoalexey.github.io/2026/06/making-karpathy-autoresearch-production-ready/
