---
date: 2026-05-01
url: https://www.linkedin.com/feed/update/urn:li:activity:7455092576707342336/
title: "TPU model performance auto-optimization"
related_post: 2026-05-01-tpu-model-performance-auto-optimization
repo: https://github.com/vlasenkoalexey/tpu_performance_autoresearch_wiki
---

A big part of my work involves optimizing ML models to run efficiently on TPUs. It's not a trivial effort — every meaningful speedup requires a lot of domain knowledge spanning the compiler, the hardware, the framework internals, Pallas kernels, ML-specific knowledge and so on.

So I had an idea: what if Andrej Karpathy's autoresearch loop could automate this?

Turns out it works, but only when the autoresearch idea is extended into a more complete architecture:

Autoresearch — an optimization loop modified to tune model performance (tokens/sec, MFU), instead of optimizing how fast a model can train. It proposes ranked, falsifiable experiments, runs them on TPU, and either accepts or rejects them based on results. See https://lnkd.in/ecZE8hiG

LLM wiki — a knowledge base of TPU-specific domain information that informs every hypothesis (papers, codebases, kernels, plus the running record of every experiment). It also documents how to profile TPU models and interpret traces. See https://lnkd.in/efKPN3BA

XProf MCP server — an observer that grounds each iteration in real signal. For TPU perf, that's XProf wrapped as MCP, so the agent can read profiles, attribute ops back to HLO and source lines, and verify every claimed win against a real trace. See https://lnkd.in/eftmRcCt

When those ideas were combined together, everything worked like magic.
Repo is available at: https://lnkd.in/ebywzwua

In less than a day, the agent ran 60+ experiments with minimal supervision and achieved performance comparable to the highly-optimized Llama3 8B model implementations in the MaxText repo (see wiki/experiments/llama3_8B_autoresearch_optimization folder).
