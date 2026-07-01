---
date: 2026-05-01
url: # TODO: paste Hacker News submission permalink after posting
title: "Auto-optimizing TPU model performance with an LLM autoresearch loop"
submit_url: https://vlasenkoalexey.github.io/2026/05/tpu-model-performance-auto-optimization/
related_post: 2026-05-01-tpu-model-performance-auto-optimization
repo: https://github.com/vlasenkoalexey/tpu_performance_autoresearch_wiki
---

<!--
Submit the blog write-up URL (submit_url above) as the link — not a "Show HN",
since it needs a TPU + Claude Code to actually run. Use the `title` above as the
submission title (factual, no emoji, no hype). Post the comment below as the
first comment right after submitting. Weekday US morning (Pacific) is best.
Don't solicit upvotes anywhere. Stay in the thread for the first couple hours.
-->

Author here. I work on ML performance — making large models run efficiently on TPUs. It's specialized, iterative work: form a hypothesis, change the model code, run and profile it on a TPU, read the trace, decide what to try next.

I adapted Andrej Karpathy's autoresearch loop (https://github.com/karpathy/autoresearch) to automate that cycle for performance optimization. Three parts:

- The autoresearch loop, modified to optimize performance metrics (tokens/sec, MFU) instead of training-time research. It proposes ranked, falsifiable experiments, runs them on TPU, and accepts or rejects each based on measured results.
- An "LLM wiki" — a Karpathy-method knowledge base of TPU-specific domain info (compiler, kernels, papers) plus a running record of every experiment, so each hypothesis is informed by the prior ones.
- An XProf MCP server so the agent reads real profiles, attributes ops back to HLO and source lines, and verifies every claimed speedup against an actual trace rather than trusting its own guess.

In one run it ran 60+ experiments with minimal supervision and reached throughput comparable to the hand-optimized Llama3 8B implementation in MaxText.

Repo: https://github.com/vlasenkoalexey/tpu_performance_autoresearch_wiki

The hardest part wasn't getting it to work once — it was making the loop run unsupervised for hours without drifting, stopping early, or tunneling into a dead end. I wrote that part up separately. Happy to answer questions.
