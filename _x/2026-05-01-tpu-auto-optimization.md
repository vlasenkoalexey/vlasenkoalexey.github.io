---
date: 2026-05-01
url: # TODO: paste X thread permalink after posting
title: "TPU model performance auto-optimization (thread)"
post_url: https://vlasenkoalexey.github.io/2026/05/tpu-model-performance-auto-optimization/
related_post: 2026-05-01-tpu-model-performance-auto-optimization
repo: https://github.com/vlasenkoalexey/tpu_performance_autoresearch_wiki
---

<!--
Post as a THREAD, not a single tweet. Attach the hero image (or a diagram) to
tweet 1 — native media is what carries reach. Keep the external link OUT of
tweet 1 (X deprioritizes lead tweets with links); put links in the last tweet.
@karpathy mention is credit-giving, not upvote-begging — keep it genuine.
-->

1/
I adapted @karpathy's autoresearch loop to automatically optimize ML model performance on TPUs.

Left it running — it ran 60+ experiments unsupervised and matched the hand-tuned Llama3 8B implementation in MaxText.

Here's how it works 🧵

2/
Making models run fast on TPUs is slow, expert, iterative work: hypothesize → change the code → run + profile on TPU → read the trace → decide what's next → repeat.

Exactly the kind of loop worth automating.

3/
The system has 3 parts.

(1) The autoresearch loop, retargeted at performance (tokens/sec, MFU). It proposes ranked, falsifiable experiments, runs them on TPU, and accepts or rejects each based on measured results.

4/
(2) An "LLM wiki" — a Karpathy-method knowledge base of TPU domain info (compiler, kernels, papers) plus a running log of every experiment, so each new hypothesis is informed by the last.

5/
(3) An XProf MCP server, so the agent reads real profiles, attributes ops back to HLO + source lines, and verifies every claimed speedup against an actual trace — instead of trusting its own guess.

6/
Getting it to work once was easy. Making it run unsupervised for hours was the real work — it drifted from instructions, stopped early to ask permission, and tunneled into dead ends.

That fix is its own thread.

7/
Write-up: https://vlasenkoalexey.github.io/2026/05/tpu-model-performance-auto-optimization/

Repo: https://github.com/vlasenkoalexey/tpu_performance_autoresearch_wiki

Built on @karpathy's autoresearch: https://github.com/karpathy/autoresearch
