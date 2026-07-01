---
date: 2026-06-05
url: # TODO: paste X thread permalink after posting
title: "Making autoresearch production-ready (thread)"
post_url: https://vlasenkoalexey.github.io/2026/06/making-karpathy-autoresearch-production-ready/
related_post: 2026-06-05-making-karpathy-autoresearch-production-ready
repo: https://github.com/vlasenkoalexey/tpu_performance_autoresearch_wiki
---

<!--
Follow-up thread to the TPU one. Post as a THREAD. Attach the "putting it
together" guardrails diagram to tweet 1 if you export it. Link out only in the
last tweet. Can be run as a standalone thread or quote-tweeted onto the first.
-->

1/
Everyone posts the demo where an AI agent runs autonomously.

Nobody posts what it takes to keep it running for hours without going off the rails.

Here's what broke when I ran a @karpathy-style autoresearch loop on TPU optimization — and how I fixed each 🧵

2/
The first two failures came down to context pollution: as the conversation grows, the loop instructions sink out of view and the agent stops paying attention to them.

3/
(1) The agent drifted from its instructions.

Fix: periodically re-inject the loop instructions back to the top of the context (the /loop skill in Claude Code).

4/
(2) The agent stopped early to ask permission — even with explicit "never stop" instructions.

Fix: a stop hook that intercepts the hand-back and pushes it to keep going.

5/
The other two were different in nature.

(3) It tunneled into one direction and declared it was "out of ideas."

Fix: a retrospective step that makes it step back and look for areas it hadn't explored yet.

6/
(4) Running several loops in parallel made them clobber each other's git branches.

Fix: give each experiment its own forked copy of the repo.

7/
Getting something working in the Software 3.0 paradigm is easy. Making it work reliably takes real engineering.

Write-up: https://vlasenkoalexey.github.io/2026/06/making-karpathy-autoresearch-production-ready/
