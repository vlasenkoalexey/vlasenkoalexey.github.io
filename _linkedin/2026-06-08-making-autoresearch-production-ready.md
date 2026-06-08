---
date: 2026-06-08
url: https://www.linkedin.com/posts/oleksiyvlasenko_a-few-weeks-ago-i-shared-how-i-adapted-andrej-share-7469886976691879936-nZ2b/?utm_source=share&utm_medium=member_desktop&rcm=ACoAAALEEREB2wsAkwy4chr664ousfQIFHPaS84
title: "Making Andrej Karpathy's autoresearch production-ready"
related_post: 2026-06-05-making-karpathy-autoresearch-production-ready
post_url: https://vlasenkoalexey.github.io/2026/06/making-karpathy-autoresearch-production-ready/
---

A few weeks ago I shared how I adapted Andrej Karpathy's autoresearch loop to automatically optimize ML model performance on TPUs. It worked — but it needed babysitting. Link to original idea: https://vlasenkoalexey.github.io/2026/05/tpu-model-performance-auto-optimization/

This follow-up is about the unglamorous part the demos skip: what it actually took to make the loop run unsupervised for hours.

Original approach had several issues.

The first two came down to context pollution — as the conversation grows, the loop instructions sink out of view:

- The agent drifted from its instructions → fixed by periodically re-injecting the loop instructions back to the top of the context (the /loop skill in Claude Code).
- The agent stopped early to ask for permission → caught with a stop hook that blocks the hand-back and pushes it to keep going.

The other two were different in nature:

- The agent tunneled into one direction and declared it was "out of ideas" → addressed by adding a retrospective skill that makes agent to step back and look for areas it hadn't explored yet.
- Running several optimization loops in parallel made them clobber each other's git branches → fixed by giving each experiment its own forked copy of the repo.

Getting something working in the Software 3.0 paradigm is easy. Making it work reliably takes some creative effort.

Full write-up: https://vlasenkoalexey.github.io/2026/06/making-karpathy-autoresearch-production-ready/
