---
date: 2026-11-03
url: # TODO: paste LinkedIn permalink after posting
title: "Muon, Clearly Explained: How FlexShard Trains It at Production Scale (video)"
video_url: https://youtu.be/nj4bztH4EL0
youtube_kit: 2026-11-muon-flexshard
image: /assets/images/youtube/linkedin-muon-flexshard.png
image_alt: /assets/images/youtube/linkedin-muon-flexshard-1200x627.png
---

I have been experimenting with using AI to generate explainer videos, to see whether it can make hard concepts easier to understand than a write-up does. It turned out to work well, so I polished one and published it.

It is a 20-minute video on Muon, the optimizer behind Kimi K2/K3 and DeepSeek V4, and on FlexShard, the PyTorch library that lets it run on hundreds of GPUs.

The video builds Muon from scratch first: momentum, the SVD picture, why Newton-Schulz's odd coefficients are what they are and why five iterations are enough.

FlexShard exists because Muon and FSDP want different things. FSDP splits every weight matrix into slices across GPUs, which is fine for AdamW because each weight updates on its own. Muon updates a matrix as a whole, so a slice updated alone gives the wrong answer, and every matrix has to be put back together on some GPU before the optimizer step can run. FlexShard separates where a weight is stored from where its math runs: it stores each matrix whole on the GPU that will update it, so Muon runs locally with no extra communication.

Watch: https://youtu.be/nj4bztH4EL0
torchtitan FlexShard: https://github.com/pytorch/torchtitan/tree/main/torchtitan/distributed/flex_shard
Standalone library: https://github.com/meta-pytorch/flex_shard
