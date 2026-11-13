---
date: 2026-11-12
url: # TODO: paste LinkedIn permalink after posting
title: "spmd_types: Explicit Collectives, Checked Shardings (video)"
video_url: https://youtu.be/qWPCGUNEbV0
youtube_kit: 2026-11-spmd-types
image: /assets/images/youtube/linkedin-spmd-types-1200x627.png
---

I thought I had a pretty good understanding of how collectives work when you shard a model. Then I sat down with spmd_types, and it still took me a while to really get it. In the best way: it made me see code I thought I knew differently.

spmd_types (meta-pytorch) is a small library with a big idea: you write every collective explicitly, and a type checker verifies your shardings and your gradients. Every tensor carries a type on each axis of the device mesh, Replicate, Invariant, Varying or Partial, and the forward type decides what the gradient needs. One forward pass on a fake process group, on a CPU, is enough to prove that every backward all-reduce is where it should be.

The part that took me the longest was Replicate vs Invariant. Same values on every device, different gradients. The difference is whether the devices split the work that uses a tensor or duplicate it. Once that clicked, Megatron's copy_to, sequence parallelism and FSDP's gradient reduce-scatter all made more sense.

So I turned it into a 31-minute visual explainer: the four types, how annotation actually works, the sharding notation from the scaling book, then tensor parallel, sequence parallel, data parallel and FSDP typed step by step with the data moving between devices, and how it all compares to DTensor. If you work on distributed training, I think it is worth your time. I learned a lot making it.

Watch: https://youtu.be/qWPCGUNEbV0
spmd_types: https://github.com/meta-pytorch/spmd_types
How to Scale Your Model, sharding chapter: https://jax-ml.github.io/scaling-book/sharding/
