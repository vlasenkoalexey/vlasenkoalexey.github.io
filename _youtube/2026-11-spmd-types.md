---
title: "spmd_types: Explicit Collectives, Checked Shardings"
planned: 2026-11
published: 2026-11-12 (uploaded through YouTube Studio in Chrome, 2160p)
url: https://youtu.be/qWPCGUNEbV0
source_render: OpenMontage/projects/spmd-types-explained/renders/final_4k_youtube.mp4 (2160p30, 31:22, rendered at scale 2.0, loudnorm -14 LUFS)
drive: https://drive.google.com/file/d/13mfwJUd4UXdEPgE9lU3G6euy2TzG_wPF/view
playlist: "Distributed training internals (PyTorch)"
thumbnail: /assets/images/youtube/thumb-spmd-types.png
sources:
  - wiki/codebases/spmd_types.md and wiki/codebases/spmd_types/   (meta-pytorch/spmd_types at 0196e4ec67)
  - raw/code/spmd_types/docs/{design,local_spmd_types,key_concepts}.md
  - raw/code/torchtitan (google-pytorch fork at 2ad96c939): configs.py spmd_backend, decoder_sharding.py norm_config, trainer.py
  - How to Scale Your Model, sharding + training chapters (scaling-book at ed80595)
  - OpenMontage/projects/spmd-types-explained/research/ (facts.md, demo_*.py real checker runs)
---

# Video 2: spmd_types

## Title (as chosen; 100 char max)

**spmd_types: Explicit Collectives, Checked Shardings** - 52 chars

Alternatives considered: "spmd_types Explained: Check Distributed Gradients Without a Training Run",
"spmd_types Explained: Catch Silent Gradient Bugs in Distributed PyTorch".

## Description (paste as-is; chapter times = section starts from artifacts/props.json)

```
spmd_types is a type system for distributed (SPMD) PyTorch: you write every collective explicitly, and a type checker verifies your shardings and your gradients - in a single forward pass on a fake process group, on a CPU, without an end-to-end training run.

In this video:
- the four local SPMD types: Replicate (R), Invariant (I), Varying (V) and Partial (P), and why R vs I comes down to split work vs duplicated work
- how they map onto the sharding notation from the scaling book (How to Scale Your Model)
- how annotation works (assert_type) and what the checker does on every op
- tensor parallel, sequence parallel, data parallel and FSDP, typed step by step, with the data movement
- how spmd_types compares to DTensor

Chapters
0:00 Intro: why spmd_types
1:04 What spmd_types is
2:24 Meshes, axes and what a type describes
3:41 The scaling book sharding notation
4:49 Matmul cases as data movement: AllGather, AllReduce, ReduceScatter
6:37 From the book notation to spmd_types
7:33 Varying (V)
8:30 Replicate vs Invariant: split work vs duplicated work
10:07 R and I side by side, in real models
11:46 Partial (P)
13:12 The gradient table and design choices
14:23 How annotation works: assert_type
15:44 What the checker does
17:14 Tensor parallel: shardings and forward
19:06 Tensor parallel: backward and annotated code
20:36 Sequence parallel
22:19 Sequence + tensor parallel
23:51 Data parallel
25:06 FSDP
26:45 Escape hatches and custom autograd functions
27:41 spmd_types vs DTensor
29:08 Five differences that matter
30:28 Limits and cheat sheet

Links
spmd_types (meta-pytorch): https://github.com/meta-pytorch/spmd_types
Local SPMD types guide (porting Megatron-style code): https://github.com/meta-pytorch/spmd_types/blob/main/docs/local_spmd_types.md
spmd_types design doc: https://github.com/meta-pytorch/spmd_types/blob/main/docs/design.md
How to Scale Your Model - sharded matrices: https://jax-ml.github.io/scaling-book/sharding/
How to Scale Your Model - training (DP, FSDP, TP): https://jax-ml.github.io/scaling-book/training/
Megatron-LM tensor-parallel mappings (copy_to / reduce_from): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/mappings.py
torchtitan: https://github.com/pytorch/torchtitan
PyTorch DTensor docs: https://docs.pytorch.org/docs/stable/distributed.tensor.html
JAX explicit sharding: https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html
veScale paper (DTensor eager overhead numbers): https://arxiv.org/abs/2509.07003

Shown: spmd_types at commit 0196e4ec67 (Nov 2026), PyTorch 2.15 nightly. The checker errors and types on screen are real output from runs on a fake process group.

#PyTorch #DistributedTraining #TensorParallelism

---
Blog: https://vlasenkoalexey.github.io
Autoresearch wiki: https://github.com/vlasenkoalexey/tpu_performance_autoresearch_wiki
LinkedIn: https://www.linkedin.com/in/oleksiyvlasenko
X: https://x.com/AlekseyVlasenko
```

## Tags (500 char total max; this is ~338)

```
spmd_types, PyTorch distributed training, tensor parallelism, sequence parallelism, FSDP, DTensor, collectives, all-reduce, all-gather, reduce-scatter, sharding, Megatron-LM, torchtitan, type checking, How to Scale Your Model, scaling book, data parallelism, LLM training, deep learning systems, PyTorch, TPU, machine learning performance
```

## Thumbnail

`assets/images/youtube/thumb-spmd-types.png` (1280x720) = the first frame of the video: kicker
"A TYPE CHECKER FOR YOUR COLLECTIVES", title "spmd_types", subtitle "Explicit Collectives, Checked
Shardings", the four type chips R / I / V / P and the six-part agenda. Re-export from the render with
`ffmpeg -ss 0 -i final_4k_youtube.mp4 -frames:v 1 ... -vf scale=1280:720`.

## Upload settings

Visibility Unlisted first (review), then Public. Not made for kids. Category Science & Technology.
Playlist "Distributed training internals (PyTorch)". Chapters on (from the description).

## Pinned comment

```
Correction log for this video goes here. Every type and checker error on screen comes from real runs
of spmd_types on a fake process group (commit 0196e4ec67, PyTorch 2.15 nightly). If R vs I is still
unclear, the one-line test is: do the devices split the work that uses this tensor (R, its gradient
must be summed) or duplicate it (I, its gradient is already complete)?
```

## Script (final cut, 23 sections)

### Hook (0:01-1:03)

This video is about spmd_types, a library for distributed PyTorch built on two ideas: you write every collective explicitly, and a type checker verifies your shardings, and your gradients. Here is why that matters. Delete one line of Megatron code, a line that does nothing in the forward pass, and your gradients go silently wrong. Every forward number stays the same, and the first loss is identical. But the backward pass now skips an all-reduce on every tensor-parallel rank, and you only find out days later, from a loss curve that drifts. spmd_types catches that in a single forward pass, in one process on a CPU, and tells you which collective is missing. We will cover what its four types mean, how annotation works, how it builds on the sharding notation from the scaling book, and then tensor, sequence and data parallelism, typed one step at a time.

### What spmd_types is (1:04-2:24)

At its core, spmd_types is a language for one question, asked on every mesh axis: is this tensor's gradient complete, or still pending a sum? Plus a checker that enforces the answer. Every tensor carries, for every axis of the device mesh, one of four local types. The types are plain attributes on ordinary tensors. There is no subclass and no wrapper, so you can erase them: with the checker off, the same code runs as plain PyTorch. To check, you run the forward pass once, in one process, on a fake process group that stubs out the communication, inside the checker. The design has four goals. Be strict enough that accepted programs compute correct gradients. Be permissive enough to express the communication tricks real training code uses. Make the backward's sharding follow from the forward's, so you never reason non-locally. And stay optional, which forces every collective to be explicit in the code. It comes in two modes. Local SPMD types what each rank holds; that is the mode for porting an existing Megatron-style codebase. Global SPMD adds partition specs, so a program means the same thing on one device or many. The README says that mode is still under construction.

### Meshes, axes, and what a type describes (2:24-3:41)

Before any API, we need the vocabulary. Picture eight devices arranged as a two by four grid: a device mesh with two named axes. dp has size two, and tp has size four. An axis is a direction you communicate along. A collective on tp involves the four devices in one row. A collective on dp involves the two devices in one column. Each row, and each column, is a group that only talks among itself. A local SPMD type answers one question about a tensor, for one axis: across the devices in that group, how do the copies of this tensor relate? Are they different? Are they the same? Or are they pieces of a sum that has not been added up yet? So every tensor carries one type per mesh axis. A batch of inputs might be Varying on dp, because each data-parallel group gets different examples, and Invariant on tp, because all four tensor-parallel devices see the same examples. We write that as a little dictionary: dp, V, and tp, I. There are exactly four types. Before we meet them, let's connect them to a notation many of you already know.

### The scaling book's notation (3:41-4:49)

If you have read the scaling book, How to Scale Your Model, you already know most of this. Let's take its notation slowly. Picture four devices in a row, along one mesh axis called X, and one matrix A, with dimensions I and J. A of I sub X, J. The subscript X on I means dimension I is split across the four devices. Each device holds one quarter of the rows: different data on every device. A of I, J, with no subscript at all. Nothing is split. Every device holds the whole matrix: the same data everywhere. And C of I, K, with curly braces, U sub X. That means unreduced along X. Every device holds a full-size matrix, but each one is only a partial sum. The real C is what you get when you add all four together. Split, whole, or unreduced. Those three states are all the notation needs. What moves data between them are collectives, and the clearest way to see them is inside a matrix multiply.

### Matrix multiply cases, as data movement (4:49-6:37)

Multiply A by B, summing over J. The book sorts every sharded matrix multiply into a few cases, and each case is a different pattern of data movement. Case one: no contracting dimension is split. Say A is split along I, and B is whole. Each device multiplies its quarter of the rows of A by all of B, and gets its quarter of the rows of C. Nothing moves between devices. No communication at all. Case two: one side is split on the contracting dimension. Now A is split along J, the dimension we sum over, and B is whole. No device can finish on its own. So first, we all-gather A. Watch the pieces: every device sends its slice of A to every other device, until each one holds all of A. Then each device multiplies locally. An all-gather turns split into whole. Case three: both sides are split on the contracting dimension. Each device holds a slice of A along J, and the matching slice of B. Each one multiplies its own slices, and gets a full-size result that is only part of the sum: C, unreduced along X. There are two ways to finish. An all-reduce. The devices add up their partial results, and every device ends with the complete C. Unreduced turns into whole. Or a reduce-scatter. The devices add up the partials, but each one keeps only a quarter of the sum. Unreduced turns into split. It moves half the data of an all-reduce, because an all-reduce is exactly a reduce-scatter followed by an all-gather. And in the backward pass, they trade places. The backward of an all-gather is a reduce-scatter, and the backward of a reduce-scatter is an all-gather.

### From the book's notation to spmd_types (6:37-7:32)

Now map this onto spmd_types, one mesh axis at a time. A subscript X on dimension i is S of i on axis X, which is Varying: different data on every device. Unreduced along X is Partial on X: pieces of a sum. And the collectives keep their names. An all-gather takes S of i to whole. A reduce-scatter takes Partial to S of i. An all-reduce takes Partial to whole. That leaves one gap: whole. No subscript, the whole matrix on every device, is where it gets interesting, because in spmd_types, whole comes in two different types: Replicate, and Invariant. The book's forward notation cannot tell them apart. The difference only shows up in its backward formulas. So let's meet the types one at a time: first Varying, and then the two kinds of whole.

### Varying (7:33-8:29)

Varying, V, means the devices along the axis hold different values. The everyday case is a shard. Split a weight into four slices and give one slice to each tp device, and that weight is Varying on tp. Split the batch in two across dp, and the batch is Varying on dp. When you know which dimension was split, you can write S of i, shard on dimension i. S of i is still Varying; it simply also records how to stitch the pieces back together, by concatenating along dimension i. Plain V makes no such promise, which is exactly what you want for pieces that are uneven, or that have no global meaning at all. The gradient of Varying is Varying. Each device's piece produces a gradient for that piece only. Device two's slice of the weight gets device two's slice of the gradient. Nothing needs to move.

### Replicate versus Invariant: split work or duplicated work (8:30-10:06)

Now the two kinds of whole. Replicate and Invariant both mean that every device on this axis holds exactly the same values. In memory, they are identical. The difference is what the devices do with that value. Do they split the work that uses it, or does each device do all of it? Take an RMSNorm weight, g, and a sequence of eight tokens, on four tensor-parallel devices. On a single GPU, the gradient of g is a sum with one contribution per token: c zero, plus c one, all the way to c seven. First, with sequence parallelism. Every device holds the same g, but each one normalizes only its own two tokens. So device zero's gradient is c zero plus c one. Device one's is c two plus c three, and so on. Each device holds a quarter of the sum, and to get the real gradient, the four pieces must be added up. Split work: that is Replicate. Its gradient is Partial. Now without sequence parallelism. Every device holds the same g, and normalizes all eight tokens. So every device computes the whole sum, c zero through c seven. Each one already has the complete gradient, and there is nothing to add. Adding would count every token four times. Duplicated work: that is Invariant. Its gradient is Invariant. Same weight, same values, same line of code. The only difference is whether the devices split the work, or duplicate it.

### R and I side by side, and where you meet them (10:07-11:46)

Put them side by side. Both hold identical values. With Replicate, each device does a different slice of the work, so each local gradient is a piece, and the backward must sum it over the axis. With Invariant, each device does all of the work, so each local gradient is already complete, and the backward does nothing. Getting it wrong costs you in both directions. Replicate labelled as Invariant skips the sum, and the gradients stay partial: that is the bug from the start of this video. Invariant labelled as Replicate sums gradients that were already complete, and on four devices they come out four times too large. You meet both in every real model. Data-parallel weights are Replicate on dp: the same weights, but each group works on different examples, so DDP or FSDP sums their gradients. Norm weights are Invariant on tp without sequence parallelism, and Replicate with it. torchtitan encodes exactly that, in one line. And one tensor can switch. Between tensor-parallel blocks, the hidden state is Invariant: every device holds it, and does the same work. The moment it enters a column-parallel linear, each device multiplies it by a different weight shard. Now the work is split, so it must become Replicate. That switch is Megatron's copy to region. Its forward does nothing. Its backward is the all-reduce that sums the pieces. That is why the checker says Invariant may only meet Invariant. Once a duplicated tensor touches anything that differs from device to device, the work is no longer duplicated, and you have to say so, explicitly.

### Partial (11:46-13:12)

Partial, P, means each device holds one term of a sum that has not been added up yet. The real value is the sum over the axis. Where does it come from? A matrix multiply over a split dimension. If each tp device holds a quarter of the hidden units, and the matching quarter of the next weight's rows, each device can multiply its quarter locally. Each result is a partial product. The true output is the sum of all four. Or a local loss. Each data-parallel group computes the loss on its own examples, and the global loss is the sum, so the local loss is Partial on dp. Partial is useful because you can wait. You can add partials together, scale them, or multiply one by a replicated tensor, and add them all up later, perhaps as a cheaper reduce-scatter. But a pending sum does not survive anything nonlinear. Take two partial tensors, each holding 1 on two devices. Each one means 2, so their product should be 4. Multiply locally and then add, and you get 1 times 1, plus 1 times 1: that is 2. A gelu or a softmax breaks the same way, so the checker rejects them on Partial inputs. Reduce first. And the gradient of Partial is Replicate. The gradient of a sum flows back unchanged to every term, so every device receives the same gradient.

### The gradient table, and two design choices (13:12-14:22)

Put the four together, and you get the table that drives everything. Replicate goes to Partial. Partial goes to Replicate. Invariant stays Invariant. Varying stays Varying. This is the central design choice. The type of a gradient is decided by the type of the forward value, and nothing else. No other part of the program matters. So by checking the forward pass, you already know what every backward collective must be. You do not need to run the backward, and you never have to reason about distant code. The second design choice is which replicated type is the normal case. JAX's sharding types made a replicated forward mean a replicated gradient, and added a separate type for the pending-sum case. spmd_types goes the other way. R, replicate forwards and partial backwards, is the normal case, and I is the opt-in. A pending sum keeps options open: a sum that waits can become a reduce-scatter instead of an all-reduce. The practical advice follows. Prefer R in your forward code, and keep I mostly for parameters.

### What an annotation is (14:23-15:44)

Now, how do you actually write these types into a program? First, name your axes. spmd dot set current mesh takes the device mesh, so the strings dp and tp mean something. Give it the compute mesh: the axes your activations are split along. Then annotate with spmd dot assert type: a tensor, and a dictionary from axis name to type. Here is what that call does. It writes a small dictionary onto the tensor, as an ordinary Python attribute. The tensor stays a plain torch tensor. No subclass, no copy, and no communication. If the tensor already carries a type, the same call checks it instead of overwriting it, and complains if the two disagree. Writing S of i instead of V stores one more fact: which dimension is split. What do you annotate? Every input, and every parameter. That is enough for the checker to work out everything in between. Asserting types at module boundaries is optional, but it works like a function signature: it documents the contract, and it fails loudly the moment the code breaks it. With the checker switched off, these calls are just metadata. Your training run is unchanged.

### What the checker does (15:44-17:13)

The checker is a context manager: spmd_types dot checker dot typecheck. Inside it, every torch operation is intercepted. The operation runs normally. Then the checker reads the types of its inputs, applies the rules, and writes the result type onto the output. The rules for ordinary operations are short, because ordinary operations never communicate. Inputs that are all R give R. Mixing R with V gives V. Partial may only pass through linear operations. And Invariant cannot mix with anything else, because the gradient of I must come out identical everywhere, and mixing it with other types would quietly break that. Collectives are different. spmd dot all reduce, all gather, reduce scatter and all to all take a real process group, plus a source type and a destination type. The checker verifies that the input really is the source type, and stamps the destination type on the output. The source and destination also choose the backward. All reduce to R sends an all-reduce backward. All reduce to I sends nothing back. Anything that breaks a rule raises a type error right there, at the line that broke it, naming the axis, the types it found, and the collective you probably meant. And you run all of this once, on a fake process group, so that one process can pretend to be every rank.

### Tensor parallel: shardings and the forward pass (17:14-19:05)

Tensor parallelism on four devices, on one MLP: y equals W2, times gelu of W1 times x. Start with the shardings. W1 is split by its output rows: each device holds a quarter of the hidden units. That is S of 0 on tp. W2 is split the matching way, by its input columns: S of 1 on tp. Both are Varying. In scaling-book notation, W1 is W in, of D by F sub Y, and W2 is W out, of F sub Y by D, with Y as our tp axis. The input x is the same on every device, and so far every device has done the same work with it. So x is Invariant on tp. To multiply an Invariant x by a Varying weight, the checker first wants x converted to R. In the forward pass, that conversion is free. We will see in a moment what it costs in the backward. Now the forward data movement. Each device multiplies x by its slice of W1. No communication: it already has everything it needs. The result h is Varying: four different quarters of the hidden layer. That is the book's first case: no contracting dimension is split, so there is no communication. The gelu is applied element by element, so it stays local, and stays Varying. Then each device multiplies its quarter of h by its quarter of W2. Each gets a full-size output, but only a partial product: the true y is the sum of all four. That is the book's third case, unreduced along Y. So the one forward collective is an all-reduce, with destination I. Every device ends up with the same complete y, ready for the next block.

### Tensor parallel: the backward pass, and the annotations (19:06-20:35)

Now run the gradient backwards, and read the types straight down. y was Invariant, so its gradient is Invariant: every device starts with the same gradient. The all-reduce to I sends nothing back. Each device pushes that gradient through its quarter of W2, and through the gelu, locally. The weight gradients for W1 and W2 are Varying: each device gets its own slice. No communication. But look at the gradient of x. Each device computed it from only its own quarter of the hidden units, so each one holds a different, partial gradient. The real gradient is their sum: the scaling book writes it as d In, unreduced along Y. That is the all-reduce Megatron's copy to region performs in its backward. In spmd_types, it falls out of the types. x went from I to R, the gradient of R is Partial, so the convert's backward all-reduces it back to I. Here is the annotated code. Assert type on x: I on tp. Assert type on the two weights: S of 0, and S of 1. Convert, from I to R. Two ordinary linear layers. And an all reduce, with destination I. A handful of annotations, and the checker can prove that every backward collective is in place. Delete the convert, and you get exactly the error from the start of this video.

### Sequence parallel: the region in between (20:36-22:19)

In pure tensor parallel, everything outside the MLP and attention is duplicated. The norms and dropouts process the full sequence, identically, on all four devices. Invariant, and wasteful. Sequence parallelism puts the tp devices to work there too, by splitting the sequence. Each device holds a quarter of the tokens: S of 1 on tp, if the sequence is dimension one. An RMSNorm works token by token, so each device normalizes its own tokens, with no communication. The output is still split by sequence. The interesting part is the norm's weight. Its values are identical on every device, but each device applies it to different tokens. That is the definition of Replicate. So its gradient is Partial: each device holds a piece of it, and someone has to sum those pieces over tp. Who? The library supports both answers. Type the weight R, and a wrapper such as FSDP performs the sum, alongside the data-parallel reduction. Or type it I, and the module must do it itself: the checker rejects I mixed with a Varying input, and forces an invariant to replicate inside the module, whose backward all-reduces over tp. The design favors the wrapper: simpler module code, and one fused reduction instead of two. torchtitan does exactly this. Norm weights are R with sequence parallelism, and I without.

### Sequence plus tensor parallel (22:19-23:50)

Now join the two regions. Between blocks, activations are split by sequence. Inside the MLP, every device needs whole sequences, split by hidden units instead. So two transitions appear. Entering the MLP: an all gather, from S of 1 to R. Every device collects all four chunks of the sequence. Then the column-parallel multiply gives Varying, exactly as before. Leaving the MLP: the row-parallel multiply gives partial sums over full sequences. A reduce scatter, to S of 1: the devices add up their partials, and each keeps only its own quarter of the tokens. In the backward, the mirror images run. The reduce-scatter's backward is an all-gather, and the all-gather's backward is a reduce-scatter. This is the scaling book's tensor-parallel algorithm, step for step, except that the book splits the hidden dimension between layers, where Megatron's sequence parallelism splits the sequence. The types are the same; only the i in S of i changes. Compare this with plain tensor parallel. An all-reduce is a reduce-scatter followed by an all-gather, so the bytes moved are the same. What you gain is that every activation between blocks is a quarter of the size, and the duplicated work is gone. In code, only the edges change. The convert and the all reduce become an all gather and a reduce scatter. The middle of the block is identical.

### Data parallel (23:51-25:05)

Data parallel is the simplest case, and it shows a convention every PyTorch module already follows. The batch is split across dp: S of 0, Varying. Every parameter is identical on both dp groups, but each group applies it to different examples. That is Replicate, by definition. Forward data movement: none. Each group computes on its own examples. Each group's loss covers only its own examples, and the global loss is their sum. So the local loss is Partial on dp, and you declare it with a reinterpret, from V to P. Backward: the gradient of every R parameter is Partial, a pending sum over dp: the book's d W, unreduced along X. And notice that no module performs that sum. A wrapper, DDP or FSDP, all-reduces the gradients out of band. That is the standard calling convention. Modules take inputs that are Varying on dp, and produce parameter gradients that are still pending on dp. So annotating data parallel is formulaic: one helper walks the parameters, and marks them all R on dp.

### FSDP (25:06-26:44)

FSDP changes where the weights live, not what the module computes. Storage: each dp group keeps only a slice of every weight, S of 0 on dp. Just before a layer runs, FSDP all-gathers the slices into the full weight. The module computes with it, and the full copy is freed. In the backward, the weight is gathered again, and the full gradient, a pending sum over dp, is reduce-scattered, so each group keeps only its own slice of the summed gradient. In the book's notation, W in of D sub X by F is all-gathered before the matmul, and its gradient, unreduced along X, is reduce-scattered back. How does that look to the type system? The module only ever sees the gathered weight, so that weight must be R on dp. Its gradient is Partial, and the reduce-scatter is exactly that pending sum being paid. That is why data-parallel axes must be typed R. FSDP performs those gathers out of band, not as differentiable collectives, so its internals run under no typecheck. Recent PyTorch nightlies teach FSDP2 to keep the types: it stores annotated weights as sharded storage, restores the types after every all-gather, and reads the gradient type to set up the reduce-scatter. There is also a variant the checker can see all the way through. Gather the weight with a differentiable spmd dot all gather, from S of 0 to R. Its backward is the reduce-scatter. That is the SimpleFSDP design, and it type-checks end to end.

### Escape hatches and custom code (26:45-27:41)

Real models contain code the checker cannot see into. Local map runs a region on local tensors and stamps the output types you declare; torchtitan wraps its rotary embeddings this way. No typecheck switches checking off for a region, like an opaque attention kernel. You re-assert the types of what comes out. For your own autograd functions, add a static method called spmd typecheck, built from the spmd_types rules module. For example: this function types exactly like an all gather from S zero to R. And rules dot einsum covers fused local math. Then rulecheck tests that claim numerically, on simulated ranks, so a wrong typing rule gets caught too. And for expert parallelism, you nest set current mesh to switch onto the expert mesh, dispatch with an all to all from V to V, and use reinterpret mesh to come back.

### DTensor, and a different route (27:41-29:08)

Now, DTensor, PyTorch's built-in answer to sharded tensors. A DTensor is a tensor subclass that carries a device mesh and a placement on every axis: Shard, Replicate, or Partial. You write global code. DTensor propagates shardings operator by operator, and when the placements do not line up, it inserts a redistribute for you, in the forward and in the backward. That is convenient, and it gives correct gradients. The spmd_types docs say it plainly: if you write a program entirely in DTensor, you already get correct gradients, and local SPMD types add nothing. The costs show up elsewhere. One is eager speed. The veScale paper measured DTensor 36 percent slower than plain tensors for Llama, and 61 percent slower for Mixtral; with their own optimized dispatch, the overhead drops to about 5 percent. The other is communication you did not write. People write ordinary code, then go debugging why extra collectives appeared, and only then discover partial tensors floating around. spmd_types takes a different route, by design. Make every collective explicit, and make every gradient's type follow from the forward. Then the subclass is unnecessary: you run plain tensors, and switching the types on turns the same program into a type checker.

### Five differences that matter (29:08-30:27)

Day to day, five differences matter. One: R versus I. DTensor has a single Replicate placement. Both R and I map onto it, so who reduces the gradient is simply not expressible. Two: Partial is never inferred. DTensor turns a contraction over a sharded dimension into Partial automatically. spmd_types refuses, and the reason is structural. Types are erasable, and once the partition specs are erased, there is no way to look at the inputs and tell that a contracted dimension was sharded. So you declare the pending sum. Three: a redistribute needs its source. DTensor reads the current placement off the tensor. With erasable types there is nothing to read, so you declare both sides. Four: a missing collective. DTensor inserts it silently. spmd_types raises a type error. Five: the cost when it is off. DTensor pays subclass dispatch on every operation. spmd_types pays one attribute write per annotation. And one bonus: arbitrary uneven splits, which DTensor's Shard cannot describe, are simply V.

### Limits, uses, and the cheat sheet (30:28-31:22)

Where are the limits today? It is SPMD only, so pipeline parallelism is out of scope. The advice is to typecheck the whole model before you split it into stages, and torchtitan refuses to typecheck with pipeline parallelism turned on. The check is forward only. torchtitan runs the backward with type checking switched off, and trusts the derivation, which is sound as long as the gradient table holds. Global SPMD is still under construction. So what is it good for right now? A test. Run the forward once on a fake process group, and catch the forward no-op with a missing backward reduction: the kind of bug that neither a loss curve nor an HLO diff proves absent. Here is the cheat sheet.
