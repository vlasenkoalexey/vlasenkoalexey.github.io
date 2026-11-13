---
title: "Muon and FlexShard: why the optimizer needs whole matrices"
planned: 2026-11
published: 2026-10-31 (1080p, https://youtu.be/VZzN9X-emNA, now unlisted); re-uploaded 2026-11-02 in 4K
url: https://youtu.be/nj4bztH4EL0
source_render: OpenMontage/projects/flexshard-explained/renders/draft_v10_4k_norm.mp4 (2160p30, 20:05, same cut as draft_v10_norm.mp4 rendered at scale 2.0)
playlist: "Distributed training internals (PyTorch)"
thumbnail: /assets/images/youtube/thumb-muon-flexshard.png
sources:
  - wiki/analyses/2026-10-29-flexshard-motivation-and-mechanism.md   (torchtitan FlexShard at abd6e3658)
  - wiki/codebases/flex_shard/overview.md                             (meta-pytorch/flex_shard at 97d7647)
  - wiki/codebases/flex_shard/concepts/dist-muon-owner-storage.md
  - wiki/codebases/flex_shard/doc-concepts/muon-whole-matrices.md
  - Google Doc version written for readers new to Muon: https://docs.google.com/document/d/1K96T-Ky9W5IfLgXPS6JGCz0S2wfCqqw2g4Bts-oKeFs/edit
target_length: 16-18 min
---

# Video 1: Muon and FlexShard

## Title (pick one; 100 char max)

1. **Muon and FlexShard: why the optimizer needs whole matrices (PyTorch distributed training)** - 89 chars
2. Why Muon can't run on FSDP shards, and how FlexShard fixes it - 62 chars
3. Storage layout vs compute layout: Muon on 256 GPUs without an optimizer collective - 84 chars

## Description (as published 2026-10-31; chapter times = section starts from artifacts/props.json)

```
Why Muon can't run on FSDP shards, and how FlexShard gives the optimizer step whole matrices without an extra collective.

Every optimizer you have run on FSDP works on a slice of each weight matrix. Muon cannot: its update orthogonalizes the whole momentum matrix with a Newton-Schulz iteration, so every output row depends on every input row. This video reads FlexShard from the source - first the torchtitan version (torchtitan/distributed/flex_shard), then the standalone meta-pytorch/flex_shard repo - and explains what it does about that: keep parameters, gradients and momentum in their FSDP/EP storage layout, declare a per-parameter compute layout for the optimizer step, move each matrix to an owner with one packed all-to-all per bucket, run Newton-Schulz on whole matrices, move the result back. Then the follow-up idea: if the storage layout already holds whole matrices, the optimizer step needs no collective at all.

Chapters
00:00 Hook: elementwise optimizers are why FSDP works
00:25 AdamW primer
00:59 FSDP primer
02:09 Muon: a matrix-aware optimizer
03:06 Step 1: momentum
03:53 Step 2: orthogonalize
05:48 What an exact SVD takes
06:44 Newton-Schulz
09:28 Step 3: cost and scale
10:48 AdamW vs Muon
11:22 Why shards break
11:58 Naive options
12:30 FlexShard v1: storage layout vs compute layout
13:40 Owner storage
14:57 Result
15:30 Two FlexShards: v1 vs v2
16:32 One runtime, many placements
17:41 Same rule, low precision: FP8 and MXFP8
18:29 Others reached the same rule
19:08 Limits and recap

Code and docs
torchtitan FlexShard: https://github.com/pytorch/torchtitan/tree/main/torchtitan/distributed/flex_shard
Standalone repo: https://github.com/meta-pytorch/flex_shard
Muon (Keller Jordan): https://kellerjordan.github.io/posts/muon/
Moonlight / Muon is scalable: https://arxiv.org/abs/2502.16982
Canzona / FSDP-Canzona (the overlap scheme the runtime credits): https://arxiv.org/abs/2602.06079
torch.optim.Muon: https://docs.pytorch.org/docs/main/generated/torch.optim.Muon.html

Numbers quoted in the video come from the repos' READMEs and from running the Newton-Schulz polynomial on small random matrices; the DeepSeek-V3 671B / 256 H100 figure (slowest-GPU optimizer step from 527 ms to 301 ms) is the flex_shard README's, not mine.

---
Blog: https://vlasenkoalexey.github.io
Autoresearch wiki: https://github.com/vlasenkoalexey/tpu_performance_autoresearch_wiki
LinkedIn: https://www.linkedin.com/in/oleksiyvlasenko
X: https://x.com/AlekseyVlasenko
```

## Tags (500 char total max; this is ~330)

```
Muon optimizer, FlexShard, FSDP, PyTorch distributed training, torchtitan, Newton-Schulz, DTensor, expert parallelism, all-to-all, optimizer sharding, DistMuon, Kimi K2, Moonlight, DeepSeek V3, mixture of experts, LLM training, deep learning systems, PyTorch, TPU, machine learning performance
```

## Thumbnail

`assets/images/youtube/thumb-muon-flexshard.png` (1280x720). Left: "Muon + FlexShard /
why the optimizer needs WHOLE matrices". Right: four colored FSDP row shards (rank 0-3)
with an arrow to one whole matrix framed in orange, labelled "X X^T / Newton-Schulz".
Regenerate with `python3 _youtube/make_channel_art.py`.

## Pinned comment

```
Correction log for this video goes here. The written version of this material, with
line references into the source, is the FlexShard page in the autoresearch wiki repo.
If you are porting Muon to a sharded trainer and the storage/compute split is unclear,
ask below - the planning half (owners, routes, one all-to-all per bucket) is the part
that transfers to any backend.
```

## Script

Format: [VISUAL] cue, then narration. Target 16-18 minutes at a measured pace
(~2,400 words). Every number is from the sources listed in the front matter.

### 0. Cold open (0:00-1:30) - Elementwise optimizers are why FSDP works

[VISUAL] One weight matrix, 8 rows x 32 columns. It gets cut into 4 horizontal
bands, each band slides to a different GPU labelled rank 0..3. Caption: FSDP, Shard(0).

Here is a weight matrix in a model trained with FSDP. FSDP stores every parameter
as a slice of rows: Shard(0) over the data-parallel axis. Rank 0 holds rows 0 and 1,
rank 1 holds rows 2 and 3, and so on. The gradient is stored the same way, and so
is the optimizer state.

[VISUAL] Adam update formula appears next to one single cell of the matrix; the
cell lights up; arrows show it only reads its own gradient and its own two moments.

And the optimizer step runs on that slice. That is fine for AdamW, and for SGD, and
for almost every optimizer you have used, because their updates are elementwise.
The update of one element depends only on that element's gradient and that
element's state. Running the update on a slice gives you exactly the same numbers
as running it on the whole matrix and then slicing. Nobody has to gather anything
for the optimizer step. That property is a big part of why FSDP is cheap.

[VISUAL] Title card: "Muon and FlexShard: why the optimizer needs whole matrices".

Muon breaks that property. Its update is not elementwise, and that single fact
forces a new piece of machinery into the training step. That machinery is
FlexShard. In this video I read it from the source: first the version that
shipped inside torchtitan, then the standalone repository it became, where the
same idea goes one step further and the optimizer step runs with no communication
at all.

### 1. What Muon computes (1:30-3:30)

[VISUAL] Matrix M labelled "momentum". M = U S V^T drawn as three blocks; S is a
diagonal with bars of very different heights.

Muon keeps a momentum matrix M for every two-dimensional weight, exactly like SGD
with momentum. The difference is the direction it steps in. Write M as a singular
value decomposition, U S V^T. Muon does not step along M. It steps along U V^T:
the same directions, but with every singular value set to one.

[VISUAL] The bars in S all snap to the same height.

Why would you want that? A raw gradient matrix is usually dominated by a few
directions. Equalizing the strengths means every direction the gradient points
along moves by the same amount, instead of a few big ones drowning out the rest.
That is the whole idea, and it is why Muon is defined on matrices and not on
individual numbers.

[VISUAL] Text: "exact SVD per matrix per step: too slow". Then: "approximate U V^T
with matrix products only".

An exact SVD per matrix per step is far too slow. So Muon approximates U V^T with
nothing but matrix multiplications, and that approximation is the Newton-Schulz
iteration.

### 2. Newton-Schulz in five steps (3:30-6:00)

[VISUAL] The five-line pseudo-code, one line at a time:
  X = M / ||M||_F
  repeat 5 times:
      A = X X^T
      X = a X + (b A + c A A) X        (a, b, c) = (3.4445, -4.7750, 2.0315)

First normalize M by its Frobenius norm, so the largest singular value is at most
one. Then, five times: form A equals X times X transpose, and replace X with a X
plus b A X plus c A squared X. Three coefficients, chosen by Keller Jordan,
3.4445, minus 4.7750, 2.0315.

[VISUAL] Side-by-side: X X^T X = U S^3 V^T. Then p(s) = a s + b s^3 + c s^5.

The trick is that X X^T X equals U S cubed V^T. Any odd polynomial in X keeps the
same U and V and acts on each singular value separately. So one step maps every
singular value s to p of s, which is a s plus b s cubed plus c s to the fifth.

[VISUAL] Table animates row by row:
  start s | after 1 | 2    | 3    | 4    | 5
  0.001   | 0.003   | 0.012| 0.041| 0.140| 0.471
  0.01    | 0.034   | 0.118| 0.400| 1.093| 0.699
  0.2     | 0.651   | 1.162| 0.815| 0.953| 0.747
  1.0     | 0.701   | 1.114| 0.721| 1.090| 0.696

Watch what five steps do. A singular value of 0.2 goes to 0.65, overshoots to 1.16,
and settles at 0.75. A value of 1.0 ends at 0.70. Even 0.01 gets pulled up to 0.70.
The coefficients are tuned so that everything in the unit interval lands in a band
around one, quickly, at the price of not converging to exactly one. The textbook
polynomial, 1.5 s minus 0.5 s cubed, converges exactly, but from 0.01 five steps
only get you to 0.076. On a random 64 by 256 matrix whose singular values span a
factor of 22, Muon's five steps bring all of them into 0.68 to 1.13.

[VISUAL] Cost: 5 x s^2 x (2l + s) for an s x l matrix, s <= l.

Each step costs two s by s by l products and one s by s by s product for an s by l
matrix, with s the smaller dimension; tall matrices are transposed first. Five
steps: 5 times s squared times 2l plus s. Keep that formula; FlexShard uses it
later to balance work.

[VISUAL] Back to the 8 x 32 matrix split in 4 bands. Highlight rank 0's two rows.
Run NS on just those two rows vs the full matrix; show the difference heat map,
max difference 0.19, max entry 0.46.

Now the point of this whole video. A equals X X transpose pairs every row with
every other row. If you run the iteration on rank 0's two rows alone, you get a
different answer from the same two rows of the full result. On this 8 by 32
example the entries differ by up to 0.19, where the largest entry is 0.46. That is
not a rounding error. It is a different computation. Orthogonalizing a slice of
rows is not the same as slicing the orthogonalized matrix. So Muon has to see
whole matrices, and FSDP gives it slices.

### 3. Three obvious fixes and their cost (6:00-7:30)

[VISUAL] Three-row table:
  all-gather every gradient, compute everywhere | N-fold redundant compute + full-model temporary per rank
  all-gather to one rank, compute, scatter       | serializes the optimizer on one device
  store whole matrices instead of FSDP shards    | gives up FSDP's memory savings for params and state

There are three obvious ways out, and each one is expensive. One: all-gather every
gradient and run Newton-Schulz everywhere. Every rank does the whole model's
optimizer math, and you need a full-model temporary on every rank. Two: gather to
one rank, compute there, scatter the result. Now one device does all the math while
the others wait. Three: change the storage layout so matrices are whole. Then you
have given up the memory savings that made you pick FSDP in the first place.

[VISUAL] Text builds up: "for each matrix: pick some rank to own it, this step only;
send it the pieces; spread owners so work is even; move the result back; leave
storage exactly as FSDP and EP laid it out."

What you actually want is narrower than any of those. For each matrix, pick some
rank, or some small set of ranks, to own it during the optimizer step. Send that
owner the pieces it needs. Spread the owners around so every rank does about the
same amount of Newton-Schulz. Move the result back. And leave the persistent
storage, and the optimizer state, exactly as FSDP and expert parallelism laid them
out. That is what FlexShard does. Its README calls it "PyTorch-native building
blocks for running optimizer compute with layouts that differ from persistent
DTensor parameter storage layouts". DistMuon was its first and only consumer.

[VISUAL] Two side notes. (a) a [num_heads * head_dim, C] projection drawn as
stacked per-head blocks. (b) a [E, F, D] expert tensor drawn as E slabs, with the
E axis also cut by an EP line.

Two model shapes make "whole matrix" subtler than "one tensor". Kimi-family recipes
run Muon per attention head, so a projection stored as num_heads times head_dim by
C is really num_heads independent matrices, and the compute layout has to keep each
head's rows together. And routed experts are stored as one E by F by D tensor per
projection. Each F by D slab is its own matrix, so splitting on the expert dimension
already gives whole matrices, but under expert parallelism that dimension is also
split across the EP axis, and the optimizer has to respect that.

### 4. The idea: storage layout vs compute layout (7:30-9:00)

[VISUAL] The two-column diagram from the wiki page:
  persistent (never changes)             temporary (one optimizer step)
  param  storage layout: DTensor placements --+
  grad   same layout                          |  prepare   pack + all-to-all
  state  same layout (momentum)               +--> update -----------------> compute layout
                                                   input
  param <-- apply update, weight decay <-- unpack <-- all-to-all back <-- Newton-Schulz

Here is the idea in one picture. On the left, the storage layout. That is the
parameter's DTensor placements as FSDP and EP created them. The gradient shares it.
The momentum buffer shares it. Nothing about it ever changes, which means
checkpointing with DCP sees ordinary DTensors and knows nothing about Muon.

On the right, the compute layout. It is declared per parameter, by fully qualified
name, and it exists only for the duration of one step. It says, per mesh axis, how
the tensor is distributed while the optimizer math runs. Axes it does not mention
keep their storage placement.

[VISUAL] Highlight the two arrows crossing the middle; label "the only tensors that
cross the network".

And only one tensor per parameter crosses the network in each direction. The
prepared update input, which is the Nesterov momentum blend, goes out. The update
direction comes back. The momentum update and the final parameter update both run
locally, in the storage layout.

### 5. The vocabulary (9:00-11:00)

[VISUAL] `from torchtitan.distributed.flex_shard import ComputeLayout, Owned,
BlockShard, BucketConfig, build_dist_muon` - five names.

The public API is five names. A ComputeLayout maps mesh axis names to one of four
compute shardings.

[VISUAL] Four-row table appearing one row at a time, with the Kimi usage column:
  Owned()                    whole tensor to one rank FlexShard picks, this step only   | wq_a, wkv_a, wo, router gate, shared experts, layer-0 dense MLP
  BlockShard(dim, block)     split dim into fixed blocks, distribute whole blocks        | per-head Muon: wq_b, wkv_b with block = head dims
  Shard(dim)                 DTensor's ordinary contiguous split                        | stacked experts [E, F, D]: whole expert matrices per rank
  Replicate()                every rank gets the whole tensor                           | available, unused in Kimi

Owned: the whole subgroup-local tensor goes to one rank, chosen by FlexShard, for
this step only. That is the single matrices: the attention projections, the router
gate, shared experts, the dense MLP in layer zero. BlockShard: split a dimension into
fixed-size blocks and distribute whole blocks, never splitting one. That is per-head
Muon; the block size is the head dimension. Shard: DTensor's ordinary split, used for
stacked experts, where each rank gets whole expert matrices. And Replicate, which is
there but the Kimi config does not use.

[VISUAL] Mesh axes (efsdp, ep) over an [E, F, D] tensor; two arrows both pointing at
dim 0; label order ("ep", "efsdp"); then "_StridedShard" below.

One subtlety: when two mesh axes split the same tensor dimension, DTensor applies
them in storage-mesh order. A layout can state a different order. Kimi's routed
experts under EP use Shard(0) on both the ep and the efsdp axes, with the order ep
first, then efsdp: EP keeps its expert split, and each EP-local block of experts is
spread over the expert-FSDP ranks, even though efsdp comes before ep in the storage
mesh. FlexShard lowers that to DTensor's strided-shard placements.

[VISUAL] BucketConfig(patterns=[...], name=...) with fnmatch globs; buckets drawn
as ordered boxes; layer 0 alone, then MoE layers in pairs; routed experts in their
own bucket.

The other piece is BucketConfig. It groups parameters by name pattern into ordered
buckets. Every parameter must land in exactly one bucket, and bucket order is
execution order. A bucket controls only scheduling and overlap, with one rule: all
the redistributed parameters in a bucket must share one one-dimensional
communication group. That is why Kimi keeps the routed experts, which live on the
EP/EFSDP group, in a separate bucket from the attention and shared-expert
parameters of the same layers, which live on the DP-shard group. Layer zero gets a
bucket of its own because its dense MLP is much larger; the MoE layers go in pairs,
to amortize collective launch overhead.

[VISUAL] build_dist_muon(params, compute_sharding_by_fqn=..., bucket_configs=...,
**muon_kwargs); then a split: matrices -> DistMuon; norms, embeddings, LM head ->
AdamW.

build_dist_muon ties it together. Following the Moonlight recipe, only matrix
parameters go to DistMuon: attention projections, MLPs, routed and shared experts,
routers. Norms, embeddings and the LM head stay on AdamW.

### 6. One optimizer step (11:00-13:00)

[VISUAL] Six numbered stages on a timeline, each lighting up as narrated.

DistMuon dot step is a preflight check followed by one call into the bucketed
runtime with three callbacks. Six stages.

One, preflight. Every configured parameter must have a DTensor gradient in the
storage layout the plan was built for, and its local storage must not have moved.
Failures raise before any communication starts, so an input error cannot leave
half the buckets updated. Momentum buffers are created on the first step, in the
storage layout.

Two, prepare, in the storage layout. For each parameter: momentum lerp toward the
gradient with weight one minus beta, then the Nesterov blend, lerp of gradient and
momentum, written straight into the bucket's packed send buffer.

[VISUAL] One all_to_all_single per bucket; pieces fly from storage ranks to owner.

Three, storage to compute. One all_to_all_single per bucket moves every
parameter's pieces to its compute owners.

Four, compute, in the compute layout. Newton-Schulz on whole matrices, in bf16,
batched when the compute tensor is a stack of matrices: experts, or BlockShard
heads viewed as M by R by C without a copy.

Five, compute to storage. One more all_to_all_single per bucket sends each
direction back to the ranks that store the parameter.

Six, finalize, in the storage layout. Decoupled weight decay, then param minus
adjusted learning rate times direction. Kimi scales the learning rate by the global
matrix shape: 0.2 times the square root of max of rows and columns, the
"match RMS of AdamW" rule.

[VISUAL] A parameter whose storage already equals its compute layout bypasses
stages 3 and 5.

Parameters whose storage is already compute-ready skip the two all-to-alls and run
prepare, compute and finalize locally.

### 7. Overlap (13:00-13:45)

[VISUAL] Two-lane timeline from the wiki page:
  transfer stream: [prep+a2a B0][prep+a2a B1][a2a back B0, finalize B0][prep+a2a B2][a2a back B1 ...
  compute stream:              [NS B0      ]              [NS B1      ]             [NS B2 ...

The runtime uses a dedicated transfer stream and two rolling buffer slots. While the
caller's stream runs bucket N's Newton-Schulz, the transfer stream is already
packing and exchanging bucket N plus one into the other slot, and it sends bucket
N's directions back once the compute-done event fires. Events order the two
streams; the caller waits for a slot's return trip before reusing it. The design
credits Canzona and FSDP-Canzona.

### 8. Planning once at construction (13:45-15:00)

[VISUAL] "Construction" box feeding a frozen "plan"; "step()" box labelled "replay".

Everything about who sends what to whom is decided when the optimizer is built. At
step time the runtime replays the plan. Four parts.

[VISUAL] Bars of different heights (matrices) being placed onto 4 bins (ranks),
tallest first; formula ns_steps x s^2 x (2l + s) under it.

Owners for Owned parameters are chosen by a deterministic longest-processing-time
heuristic. The load is the estimated Newton-Schulz cost, that 5 s squared 2l plus
s formula from earlier. Bytes moved are the tiebreak, and the parameter name makes
it stable. Load carries across buckets, so the whole step is balanced, not each
bucket on its own.

[VISUAL] A tensor tiled into rectangles, each rectangle with a "from rank, to rank"
label; the reverse arrows drawn by flipping.

Each move is described as rectangular regions from storage partitions to compute
partitions. The reverse direction is derived by inverting the forward routes, so
the two cannot disagree. The plan is validated once: routes must tile the tensor
exactly, and every participant's regions must cover its partition exactly.

[VISUAL] Regions for all parameters in a bucket packed into one flat buffer grouped
by destination rank, with split sizes.

Per bucket, every parameter's outgoing regions are laid into one flat buffer grouped
by destination rank, with the split sizes for a single all_to_all_single and the
matching unpack. Packing is a foreach copy over region views. When no route crosses
ranks the schedule degrades to a local copy.

[VISUAL] Each rank hashes its plan; hashes all-gathered; mismatch -> raise at
construction.

And every rank hashes its dtypes, devices, routes and item signatures and
all-gathers the hash within each bucket's group. A mismatch raises at construction,
instead of hanging or corrupting data inside a packed all-to-all at step time.

### 9. Limits and contracts (15:00-15:45)

[VISUAL] Bullet list.

The contracts are worth knowing before you adopt it. Compute layouts and buckets are
frozen at build time and are not in the optimizer state dict; restoring a checkpoint
means rebuilding the optimizer with the same values. One parameter group, no
add_param_group. Exceptions are fatal: once a step has started, a failure may have
updated some parameters with communication in flight, so the optimizer and the job
must not be reused. BlockShard in DistMuon is narrow: a 2D parameter with contiguous
local storage, blocks on dim 0, on at most one non-unit mesh axis. Numerics: batched
bf16 Newton-Schulz implements the same update as torch.optim.Muon one matrix at a
time, but bitwise equality between the two kernel schedules is not promised. And it
is eager and stream-based: it runs inside an eager optimizer.step with side streams;
there is no traced or compiled path.

### 10. Where it went: meta-pytorch/flex_shard (15:45-17:00)

[VISUAL] Repo card: github.com/meta-pytorch/flex_shard. Then the one-line API:
flex_shard(model, buckets=[BucketSpec(...)]).

The torchtitan README said the directory would be annexed into its own package, and
it was. In meta-pytorch/flex_shard, FlexShard is no longer only an optimizer trick.
It is FSDP2's runtime with one decision pulled out: how a parameter is stored
between uses and gathered for use. You call flex_shard once with a list of buckets.
Each bucket names a set of parameters, a one-dimensional mesh and a placement, and
lives in one byte buffer per rank that issues exactly one all-gather in forward and
one reduce-scatter in backward.

[VISUAL] Placement list: Shard (FSDP2 byte-for-byte parity), BucketedBlockShard,
BlockShard, BucketedOwned, blockwise FP8 gathers.

With Shard placements a bucket reproduces FSDP2 byte for byte. The other placements
exist because two consumers need whole units on one rank: Muon needs whole
matrices, and blockwise FP8 needs whole 128 by 128 tiles, so you can quantize
locally and all-gather FP8 at about half the bytes of bf16.

[VISUAL] The same six-stage timeline from section 6, with stages 3 and 5 greyed out
and the caption "plan is empty: no collective".

And here is the payoff for Muon. If the storage layout already holds each matrix
whole on the rank that will compute it, owner storage via BucketedOwned or
BlockShard, then every parameter's transition resolves to "no redistribution",
every bucket's plan is empty, and the optimizer step launches no collective at all.
The owner placement is decided at storage time instead of at step time. That removes
the exchange bytes, and it removes something less obvious: the per-bucket barrier
where every bucket waits for its slowest owner.

[VISUAL] "DeepSeek-V3 671B, 256 H100s, slowest-GPU optimizer step: 527 ms -> 301 ms
(README). Training step ~69 s." Then small: "about 0.3 % of the step".

The README's number: DeepSeek-V3 671B on 256 H100s, slowest-GPU optimizer step from
527 to 301 milliseconds. Be honest about the scale: that step is about 69 seconds,
so this is roughly 0.3 percent of it. The structural point matters more than the
number: the optimizer step is now local work, and the cost moved to uneven storage
per rank, which is why the owner balancing exists.

### 11. What carries over to TPU graph trainers (17:00-17:45)

[VISUAL] Two columns: "carries over" / "does not".

Last thing, because this channel is mostly about TPUs. What carries over is the
vocabulary. The storage-versus-compute split is the same split a graph trainer with
explicit FSDP already makes: the shard spec is the storage layout, unshard produces
a compute layout of Replicate on the FSDP axis. When an expert-parallel axis gets
added to that spec, FlexShard's convention, placements keyed by named mesh axis plus
an explicit order for axes that share a tensor dimension, is the one to adopt.

What does not carry over is the runtime. On a TPU graph trainer the optimizer is
part of one traced XLA program per rank, not an eager step with side streams. But
the planning half, owners, exact routes, one packed all-to-all per bucket, is
backend-neutral geometry. It could be emitted as all-to-all ops inside the traced
step and let XLA's scheduler provide the overlap that the two-slot stream pipeline
provides on GPU. That is an assessment, not a result; nobody has tried it yet.

### 12. Close (17:45-18:00)

[VISUAL] End card: blog URL, wiki repo, the playlist.

The written version of this, with line references into both repositories, is in
the autoresearch wiki linked below. Next in this playlist: explicit FSDP for a
graph trainer, parameter specs instead of hooks. If you are porting Muon to a
sharded trainer, say which part is unclear in the comments and I will cover it.

## Recording notes

- Record the Newton-Schulz table and the 8x32 slice-vs-whole example live in a
  notebook (`torch.linalg.svdvals`, 10 lines); it is the one demo that makes the
  "different computation, not rounding" point land.
- Screen captures: `optimizer_reshard.py` (ComputeLayout, BucketConfig),
  `dist_muon.py` (step, _assign_balanced_owner_ranks, _zeropower_via_newtonschulz),
  `_optimizer_reshard_schedule.py` (_build_packed_all_to_all_schedule,
  _validate_bucket_plans_across_ranks), `_optimizer_reshard_runtime.py` (run).
  Line numbers in the wiki page are for torchtitan `abd6e3658`; re-check on main.
- Pronounce it "Newton-Schulz" with a hard "sh". DSv3, not "DeepSeek three".
- Caption the 527 -> 301 ms figure as the README's measurement every time it is on
  screen.
