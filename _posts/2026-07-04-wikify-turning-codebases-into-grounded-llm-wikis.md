---
title: "🧠 wikify repo — turning codebases into grounded LLM wikis"
date: 2026-07-04
categories: [Machine Learning, Agents]
tags: [autoresearch, llm-wiki, wikify, scip, grounding, code-comprehension]
image:
  path: /assets/images/wikify-turning-codebases-into-grounded-llm-wikis/hero.jpeg
  alt: "wikify repo — turning codebases into grounded LLM wikis"
---

For the [TPU Model Performance Auto-optimization](https://github.com/vlasenkoalexey/tpu_performance_autoresearch_wiki) project introduced in the [first post]({% post_url 2026-05-01-tpu-model-performance-auto-optimization %}) of this series we proposed to use an **LLM wiki** to store information about model optimizations, experiment results and relevant codebases. But we didn't spend any time justifying that decision.

In this post we are going to address that and answer the following questions:

- Why we should use an LLM wiki and not a RAG-based solution
- What benefits we get if we ingest codebases into the LLM wiki
- What it actually takes to ingest a codebase
- And finally we propose our own solution named [**wikify-repo**](https://github.com/vlasenkoalexey/wikify-repo) for ingesting codebases

## 📚 Why an LLM wiki and not RAG

Let's start from the choice that predates this whole project. When LLM responses have to be grounded in data that wasn't present during training, the standard answer is RAG: chunk the dataset, ingest it into a vector database that pre-computes embeddings, and enrich each request with the chunks closest to the prompt in embedding space. That's the technology behind MaxCode and MaxKernel in [accelerator-agents](https://github.com/AI-Hypercomputer/accelerator-agents).

RAG is well studied, but it has real issues. From a practical user's point of view, a vector database is a black box: not straightforward to update, and almost impossible to inspect or edit. And there are conceptual issues too — Pinecone, the company behind one of the most popular vector databases, [publicly concluded](https://www.pinecone.io/learn/series/beyond-retrieval/rag-applicability-problem/) that retrieval alone isn't enough and RAG needs a meta layer to work efficiently.

The lightweight alternative is the [**LLM wiki**](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) pattern popularized by Andrej Karpathy. Three stages:

- **Ingest** — raw sources go into an immutable collection: articles, papers, notes. The LLM reads them, never modifies them, and produces summary pages with extracted concepts, cross-linked to related concepts and back to the sources. Immutability is deliberate: you can always re-compile the wiki from scratch.
- **Query** — you ask a question. The LLM doesn't search raw documents; it reads the already-synthesized wiki. Everything is indexed, so it loads only the pages it needs.
- **Lint** — periodically the LLM audits the whole wiki: contradictions, orphan pages, concepts mentioned but missing their own page. The wiki stays healthy because the LLM does the maintenance no human ever wants to do.

Side by side:

| RAG | LLM wiki |
|---|---|
| Raw docs stay raw | Raw docs **compiled** into structured wiki pages |
| Retrieves chunks per query | Reads pre-synthesized pages |
| Black box; extra infrastructure to query and update | Indexed markdown files — easy to read, navigate, diff |
| Unit of operation: a chunk with an embedding | Unit of operation: a **concept**, connected to other concepts |
| Stateless — every query starts from scratch | Stateful — knowledge compounds over time |
| Cheap per query | Expensive ingest, **cheap query** |
| Hallucinations stay local to one answer | Hallucinations can get **baked in as "facts"** during ingest |
| Scalability: millions of entries | Scalability: thousands of entries |
| Best for: large, fast changing corpora, fact lookup, millions of docs | Best for: limited number of mostly static curated sources, research projects, personal knowledge, books |

The economics are the interesting row: RAG does its work at query time, over and over; the wiki does the work once at ingest and amortizes it over every future query. For a limited, mostly static, curated corpus — exactly what a research project accumulates — that trade wins. And the compiled artifact is just markdown, so it can be shared directly: **anyone who clones the wiki gets the full benefit of the ingestion cost someone else already paid.** 

It is also perfect for distribution: you share a repo of markdown files, not a binary database snapshot.

Conceptually, the wiki is an **annotated map** that points into raw data and lets the agent find what it needs with minimal effort. No code, no database — just context engineering. And that's a perfect fit for agentic engineering.

It might sound complex, but in reality building wikis is pretty straightforward. You can check this visualization for the one used in the TPU Model Performance Auto-optimization project: 
[![The wiki graph: every markdown page the agent has built or read, and the cross-links between them. Click to interact.](/assets/images/tpu-model-performance-auto-optimization/wiki.png)](https://vlasenkoalexey.github.io/tpu_performance_autoresearch_wiki/tools/graph/index.html)


## 🧑‍💻 What benefits we get if we ingest codebases into the LLM wiki

The wiki pattern above was described for prose — articles, papers, notes. But for a model-optimization project, some of the most valuable "documents" are codebases: the model you're optimizing, the framework it runs on, and the reference implementations you're chasing. Out of the box an LLM knows these codebases only as fuzzy training-data memories, usually of some older version. Keeping them as plain submodules helps — the agent can grep — but grep gives you text matches, not understanding, and re-discovering the same code structure from scratch in every session burns tokens and time.

Ingesting a codebase into the wiki — turning it into an **annotated navigation map** the agent reads before touching the source — unlocks a set of capabilities that plain checkout access doesn't give you:

- **Efficient navigation.** The agent opens the repo's overview page, follows the map to the right concept or module page, and only then drops into the actual source at the exact file and line. Minimal context spent, no directory-listing archaeology at the start of every session.
- **Grounded answers about how a codebase works.** "How does chunked cross-entropy work here?", "what happens between optimizer step and device sync?" — answered from pages that cite real symbols, with a pinned source link one hop away when line-level certainty matters. This matters double for internals questions where a wrong-but-confident answer leads to a wrong code edit.
- **Cross-repo stack-trace tracking.** An ML program doesn't execute in one repo: a TorchTitan model runs through PyTorch, dispatches into the TorchTPU backend, lowers to StableHLO, and lands in XLA and the TPU driver. With each layer ingested into the same wiki, the agent can follow a crash or a suspicious HLO op across those boundaries and reason about the *end-to-end* execution path — TorchTitan → PyTorch → TorchTPU → XLA → driver — instead of stopping at the first repo's edge.
- **Extracting optimization techniques.** A reference codebase is a catalog of applied optimizations: kernels, sharding strategies, precision tricks. With the reference ingested, those techniques become named, citable concepts the agent can find, compare against your model's profile, and adapt — instead of vague "MaxText does something clever with attention" recollections.
- **Model and optimization migration between codebases.** Converting a HuggingFace PyTorch model to JAX, or porting an optimization from MaxText into TorchTitan, needs both ends understood: what the source actually does, and where the equivalent mechanism lives in the target. With both codebases in the wiki this becomes a mapping exercise between two grounded descriptions — the generic version of the problem MaxCode solves specifically for PyTorch → MaxText conversion.

Not every repo plays the same role, and it's useful to name them, because the role determines how much ingestion depth each one deserves:

| Role | Examples | Why ingest |
|---|---|---|
| Documentation sources | scaling-book | domain knowledge, ingested as prose |
| Reference implementations | MaxText, MaxDiffusion | the SOTA baseline to compare against and borrow from |
| Code to draw from | tokamax | kernels and utilities to transplant |
| Frameworks | JAX, PyTorch, TorchTPU, torchax | resolve stack traces and dispatch paths below the model layer |
| Code we actually change | \<your repo\> | the optimization target itself — edited on per-experiment branches |


## ⛏️ What it actually takes to ingest a codebase

My first attempt was exactly the naive thing: give the agent high-level instructions to "ingest this codebase" the same way it ingests articles. It was better than nothing — but following the original method literally would mean the LLM reads every file, extracts concepts, and builds connections by hand. Wasteful, shallow, and the summaries drift from the code the moment it changes.

There are a few OSS tools that solve this problem, the most popular being [graphify](https://github.com/safishamsi/graphify) and [understand-anything](https://github.com/labolado/understand-anything). The full list of tools and how they compare to each other can be found in [codebase-cartography-wiki](https://github.com/vlasenkoalexey/codebase-cartography-wiki).

I could have used one of those tools, but none of them worked well for my use case. All of them used a proprietary "black box" data format which is not very friendly for sharing, and they used shallow AST code parsing.

Their high-level idea is to extract an [AST (abstract syntax tree)](https://en.wikipedia.org/wiki/Abstract_syntax_tree) map from the codebase, identify the most important nodes, have an LLM agent annotate them, and store the data in a format that is easy to query. Fast and build-free, but the view is syntactic: the AST tells you a name *appears*; it can't tell you which symbol that name actually *refers to*. No cross-file resolution, no real call graph, no type information; C++ macros and templates break textual extraction; Python decorators and metaclasses trip the walkers. You get a map where every `forward` is potentially every other `forward`.

Turns out there is a better solution — [SCIP](https://github.com/sourcegraph/scip), the code-intelligence format behind Sourcegraph. It fixes this by running the *actual language toolchain*: `scip-python` is pyright, `scip-clang` is a real compilation. Symbol resolution, cross-file references, types, macros, templates all come out the way the compiler sees them. The trade-off is upfront cost (you index the codebase once, C++ needs a build), but for a stable corpus of reference codebases that's paid once and amortized — and wiki *consumers* pay nothing at all.

## 🛠️ Introducing wikify-repo

To make codebase ingestion a first-class citizen of my LLM-wiki-based solution, I built a specialized tool named [**wikify-repo**](https://github.com/vlasenkoalexey/wikify-repo).

The idea is similar: record every class, method, and their relationships with SCIP, then spend the LLM annotating only the most central ~20% of nodes — enough to explain ~80% of the repo, while the rest still get a deterministic catalog page so nothing is dropped. Then drop everything into the existing LLM wiki and connect core concepts. The big selling point is that information retrieval doesn't require any specialized tools or binaries, since the output is plain markdown. It offers a single `/wikify-ingest-repo` skill which handles the whole ingestion process end-to-end.

Here is how it compares to [graphify](https://github.com/safishamsi/graphify), [understand-anything](https://github.com/labolado/understand-anything), and Google's hosted Code Wiki:

| | **wikify-repo** | graphify | understand-anything | Google Code Wiki |
|---|---|---|---|---|
| **Specialization** | Grounded markdown wiki you own — for trusted agent retrieval | Multi-modal knowledge graph (code + docs + media) | Visual codebase onboarding — explore it as a graph | Zero-setup hosted docs for public repos |
| **Output** | ✅ Markdown wiki — pages in your git repo | ➖ Knowledge graph (HTML + JSON) | ➖ React-Flow graph dashboard | ❌ Hosted web docs only |
| **Code structure from** | ✅ **SCIP** — compiler-grade symbol resolution (scip-python / scip-clang). **Full semantic mapping** | ➖ tree-sitter AST, **name-based** (20 languages). Syntactic mapping. | ➖ tree-sitter AST, **name-based**. Syntactic mapping. | ❔ Gemini (closed) |
| **Faithfulness** | ✅ **Citation linter is a hard build gate**; uncited → `[!inferred]` | ➖ `EXTRACTED / INFERRED / AMBIGUOUS` labels — honest, not gated | ❌ LLM per-node summaries, unverified | ❌ *"AI-generated map, not a source of truth"* |
| **Coverage** | ✅ **Deterministic set-difference** — every module gets a page | ➖ Leiden community clustering | ➖ analyzes discovered files — no stated completeness | ❔ not specified |
| **Inputs** | ➖ code + prose (docs / articles) | ✅ **widest** — code, SQL, shell, docs, papers, images, audio/video | ➖ code + docs / LLM-wikis | ➖ code repos only |
| **Retrieval** | ✅ `grep` + `index.md` — **no embeddings, no DB, no additional tools** | ➖ graph queries + clusters (no embeddings) | ➖ name + semantic search in the dashboard | ➖ hosted UI + Gemini chat — no MCP / API |
| **Updates** | ✅ **idempotent reconcile** — `--ref` rebuilds only changed *symbols* | ✅ `--update` re-extracts only changed *files* (caches semantic passes) | ✅ incremental — re-analyzes only changed *files* | ✅ auto-maintained (hosted) |
| **Ownership** | ✅ plain markdown in your repo — offline, git-diffable | ➖ local graph files | ➖ local dashboard | ❌ **Google-hosted** (private repos waitlisted) |

<sub>✅ strong · ➖ partial / trade-off · ❌ weak or absent · ❔ unknown / closed</sub>

And here is the repo where I did a more in-depth analysis of similar OSS tools and their implementations: [codebase-cartography-wiki](https://github.com/vlasenkoalexey/codebase-cartography-wiki).

**wikify-repo** is meant to be used to ingest codebases into LLM wiki projects. To start from scratch refer to the demo/template project [wikify-repo-demo](https://github.com/vlasenkoalexey/wikify-repo-demo): just click the "Use this template" button, update the README in your project, and you should be good.

## 🏁 Closing thoughts

I didn't set out to build another code-mapping tool. What pushed me was a practical wall: the existing tools store what they learn about your code in proprietary formats — a graph database here, a dashboard bundle there — and that made the result impossible to share the way the rest of my LLM wiki is shared. A knowledge base you can't hand to a teammate as a git repo, diff in review, or point any agent at without installing the tool that produced it wasn't going to work for me. And once I was building for markdown output anyway, the other steps followed: SCIP instead of name-based AST parsing, and citations gated by a linter instead of unverified summaries. wikify-repo is simply those steps taken together.

Now all repos referenced in the [TPU Performance Auto-optimization](https://github.com/vlasenkoalexey/tpu_performance_autoresearch_wiki) project are re-ingested using the wikify-repo tool. But to be clear, nothing about it is specific to TPUs or this optimization: wikify-repo fits anywhere you'd reach for graphify or understand-anything — onboarding onto an unfamiliar codebase, documenting your own, giving a coding agent a reliable map of a framework it keeps touching. Same job, except the artifact it produces is plain markdown in your own repo: shareable, diffable, and readable by any agent — or human — with nothing installed.

A fitting way to test it was to point it at the alternatives themselves: [codebase-cartography-wiki](https://github.com/vlasenkoalexey/codebase-cartography-wiki) compares the code-mapping tools by ingesting their source code — every claim in the comparison cites the implementation it describes. And to see what the output looks like on a large production ML codebase, browse the [MaxText silo](https://github.com/vlasenkoalexey/tpu_performance_autoresearch_wiki/blob/main/wiki/codebases/maxtext/overview.md) in the TPU performance wiki — a grounded map of exactly the kind of repo this series optimizes.

Starting your own is two steps: "Use this template" on [wikify-repo-demo](https://github.com/vlasenkoalexey/wikify-repo-demo), then `ingest <your-repo>` from Claude Code, Codex, or Antigravity.