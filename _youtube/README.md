# YouTube channel kit

Channel: https://studio.youtube.com/channel/UCOFAG2zVaGD4DYGvthcQ4pA
(public: https://www.youtube.com/channel/UCOFAG2zVaGD4DYGvthcQ4pA).

State on 2026-10-31 (applied through YouTube Studio in Chrome): handle
`@vlasenkoalexey` (`@alekseyvlasenko` was taken), description, 5 links,
contact email, banner, avatar = `avatar-photo.png` (the 144 px GitHub photo
upscaled; YouTube keeps one source image and rescales it for every slot; the
user tried a sharper 886 px headshot and chose to keep this one), country US, keywords,
upload defaults, Home tab on, 6 public playlists below. Phone-verified 2026-10-31.
Video 1 published 2026-10-31 at 1080p (https://youtu.be/VZzN9X-emNA, unlisted since 2026-11-02 with a pointer to the 4K re-upload); live version re-uploaded 2026-11-02 in 4K: https://youtu.be/nj4bztH4EL0 (public, playlist 1). Upload at 1440p or above: YouTube then uses its higher-quality VP9/AV1 ladder and the player starts sharper.
The channel also carries 8 pre-existing PRIVATE personal playlists - leave them.

This folder mirrors `_linkedin/` and `_x/`: per-platform copy that is NOT
rendered by Jekyll (underscore dir, not a declared collection). Everything
below is ready to paste into Studio, or to push with `yt_channel_apply.py`
(see "Pushing through the API").

Files:

| File | What |
|---|---|
| `channel.json` | Machine-readable source of truth: description, keywords, links, playlists, upload defaults |
| `yt_channel_apply.py` | Pushes `channel.json` + banner to the channel via YouTube Data API v3 |
| `make_channel_art.py` | Regenerates the banner, avatar and the first thumbnail |
| `2026-11-muon-flexshard.md` | First video: title, description, tags, chapters, thumbnail, full script |
| `2026-11-spmd-types.md` | Second video (spmd_types): title, description, tags, chapters, thumbnail, full script |
| `../assets/images/youtube/banner.png` | 2048x1152 banner, text inside the 1235x338 safe area |
| `../assets/images/youtube/avatar-photo.png` | 800x800 upscaled GitHub photo, the live avatar (alt: `avatar.png` monogram) |
| `../assets/images/youtube/thumb-muon-flexshard.png` | 1280x720 thumbnail for video 1 |
| `../assets/images/youtube/thumb-spmd-types.png` | 1280x720 thumbnail for video 2 (first frame of the video) |

## 1. Identity (Studio -> Customization -> Basic info)

- Name: **Aleksey Vlasenko** (same as the blog; keep the personal brand, the
  blog tagline already says what the channel is about).
- Handle: pick the first free one of `@alekseyvlasenko`, `@vlasenkoalexey`
  (matches GitHub), `@aleksey_vlasenko`. The handle can only be set in Studio.
- Country: United States. Language: English.
- Profile picture: `assets/images/youtube/avatar-photo.png` (Studio -> Branding),
  the 144x144 GitHub photo upscaled; `avatar.png` is a monogram alternative.
- Banner: `assets/images/youtube/banner.png` (Studio -> Branding, or the script).

## 2. Description (max 1000 chars; this is 825)

```
Technical deep dives on training large models on TPUs with PyTorch: TorchTPU and torchtitan, distributed training internals (FSDP, expert parallelism, Muon and FlexShard), profiling with XProf, XLA and Pallas kernels, and autoresearch - letting coding agents run the performance-optimization loop end to end.

I am a Tech Lead Manager on the PyTorch TPU team at Google. These videos are the long-form, visual companion to my blog, where every topic is also written up with code, profiles and numbers.

Blog: https://vlasenkoalexey.github.io
Autoresearch wiki: https://github.com/vlasenkoalexey/tpu_performance_autoresearch_wiki
GitHub: https://github.com/vlasenkoalexey
LinkedIn: https://www.linkedin.com/in/oleksiyvlasenko
X: https://x.com/AlekseyVlasenko

New videos every few weeks. Opinions are my own, not my employer's.
```

## 3. Links (Studio -> Customization -> Basic info -> Links; first one shows on the banner)

| Title | URL |
|---|---|
| Blog | https://vlasenkoalexey.github.io |
| Autoresearch wiki (GitHub) | https://github.com/vlasenkoalexey/tpu_performance_autoresearch_wiki |
| GitHub | https://github.com/vlasenkoalexey |
| LinkedIn | https://www.linkedin.com/in/oleksiyvlasenko |
| X | https://x.com/AlekseyVlasenko |

Contact email (Studio -> Basic info -> Contact info): vlasenkoalexey@gmail.com.

## 4. Channel keywords (Studio -> Settings -> Channel -> Basic info; 257 chars)

```
PyTorch TPU "distributed training" FSDP "Muon optimizer" FlexShard torchtitan TorchTPU XLA Pallas XProf profiling autoresearch "LLM agents" "machine learning performance" MFU "mixture of experts" "expert parallelism" "model training" "deep learning systems"
```

## 5. Playlists (Studio -> Content -> Playlists -> New)

| # | Title | Description | Seeded by |
|---|---|---|---|
| 1 | Distributed training internals (PyTorch) | How the pieces of a distributed PyTorch training step actually work: FSDP storage layouts, expert parallelism, optimizer resharding, Muon and FlexShard, DTensor placements, collectives and their overlap. Read from the source, with diagrams. | video 1 (Muon/FlexShard) |
| 2 | TPU auto-optimization with agents (autoresearch) | The autoresearch series: an autonomous loop that proposes a hypothesis, patches model code, runs on real TPUs, reads the profile and files a verdict. Companion to the blog series at https://vlasenkoalexey.github.io/tags/autoresearch/ | blog posts 2026-05-01, 06-05, 06-26 |
| 3 | PyTorch on TPU: TorchTPU and torchtitan | Training PyTorch models on Cloud TPUs: the TorchTPU backend, torchtitan recipes, scan over layers, graph trainers, launching on GKE, and what the profiles say. | wiki: tpu lane, scan, explicit FSDP |
| 4 | Profiling and compilers: XProf, XLA, HLO | Reading an xprof trace, understanding HLO and fusions, XLA flags that matter, memory profiles, and the tooling (xprof-cli, perfsim overlays) used to find headroom. | wiki: profile-analyzer-index, perfsim-overlay |
| 5 | Pallas kernels on TPU | Writing and tuning Pallas/Mosaic kernels for TPU: attention, GEMM, normalization, MoE kernels, block sizes, VMEM, and having agents author kernels with adversarial verification. | blog post 2026-07-31, wiki kernels/ |
| 6 | Agent tooling: wikify, xprof-mcp, skills | The tools that make coding agents useful for performance work: grounded codebase wikis (wikify-repo), the XProf MCP server and CLI, skills and sub-agents, process auditors. | blog post 2026-07-04 |

Layout (Studio -> Customization -> Layout): one "Series" section listing all
six playlists; after video 1 is up, set it as the channel trailer for new
visitors and as the featured video for returning subscribers.

## 6. Upload defaults (Studio -> Settings -> Upload defaults)

- Visibility public, category Science & Technology, language English,
  standard license, not made for kids, embedding allowed, comments on with
  hold-for-review of potentially inappropriate comments.
- Default tags: PyTorch, TPU, distributed training, machine learning, deep learning, performance.
- Description footer (appended to every video):

```
---
Blog: https://vlasenkoalexey.github.io
Autoresearch wiki: https://github.com/vlasenkoalexey/tpu_performance_autoresearch_wiki
LinkedIn: https://www.linkedin.com/in/oleksiyvlasenko
X: https://x.com/AlekseyVlasenko
```

## 7. Video backlog (derived from the blog and the wiki)

| # | Working title | Playlist | Source material |
|---|---|---|---|
| 1 | Muon and FlexShard: why the optimizer needs whole matrices - PUBLISHED https://youtu.be/nj4bztH4EL0 (4K re-upload; 1080p original VZzN9X-emNA unlisted) | 1 | `2026-11-muon-flexshard.md`; wiki analysis 2026-10-29 FlexShard; `codebases/flex_shard/` silo |
| 2 | An agent optimized Llama 3 8B on TPU while I slept: the autoresearch loop | 2 | blog 2026-05-01 |
| 3 | Keeping an autonomous agent running for 10 hours: loop, stop hook, retrospectives | 2 | blog 2026-06-05 |
| 4 | Same loop, four agents: Claude Code, Codex, Gemini CLI, Antigravity | 2 | blog 2026-06-26 |
| 5 | wikify-repo: grounded LLM wikis instead of RAG for codebases | 6 | blog 2026-07-04 |
| 6 | Teaching the loop to write Pallas kernels (and audit itself) | 5 | blog 2026-07-31 |
| 7 | Reading a TPU training profile in 20 minutes (xprof walkthrough) | 4 | wiki `profile-analyzer-index.md`, xprof-cli |
| 8 | How far is a training step from perfsim? Measured vs predicted, op class by op class | 4 | wiki `/perfsim-overlay` skill, Flint/Qwen3-235B/Muse Glimmer charts |
| 9 | Explicit FSDP for a graph trainer: parameter specs instead of hooks | 3 | wiki analysis 2026-10-26 explicit parallelism by parameter specs |
| 10 | Scan over layers on TPU: compile time, collectives, and what XLA does with the loop carry | 3 | wiki concepts scan-over-layers, qwen3 v022/v023 |
| 11 | MoE on TPU: ragged_dot, expert parallelism, and why per-expert size sets your MFU | 3 | wiki concepts expert-parallelism, qwen3 maxtext v001 cross-arch note |
| 12 | Splash attention block sizes and VMEM: tuning one kernel across v5p, v6e, v7x | 5 | wiki kernels/, observations on block sizes and scoped VMEM |

## 8. Pushing through the API

```
pip install google-api-python-client google-auth-oauthlib   # done on this box
python3 _youtube/yt_channel_apply.py auth-url          # open URL as the channel owner, approve
python3 _youtube/yt_channel_apply.py auth-code '<the http://localhost:8765/?... url you land on>'
python3 _youtube/yt_channel_apply.py apply --dry-run
python3 _youtube/yt_channel_apply.py apply
```

The script sets description, keywords, country, language, uploads the banner,
creates the six playlists (skips ones that exist by title) and a "Series"
section. It reuses the gws OAuth client and enables the YouTube Data API on
its project on first run. Name, handle, avatar and trailer are Studio-only.

## 9. Upload notes (learned 2026-10-31)

- Unverified channels are capped at 15 min; a 20-min upload is accepted then
  abandoned with "Video is too long". Phone-verify first (youtube.com/verify).
- First description paragraph = a one-line hook (title-like), blank line, then the body.
- Studio rejects descriptions containing `<` or `>` ("Angled brackets aren't
  allowed") - write "from X to Y", not "X -> Y".
- Chapter timestamps come from the composition's section starts
  (`artifacts/props.json` in the OpenMontage project), not from the script plan.
- Agent upload path: the Chrome extension bridge caps file uploads at 10 MB, and
  fetch() from studio.youtube.com to localhost is blocked by Chrome's local
  network permission; what works is focusing "Select files" and driving the
  native GTK dialog with xdotool on DISPLAY=:20 (ctrl+L, type path, Return).
