---
title: "Making Andrej Karpathy's autoresearch production-ready"
date: 2026-06-05
categories: [Machine Learning, Agents]
tags: [autoresearch, llm-agents, optimization]
hidden: true # remove this line to publish
---

Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) is a
genuinely brilliant idea. I was able to adapt to to model performance optimization domain and it proved to work in its original form — see my earlier write-up on
[TPU model performance auto-optimization]({% post_url 2026-05-01-tpu-model-performance-auto-optimization %}).

Also as I mentioned there were some issues with running it - I often had to babysit the process where model complained that it can't make further progress, that it exhaused all ideas, or simply asked a confirmation for a simple action.

[Autoresearch prompt](https://github.com/karpathy/autoresearch/blob/master/program.md) has a snippet explicitly telling model that it should never stop, I really like it:
```
NEVER STOP: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
```

And nevertherless it did stop, complain, asked for permission. Same as working with engineers 🙂

Before getting into details I'd like to highlight that this is not a ctitique of the autoresearch work, it is a practical solution I ended up using to solve those problems in order to productionize autoresearch loop for my specific use-caes. Autoresearch is perfect in its simplicity, it is a minimal repo to demonstrate an idea, and it doesn't need bells and whistles that would otherwise distract from the main point ❤️

## Background

When I first demoed auto-optimization repo some people were confused. Where is the code that sould be run here. The whole repo is just a collection of markdown files, how it is even supposed to work?

In the end everything is just a prompt for LLM, and everything we are doing here is a clever prompt engineering. That is what Andrej Karpathy calls [Software 3.0](https://www.youtube.com/watch?v=LCEmiRjPEtQ).

- Software 1.0 is writing traditional programs
- Software 2.0 is trainign models to solve specific problems - object detection for example, this is kind of how ML was evolving before ChatGPT become a mainstream
- Software 3.0 is a prompt or collection of prompts given to LLM agent, usually running in some kind of agentic harness like Claude Code, Codex and Gemini CLI

With this distinction in mind we can now see that Software 1.0 and Software 3.0 have different properties.

Software 1.0 is a determenistic and rigid, does exactly what it was coded to do. It is totally possible to code autoresearch loop using traditional code, and it would solve problem with auto-stop. But should we? Non-deteremenism is such autoresearch loop is a feature, not a bug. 

Another important distinction is that with Softare 3.0 it is often easy to get something working, and that's one of the reasons why AI prototypes are poping out left and right. But it is much harder to make it working reliably, measure performance, and find ways to improve performance.

With that in mind, let's explore problems that auto-research loop has in TPU model auto-optimization framework, and how they were addressed or at least mitigated.

## Agent not following instructions

Complaining, asking for permission, stopping optimization process are all symptoms of one problem - context pollution. Once I started investigate what's going on turned out that model is often skipping steps, it is not collecting profiles, doing blind flag sweeps, not properly generating reports, diverging from instructions because it decided that it knows better. 

What's happening is that model always have loop instructions in the context. When you start optimization loop, those instructions are at the top of the context, and model always treats it as highest priority. But as conversation goes, autoresearch instructions are pushed deeper and deeper into the context. Context window for modern LLMs is pretty large, like 1M tokens for Opus 4.8. So why model should pay attention to that tiny snippet of instructions compared to everything else it already accumulated into the context on top of it?

The fix is simple, force model to keep instructions at the top of the context window. There should be an external mechanism not related to model loop (because otherwise it can just ignore those instructions) to force that. What I ended up with is adding a scheduller that pushes autoreseach loop instructions on top of the context, which would be same as if you were to tell model to re-read instructions manually. All harnesses support this feature now, in Claude Code you do it using /loop skill. 

To automate this for every new experiment I came up with [/start-experiment](https://github.com/vlasenkoalexey/tpu_performance_autoresearch_wiki/blob/main/.claude/skills/start-experiment/SKILL.md) and [/stop-experiment](https://github.com/vlasenkoalexey/tpu_performance_autoresearch_wiki/blob/main/.claude/skills/stop-experiment/SKILL.md) skills. 

All preparation and validation logic moved to /start-experiment, and it also initializes /loop to force auto-research loop reload, and starts iteration itself. /stop-experiment stops loops, and performs cleanup.

This is a hardcore solution that obviously wastes tokens, and maybe there is a better one like using traditional code to orchestrate the optimization process. But it is good enough, and honestly tokens should not be of a concern with auto-optimization anyways, it is going to waste a lot of them.

## Agent stopping early

Similarly to previous problem, there are clear instructions for agent to never stop the optimization process. But even with /loop in place it still does. Solution for this is simple, at least for Claude Code that supports [stop hooks](https://code.claude.com/docs/en/hooks). When LLM is trying to pass control back to the user, stop hook is triggered. A minor inconvenience is that stop hook is global, and that it affects all sessions. But since it allows running some code, session can be filtered so it looks like stop hook logic only triggers for specific session. Similar to /loop, /start-experiment skill can optionally register a stop hook for long running optimization processes. See implementation here: https://github.com/vlasenkoalexey/tpu_performance_autoresearch_wiki/blob/main/.claude/stop_hook.sh

## Agent narrowing down to one specific direction

Often agent complained that it exhaused all ideas and can't make progress. And in some sense it could be true, if agent gets to do a flag sweep for one of the hyper-parameters, once it tries all values that make sense it can say that I'm done, and there is no more work to do. It was a common problem that agent spends a lot of time exploring one domain, but completely ignores others. To make analogy with engineering work if you get too deep into the problem and get stuck, it is a good idea to make a step back and review your data at a higher level. Same solution applies to models. Extending on a stop hook idea, it is possible to write any instruction into the stop hook to be returned back to agent. 
To step back you can tell agent to do a retrospective over all experiments it did so far, check areas that it explore in depth, and choose areas that could have some gaps. This can be coded as a skill, see [/create-retrospective](https://github.com/vlasenkoalexey/tpu_performance_autoresearch_wiki/blob/main/.claude/skills/create-retrospective/SKILL.md) skill that is bundled into the stop hook. Sill itself can be called into manually at any moment.

## Parallelizing research

Original auto-research solution implies that there is a single optimization loop running. In my case the codebase I worked with had multiple models, and it was natural to try running separate loops for multiple models. Due to the way autoresearch is setup - create a new branch for each experiment, doing that directly would mess up branches. First I started with running separate forks of whole repo, but that was wasteful and inconvenient.

I experimented with few approaches:

- one clever orchestrator managing branches, and running optimization loop; and dumb workers executing instructions - didn't scale because of context pollution
- one dumb orchestrator only managing branches, and clever workers that managed optimization loops - this worked, but in the end you'd need to build a dashboard to track work of every worker
- fork/copy repo to the experiment folder and have a separate optimization session per model - this is what I ended up using. 

Logic for bootstrapping per-experiment fork is hardcoded into /start-experiment skill. Main repo that is being modified lives under /raw/code/<repo_name>. Once experiment is started, that repo is copied to /wiki/experiment/<experiment_name>/<experiment_lane>/.repo/<repo_name>. With this setup in place, it doesn't matter anymore what's the top branch in main repo anymore, each session is doing code manipulation its own repo now. And only successful experiments are merge back into main repo.

This is a bit wasteful in a sense that we have to copy/fork the whole repo for each experiment, but disk space is cheap, and can drop local repo copy once experiment is complete.


## Closing thoughts

Changes described above addressed problem of mis-behaved agents for Claude Code, but that was still did not make Codex or Gemini CLI to run auto-optimization loop reliably. Will describe solution for that in a separate post.