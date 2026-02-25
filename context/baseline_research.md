# Baseline Research for INFRAMIND Paper

**Date**: 2026-02-25
**Status**: Research complete, implementation pending

---

## Current Baseline

### MasRouter (ACL 2025)
- **Paper**: https://arxiv.org/abs/2502.11133
- **Code**: Already in `MAR/MasRouter/`
- **What it does**: VAE-based cascaded controller — determines collaboration mode, allocates roles, routes to specific LLMs based on task semantics only
- **Weakness**: No cost/budget awareness, no infrastructure awareness

---

## New Baseline 1: GPTSwarm (ICML 2024 Oral, top 1.5%)

### Overview
- **Paper**: https://arxiv.org/abs/2402.16823
- **Code**: https://github.com/metauto-ai/GPTSwarm (MIT License, ~1k stars)
- **Venue**: ICML 2024 Oral (top 1.5%, 144 out of 9,473)
- **Install**: `pip install gptswarm`

### How It Works
- Models multi-agent systems as **computational graphs** (nodes = LLM operations, edges = information flow)
- **Edge optimization**: Learns inter-agent connectivity via gradient-based optimization (which agents should talk to whom)
- **Node optimization**: Refines LLM prompts per node
- Within an agent, edges are fixed; inter-agent connections are optimized toward edge pruning (0) or creation (1)
- OpenAI-compatible API calls → easy vLLM adaptation

### Why It's Good for Us
- **Learns topology** like our planner, but has **zero infrastructure awareness**
- Graph optimization is purely quality-driven — ignores queue depth, latency, cost
- Under load, optimized graph topology becomes suboptimal (the "best" agent might be overloaded)
- **Already a baseline in MasRouter's paper** → reviewers expect it
- Clean modular code, MIT license

### Training Required
- Edge optimization = differentiable graph optimization over ~50-100 evaluation examples
- ~2-4 hours on a single GPU (no LLM fine-tuning, just edge weight optimization)
- Per-dataset training needed (5 datasets × ~2-4 hrs = ~10-20 hrs total)

### Implementation Plan (~3-4 days)
1. Install gptswarm, swap API calls to vLLM endpoints (OpenAI-compatible)
2. Add MBPP/GSM8K/MATH/HumanEval/MMLU dataset loaders (reuse existing)
3. Run edge optimization training on each dataset
4. Integration with evaluation harness + arrival rate sweep testing

### Benchmarks Confirmed in Paper
- HumanEval, MMLU (confirmed in paper/repo)
- MBPP, GSM8K, MATH (need to add via existing dataset loaders)

### Key Paper Story
*"Even learned multi-agent topologies degrade under system load because the graph was optimized for quality in isolation, never observing infrastructure state."*

---

## New Baseline 2: Mixture of Agents — MoA (Together AI, 2024)

### Overview
- **Paper**: https://arxiv.org/abs/2406.04692
- **Code**: https://github.com/togethercomputer/MoA (Apache 2.0, ~2.9k stars)
- **Venue**: Highly cited 2024, well-known in community

### How It Works
- **Layer 1 (Proposers)**: ALL N models generate independent responses in parallel
- **Layer 2+ (Aggregation)**: Aggregator model synthesizes all Layer 1 outputs into refined answer
- Fixed topology: always uses ALL models, no routing, no learning
- Prompt-based aggregation (no voting, no weighting)
- Core code is ~200 lines Python

### Why It's Good for Us
- **Zero routing, zero learning** — pure brute-force multi-agent ensemble
- Always queries ALL 5 models → **saturates all queues simultaneously** under load
- Latency = slowest model × number of layers (4x overhead without parallelization)
- Perfect "upper bound compute" baseline
- Extremely simple to implement

### Training Required
- **NONE** — inference only, zero training

### Implementation Plan (~2-3 days)
1. Swap Together API → vLLM endpoints (trivial, OpenAI-compatible `/v1/chat/completions`)
2. Write MoA runner with our 5 models as proposers, best model as aggregator
3. Add answer extraction for each benchmark (code exec for MBPP/HumanEval, exact match for math, etc.)
4. Integration + arrival rate sweep testing

### Known Risk
- GitHub issue #41: MoA degrades on exact-answer tasks (MATH, QA)
- Mitigation: Redesign aggregation prompt from "synthesize" to "select best answer" (majority voting with LLM judge)

### Key Paper Story
*"Brute-force ensembling of all models is catastrophically expensive under load — INFRAMIND achieves comparable quality by intelligently routing to the right model at the right time."*

---

## Baseline Comparison Matrix

| Baseline | Multi-Agent | Learned | Cost-Aware | Infra-Aware | Training | Venue |
|----------|:-----------:|:-------:|:----------:|:-----------:|:--------:|-------|
| **MoA** | Yes (all models, fixed) | No | No | No | None | arXiv 2024 |
| **GPTSwarm** | Yes (optimized graph) | Yes (edges) | No | No | Light (~2-4 hrs) | ICML 2024 Oral |
| **MasRouter** | Yes (VAE routing) | Yes (VAE) | No | No | Medium | ACL 2025 |
| **INFRAMIND** | Yes (CMDP) | Yes (PPO-Lag) | Yes (budget) | **Yes** | Full | Ours |

### Paper Narrative (Progressive Story)
1. **MoA**: No intelligence → uses everything → dies under load
2. **GPTSwarm**: Learned topology → smart in isolation → still degrades under load (can't see queues)
3. **MasRouter**: Learned routing → task-aware → but budget/infra blind
4. **INFRAMIND**: Full stack → task + budget + infrastructure → adapts under load

Each baseline adds one dimension. Only INFRAMIND has the infrastructure piece.

---

## Rejected Candidates (with reasons)

| Candidate | Venue | Why Rejected |
|-----------|-------|-------------|
| **Optima** (Tsinghua) | ACL 2025 Findings | Fine-tunes a single model (Llama-3-8B), doesn't route between models. Fixed 2-agent debate. No license. Messy code. Fundamentally different problem. |
| **xRouter** (Salesforce) | arXiv Oct 2025 | Routes to 20+ cloud APIs (GPT-5, o3). Needs full DAPO retraining. Heavy deps (VERL + SGLang + Ray). Multi-week effort. |
| **AFlow** | ICLR 2025 Oral | Depends on MetaGPT framework. MCTS workflow search is offline/slow/complex. Authors note bugs during migration. |
| **MacNet** | ICLR 2025 | Separate ChatDev branch, deeply integrated. No standalone implementation. |
| **DAAO** | arXiv Sep 2025 | No published code yet. |
| **AgentPrune** | 2024 | Pruning-only approach, not full orchestration. |
| **MARTI** | ICLR 2026 | RL training of MAS but fine-tunes LLM weights, not router. Different paradigm. |
| **MARFT** | arXiv Apr 2025 | RL methodology paper, no ready-to-use routing baseline. |
| **RouteLLM** | ICLR 2025 | Single-query router (strong vs weak), NOT multi-agent. |
| **FrugalGPT** | TMLR 2024 | Sequential cascade, NOT multi-agent collaboration. |

---

## Baselines Used by MasRouter Paper (for reference)

MasRouter compared against ~20 baselines:
- **Multi-agent**: LLM Debate, MacNet, GPTSwarm, AgentPrune, AFlow
- **Routing**: RouterDC, RouteLLM, FrugalGPT, PromptLLM
- **Single-agent**: IO, CoT, Self-Consistency, Reflexion
- **Strongest competitors**: AFlow (+1.8% gap on MBPP), GPTSwarm, RouterDC (+3.51% avg gap)

---

## Repo Organization Plan

```
MAR/
├── InfraMind/          # Our system (existing)
├── MasRouter/          # Baseline - VAE routing (existing)
├── MoA/                # Baseline - brute-force ensemble (new)
│   ├── __init__.py
│   ├── moa_runner.py   # Core MoA logic adapted for vLLM
│   └── aggregation.py  # Answer extraction per benchmark
├── GPTSwarm/           # Baseline - learned graph topology (new)
│   ├── __init__.py
│   ├── swarm_runner.py # Wrapper around gptswarm library
│   └── configs/        # Per-dataset graph configs

Experiments/
├── train_system_router_*.py      # INFRAMIND (existing)
├── run_*.py                      # MasRouter baseline (existing)
├── run_moa_*.py                  # MoA baseline (new)
├── run_gptswarm_*.py             # GPTSwarm baseline (new)

scripts/
├── baseline_train/               # MasRouter training (existing)
├── gptswarm_train/               # GPTSwarm edge optimization (new)
│   └── {dataset}/optimize_gptswarm_{dataset}.slurm
├── test/
│   └── {dataset}/
│       ├── submit_baseline_test_{dataset}.slurm    (existing)
│       ├── submit_inframind_test_{dataset}.slurm    (existing)
│       ├── submit_moa_test_{dataset}.slurm          (new)
│       └── submit_gptswarm_test_{dataset}.slurm     (new)
```

---

## Key Research Gap (Our Contribution)

After surveying 30+ papers from 2024-2026 (NeurIPS, ICML, ICLR, ACL):

**No existing paper combines multi-agent LLM collaboration with real-time infrastructure monitoring (queue depth, KV cache, latency).**

| Dimension | Existing Papers |
|-----------|----------------|
| Multi-agent collaboration | All surveyed papers |
| Cost-aware routing | xRouter, DAAO, MoMA |
| RL-trained orchestration | xRouter, Puppeteer, MAGRPO, MARTI |
| Dynamic topology | EvoMAC, AgentNet, MacNet |
| Live vLLM system metrics | **NONE** |
| Constrained optimization (Lagrangian) | **NONE** in multi-agent LLM context |
| Joint model+strategy selection | **NONE** |

INFRAMIND is the first system to make multi-agent orchestration decisions based on real-time infrastructure state using constrained optimization (PPO-Lagrangian with shared lambda).
