# INFRAMIND: Infrastructure-Aware Multi-Agent Orchestration

**INFRAMIND** is a research framework for dynamically orchestrating LLM collaboration in multi-agent systems based on real-time infrastructure metrics and resource constraints.

Unlike traditional routing approaches that only consider task characteristics, INFRAMIND monitors vLLM infrastructure state (queue depth, KV cache usage, latency) and uses a hierarchical Constrained Markov Decision Process (CMDP) to adaptively balance accuracy, latency, and cost under dynamic system loads.

## Overview

### Key Contributions

- **System-Aware Routing**: Real-time monitoring of vLLM metrics (queue depth, KV cache usage, latency) to inform routing decisions
- **Hierarchical CMDP Architecture**: Two-level decision making — Planner (topology + role selection) and Executor (LLM + strategy routing)
- **PPO-Lagrangian Training**: Shared Lagrange multiplier automatically balances quality vs cost without hand-tuned reward coefficients
- **Hybrid Model Pool**: Support for both whitebox (self-hosted vLLM) and blackbox (API-based) models with unified profiling
- **Comprehensive Evaluation**: Tested across 5 datasets (MATH, MBPP, GSM-Hard, HumanEval, MMLU-Pro) under various load conditions
- **Multiple Baselines**: Comparison against MasRouter, GPTSwarm, and MoA

## Architecture

INFRAMIND uses a two-level decision process:

**Planner** (quality-driven, t=0):
- Selects collaboration topology (single-agent CoT, debate, hierarchical review)
- Assigns agent roles from query embedding
- Optimizes purely for quality — no budget awareness

**Executor** (infrastructure-aware, runtime):
- Selects (LLM, strategy) jointly per role from a 5×3=15 action space (5 models × 3 strategies)
- Input: query embedding + role embedding + remaining budget + system metrics
- Adapts to infrastructure state: smaller/faster models under high load, larger models when resources available

**Prompt Strategies**:
- **Flash**: Minimal reasoning (fastest, cheapest)
- **Concise**: Balanced reasoning
- **DeepThink**: Detailed step-by-step reasoning (slowest, most accurate)

### Infrastructure Monitoring

Real-time metrics collection from vLLM `/metrics` endpoints:
- Queue depth (running + waiting requests per model)
- KV cache usage (GPU memory utilization)
- Time-to-first-token (TTFT)
- Inter-token latency (ITL)
- End-to-end latency

## Quick Start

### Environment Setup

```bash
# Create virtual environment (Python 3.11)
uv venv --python 3.11
source .venv/bin/activate
uv sync --frozen

# For local vLLM serving
uv sync --frozen --extra serve
```

### Configuration

Copy `template.env` to `.env` and configure:

```bash
cp template.env .env
```

Required keys:
- `URL` / `KEY`: vLLM backend URL and API key
- `GEMINI_API_KEY`: For blackbox experiments with Gemini models
- `OPENROUTER_API_KEY`: For blackbox experiments via OpenRouter

### Local vLLM Model Pool

```bash
bash scripts/vllm/serve_full_pool.sh    # Start all models (ports 8001-8005)
bash scripts/check_vllm_status.sh       # Check health
bash scripts/vllm/stop_pool.sh          # Stop all
```

## Running Experiments

### INFRAMIND Training (Whitebox)

```bash
python Experiments/train_inframind_mbpp.py
python Experiments/train_inframind_gsm_hard.py
python Experiments/train_inframind_humaneval.py --dataset-path Datasets/humaneval/humaneval-py.jsonl
python Experiments/train_inframind_math.py --dataset-root Datasets/MATH
python Experiments/train_inframind_mmlu_pro.py
```

### INFRAMIND Training (Blackbox API Models)

```bash
python Experiments/train_inframind_blackbox_math.py
```

### Baseline: MasRouter

```bash
python Experiments/run_mbpp.py
python Experiments/run_gsm_hard.py
python Experiments/run_humaneval.py
python Experiments/run_math.py
python Experiments/run_mmlu_pro.py
```

### Baseline: GPTSwarm

```bash
python Experiments/train_gptswarm_math.py        # Training
python Experiments/run_gptswarm_math.py           # Testing
```

### Baseline: MoA (Zero-Training Ensemble)

```bash
python Experiments/run_moa_math.py
```

### Blackbox Baseline (MasRouter on API Models)

```bash
python Experiments/run_blackbox_math.py
```

### SLURM (HPC)

```bash
# Baseline training
sbatch scripts/baseline_train/math/train_mas_math.slurm

# INFRAMIND training
sbatch scripts/inframind_training/math/train_inframind.slurm

# Blackbox experiments
sbatch scripts/blackbox/math/1_train_mas.slurm
sbatch scripts/blackbox/math/3_train_inframind.slurm

# Test sweeps (arrival rate × cost rate)
sbatch scripts/test/math/submit_baseline_test_math.slurm
sbatch scripts/test/math/submit_inframind_test_math.slurm
```

## Datasets

| Dataset  | Train | Val | Test | Source |
|----------|-------|-----|------|--------|
| MATH     | 519   | 131 | 500  | Local (`Datasets/MATH/`) |
| MBPP     | 374   | 94  | 500  | HuggingFace (auto-download) |
| GSM-Hard | 500   | 125 | 500  | HuggingFace `reasoning-machines/gsm-hard` |
| HumanEval| 33    | 10  | 131  | Local JSONL |
| MMLU-Pro | 500   | 70  | 500  | HuggingFace `TIGER-Lab/MMLU-Pro` |

## Project Structure

```
.
├── MAR/                              # Core framework
│   ├── InfraMind/                    # INFRAMIND router (main contribution)
│   │   ├── inframind_router.py       #   Hierarchical CMDP (planner + executor)
│   │   ├── metrics_watcher.py        #   Real-time vLLM metrics collection
│   │   ├── blackbox_metrics.py       #   RPM-based metrics for API models
│   │   ├── trainer.py                #   PPO-Lagrangian training
│   │   └── training.py               #   Training loop orchestration
│   ├── MasRouter/                    # Baseline MAS Router (VAE-based)
│   ├── GPTSwarm/                     # GPTSwarm baseline (REINFORCE edges)
│   ├── MoA/                          # MoA baseline (brute-force ensemble)
│   ├── Graph/                        # Multi-agent execution framework
│   ├── Agent/                        # Base agent implementation
│   ├── LLM/                          # LLM interface layer
│   │   ├── model_pool.py             #   Unified whitebox/blackbox model pool
│   │   ├── blackbox_setup.py         #   API key management for blackbox models
│   │   ├── llm_profile_full.json     #   Whitebox model profiles (vLLM)
│   │   └── llm_profile_blackbox.json #   Blackbox model profiles (API)
│   ├── Roles/                        # Domain-specific agent roles
│   └── Prompts/                      # Prompt templates
│
├── Experiments/                      # Training and evaluation scripts
│   ├── train_inframind_*.py          #   INFRAMIND training (whitebox + blackbox)
│   ├── run_*.py                      #   MasRouter baseline
│   ├── train_gptswarm_*.py           #   GPTSwarm training
│   ├── run_gptswarm_*.py             #   GPTSwarm testing
│   └── run_moa_*.py                  #   MoA testing
│
├── Datasets/                         # Dataset loaders
│   ├── math_dataset.py
│   ├── mbpp_dataset.py
│   ├── gsm_hard_dataset.py
│   ├── humaneval_dataset.py
│   └── mmlu_pro_dataset.py
│
├── results/                          # Experiment results and plots
│   ├── aggregate_results.py
│   ├── plots/                        #   Generated figures (PDF/PNG/SVG)
│   └── tables.md                     #   Result tables
│
├── scripts/                          # SLURM and utility scripts
│   ├── baseline_train/{dataset}/     #   MasRouter SLURM training
│   ├── inframind_training/{dataset}/ #   INFRAMIND SLURM training
│   ├── blackbox/{dataset}/           #   Blackbox experiment scripts
│   ├── test/{dataset}/               #   Test sweep scripts
│   └── vllm/                         #   vLLM server management
│
└── visualization/                    # Plotting utilities
```

## Acknowledgments

This work builds upon the **MAS Router** framework:

```bibtex
@misc{yue2025masrouter,
  title={MasRouter: Learning to Route LLMs for Multi-Agent Systems},
  author={Yanwei Yue and Guibin Zhang and Boyang Liu and Guancheng Wan and Kun Wang and Dawei Cheng and Yiyan Qi},
  year={2025},
  eprint={2502.11133},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2502.11133}
}
```

Additional projects:
- [GPTSwarm](https://github.com/metauto-ai/GPTSwarm) — Agent graph optimization (ICML 2024)
- [vLLM](https://github.com/vllm-project/vllm) — Efficient LLM serving

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
