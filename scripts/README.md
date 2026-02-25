# Scripts Directory

SLURM job scripts, bash utilities, and diagnostic tools for InfraMind.

## Directory Structure

```
scripts/
├── setup_hpc_env.sh              # HPC environment (BLUE_STORAGE, HF_HOME, etc.)
├── check_vllm_status.sh          # Health check for vLLM servers
│
├── baseline_train/               # Baseline MAS Router training
│   ├── math/train_mas_math.slurm
│   ├── mbpp/train_mas_mbpp.slurm
│   ├── gsm8k/train_mas_gsm8k.slurm
│   ├── humaneval/train_mas_humaneval.slurm
│   └── mmlu/train_mas_mmlu.slurm
│
├── inframind_training/           # InfraMind PPO-Lagrangian training
│   ├── math/train_inframind.slurm
│   ├── mbpp/train_inframind.slurm
│   ├── gsm8k/train_inframind.slurm
│   ├── humaneval/train_inframind.slurm
│   └── mmlu/train_inframind.slurm
│
├── test/                         # Test sweeps (arrival rate × budget/cost)
│   ├── math/
│   │   ├── submit_baseline_test_math.slurm
│   │   └── submit_inframind_test_math.slurm
│   ├── mbpp/
│   │   ├── submit_baseline_test_mbpp.slurm
│   │   └── submit_inframind_test_mbpp.slurm
│   ├── gsm8k/
│   │   ├── submit_baseline_test_gsm8k.slurm
│   │   └── submit_inframind_test_gsm8k.slurm
│   ├── humaneval/
│   │   ├── submit_baseline_test_humaneval.slurm
│   │   └── submit_inframind_test_humaneval.slurm
│   └── mmlu/
│       ├── submit_baseline_test_mmlu.slurm
│       └── submit_inframind_test_mmlu.slurm
│
├── vllm/                         # vLLM server management
│   ├── serve_full_pool.sh        # Start all 5 models (ports 8001-8005)
│   └── stop_pool.sh              # Stop all models
│
├── utils/                        # Diagnostic utilities
│   ├── test_dataset_loaders.py   # Verify dataset loaders before SLURM jobs
│   ├── test_vllm_metrics.py      # Test vLLM metrics endpoints
│   └── setup_jupyter_kernel.sh   # One-time Jupyter kernel setup
│
└── _deprecated/                  # Old scripts kept for reference
```

## Workflow

### 1. Train Baseline MAS Router

```bash
sbatch scripts/baseline_train/math/train_mas_math.slurm
sbatch scripts/baseline_train/mbpp/train_mas_mbpp.slurm
sbatch scripts/baseline_train/gsm8k/train_mas_gsm8k.slurm
sbatch scripts/baseline_train/humaneval/train_mas_humaneval.slurm
sbatch scripts/baseline_train/mmlu/train_mas_mmlu.slurm
```

Each trains 3 checkpoints (cost_rate=100, 400, 700).

### 2. Train InfraMind

Requires baseline MAS checkpoint (cost_rate=100) for planner initialization.

```bash
sbatch scripts/inframind_training/math/train_inframind.slurm
sbatch scripts/inframind_training/mbpp/train_inframind.slurm
sbatch scripts/inframind_training/gsm8k/train_inframind.slurm
sbatch scripts/inframind_training/humaneval/train_inframind.slurm
sbatch scripts/inframind_training/mmlu/train_inframind.slurm
```

### 3. Test Sweeps

Baseline sweeps: 3 cost_rates x 6 arrival_rates.
InfraMind sweeps: 8 budget_tiers x 6 arrival_rates.

```bash
# Baseline
sbatch scripts/test/{dataset}/submit_baseline_test_{dataset}.slurm

# InfraMind
sbatch scripts/test/{dataset}/submit_inframind_test_{dataset}.slurm
```

## Dataset Configuration

Train/test limits are centralized in `Experiments/dataset_config.json`.

| Dataset   | Train | Val | Test | Notes |
|-----------|-------|-----|------|-------|
| MATH      | 519   | 131 | 500  | `--dataset-root` |
| MBPP      | 374   | 94  | 500  | HF auto-download |
| GSM8K     | 500   | 125 | 500  | `--dataset-path` |
| HumanEval | 33    | 10  | 131  | `--split-ratio 0.2` |
| MMLU      | 500   | 125 | 500  | `--dataset-root` |

## Log Locations

- Baseline training: `logs/baseline_train/{dataset}/`
- InfraMind training: `logs/inframind_training/{dataset}/`
- Test sweeps: `logs/test/{dataset}/`
- vLLM: `logs/vllm/`
