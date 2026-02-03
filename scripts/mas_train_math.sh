#!/bin/bash
# Repo root (script is in scripts/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Blue storage configuration
BLUE_STORAGE="/blue/qi855292.ucf/ah872032.ucf"

# Checkpoint directory
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/mas_router"
mkdir -p "${CHECKPOINT_DIR}"

# vLLM API configuration (EMPTY means no authentication required)
export KEY="EMPTY"

# Dataset root (override with MATH_DATASET_ROOT)
MATH_DATASET_ROOT="${MATH_DATASET_ROOT:-${BLUE_STORAGE}/datasets/MATH}"
if [[ ! -d "${MATH_DATASET_ROOT}/train" || ! -d "${MATH_DATASET_ROOT}/test" ]]; then
    echo "MATH dataset not found at ${MATH_DATASET_ROOT}."
    echo "Expected ${MATH_DATASET_ROOT}/train and ${MATH_DATASET_ROOT}/test."
    echo "Download and extract the dataset there, or set MATH_DATASET_ROOT."
    exit 1
fi

python Experiments/run_math.py \
  --epochs 1 \
  --batch_size 16 \
  --train_limit 3 \
  --lr 0.01 \
  --test_limit 1 \
  --dataset-root "${MATH_DATASET_ROOT}" \
  --train-telemetry-csv logs/baseline_math_train_output_test_3.csv \
  --save-checkpoint "${CHECKPOINT_DIR}/baseline_mas_math_train_test_3.pth"
