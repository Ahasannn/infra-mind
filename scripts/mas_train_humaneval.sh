#!/bin/bash
# Blue storage configuration
BLUE_STORAGE="/blue/qi855292.ucf/ji757406.ucf"

# Checkpoint directory
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/mas_router"
mkdir -p "${CHECKPOINT_DIR}"

# Dataset directory (offline)
export HUMANEVAL_DATASET_PATH="${BLUE_STORAGE}/datasets/humaneval"

# vLLM API configuration (EMPTY means no authentication required)
export KEY="EMPTY"

python Experiments/run_humaneval.py \
  --epochs 1 \
  --batch_size 16 \
  --train_limit 100 \
  --lr 0.01 \
  --test_limit 1 \
  --split-ratio 0.8 \
  --train-telemetry-csv logs/humaneval_train_output_full_100.csv \
  --save-checkpoint "${CHECKPOINT_DIR}/mas_humaneval_train_full_100.pth"
