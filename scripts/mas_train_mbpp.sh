#!/bin/bash
# Blue storage configuration
BLUE_STORAGE="/blue/qi855292.ucf/ji757406.ucf"

# Checkpoint directory
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/mas_router"
mkdir -p "${CHECKPOINT_DIR}"

# Dataset directory (offline)
export MBPP_DATASET_PATH="${BLUE_STORAGE}/datasets/mbpp/full"

# vLLM API configuration (EMPTY means no authentication required)
export KEY="EMPTY"

# Clear Python cache to ensure fresh configuration
export PYTHONDONTWRITEBYTECODE=1

python Experiments/run_mbpp.py \
  --epochs 1 \
  --batch_size 16 \
  --lr 0.01 \
  --test_limit 1 \
  --train-telemetry-csv logs/my_train_output_full.csv \
  --save-checkpoint "${CHECKPOINT_DIR}/mas_mbpp_train_full.pth"
