#!/bin/bash
# Blue storage configuration
BLUE_STORAGE="/blue/qi855292.ucf/ah872032.ucf"

# Checkpoint directory
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/mas_router"

# vLLM API configuration (EMPTY means no authentication required)
export KEY="EMPTY"

python Experiments/run_humaneval.py \
  --test_limit 33 \
  --concurrency 70 \
  --arrival-rate 10 30 50 70 100 150 200 \
  --max_agent 5 \
  --split-ratio 0.8 \
  --checkpoint "${CHECKPOINT_DIR}/mas_humaneval_train_full_100.pth" \
  --epochs 0
