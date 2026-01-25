#!/bin/bash
# Blue storage configuration
BLUE_STORAGE="/blue/qi855292.ucf/ji757406.ucf"

# Checkpoint directory
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/mas_router"

# vLLM API configuration (EMPTY means no authentication required)
export KEY="EMPTY"

python Experiments/run_humaneval.py \
  --test_limit 131 \
  --concurrency 70 \
  --arrival-rate 10 30 50 70 100 150 200 \
  --max_agent 5 \
  --split-ratio 0.2 \
  --checkpoint "${CHECKPOINT_DIR}/mas_humaneval_train.pth" \
  --epochs 0
