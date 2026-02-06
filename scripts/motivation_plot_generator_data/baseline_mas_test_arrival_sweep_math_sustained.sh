#!/bin/bash
# Baseline MAS Test Arrival Rate Sweep for MATH (Sustained High Load)
# This script tests with sustained arrival pattern at high request rate

# Blue storage configuration
BLUE_STORAGE="/blue/qi855292.ucf/ah872032.ucf"

# Checkpoint directory
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/mas_router"

# Dataset directory (offline)
export MATH_DATASET_ROOT="${BLUE_STORAGE}/datasets/MATH"

# Output CSV file
OUTPUT_CSV="logs/motivation_plot_generator_data/baseline_motivation_sweep_math_test_1000_poisson_2.csv"

# Arrival patterns, rates, and concurrency tuples: (pattern, arrival_rate, concurrency)
# Key insight: Shared connection pool prevents connection explosion.
# Requests queue inside httpx pool → flow into vLLM scheduler's waiting queue (CPU memory).
# - arrival_rate: requests/minute (how fast requests arrive)
# - concurrency: max simultaneous threads (400 safe with shared pool)
# - With 5 models x 32 max_num_seqs = 160 running; rest wait in vLLM queues
# - Progressive rates show latency degradation under increasing load
RUN_CONFIGS=(
    "poisson 10 1000"
    "poisson 50 1000"
    "poisson 100 1000"
    "poisson 200 1000"
)

# Loop through each configuration
for pair in "${RUN_CONFIGS[@]}"; do
    read -r pattern arrival_rate concurrency <<< "$pair"

    echo "========================================"
    echo "Running with arrival_rate=$arrival_rate, concurrency=$concurrency, pattern=$pattern"
    echo "========================================"

    python Experiments/run_math.py \
        --concurrency "$concurrency" \
        --arrival-rate "$arrival_rate" \
        --arrival-pattern "$pattern" \
        --checkpoint "${CHECKPOINT_DIR}/mas_math_train_full.pth" \
        --test-telemetry-csv "$OUTPUT_CSV" \
        --epochs 0 \
        --test_limit 1000

    echo "Completed run with arrival_rate=$arrival_rate, concurrency=$concurrency, pattern=$pattern"
    echo ""
done

echo "========================================"
echo "All sweeps completed!"
echo "Results saved to: $OUTPUT_CSV"
echo "========================================"
