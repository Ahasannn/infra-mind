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
OUTPUT_CSV="logs/motivation_plot_generator_data/baseline_motivation_sweep_math_test_500_sustained.csv"

# Arrival patterns, rates, and concurrency tuples: (pattern, arrival_rate, concurrency)
RUN_CONFIGS=(
    "sustained 20 1000"
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
        --checkpoint "${CHECKPOINT_DIR}/mas_math_train_debug.pth" \
        --test-telemetry-csv "$OUTPUT_CSV" \
        --epochs 0 \
        --batch_size 1 \
        --test_limit 500

    echo "Completed run with arrival_rate=$arrival_rate, concurrency=$concurrency, pattern=$pattern"
    echo ""
done

echo "========================================"
echo "All sweeps completed!"
echo "Results saved to: $OUTPUT_CSV"
echo "========================================"
