#!/bin/bash
# train_inframind_math.sh
# Train the InfraMind (CMDP) on the MATH dataset.
# Expects vLLM servers already running.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate

# ===== Load shared dataset config =====
DATASET_CONFIG="${REPO_ROOT}/Experiments/dataset_config.json"
DATASET_NAME="math"

TRAIN_LIMIT=$(python3 -c "import json; print(json.load(open('${DATASET_CONFIG}'))['${DATASET_NAME}']['train_limit'])")
ARRIVAL_PATTERN=$(python3 -c "import json; print(json.load(open('${DATASET_CONFIG}'))['${DATASET_NAME}']['arrival_pattern'])")
ARRIVAL_RATES_CSV=$(python3 -c "import json; print(','.join(str(r) for r in json.load(open('${DATASET_CONFIG}'))['${DATASET_NAME}']['arrival_rates']))")

# ===== Paths =====
BLUE_STORAGE="${BLUE_STORAGE:-/blue/qi855292.ucf/ah872032.ucf}"

# Predictor checkpoints
LATENCY_PREDICTOR="${BLUE_STORAGE}/checkpoints/predictors/latency_estimator.pth"
LENGTH_PREDICTOR="${BLUE_STORAGE}/checkpoints/predictors/length_estimator.pth"

# Budget sweep values (seconds) — sweep through fixed budgets instead of per-query CSV
# Aligned with test configuration for optimal performance
BUDGET_SWEEP="20,40,60,100,150"

# Dataset
MATH_DATASET_ROOT="${MATH_DATASET_ROOT:-${BLUE_STORAGE}/datasets/MATH}"

# Checkpoint (single path, progressively improved across configs)
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/inframind"
mkdir -p "${CHECKPOINT_DIR}"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/inframind_math.pt"

# Telemetry
TELEMETRY_DIR="logs/inframind_training/math"
mkdir -p "${TELEMETRY_DIR}"
TELEMETRY_CSV="${TELEMETRY_DIR}/inframind_train_math.csv"

# ===== Validation =====
for f in "$LATENCY_PREDICTOR" "$LENGTH_PREDICTOR"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Required file not found: $f"
        exit 1
    fi
done
if [[ ! -d "${MATH_DATASET_ROOT}/train" ]]; then
    echo "ERROR: MATH dataset not found at ${MATH_DATASET_ROOT}"
    exit 1
fi

# ===== Environment =====
export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"

echo "========================================="
echo "InfraMind Training — MATH"
echo "========================================="
echo "Train limit: ${TRAIN_LIMIT}"
echo "Arrival rates: ${ARRIVAL_RATES_CSV}"
echo "Arrival pattern: ${ARRIVAL_PATTERN}"
echo "Budget sweep: ${BUDGET_SWEEP}"
echo "========================================="
echo ""

# ===== Training =====
CMD="python -m MAR.InfraMind.training \
  --dataset math \
  --dataset-root ${MATH_DATASET_ROOT} \
  --limit ${TRAIN_LIMIT} \
  --epochs 1 \
  --max-tokens 4096 \
  --arrival-rates ${ARRIVAL_RATES_CSV} \
  --arrival-pattern ${ARRIVAL_PATTERN} \
  --budget-sweep ${BUDGET_SWEEP} \
  --concurrency 1000 \
  --latency-predictor ${LATENCY_PREDICTOR} \
  --length-predictor ${LENGTH_PREDICTOR} \
  --checkpoint-path ${CHECKPOINT_PATH} \
  --checkpoint-every 50 \
  --telemetry-csv ${TELEMETRY_CSV}"

# Resume from existing checkpoint if present
if [[ -f "${CHECKPOINT_PATH}" ]]; then
    echo "Resuming from checkpoint: ${CHECKPOINT_PATH}"
    CMD="${CMD} --resume-checkpoint"
fi

echo "Command: ${CMD}"
echo ""

$CMD
echo ""
echo "Training complete."
