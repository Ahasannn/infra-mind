#!/bin/bash
# train_inframind_math_dryrun.sh
# Dry-run: validate the InfraMind training pipeline with a tiny slice.
# Expects vLLM servers already running.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate

# ===== Dry-run config — small slice =====
DATASET_NAME="math"
TRAIN_LIMIT=5           # tiny slice to validate pipeline
ARRIVAL_RATES_CSV="10"  # single rate for speed
ARRIVAL_PATTERN="poisson"
BUDGET_SWEEP="10,30"    # just 2 budgets to test sweep logic

# ===== Paths =====
BLUE_STORAGE="${BLUE_STORAGE:-/blue/qi855292.ucf/ah872032.ucf}"

# Predictor checkpoints
LATENCY_PREDICTOR="${BLUE_STORAGE}/checkpoints/predictors/latency_estimator.pth"
LENGTH_PREDICTOR="${BLUE_STORAGE}/checkpoints/predictors/length_estimator.pth"

# Dataset
MATH_DATASET_ROOT="${MATH_DATASET_ROOT:-${BLUE_STORAGE}/datasets/MATH}"

# Checkpoint (dry-run uses separate file to avoid overwriting real checkpoint)
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/inframind"
mkdir -p "${CHECKPOINT_DIR}"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/inframind_math_dryrun.pt"

# Telemetry
TELEMETRY_DIR="logs/inframind_training/math"
mkdir -p "${TELEMETRY_DIR}"
TELEMETRY_CSV="${TELEMETRY_DIR}/inframind_train_math_dryrun.csv"

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
echo "InfraMind DRY-RUN Training — MATH"
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
  --max-tokens 256 \
  --arrival-rates ${ARRIVAL_RATES_CSV} \
  --arrival-pattern ${ARRIVAL_PATTERN} \
  --budget-sweep ${BUDGET_SWEEP} \
  --concurrency 2 \
  --latency-predictor ${LATENCY_PREDICTOR} \
  --length-predictor ${LENGTH_PREDICTOR} \
  --checkpoint-path ${CHECKPOINT_PATH} \
  --checkpoint-every 5 \
  --telemetry-csv ${TELEMETRY_CSV}"

echo "Command: ${CMD}"
echo ""

$CMD
echo ""
echo "Dry-run training complete."
