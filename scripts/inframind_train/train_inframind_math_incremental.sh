#!/bin/bash
# train_inframind_math_incremental.sh
# Incremental training: Resume from existing checkpoint and train ONLY on new (rate, budget) pairs.
# This saves compute by avoiding retraining on already-seen combinations.
#
# Already trained (24 pairs):
#   Rates: [10, 50, 100, 200] × Budgets: [10, 20, 30, 50, 100, 200]
#
# New pairs to train (24 pairs):
#   - New rates [30, 150] × all budgets [10, 20, 30, 50, 70, 100, 150, 200]
#   - Old rates [10, 50, 100, 200] × new budgets [70, 150]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate

# ===== Paths =====
BLUE_STORAGE="${BLUE_STORAGE:-/blue/qi855292.ucf/ah872032.ucf}"

LATENCY_PREDICTOR="${BLUE_STORAGE}/checkpoints/predictors/latency_estimator.pth"
LENGTH_PREDICTOR="${BLUE_STORAGE}/checkpoints/predictors/length_estimator.pth"
MATH_DATASET_ROOT="${MATH_DATASET_ROOT:-${BLUE_STORAGE}/datasets/MATH}"

CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/inframind"
mkdir -p "${CHECKPOINT_DIR}"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/inframind_math.pt"

TELEMETRY_DIR="logs/inframind_training/math"
mkdir -p "${TELEMETRY_DIR}"
TELEMETRY_CSV="${TELEMETRY_DIR}/inframind_train_math_incremental.csv"

# ===== Validation =====
for f in "$LATENCY_PREDICTOR" "$LENGTH_PREDICTOR" "$CHECKPOINT_PATH"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Required file not found: $f"
        exit 1
    fi
done

# ===== Environment =====
export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"

TRAIN_LIMIT=519
ARRIVAL_PATTERN="poisson"

echo "========================================="
echo "InfraMind Incremental Training — MATH"
echo "========================================="
echo "Resuming from: ${CHECKPOINT_PATH}"
echo "Train limit: ${TRAIN_LIMIT}"
echo "========================================="
echo ""

# ===== Phase 1: Train new rates [30, 150] with all budgets =====
echo "Phase 1: Training new rates [30, 150] × budgets [10, 20, 30, 50, 70, 100, 150, 200]"
echo "  → 2 rates × 8 budgets × 519 items = 8,304 episodes"
echo ""

python -m MAR.InfraMind.training \
  --dataset math \
  --dataset-root "${MATH_DATASET_ROOT}" \
  --limit "${TRAIN_LIMIT}" \
  --epochs 1 \
  --max-tokens 4096 \
  --arrival-rates "30,150" \
  --arrival-pattern "${ARRIVAL_PATTERN}" \
  --budget-sweep "10,20,30,50,70,100,150,200" \
  --concurrency 1000 \
  --latency-predictor "${LATENCY_PREDICTOR}" \
  --length-predictor "${LENGTH_PREDICTOR}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --resume-checkpoint \
  --checkpoint-every 50 \
  --telemetry-csv "${TELEMETRY_CSV}"

echo ""
echo "Phase 1 complete."
echo ""

# ===== Phase 2: Train old rates [10, 50, 100, 200] with new budgets [70, 150] =====
echo "Phase 2: Training old rates [10, 50, 100, 200] × new budgets [70, 150]"
echo "  → 4 rates × 2 budgets × 519 items = 4,152 episodes"
echo ""

python -m MAR.InfraMind.training \
  --dataset math \
  --dataset-root "${MATH_DATASET_ROOT}" \
  --limit "${TRAIN_LIMIT}" \
  --epochs 1 \
  --max-tokens 4096 \
  --arrival-rates "10,50,100,200" \
  --arrival-pattern "${ARRIVAL_PATTERN}" \
  --budget-sweep "70,150" \
  --concurrency 1000 \
  --latency-predictor "${LATENCY_PREDICTOR}" \
  --length-predictor "${LENGTH_PREDICTOR}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --resume-checkpoint \
  --checkpoint-every 50 \
  --telemetry-csv "${TELEMETRY_CSV}"

echo ""
echo "========================================="
echo "Incremental Training Complete"
echo "========================================="
echo "Total new episodes trained: 12,456"
echo "  Phase 1: 8,304 episodes (new rates × all budgets)"
echo "  Phase 2: 4,152 episodes (old rates × new budgets)"
echo "Coverage: 48 total (rate, budget) pairs (24 old + 24 new)"
echo "Updated checkpoint: ${CHECKPOINT_PATH}"
echo "Telemetry: ${TELEMETRY_CSV}"
echo "========================================="
