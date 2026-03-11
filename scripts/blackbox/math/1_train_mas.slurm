#!/bin/bash
#SBATCH --job-name=bb_mas_train
#SBATCH --output=logs/blackbox/mas/math/train_slurm-%j.out
#SBATCH --error=logs/blackbox/mas/math/train_slurm-%j.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-default
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

# =========================================================================
# Step 1: Blackbox MAS Router Training — MATH
#
#   Trains MAS baseline using blackbox API models (no GPU needed).
#   5 models: gemini-2.5-flash-lite, gemini-2.5-flash, gpt-4o-mini,
#             x-ai/grok-4.1-fast, deepseek-v3.2
#   RPM=10 per model. Response cache enabled.
#
#   Produces checkpoint used by:
#     - Step 2 (MAS test)
#     - Step 3 (InfraMind training — planner init)
#
#   Output:
#     Checkpoint: checkpoints/mas_router_blackbox/mas_blackbox_math.pth
#     CSV:        logs/blackbox/mas/math/mas_train_<jobid>.csv
# =========================================================================

echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================="
echo ""

REPO_ROOT="/home/ah872032.ucf/system-aware-mas"
cd "$REPO_ROOT" || exit 1

source scripts/setup_hpc_env.sh
source .venv/bin/activate || exit 1
echo "Python: $(which python)"
python --version
echo ""

export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED=1
export MATH_DATASET_ROOT="${BLUE_STORAGE}/datasets/MATH"

# ===== Configuration (override via env vars) =====
TRAIN_LIMIT="${TRAIN_LIMIT:-100}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-0.01}"
COST_RATE="${COST_RATE:-100}"
CONCURRENCY="${CONCURRENCY:-1000}"

# ===== Paths =====
CHECKPOINT_DIR="checkpoints/mas_router_blackbox"
mkdir -p "${CHECKPOINT_DIR}"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/mas_blackbox_math.pth"

LOG_DIR="logs/blackbox/mas/math"
mkdir -p "${LOG_DIR}"
TRAIN_CSV="${LOG_DIR}/mas_train_${SLURM_JOB_ID}.csv"

# ===== Validate dataset =====
if [[ ! -d "${MATH_DATASET_ROOT}/train" || ! -d "${MATH_DATASET_ROOT}/test" ]]; then
    echo "ERROR: MATH dataset not found at ${MATH_DATASET_ROOT}"
    exit 1
fi

echo "========================================="
echo "Blackbox MAS Training — MATH"
echo "========================================="
echo "Train limit:  ${TRAIN_LIMIT}"
echo "Epochs:       ${EPOCHS}"
echo "Batch size:   ${BATCH_SIZE}"
echo "LR:           ${LR}"
echo "Cost rate:    ${COST_RATE}"
echo "Concurrency:  ${CONCURRENCY}"
echo "Checkpoint:   ${CHECKPOINT_PATH}"
echo "Train CSV:    ${TRAIN_CSV}"
echo "Dataset:      ${MATH_DATASET_ROOT}"
echo "========================================="
echo ""

CMD="python Experiments/run_blackbox_math.py \
  --train_limit ${TRAIN_LIMIT} \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --cost_rate ${COST_RATE} \
  --concurrency ${CONCURRENCY} \
  --dataset-root ${MATH_DATASET_ROOT} \
  --train-telemetry-csv ${TRAIN_CSV} \
  --skip-test \
  --save-checkpoint ${CHECKPOINT_PATH}"

# Auto-resume from existing checkpoint
if [ -f "${CHECKPOINT_PATH}" ]; then
    echo "Found existing checkpoint: ${CHECKPOINT_PATH}"
    echo "Resuming training..."
    CMD="${CMD} --checkpoint ${CHECKPOINT_PATH}"
fi

echo "Command: ${CMD}"
echo ""

set +e
$CMD
EXIT_CODE=$?
set -e

echo ""
echo "========================================="
echo "MAS Training completed (exit: ${EXIT_CODE})"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "CSV:        ${TRAIN_CSV}"
echo "End Time:   $(date)"
echo "========================================="

if [ $EXIT_CODE -eq 0 ] && [ -f "${CHECKPOINT_PATH}" ]; then
    echo "Checkpoint size: $(du -h "${CHECKPOINT_PATH}" | cut -f1)"
    # Symlink latest for easy reference
    ln -sf "$(basename "${CHECKPOINT_PATH}")" "${CHECKPOINT_DIR}/latest.pth"
fi

exit $EXIT_CODE
