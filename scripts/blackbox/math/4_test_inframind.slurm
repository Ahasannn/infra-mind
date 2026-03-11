#!/bin/bash
#SBATCH --job-name=bb_im_test
#SBATCH --output=logs/blackbox/inframind/math/test_slurm-%j.out
#SBATCH --error=logs/blackbox/inframind/math/test_slurm-%j.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-default
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

# =========================================================================
# Step 4: Blackbox InfraMind Testing — MATH
#
#   Tests InfraMind using checkpoint from Step 3.
#   Sweeps over arrival rates × budget tiers.
#   5 models × RPM=10 = 50 RPM total pool capacity.
#
#   Requires:
#     MAS checkpoint from Step 1 (for planner architecture)
#     InfraMind checkpoint from Step 3
#
#   Output:
#     CSV: logs/blackbox/inframind/math/inframind_test_<jobid>.csv
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

# ===== Configuration =====
LIMIT="${TEST_LIMIT:-100}"
CONCURRENCY="${CONCURRENCY:-1000}"

# Test sweep: arrival rates × budget tiers (2x2 matrix matching training)
BUDGET_TIERS="${BUDGET_TIERS:-120,300}"
ARRIVAL_RATES="${ARRIVAL_RATES:-10,50}"

# Fixed lambda/beta (must match training)
LAMBDA_INIT="${LAMBDA_INIT:-0.5}"
BETA_COST="${BETA_COST:-100}"

# ===== Paths =====
MAS_CHECKPOINT_DIR="checkpoints/mas_router_blackbox"
MAS_CHECKPOINT="${MAS_CHECKPOINT:-${MAS_CHECKPOINT_DIR}/mas_blackbox_math.pth}"

INFRAMIND_CHECKPOINT_DIR="checkpoints/inframind_blackbox"
# Find the best checkpoint (naming: inframind_math_{run_id}_best.pt)
if [ -z "${INFRAMIND_CHECKPOINT}" ]; then
    INFRAMIND_CHECKPOINT=$(ls -t "${INFRAMIND_CHECKPOINT_DIR}"/inframind_math_*_best.pt 2>/dev/null | head -1)
    # Fall back to latest non-best checkpoint if no _best.pt exists (e.g., val-limit=0)
    if [ -z "${INFRAMIND_CHECKPOINT}" ]; then
        INFRAMIND_CHECKPOINT=$(ls -t "${INFRAMIND_CHECKPOINT_DIR}"/inframind_math_*.pt 2>/dev/null | head -1)
    fi
fi
if [ -z "${INFRAMIND_CHECKPOINT}" ]; then
    echo "ERROR: No InfraMind checkpoint found in ${INFRAMIND_CHECKPOINT_DIR}"
    echo "Run Step 3 (3_train_inframind.slurm) first."
    exit 1
fi

LOG_DIR="logs/blackbox/inframind/math"
mkdir -p "${LOG_DIR}"
TELEMETRY_CSV="${LOG_DIR}/inframind_test_${SLURM_JOB_ID}.csv"

# ===== Validate =====
if [ ! -f "${MAS_CHECKPOINT}" ]; then
    echo "ERROR: MAS checkpoint not found: ${MAS_CHECKPOINT}"
    echo "Run Step 1 (1_train_mas.slurm) first."
    exit 1
fi
echo "MAS checkpoint:      ${MAS_CHECKPOINT} ($(du -h "${MAS_CHECKPOINT}" | cut -f1))"

if [ ! -f "${INFRAMIND_CHECKPOINT}" ]; then
    echo "ERROR: InfraMind checkpoint not found: ${INFRAMIND_CHECKPOINT}"
    echo "Run Step 3 (3_train_inframind.slurm) first."
    exit 1
fi
echo "InfraMind checkpoint: ${INFRAMIND_CHECKPOINT} ($(du -h "${INFRAMIND_CHECKPOINT}" | cut -f1))"

echo ""
echo "========================================="
echo "Blackbox InfraMind Testing — MATH"
echo "========================================="
echo "Test limit:     ${LIMIT}"
echo "Concurrency:    ${CONCURRENCY}"
echo "Budget tiers:   ${BUDGET_TIERS}"
echo "Arrival rates:  ${ARRIVAL_RATES}"
echo "Lambda:         ${LAMBDA_INIT} (fixed)"
echo "Beta (dollar):  ${BETA_COST}"
echo "MAS ckpt:       ${MAS_CHECKPOINT}"
echo "InfraMind ckpt: ${INFRAMIND_CHECKPOINT}"
echo "CSV:            ${TELEMETRY_CSV}"
echo "========================================="
echo ""

CMD="python Experiments/train_inframind_blackbox_math.py \
  --limit ${LIMIT} \
  --epochs 1 \
  --skip-training \
  --dataset-root ${MATH_DATASET_ROOT} \
  --budget-tiers ${BUDGET_TIERS} \
  --arrival-rates ${ARRIVAL_RATES} \
  --concurrency ${CONCURRENCY} \
  --lambda-init ${LAMBDA_INIT} \
  --fixed-lambda \
  --beta-cost ${BETA_COST} \
  --split test \
  --mas-checkpoint ${MAS_CHECKPOINT} \
  --checkpoint-path ${INFRAMIND_CHECKPOINT} \
  --resume-checkpoint \
  --telemetry-csv ${TELEMETRY_CSV}"

echo "Command: ${CMD}"
echo ""

set +e
$CMD
EXIT_CODE=$?
set -e

echo ""
echo "========================================="
echo "InfraMind Testing completed (exit: ${EXIT_CODE})"
echo "CSV:      ${TELEMETRY_CSV}"
echo "End Time: $(date)"
echo "========================================="

exit $EXIT_CODE
