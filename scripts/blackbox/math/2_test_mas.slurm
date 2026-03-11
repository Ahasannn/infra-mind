#!/bin/bash
#SBATCH --job-name=bb_mas_test
#SBATCH --output=logs/blackbox/mas/math/test_slurm-%j.out
#SBATCH --error=logs/blackbox/mas/math/test_slurm-%j.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-default
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

# =========================================================================
# Step 2: Blackbox MAS Router Testing — MATH
#
#   Tests MAS baseline using checkpoint from Step 1.
#   Sweeps over arrival rates to measure quality + latency under load.
#   5 models × RPM=10 = 50 RPM total pool capacity.
#
#   Requires:
#     Checkpoint from Step 1: checkpoints/mas_router_blackbox/mas_blackbox_math.pth
#
#   Output:
#     CSV: logs/blackbox/mas/math/mas_test_<jobid>.csv
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
TEST_LIMIT="${TEST_LIMIT:-100}"
CONCURRENCY="${CONCURRENCY:-1000}"
ARRIVAL_PATTERN="poisson"

# Arrival rates to sweep (match InfraMind: low=10, high=50)
ARRIVAL_RATES="${ARRIVAL_RATES:-10 50}"

# ===== Paths =====
CHECKPOINT_DIR="checkpoints/mas_router_blackbox"
CHECKPOINT_PATH="${MAS_CHECKPOINT:-${CHECKPOINT_DIR}/mas_blackbox_math.pth}"

LOG_DIR="logs/blackbox/mas/math"
mkdir -p "${LOG_DIR}"
TEST_CSV="${LOG_DIR}/mas_test_${SLURM_JOB_ID}.csv"

# ===== Validate =====
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: MAS checkpoint not found: ${CHECKPOINT_PATH}"
    echo "Run Step 1 (1_train_mas.slurm) first."
    exit 1
fi
echo "Checkpoint: ${CHECKPOINT_PATH} ($(du -h "${CHECKPOINT_PATH}" | cut -f1))"

if [[ ! -d "${MATH_DATASET_ROOT}/test" ]]; then
    echo "ERROR: MATH test dataset not found at ${MATH_DATASET_ROOT}/test"
    exit 1
fi

echo ""
echo "========================================="
echo "Blackbox MAS Testing — MATH"
echo "========================================="
echo "Test limit:    ${TEST_LIMIT}"
echo "Concurrency:   ${CONCURRENCY}"
echo "Arrival rates: ${ARRIVAL_RATES}"
echo "Pattern:       ${ARRIVAL_PATTERN}"
echo "Checkpoint:    ${CHECKPOINT_PATH}"
echo "Test CSV:      ${TEST_CSV}"
echo "========================================="
echo ""

CMD="python Experiments/run_blackbox_math.py \
  --epochs 0 \
  --test_limit ${TEST_LIMIT} \
  --concurrency ${CONCURRENCY} \
  --checkpoint ${CHECKPOINT_PATH} \
  --arrival-rate ${ARRIVAL_RATES} \
  --arrival-pattern ${ARRIVAL_PATTERN} \
  --dataset-root ${MATH_DATASET_ROOT} \
  --test-telemetry-csv ${TEST_CSV}"

echo "Command: ${CMD}"
echo ""

set +e
$CMD
EXIT_CODE=$?
set -e

echo ""
echo "========================================="
echo "MAS Testing completed (exit: ${EXIT_CODE})"
echo "CSV:      ${TEST_CSV}"
echo "End Time: $(date)"
echo "========================================="

exit $EXIT_CODE
