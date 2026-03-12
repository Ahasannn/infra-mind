#!/bin/bash
#SBATCH --job-name=bb_im_train
#SBATCH --output=logs/blackbox/inframind/math/train_slurm-%j.out
#SBATCH --error=logs/blackbox/inframind/math/train_slurm-%j.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-default
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

# =========================================================================
# Step 3: Blackbox InfraMind Training — MATH
#
#   Trains InfraMind PPO-Lagrangian using blackbox API models.
#   Uses MAS checkpoint (cost_rate=15) from Step 1 as planner initialization.
#   Fixed lambda=0.3 (latency) + beta=30 (dollar cost).
#   5 models × RPM=10 = 50 RPM total pool capacity.
#
#   Requires:
#     MAS checkpoint from Step 1: checkpoints/mas_router_blackbox/mas_blackbox_math_cost15.pth
#
#   Produces checkpoint used by:
#     - Step 4 (InfraMind test)
#
#   Output:
#     Checkpoint: checkpoints/inframind_blackbox/inframind_blackbox_math_best.pt
#     CSV:        logs/blackbox/inframind/math/inframind_train_<jobid>.csv
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
LIMIT="${LIMIT:-100}"
VAL_LIMIT="${VAL_LIMIT:-0}"
EPOCHS="${EPOCHS:-3}"
CONCURRENCY="${CONCURRENCY:-1000}"
TRAINING_BATCH_SIZE="${TRAINING_BATCH_SIZE:-16}"
PATIENCE="${PATIENCE:-3}"

# Budget tiers and arrival rates
# MAS baseline latencies: rate=10→123s, rate=50→280s, rate=100→312s
# Budget 200s = tight (forces cheaper models), 500s = loose (allows quality models)
# 3 arrival rates matching MAS test sweep
BUDGET_TIERS="${BUDGET_TIERS:-200,500}"
ARRIVAL_RATES="${ARRIVAL_RATES:-10,50,100}"

# PPO-Lagrangian with fixed lambda
PPO_EPOCHS="${PPO_EPOCHS:-3}"
PPO_CLIP="${PPO_CLIP:-0.2}"
LAMBDA_INIT="${LAMBDA_INIT:-0.3}"
BETA_COST="${BETA_COST:-30}"

# ===== Paths =====
MAS_CHECKPOINT_DIR="checkpoints/mas_router_blackbox"
MAS_CHECKPOINT="${MAS_CHECKPOINT:-${MAS_CHECKPOINT_DIR}/mas_blackbox_math_cost15.pth}"

CHECKPOINT_DIR="checkpoints/inframind_blackbox"
mkdir -p "${CHECKPOINT_DIR}"

LOG_DIR="logs/blackbox/inframind/math"
mkdir -p "${LOG_DIR}"
TELEMETRY_CSV="${LOG_DIR}/inframind_train_${SLURM_JOB_ID}.csv"

# ===== Validate MAS checkpoint =====
if [ ! -f "${MAS_CHECKPOINT}" ]; then
    echo "ERROR: MAS checkpoint not found: ${MAS_CHECKPOINT}"
    echo "Run Step 1 (1_train_mas.slurm) first."
    exit 1
fi
echo "MAS checkpoint: ${MAS_CHECKPOINT} ($(du -h "${MAS_CHECKPOINT}" | cut -f1))"

echo ""
echo "========================================="
echo "Blackbox InfraMind Training — MATH"
echo "========================================="
echo "Limit:          ${LIMIT} train, ${VAL_LIMIT} val"
echo "Epochs:         ${EPOCHS}, Patience: ${PATIENCE}"
echo "Budget tiers:   ${BUDGET_TIERS}"
echo "Arrival rates:  ${ARRIVAL_RATES}"
echo "Concurrency:    ${CONCURRENCY}"
echo "Batch size:     ${TRAINING_BATCH_SIZE}"
echo "PPO:            epochs=${PPO_EPOCHS} clip=${PPO_CLIP}"
echo "Lambda:         ${LAMBDA_INIT} (fixed)"
echo "Beta (dollar):  ${BETA_COST}"
echo "MAS init:       ${MAS_CHECKPOINT}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "CSV:            ${TELEMETRY_CSV}"
echo "========================================="
echo ""

CMD="python Experiments/train_inframind_blackbox_math.py \
  --limit ${LIMIT} \
  --val-limit ${VAL_LIMIT} \
  --epochs ${EPOCHS} \
  --dataset-root ${MATH_DATASET_ROOT} \
  --budget-tiers ${BUDGET_TIERS} \
  --arrival-rates ${ARRIVAL_RATES} \
  --concurrency ${CONCURRENCY} \
  --training-batch-size ${TRAINING_BATCH_SIZE} \
  --patience ${PATIENCE} \
  --ppo-epochs ${PPO_EPOCHS} \
  --ppo-clip ${PPO_CLIP} \
  --lambda-init ${LAMBDA_INIT} \
  --fixed-lambda \
  --beta-cost ${BETA_COST} \
  --mas-checkpoint ${MAS_CHECKPOINT} \
  --checkpoint-dir ${CHECKPOINT_DIR} \
  --telemetry-csv ${TELEMETRY_CSV}"

# Resume from existing InfraMind checkpoint if provided
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
if [ -n "${RESUME_CHECKPOINT}" ] && [ -f "${RESUME_CHECKPOINT}" ]; then
    CMD="${CMD} --checkpoint-path ${RESUME_CHECKPOINT} --resume-checkpoint"
    echo "Resuming from: ${RESUME_CHECKPOINT}"
fi

echo "Command: ${CMD}"
echo ""

set +e
$CMD
EXIT_CODE=$?
set -e

echo ""
echo "========================================="
echo "InfraMind Training completed (exit: ${EXIT_CODE})"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "CSV:            ${TELEMETRY_CSV}"
echo "End Time:       $(date)"
echo "========================================="

# List produced checkpoints
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Checkpoints:"
    ls -lh "${CHECKPOINT_DIR}"/inframind_blackbox_math* 2>/dev/null || echo "  (none found)"
fi

exit $EXIT_CODE
