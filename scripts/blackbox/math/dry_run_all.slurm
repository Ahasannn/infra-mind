#!/bin/bash
#SBATCH --job-name=bb_dry_all
#SBATCH --output=logs/blackbox/inframind/temp/dry_run_slurm-%j.out
#SBATCH --error=logs/blackbox/inframind/temp/dry_run_slurm-%j.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-default
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1

# =========================================================================
# Dry run: All 4 blackbox steps sequentially with tiny limits
#   Step 1: MAS train (10 items, 1 epoch)
#   Step 2: MAS test  (10 items, 2 arrival rates)
#   Step 3: InfraMind train (10 items, 1 epoch)
#   Step 4: InfraMind test  (10 items, 2 arrival rates × 2 budgets)
#
#   Validates the full pipeline end-to-end.
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

# ===== Dry run paths =====
TEMP_DIR="logs/blackbox/inframind/temp"
mkdir -p "${TEMP_DIR}"

MAS_CKPT="checkpoints/mas_router_blackbox/mas_blackbox_math_dry_v2.pth"
INFRAMIND_CKPT_DIR="checkpoints/inframind_blackbox"
mkdir -p "checkpoints/mas_router_blackbox" "${INFRAMIND_CKPT_DIR}"

OVERALL_EXIT=0

# ==========================================
# Step 1: MAS Train (10 items, 1 epoch)
# ==========================================
echo ""
echo "####################################"
echo "# STEP 1: MAS Train (dry)         #"
echo "####################################"
echo ""

python Experiments/run_blackbox_math.py \
  --train_limit 10 \
  --epochs 1 \
  --batch_size 8 \
  --cost_rate 100 \
  --concurrency 1000 \
  --dataset-root "${MATH_DATASET_ROOT}" \
  --train-telemetry-csv "${TEMP_DIR}/dry_mas_train.csv" \
  --skip-test \
  --save-checkpoint "${MAS_CKPT}"

if [ $? -ne 0 ]; then
    echo "STEP 1 FAILED"
    OVERALL_EXIT=1
else
    echo "STEP 1 OK — checkpoint: ${MAS_CKPT}"
fi

if [ $OVERALL_EXIT -ne 0 ]; then
    echo "Aborting: Step 1 failed."
    exit $OVERALL_EXIT
fi

# ==========================================
# Step 2: MAS Test (10 items, 2 rates)
# ==========================================
echo ""
echo "####################################"
echo "# STEP 2: MAS Test (dry)          #"
echo "####################################"
echo ""

python Experiments/run_blackbox_math.py \
  --epochs 0 \
  --test_limit 10 \
  --concurrency 1000 \
  --checkpoint "${MAS_CKPT}" \
  --arrival-rate 2 100 \
  --arrival-pattern poisson \
  --dataset-root "${MATH_DATASET_ROOT}" \
  --test-telemetry-csv "${TEMP_DIR}/dry_mas_test.csv"

if [ $? -ne 0 ]; then
    echo "STEP 2 FAILED"
    OVERALL_EXIT=1
else
    echo "STEP 2 OK — CSV: ${TEMP_DIR}/dry_mas_test.csv"
fi

if [ $OVERALL_EXIT -ne 0 ]; then
    echo "Aborting: Step 2 failed."
    exit $OVERALL_EXIT
fi

# ==========================================
# Step 3: InfraMind Train (10 items, 1 epoch)
# ==========================================
echo ""
echo "####################################"
echo "# STEP 3: InfraMind Train (dry)   #"
echo "####################################"
echo ""

python Experiments/train_inframind_blackbox_math.py \
  --limit 10 \
  --val-limit 5 \
  --epochs 1 \
  --budget-tiers 30,300 \
  --arrival-rates 10,100 \
  --concurrency 1000 \
  --training-batch-size 8 \
  --patience 3 \
  --ppo-epochs 3 \
  --ppo-clip 0.2 \
  --lambda-init 0.5 \
  --fixed-lambda \
  --beta-cost 100 \
  --mas-checkpoint "${MAS_CKPT}" \
  --checkpoint-dir "${INFRAMIND_CKPT_DIR}" \
  --telemetry-csv "${TEMP_DIR}/dry_inframind_train.csv"

if [ $? -ne 0 ]; then
    echo "STEP 3 FAILED"
    OVERALL_EXIT=1
else
    echo "STEP 3 OK"
fi

if [ $OVERALL_EXIT -ne 0 ]; then
    echo "Aborting: Step 3 failed."
    exit $OVERALL_EXIT
fi

# Find the best InfraMind checkpoint
INFRAMIND_CKPT=$(ls -t "${INFRAMIND_CKPT_DIR}"/inframind_math_*_best.pt 2>/dev/null | head -1)
if [ -z "${INFRAMIND_CKPT}" ]; then
    # Fallback: any math checkpoint
    INFRAMIND_CKPT=$(ls -t "${INFRAMIND_CKPT_DIR}"/inframind_math_*.pt 2>/dev/null | head -1)
fi
if [ -z "${INFRAMIND_CKPT}" ]; then
    echo "ERROR: No InfraMind checkpoint found in ${INFRAMIND_CKPT_DIR}"
    echo "Step 3 may not have produced a best checkpoint (no validation improvement)."
    exit 1
fi
echo "InfraMind checkpoint: ${INFRAMIND_CKPT}"

# ==========================================
# Step 4: InfraMind Test (10 items, 2 rates × 2 budgets)
# ==========================================
echo ""
echo "####################################"
echo "# STEP 4: InfraMind Test (dry)    #"
echo "####################################"
echo ""

python Experiments/train_inframind_blackbox_math.py \
  --limit 10 \
  --epochs 1 \
  --skip-training \
  --budget-tiers 30,300 \
  --arrival-rates 2,100 \
  --concurrency 1000 \
  --lambda-init 0.5 \
  --fixed-lambda \
  --beta-cost 100 \
  --mas-checkpoint "${MAS_CKPT}" \
  --checkpoint-path "${INFRAMIND_CKPT}" \
  --telemetry-csv "${TEMP_DIR}/dry_inframind_test.csv"

if [ $? -ne 0 ]; then
    echo "STEP 4 FAILED"
    OVERALL_EXIT=1
else
    echo "STEP 4 OK — CSV: ${TEMP_DIR}/dry_inframind_test.csv"
fi

# ==========================================
# Summary
# ==========================================
echo ""
echo "========================================="
echo "DRY RUN COMPLETE (exit: ${OVERALL_EXIT})"
echo "========================================="
echo "Step 1 (MAS Train):      ${MAS_CKPT}"
echo "Step 2 (MAS Test):       ${TEMP_DIR}/dry_mas_test.csv"
echo "Step 3 (InfraMind Train): ${INFRAMIND_CKPT}"
echo "Step 4 (InfraMind Test):  ${TEMP_DIR}/dry_inframind_test.csv"
echo ""
echo "CSVs produced:"
ls -lh "${TEMP_DIR}"/dry_*.csv 2>/dev/null || echo "  (none)"
echo ""
echo "End Time: $(date)"
echo "========================================="

exit $OVERALL_EXIT
