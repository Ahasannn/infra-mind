#!/bin/bash
#SBATCH --job-name=test_im_mmlu
#SBATCH --output=logs/test/mmlu/slurm-%j-inframind.out
#SBATCH --error=logs/test/mmlu/slurm-%j-inframind.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=32:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1

# =========================================================================
# InfraMind — Test Sweep for MMLU
#   Arrival rates: 10, 30, 50, 100, 150, 200 req/min
#   Budgets:       10, 30, 50, 100, 200, 300, 600, 1000 seconds
#   max_num_seqs=16
#   Each (arrival_rate, budget) pair is a separate sweep with fixed budget.
#   vLLM is drained between each config via _wait_for_vllm_drain().
#
#   Total: 6 rates x 8 budgets x 500 items = 24,000 episodes (48 sweeps)
#
#   Output CSV:
#     logs/test/mmlu/inframind_mmlu_test.csv
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

# Match training config: max_num_seqs=16
export VLLM_MAX_NUM_SEQS=16

# Ensure log directory exists
mkdir -p logs/test/mmlu

# ===== Run settings =====
TEST_LIMIT=500
CONCURRENCY=1000
ARRIVAL_RATES_CSV="10,30,50,100,150,200"
BUDGET_SWEEP="10,30,50,100,200,300,600,1000"
ARRIVAL_PATTERN="poisson"

# ===== MMLU dataset =====
DATASET_ROOT="${BLUE_STORAGE}/datasets/MMLU/data"
if [[ ! -d "${DATASET_ROOT}" ]]; then
    echo "ERROR: MMLU dataset not found at ${DATASET_ROOT}"
    exit 1
fi

# Ensure Datasets/MMLU/data symlink exists for MMLUDataset
if [[ ! -d "Datasets/MMLU/data" ]]; then
    mkdir -p Datasets/MMLU
    ln -sf "${DATASET_ROOT}" Datasets/MMLU/data
    echo "Created symlink: Datasets/MMLU/data -> ${DATASET_ROOT}"
fi

# ===== Paths =====
# TODO: Update checkpoint path after InfraMind training completes
CHECKPOINT="${BLUE_STORAGE}/checkpoints/inframind_ppo_lag/inframind_mmlu_best.pt"
MAS_CHECKPOINT="${BLUE_STORAGE}/checkpoints/mas_router/mas_mmlu_train_500_cost100.pth"

# ===== Output =====
OUTPUT_CSV="logs/test/mmlu/inframind_mmlu_test.csv"

# ===== Validation =====
for f in "$CHECKPOINT" "$MAS_CHECKPOINT"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Required file not found: $f"
        exit 1
    fi
done

# ===== Cleanup function =====
cleanup_vllm() {
    echo ""
    echo "Cleaning up vLLM servers..."
    local vllm_log_dir="logs/vllm/job_${SLURM_JOB_ID}"
    if ls "${vllm_log_dir}"/*.pid >/dev/null 2>&1; then
        for pidfile in "${vllm_log_dir}"/*.pid; do
            if [ -f "$pidfile" ]; then
                pid=$(cat "$pidfile")
                name=$(basename "$pidfile" .pid)
                if kill -0 "$pid" 2>/dev/null; then
                    echo "Stopping $name (PID $pid)"
                    kill -TERM "$pid" 2>/dev/null
                    sleep 2
                    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null
                fi
                rm -f "$pidfile"
            fi
        done
    fi
    echo "Cleanup complete"
}
trap cleanup_vllm EXIT INT TERM

# ===== Start vLLM =====
echo "Starting vLLM model pool (VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS})..."
bash scripts/vllm/serve_full_pool.sh || { echo "ERROR: Failed to start vLLM"; exit 1; }
echo "vLLM servers ready!"
bash scripts/check_vllm_status.sh
echo ""

echo "========================================="
echo "InfraMind — MMLU Test Sweep"
echo "========================================="
echo "Checkpoint:      ${CHECKPOINT}"
echo "MAS planner:     ${MAS_CHECKPOINT}"
echo "Dataset root:    ${DATASET_ROOT}"
echo "Predictors:      None (raw system metrics)"
echo "Test limit:      ${TEST_LIMIT}"
echo "Concurrency:     ${CONCURRENCY}"
echo "Arrival rates:   ${ARRIVAL_RATES_CSV}"
echo "Arrival pattern: ${ARRIVAL_PATTERN}"
echo "Budget sweep:    ${BUDGET_SWEEP}"
echo "Max-num-seqs:    ${VLLM_MAX_NUM_SEQS}"
echo "Exec temperature: 0.7"
echo "Plan temperature: 0.7"
echo "Output CSV:      ${OUTPUT_CSV}"
echo "========================================="
echo ""

set +e

python Experiments/train_inframind_mmlu.py \
    --dataset mmlu \
    --dataset-root "${DATASET_ROOT}" \
    --split test \
    --limit "${TEST_LIMIT}" \
    --epochs 1 \
    --arrival-rates "${ARRIVAL_RATES_CSV}" \
    --arrival-pattern "${ARRIVAL_PATTERN}" \
    --budget-sweep "${BUDGET_SWEEP}" \
    --concurrency "${CONCURRENCY}" \
    --mas-checkpoint "${MAS_CHECKPOINT}" \
    --checkpoint-path "${CHECKPOINT}" \
    --resume-checkpoint \
    --skip-training \
    --executor-temperature 0.7 \
    --planner-temperature 0.7 \
    --telemetry-csv "${OUTPUT_CSV}"

INFRAMIND_EXIT=$?

set -e

echo ""
echo "========================================="
echo "InfraMind Test Sweep Complete"
echo "========================================="
echo "Output CSV:  ${OUTPUT_CSV}"
echo "Exit status: ${INFRAMIND_EXIT}"
echo "End Time:    $(date)"
echo "========================================="

exit $INFRAMIND_EXIT
