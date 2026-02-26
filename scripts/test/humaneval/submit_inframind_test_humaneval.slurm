#!/bin/bash
#SBATCH --job-name=test_im_heval
#SBATCH --output=logs/test/humaneval/slurm-%j-inframind.out
#SBATCH --error=logs/test/humaneval/slurm-%j-inframind.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1

# =========================================================================
# InfraMind — Test Sweep for HumanEval
#   Small dataset: 131 test items (split-ratio 0.2)
#   Arrival rates: 10, 30, 50, 100, 150, 200 req/min
#   Budgets:       10, 30, 50, 100, 200, 300, 600, 1000 seconds
#   max_num_seqs=16
#   Each (arrival_rate, budget) pair is a separate sweep with fixed budget.
#   vLLM is drained between each config via _wait_for_vllm_drain().
#
#   Total: 6 rates x 8 budgets x 131 items = 6,288 episodes (48 sweeps)
#
#   Output CSV:
#     logs/test/humaneval/inframind_humaneval_test.csv
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
export CODE_EXECUTION_TIMEOUT="10"

# Match training config: max_num_seqs=16
export VLLM_MAX_NUM_SEQS=16

# Ensure log directory exists
mkdir -p logs/test/humaneval

# ===== Run settings =====
TEST_LIMIT=131
SPLIT_RATIO=0.2
CONCURRENCY=1000
ARRIVAL_RATES_CSV="10,30,50,100,150,200"
BUDGET_SWEEP="10,30,50,100,200,300,600,1000"
ARRIVAL_PATTERN="poisson"

# ===== Paths =====
DATASET_PATH="${BLUE_STORAGE}/datasets/humaneval/humaneval-py.jsonl"
# TODO: Update checkpoint path after InfraMind training completes
CHECKPOINT="${BLUE_STORAGE}/checkpoints/inframind_ppo_lag/inframind_humaneval_best.pt"
MAS_CHECKPOINT="${BLUE_STORAGE}/checkpoints/mas_router/mas_humaneval_train_33_cost0.pth"

# ===== Output =====
OUTPUT_CSV="logs/test/humaneval/inframind_humaneval_test.csv"

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
echo "InfraMind — HumanEval Test Sweep"
echo "========================================="
echo "Checkpoint:      ${CHECKPOINT}"
echo "MAS planner:     ${MAS_CHECKPOINT}"
echo "Dataset path:    ${DATASET_PATH}"
echo "Predictors:      None (raw system metrics)"
echo "Test limit:      ${TEST_LIMIT}"
echo "Split ratio:     ${SPLIT_RATIO}"
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

python Experiments/train_inframind_humaneval.py \
    --dataset humaneval \
    --dataset-path "${DATASET_PATH}" \
    --split-ratio "${SPLIT_RATIO}" \
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
