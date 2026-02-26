#!/bin/bash
#SBATCH --job-name=mas_train_heval
#SBATCH --output=logs/baseline_train/humaneval/slurm-%j.out
#SBATCH --error=logs/baseline_train/humaneval/slurm-%j.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1

# =========================================================================
# Baseline MAS Router training for HumanEval dataset
#   Small dataset: 164 total (33 train / 131 test with split-ratio 0.2)
#   Trains sequentially with cost_rates: 0, 100, 400, 700
#   10 epochs each (more epochs for tiny dataset)
#   Batch size 16 (smaller for small dataset)
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

echo "Blue Storage: ${BLUE_STORAGE}"
echo ""

# Read train_limit from dataset config
CONFIG_FILE="${REPO_ROOT}/Experiments/dataset_config.json"
TRAIN_LIMIT=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['humaneval']['train_limit'])")
echo "Train Limit (from config): ${TRAIN_LIMIT}"
echo ""

export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"
export CODE_EXECUTION_TIMEOUT="10"

CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/mas_router"
mkdir -p "${CHECKPOINT_DIR}"

LOG_DIR="logs/baseline_train/humaneval"
mkdir -p "${LOG_DIR}"

# --- vLLM cleanup ---
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
    else
        echo "No PID files found in ${vllm_log_dir}"
    fi
    pkill -u $USER -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    echo "Cleanup complete"
}
trap cleanup_vllm EXIT INT TERM

# --- Start vLLM ---
echo "Starting vLLM model pool..."
bash scripts/vllm/serve_full_pool.sh || {
    echo "ERROR: Failed to start vLLM model pool"
    exit 1
}
echo "vLLM servers ready!"
bash scripts/check_vllm_status.sh
echo ""

# --- Training parameters (adjusted for small dataset) ---
EPOCHS=10
BATCH_SIZE=16
LR=0.01
SPLIT_RATIO=0.2
COST_RATES=(0 100 400 700)

echo "========================================="
echo "Baseline MAS Training — HumanEval"
echo "Training ${TRAIN_LIMIT} samples, ${EPOCHS} epochs each"
echo "Split ratio: ${SPLIT_RATIO} (20% train / 80% test)"
echo "Cost rates: ${COST_RATES[*]}"
echo "========================================="
echo ""

set +e
FINAL_EXIT=0

for COST_RATE in "${COST_RATES[@]}"; do
    echo "========================================="
    echo "Training with cost_rate=${COST_RATE}"
    echo "========================================="

    COST_TAG="cost${COST_RATE}"
    CHECKPOINT_PATH="${CHECKPOINT_DIR}/mas_humaneval_train_${TRAIN_LIMIT}_${COST_TAG}.pth"
    TELEMETRY_CSV="${LOG_DIR}/mas_train_humaneval_${TRAIN_LIMIT}_${COST_TAG}.csv"

    CMD="python Experiments/run_humaneval.py \
      --train_limit ${TRAIN_LIMIT} \
      --epochs ${EPOCHS} \
      --batch_size ${BATCH_SIZE} \
      --lr ${LR} \
      --cost_rate ${COST_RATE} \
      --split-ratio ${SPLIT_RATIO} \
      --train-telemetry-csv ${TELEMETRY_CSV} \
      --skip-test \
      --save-checkpoint ${CHECKPOINT_PATH}"

    # Auto-resume from existing checkpoint
    if [ -f "${CHECKPOINT_PATH}" ]; then
        echo "Found existing checkpoint: ${CHECKPOINT_PATH}"
        echo "Resuming training..."
        CMD="${CMD} --checkpoint ${CHECKPOINT_PATH}"
    else
        echo "No existing checkpoint. Starting fresh..."
    fi

    echo "Command: ${CMD}"
    echo ""
    $CMD
    RC=$?

    if [ $RC -eq 0 ]; then
        echo "Successfully trained cost_rate=${COST_RATE}"
        echo "Checkpoint: ${CHECKPOINT_PATH}"
    else
        echo "Training FAILED for cost_rate=${COST_RATE} (exit code: ${RC})"
        FINAL_EXIT=$RC
        break
    fi
    echo ""
done

set -e

echo ""
echo "========================================="
echo "Training completed with exit code: $FINAL_EXIT"
echo "End Time: $(date)"
echo "========================================="

if [ ${FINAL_EXIT} -eq 0 ]; then
    echo ""
    echo "Created checkpoints:"
    for COST_RATE in "${COST_RATES[@]}"; do
        echo "  - ${CHECKPOINT_DIR}/mas_humaneval_train_${TRAIN_LIMIT}_cost${COST_RATE}.pth"
    done
fi

exit $FINAL_EXIT
