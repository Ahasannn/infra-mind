#!/bin/bash
#SBATCH --job-name=mas_math_c0
#SBATCH --output=logs/baseline_train/math/slurm-%j-cost0.out
#SBATCH --error=logs/baseline_train/math/slurm-%j-cost0.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1

# =========================================================================
# Temp: MAS Router training for MATH — cost_rate=0 only
#   5 epochs, 519 train items
#   ~41 min/epoch → ~3.5h total + vLLM startup → 5h wall time
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
export VLLM_MAX_NUM_SEQS=16

CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/mas_router"
mkdir -p "${CHECKPOINT_DIR}"

mkdir -p logs/baseline_train/math

MATH_DATASET_ROOT="${MATH_DATASET_ROOT:-${BLUE_STORAGE}/datasets/MATH}"
if [[ ! -d "${MATH_DATASET_ROOT}/train" || ! -d "${MATH_DATASET_ROOT}/test" ]]; then
    echo "ERROR: MATH dataset not found at ${MATH_DATASET_ROOT}"
    exit 1
fi

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

# --- Training ---
TRAIN_LIMIT=519
EPOCHS=5
BATCH_SIZE=32
LR=0.01
COST_RATE=0

CHECKPOINT_PATH="${CHECKPOINT_DIR}/mas_math_train_${TRAIN_LIMIT}_cost${COST_RATE}.pth"
TELEMETRY_CSV="logs/baseline_train/math/mas_train_math_${TRAIN_LIMIT}_cost${COST_RATE}.csv"

echo "========================================="
echo "MAS MATH — cost_rate=${COST_RATE} (quality-only)"
echo "Train limit: ${TRAIN_LIMIT}"
echo "Epochs: ${EPOCHS}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "========================================="
echo ""

CMD="python Experiments/run_math.py \
  --train_limit ${TRAIN_LIMIT} \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --cost_rate ${COST_RATE} \
  --dataset-root ${MATH_DATASET_ROOT} \
  --train-telemetry-csv ${TELEMETRY_CSV} \
  --skip-test \
  --save-checkpoint ${CHECKPOINT_PATH}"

if [ -f "${CHECKPOINT_PATH}" ]; then
    echo "Found existing checkpoint — resuming..."
    CMD="${CMD} --checkpoint ${CHECKPOINT_PATH}"
fi

echo "Command: ${CMD}"
echo ""
$CMD
RC=$?

echo ""
echo "========================================="
if [ $RC -eq 0 ]; then
    echo "SUCCESS: cost_rate=${COST_RATE} training complete"
    echo "Checkpoint: ${CHECKPOINT_PATH}"
else
    echo "FAILED: exit code ${RC}"
fi
echo "End Time: $(date)"
echo "========================================="

exit $RC
