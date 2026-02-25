#!/bin/bash
#SBATCH --job-name=im_train_heval
#SBATCH --output=logs/inframind_training/humaneval/slurm-%j.out
#SBATCH --error=logs/inframind_training/humaneval/slurm-%j.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1

# =========================================================================
# InfraMind PPO-Lagrangian training for HumanEval
#   Small dataset: 164 total (33 train / 131 test with split-ratio 0.2)
#   Reduced epochs (20) and patience (8) due to tiny dataset.
#   Dual-pathway executor: semantic encoder + resource encoder.
#   Cost = lat/budget (no urgency multiplier).
#   Budget tiers: 10,30,50,100,200,300,600,1000 (8 tiers).
#   Arrival rates: 10,30,50,100,150,200 (6 rates).
#   No warmup: lambda active from epoch 0.
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

# --- Configuration ---
DATASET="humaneval"
DATASET_PATH="${DATASET_PATH:-${BLUE_STORAGE}/datasets/humaneval/humaneval-py.jsonl}"
MAS_CHECKPOINT="${MAS_CHECKPOINT:-${BLUE_STORAGE}/checkpoints/mas_router/mas_humaneval_train_33_cost100.pth}"

LIMIT="${LIMIT:-33}"
VAL_LIMIT="${VAL_LIMIT:-10}"
SPLIT_RATIO="${SPLIT_RATIO:-0.2}"
EPOCHS="${EPOCHS:-20}"
ARRIVAL_RATES="${ARRIVAL_RATES:-10,30,50,100,150,200}"
BUDGET_TIERS="${BUDGET_TIERS:-10,30,50,100,200,300,600,1000}"
CONCURRENCY="${CONCURRENCY:-1000}"
TRAINING_BATCH_SIZE="${TRAINING_BATCH_SIZE:-64}"
PATIENCE="${PATIENCE:-8}"

# PPO-Lagrangian hyperparameters
PPO_EPOCHS="${PPO_EPOCHS:-3}"
PPO_CLIP="${PPO_CLIP:-0.2}"
LAMBDA_INIT="${LAMBDA_INIT:-0.2}"
LR_LAMBDA="${LR_LAMBDA:-0.001}"
LAMBDA_MAX="${LAMBDA_MAX:-1.0}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"

# vLLM concurrency per server
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"

# Code execution timeout for HumanEval
export CODE_EXECUTION_TIMEOUT="10"

# Checkpoint dir
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/inframind_ppo_lag"
mkdir -p "${CHECKPOINT_DIR}"

LOG_DIR="logs/inframind_training/${DATASET}"
mkdir -p "${LOG_DIR}"
TELEMETRY_CSV="${LOG_DIR}/ppo_lag_${SLURM_JOB_ID}.csv"

export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"

# Verify MAS checkpoint for planner initialization
if [ ! -f "${MAS_CHECKPOINT}" ]; then
    echo "ERROR: MAS checkpoint not found: ${MAS_CHECKPOINT}"
    exit 1
fi
echo "  MAS checkpoint: ${MAS_CHECKPOINT} ($(du -h "${MAS_CHECKPOINT}" | cut -f1))"
echo ""

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
    echo "Cleanup complete"
}
trap cleanup_vllm EXIT INT TERM

# --- Start vLLM ---
echo "Starting vLLM model pool..."
bash scripts/vllm/serve_full_pool.sh || { echo "ERROR: Failed to start vLLM"; exit 1; }
echo "vLLM servers ready!"
bash scripts/check_vllm_status.sh
echo ""

# --- PPO-Lagrangian training ---
echo "========================================="
echo "InfraMind PPO-Lagrangian Training — HumanEval"
echo "Dataset: ${DATASET} (small: ${LIMIT} train)"
echo "Limit: ${LIMIT} train, ${VAL_LIMIT} val"
echo "Split ratio: ${SPLIT_RATIO}"
echo "Epochs: ${EPOCHS}, Patience: ${PATIENCE}"
echo "Budget tiers: ${BUDGET_TIERS}"
echo "PPO: epochs=${PPO_EPOCHS} clip=${PPO_CLIP}"
echo "Lagrangian: lambda_init=${LAMBDA_INIT} lr_lambda=${LR_LAMBDA} lambda_max=${LAMBDA_MAX}"
echo "Warmup: ${WARMUP_EPOCHS} epochs"
echo "Cost: lat/budget (no urgency)"
echo "vLLM max_num_seqs: ${VLLM_MAX_NUM_SEQS}"
echo "Arrival rates: ${ARRIVAL_RATES}"
echo "Concurrency: ${CONCURRENCY}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

echo "MAS planner init: ${MAS_CHECKPOINT}"
if [ -n "${RESUME_CHECKPOINT}" ]; then
echo "Resume from: ${RESUME_CHECKPOINT}"
fi
echo "========================================="

CMD="python Experiments/train_inframind_humaneval.py \
  --epochs ${EPOCHS} \
  --limit ${LIMIT} \
  --val-limit ${VAL_LIMIT} \
  --split-ratio ${SPLIT_RATIO} \
  --budget-tiers ${BUDGET_TIERS} \
  --arrival-rates ${ARRIVAL_RATES} \
  --concurrency ${CONCURRENCY} \
  --training-batch-size ${TRAINING_BATCH_SIZE} \
  --patience ${PATIENCE} \
  --lr-scheduler --lr-patience 3 --lr-factor 0.5 \
  --ppo-epochs ${PPO_EPOCHS} \
  --ppo-clip ${PPO_CLIP} \
  --lambda-init ${LAMBDA_INIT} \
  --lr-lambda ${LR_LAMBDA} \
  --lambda-max ${LAMBDA_MAX} \
  --warmup-epochs ${WARMUP_EPOCHS} \
  --mas-checkpoint ${MAS_CHECKPOINT} \
  --checkpoint-dir ${CHECKPOINT_DIR} \
  --telemetry-csv ${TELEMETRY_CSV} \
  --dataset-path ${DATASET_PATH}"

# Append resume flags if checkpoint provided
if [ -n "${RESUME_CHECKPOINT}" ] && [ -f "${RESUME_CHECKPOINT}" ]; then
  CMD="${CMD} --checkpoint-path ${RESUME_CHECKPOINT} --resume-checkpoint"
  echo "Resuming from: ${RESUME_CHECKPOINT}"
fi

echo "Command: $CMD"

# --- Watchdog: kill job if no log update for 1 hour ---
WATCHDOG_TIMEOUT=3600  # seconds
ERR_LOG="logs/inframind_training/${DATASET}/slurm-${SLURM_JOB_ID}.err"
(
  while true; do
    sleep 300  # check every 5 minutes
    if [ -f "$ERR_LOG" ]; then
      last_mod=$(stat -c %Y "$ERR_LOG" 2>/dev/null || echo 0)
      now=$(date +%s)
      stale=$((now - last_mod))
      if [ "$stale" -ge "$WATCHDOG_TIMEOUT" ]; then
        echo "[WATCHDOG] No log update for ${stale}s (limit: ${WATCHDOG_TIMEOUT}s). Killing job." >&2
        scancel "$SLURM_JOB_ID"
        exit 1
      fi
    fi
  done
) &
WATCHDOG_PID=$!
echo "Watchdog started (PID $WATCHDOG_PID, timeout=${WATCHDOG_TIMEOUT}s)"

set +e
$CMD
EXIT_CODE=$?
set -e

# Kill watchdog
kill $WATCHDOG_PID 2>/dev/null

# Create well-known symlink for easy access
LATEST_LINK="${LOG_DIR}/ppo_lag_train.csv"
if [ $EXIT_CODE -eq 0 ] && [ -f "${TELEMETRY_CSV}" ]; then
    ln -sf "$(basename "${TELEMETRY_CSV}")" "${LATEST_LINK}"
    echo "Symlinked: ${LATEST_LINK} -> $(basename "${TELEMETRY_CSV}")"
fi

echo ""
echo "========================================="
echo "PPO-Lagrangian training completed with exit code: $EXIT_CODE"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "Telemetry: ${TELEMETRY_CSV}"
echo "Latest link: ${LATEST_LINK}"
echo "End Time: $(date)"
echo "========================================="

exit $EXIT_CODE
