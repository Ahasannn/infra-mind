#!/bin/bash
#SBATCH --job-name=gs_train_mbpp
#SBATCH --output=logs/gptswarm_training/mbpp/slurm-%j.out
#SBATCH --error=logs/gptswarm_training/mbpp/slurm-%j.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1

# =========================================================================
# GPTSwarm (ICML 2024) — REINFORCE edge optimization for MBPP
#   Learns inter-agent topology via edge probability optimization.
#   ~100 iterations with batch_size=4 per iteration.
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
DATASET="mbpp"
ITERATIONS="${ITERATIONS:-100}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-0.1}"
VAL_INTERVAL="${VAL_INTERVAL:-10}"
TRAIN_LIMIT="${TRAIN_LIMIT:-374}"
VAL_LIMIT="${VAL_LIMIT:-94}"

export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
export CODE_EXECUTION_TIMEOUT="10"

CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/gptswarm"
mkdir -p "${CHECKPOINT_DIR}"

LOG_DIR="logs/gptswarm_training/${DATASET}"
mkdir -p "${LOG_DIR}"
TELEMETRY_CSV="${LOG_DIR}/gptswarm_${SLURM_JOB_ID}.csv"

export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"

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

# --- Training ---
echo "========================================="
echo "GPTSwarm Training — MBPP"
echo "Iterations: ${ITERATIONS}"
echo "Batch size: ${BATCH_SIZE}"
echo "LR: ${LR}"
echo "Val interval: ${VAL_INTERVAL}"
echo "Train limit: ${TRAIN_LIMIT}, Val limit: ${VAL_LIMIT}"
echo "vLLM max_num_seqs: ${VLLM_MAX_NUM_SEQS}"
echo "========================================="

CMD="python Experiments/train_gptswarm_mbpp.py \
  --iterations ${ITERATIONS} \
  --batch-size ${BATCH_SIZE} \
  --lr ${LR} \
  --val-interval ${VAL_INTERVAL} \
  --train-limit ${TRAIN_LIMIT} \
  --val-limit ${VAL_LIMIT} \
  --checkpoint-dir ${CHECKPOINT_DIR} \
  --telemetry-csv ${TELEMETRY_CSV}"

RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
if [ -n "${RESUME_CHECKPOINT}" ] && [ -f "${RESUME_CHECKPOINT}" ]; then
    CMD="${CMD} --checkpoint-path ${RESUME_CHECKPOINT}"
    echo "Resuming from: ${RESUME_CHECKPOINT}"
fi

echo "Command: $CMD"

# --- Watchdog ---
WATCHDOG_TIMEOUT=3600
ERR_LOG="${LOG_DIR}/slurm-${SLURM_JOB_ID}.err"
(
  while true; do
    sleep 300
    if [ -f "$ERR_LOG" ]; then
      last_mod=$(stat -c %Y "$ERR_LOG" 2>/dev/null || echo 0)
      now=$(date +%s)
      stale=$((now - last_mod))
      if [ "$stale" -ge "$WATCHDOG_TIMEOUT" ]; then
        echo "[WATCHDOG] No log update for ${stale}s. Killing job." >&2
        scancel "$SLURM_JOB_ID"
        exit 1
      fi
    fi
  done
) &
WATCHDOG_PID=$!

set +e
$CMD
EXIT_CODE=$?
set -e

kill $WATCHDOG_PID 2>/dev/null

echo ""
echo "========================================="
echo "GPTSwarm training completed with exit code: $EXIT_CODE"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "Telemetry: ${TELEMETRY_CSV}"
echo "End Time: $(date)"
echo "========================================="

exit $EXIT_CODE
