#!/bin/bash
#SBATCH --job-name=test_bl_heval
#SBATCH --output=logs/test/humaneval/slurm-%j-baseline.out
#SBATCH --error=logs/test/humaneval/slurm-%j-baseline.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1

# =========================================================================
# Baseline MAS Router — Test Sweep for HumanEval
#   Small dataset: 131 test items (split-ratio 0.2)
#   Cost rates:    100, 400, 700  (one checkpoint each)
#   Arrival rates: 10, 30, 50, 100, 150, 200 req/min
#   max_num_seqs=16
#   vLLM is drained between each (cost_rate, arrival_rate) config.
#
#   Total: 3 cost_rates x 6 arrival_rates x 131 items = 2,358 episodes
#
#   Output CSV:
#     logs/test/humaneval/baseline_humaneval_test.csv
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
COST_RATES=(100 400 700)
ARRIVAL_RATES=(10 30 50 100 150 200)
ARRIVAL_PATTERN="poisson"

# ===== Checkpoint directory =====
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/mas_router"

# Read train_limit from config for checkpoint naming
CONFIG_FILE="${REPO_ROOT}/Experiments/dataset_config.json"
TRAIN_LIMIT=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['humaneval']['train_limit'])")

# ===== Output CSV =====
OUTPUT_CSV="logs/test/humaneval/baseline_humaneval_test.csv"

# ===== Validation =====
for cost in "${COST_RATES[@]}"; do
    CKPT="${CHECKPOINT_DIR}/mas_humaneval_train_${TRAIN_LIMIT}_cost${cost}.pth"
    if [[ ! -f "${CKPT}" ]]; then
        echo "ERROR: Checkpoint not found: ${CKPT}"
        exit 1
    fi
    echo "  checkpoint cost=${cost}: ${CKPT}"
done
echo ""

# ===== vLLM drain function =====
VLLM_PORTS=(8001 8002 8003 8004 8005)

drain_vllm() {
    echo "[Drain] Waiting for all vLLM queues to empty..."
    local start_time
    start_time=$(date +%s)
    local last_log=0

    while true; do
        local total_active=0
        local details=""

        for port in "${VLLM_PORTS[@]}"; do
            local metrics
            metrics=$(curl -s -m 5 "http://127.0.0.1:${port}/metrics" 2>/dev/null || echo "")

            local running=0
            local waiting=0
            if [[ -n "$metrics" ]]; then
                running=$(echo "$metrics" | grep -E "^vllm:num_requests_running" | head -1 | awk '{print int($2)}')
                waiting=$(echo "$metrics" | grep -E "^vllm:num_requests_waiting" | head -1 | awk '{print int($2)}')
                running=${running:-0}
                waiting=${waiting:-0}
            fi

            local active=$((running + waiting))
            total_active=$((total_active + active))
            if [[ $active -gt 0 ]]; then
                details="${details} port${port}:${running}r+${waiting}w"
            fi
        done

        if [[ $total_active -eq 0 ]]; then
            local elapsed=$(( $(date +%s) - start_time ))
            if [[ $elapsed -gt 5 ]]; then
                echo "[Drain] All queues empty after ${elapsed}s"
            fi
            return 0
        fi

        local elapsed=$(( $(date +%s) - start_time ))
        if [[ $((elapsed - last_log)) -ge 30 ]]; then
            echo "[Drain] Waiting (${elapsed}s elapsed, ${total_active} active):${details}"
            last_log=$elapsed
        fi

        sleep 5
    done
}

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

TOTAL_CONFIGS=$(( ${#COST_RATES[@]} * ${#ARRIVAL_RATES[@]} ))

echo "========================================="
echo "Baseline MAS Router — HumanEval Test Sweep"
echo "========================================="
echo "Checkpoints:     ${CHECKPOINT_DIR}/mas_humaneval_train_${TRAIN_LIMIT}_cost{100,400,700}.pth"
echo "Cost rates:      ${COST_RATES[*]}"
echo "Arrival rates:   ${ARRIVAL_RATES[*]}"
echo "Pattern:         ${ARRIVAL_PATTERN}"
echo "Test limit:      ${TEST_LIMIT}"
echo "Split ratio:     ${SPLIT_RATIO}"
echo "Concurrency:     ${CONCURRENCY}"
echo "Max-num-seqs:    ${VLLM_MAX_NUM_SEQS}"
echo "Total configs:   ${TOTAL_CONFIGS}"
echo "Output CSV:      ${OUTPUT_CSV}"
echo "========================================="
echo ""

set +e

OVERALL_EXIT=0
CONFIG_NUM=0

for cost in "${COST_RATES[@]}"; do
    CHECKPOINT="${CHECKPOINT_DIR}/mas_humaneval_train_${TRAIN_LIMIT}_cost${cost}.pth"

    for arrival_rate in "${ARRIVAL_RATES[@]}"; do
        CONFIG_NUM=$((CONFIG_NUM + 1))

        if [[ $CONFIG_NUM -gt 1 ]]; then
            echo ""
            echo "[Config ${CONFIG_NUM}/${TOTAL_CONFIGS}] Draining vLLM before cost=${cost} rate=${arrival_rate}..."
            drain_vllm
        fi

        echo ""
        echo "========================================="
        echo "[Config ${CONFIG_NUM}/${TOTAL_CONFIGS}] cost_rate=${cost} arrival_rate=${arrival_rate}"
        echo "Checkpoint: ${CHECKPOINT}"
        echo "Start: $(date)"
        echo "========================================="

        python Experiments/run_humaneval.py \
            --epochs 0 \
            --checkpoint "${CHECKPOINT}" \
            --cost_rate "${cost}" \
            --test_limit "${TEST_LIMIT}" \
            --split-ratio "${SPLIT_RATIO}" \
            --concurrency "${CONCURRENCY}" \
            --arrival-rate "${arrival_rate}" \
            --arrival-pattern "${ARRIVAL_PATTERN}" \
            --test-telemetry-csv "${OUTPUT_CSV}"

        EXIT_CODE=$?
        if [[ $EXIT_CODE -ne 0 ]]; then
            echo "WARNING: cost=${cost} arrival_rate=${arrival_rate} exited with code ${EXIT_CODE}"
            OVERALL_EXIT=1
        fi

        echo "[Config ${CONFIG_NUM}/${TOTAL_CONFIGS}] Completed cost=${cost} rate=${arrival_rate} at $(date)"
    done
done

set -e

echo ""
echo "========================================="
echo "Baseline Test Sweep Complete"
echo "========================================="
echo "Output CSV:  ${OUTPUT_CSV}"
echo "Configs run: ${TOTAL_CONFIGS}"
echo "Exit status: ${OVERALL_EXIT}"
echo "End Time:    $(date)"
echo "========================================="

exit $OVERALL_EXIT
