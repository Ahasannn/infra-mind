#!/bin/bash
#SBATCH --job-name=test_bl_math
#SBATCH --output=logs/test/math/slurm-%j-baseline.out
#SBATCH --error=logs/test/math/slurm-%j-baseline.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=18:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1

# =========================================================================
# Baseline MAS Router — Test Sweep for MATH
#   Same configs as InfraMind training (step4_train_inframind.slurm):
#     Cost rates:    100, 400, 700  (outer loop — one checkpoint each)
#     Arrival rates: 10, 30, 50, 100, 150, 200 req/min
#     max_num_seqs=16
#   vLLM is drained between each (cost_rate, arrival_rate) config.
#   The cost_rate column in the CSV identifies which checkpoint was used.
#
#   Total: 3 cost_rates × 6 arrival_rates × 500 items = 9,000 episodes
#
#   Output CSV:
#     logs/test/math/baseline_math_test.csv
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

# Source HPC env (sets BLUE_STORAGE, HF_HOME, cache dirs, etc.)
source scripts/setup_hpc_env.sh

# Activate virtual environment
source .venv/bin/activate || exit 1
echo "Python: $(which python)"
python --version
echo ""

export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"
export CODE_EXECUTION_TIMEOUT="10"
export MATH_DATASET_ROOT="${BLUE_STORAGE}/datasets/MATH"

# Match training config: max_num_seqs=16
export VLLM_MAX_NUM_SEQS=16

# Ensure log directory exists
mkdir -p logs/test/math

# ===== Run settings (matching training config) =====
TEST_LIMIT=500
CONCURRENCY=1000
COST_RATES=(100 400 700)
ARRIVAL_RATES=(10 30 50 100 150 200)
ARRIVAL_PATTERN="poisson"

# ===== Checkpoint directory =====
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/mas_router"

# ===== Output (single CSV — cost_rate column distinguishes checkpoints) =====
OUTPUT_CSV="logs/test/math/baseline_math_test.csv"

# ===== Validation =====
for cost in "${COST_RATES[@]}"; do
    CKPT="${CHECKPOINT_DIR}/mas_math_train_519_cost${cost}.pth"
    if [[ ! -f "${CKPT}" ]]; then
        echo "ERROR: Checkpoint not found: ${CKPT}"
        exit 1
    fi
    echo "  ✓ cost=${cost}: ${CKPT}"
done
if [[ ! -d "${MATH_DATASET_ROOT}/test" ]]; then
    echo "ERROR: MATH test dataset not found at ${MATH_DATASET_ROOT}/test"
    exit 1
fi
echo ""

# ===== vLLM drain function =====
# Polls /metrics on all 5 model ports until running+waiting == 0.
# Matches the Python _wait_for_vllm_drain() in MAR/InfraMind/training.py.
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
            # Parse Prometheus metrics for running + waiting counts
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

        # Log progress every 30s
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
echo "Baseline MAS Router — MATH Test Sweep"
echo "========================================="
echo "Checkpoints:     ${CHECKPOINT_DIR}/mas_math_train_519_cost{100,400,700}.pth"
echo "Cost rates:      ${COST_RATES[*]}"
echo "Arrival rates:   ${ARRIVAL_RATES[*]}"
echo "Pattern:         ${ARRIVAL_PATTERN}"
echo "Test limit:      ${TEST_LIMIT}"
echo "Concurrency:     ${CONCURRENCY}"
echo "Max-num-seqs:    ${VLLM_MAX_NUM_SEQS}"
echo "Total configs:   ${TOTAL_CONFIGS}"
echo "Output CSV:      ${OUTPUT_CSV}"
echo "========================================="
echo ""

set +e  # Don't exit on error; we want cleanup to run

OVERALL_EXIT=0
CONFIG_NUM=0

for cost in "${COST_RATES[@]}"; do
    CHECKPOINT="${CHECKPOINT_DIR}/mas_math_train_519_cost${cost}.pth"

    for arrival_rate in "${ARRIVAL_RATES[@]}"; do
        CONFIG_NUM=$((CONFIG_NUM + 1))

        # Drain vLLM before each config (except the first)
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

        python Experiments/run_math.py \
            --epochs 0 \
            --checkpoint "${CHECKPOINT}" \
            --cost_rate "${cost}" \
            --test_limit "${TEST_LIMIT}" \
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

# ===== Summary =====
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
