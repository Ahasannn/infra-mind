#!/bin/bash
#SBATCH --job-name=test_gs_mbpp
#SBATCH --output=logs/test/mbpp/gptswarm/slurm-%j.out
#SBATCH --error=logs/test/mbpp/gptswarm/slurm-%j.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=18:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1

# =========================================================================
# GPTSwarm — Test Sweep for MBPP
#   Requires trained checkpoint with learned edge logits.
#   Arrival rates: 10, 30, 50, 100, 150, 200 req/min
#   max_num_seqs=16
#   vLLM is drained between each arrival rate.
#
#   Total: 6 arrival_rates x 500 items = 3,000 episodes
#
#   Output CSV:
#     logs/test/mbpp/gptswarm/gptswarm_mbpp_test.csv
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
export VLLM_MAX_NUM_SEQS=16

mkdir -p logs/test/mbpp/gptswarm

# ===== Run settings =====
TEST_LIMIT=500
CONCURRENCY=1000
ARRIVAL_RATES=(10 30 50 100 150 200)
ARRIVAL_PATTERN="poisson"

CHECKPOINT="${BLUE_STORAGE}/checkpoints/gptswarm/gptswarm_mbpp_best.pt"
OUTPUT_CSV="logs/test/mbpp/gptswarm/gptswarm_mbpp_test.csv"

# Verify checkpoint
if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: GPTSwarm checkpoint not found: ${CHECKPOINT}"
    exit 1
fi
echo "Checkpoint: ${CHECKPOINT}"

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

echo "========================================="
echo "GPTSwarm — MBPP Test Sweep"
echo "========================================="
echo "Arrival rates:   ${ARRIVAL_RATES[*]}"
echo "Pattern:         ${ARRIVAL_PATTERN}"
echo "Test limit:      ${TEST_LIMIT}"
echo "Concurrency:     ${CONCURRENCY}"
echo "Max-num-seqs:    ${VLLM_MAX_NUM_SEQS}"
echo "Total configs:   ${#ARRIVAL_RATES[@]}"
echo "Checkpoint:      ${CHECKPOINT}"
echo "Output CSV:      ${OUTPUT_CSV}"
echo "========================================="
echo ""

set +e

OVERALL_EXIT=0
CONFIG_NUM=0

for arrival_rate in "${ARRIVAL_RATES[@]}"; do
    CONFIG_NUM=$((CONFIG_NUM + 1))

    if [[ $CONFIG_NUM -gt 1 ]]; then
        echo ""
        echo "[Config ${CONFIG_NUM}/${#ARRIVAL_RATES[@]}] Draining vLLM before rate=${arrival_rate}..."
        drain_vllm
    fi

    echo ""
    echo "========================================="
    echo "[Config ${CONFIG_NUM}/${#ARRIVAL_RATES[@]}] arrival_rate=${arrival_rate}"
    echo "Start: $(date)"
    echo "========================================="

    python Experiments/run_gptswarm_mbpp.py \
        --test_limit "${TEST_LIMIT}" \
        --checkpoint "${CHECKPOINT}" \
        --concurrency "${CONCURRENCY}" \
        --arrival-rate "${arrival_rate}" \
        --arrival-pattern "${ARRIVAL_PATTERN}" \
        --test-telemetry-csv "${OUTPUT_CSV}"

    EXIT_CODE=$?
    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "WARNING: arrival_rate=${arrival_rate} exited with code ${EXIT_CODE}"
        OVERALL_EXIT=1
    fi

    echo "[Config ${CONFIG_NUM}/${#ARRIVAL_RATES[@]}] Completed rate=${arrival_rate} at $(date)"
done

set -e

echo ""
echo "========================================="
echo "GPTSwarm Test Sweep Complete"
echo "========================================="
echo "Output CSV:  ${OUTPUT_CSV}"
echo "Configs run: ${#ARRIVAL_RATES[@]}"
echo "Exit status: ${OVERALL_EXIT}"
echo "End Time:    $(date)"
echo "========================================="

exit $OVERALL_EXIT
