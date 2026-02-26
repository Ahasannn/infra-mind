#!/bin/bash
#SBATCH --job-name=test_gs_math
#SBATCH --output=logs/test/math/gptswarm/slurm-%j.out
#SBATCH --error=logs/test/math/gptswarm/slurm-%j.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=18:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1

# =========================================================================
# GPTSwarm — Test Sweep for MATH
#   Arrival rates: 10, 30, 50, 100, 150, 200 req/min
#   Total: 6 arrival_rates x 500 items = 3,000 episodes
#   Output CSV: logs/test/math/gptswarm/gptswarm_math_test.csv
# =========================================================================

echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================="

REPO_ROOT="/home/ah872032.ucf/system-aware-mas"
cd "$REPO_ROOT" || exit 1

source scripts/setup_hpc_env.sh
source .venv/bin/activate || exit 1

export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"
export VLLM_MAX_NUM_SEQS=16

mkdir -p logs/test/math/gptswarm

TEST_LIMIT=500
CONCURRENCY=1000
ARRIVAL_RATES=(10 30 50 100 150 200)
ARRIVAL_PATTERN="poisson"

CHECKPOINT="${BLUE_STORAGE}/checkpoints/gptswarm/gptswarm_math_best.pt"
OUTPUT_CSV="logs/test/math/gptswarm/gptswarm_math_test.csv"

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: GPTSwarm checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

VLLM_PORTS=(8001 8002 8003 8004 8005)

drain_vllm() {
    echo "[Drain] Waiting for all vLLM queues to empty..."
    local start_time=$(date +%s)
    local last_log=0
    while true; do
        local total_active=0
        local details=""
        for port in "${VLLM_PORTS[@]}"; do
            local metrics=$(curl -s -m 5 "http://127.0.0.1:${port}/metrics" 2>/dev/null || echo "")
            local running=0 waiting=0
            if [[ -n "$metrics" ]]; then
                running=$(echo "$metrics" | grep -E "^vllm:num_requests_running" | head -1 | awk '{print int($2)}')
                waiting=$(echo "$metrics" | grep -E "^vllm:num_requests_waiting" | head -1 | awk '{print int($2)}')
                running=${running:-0}; waiting=${waiting:-0}
            fi
            local active=$((running + waiting))
            total_active=$((total_active + active))
            [[ $active -gt 0 ]] && details="${details} port${port}:${running}r+${waiting}w"
        done
        if [[ $total_active -eq 0 ]]; then
            local elapsed=$(( $(date +%s) - start_time ))
            [[ $elapsed -gt 5 ]] && echo "[Drain] All queues empty after ${elapsed}s"
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

cleanup_vllm() {
    echo "Cleaning up vLLM servers..."
    local vllm_log_dir="logs/vllm/job_${SLURM_JOB_ID}"
    if ls "${vllm_log_dir}"/*.pid >/dev/null 2>&1; then
        for pidfile in "${vllm_log_dir}"/*.pid; do
            [ -f "$pidfile" ] || continue
            pid=$(cat "$pidfile")
            kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null
            sleep 2
            kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null
            rm -f "$pidfile"
        done
    fi
    echo "Cleanup complete"
}
trap cleanup_vllm EXIT INT TERM

bash scripts/vllm/serve_full_pool.sh || { echo "ERROR: Failed to start vLLM"; exit 1; }
bash scripts/check_vllm_status.sh

echo "GPTSwarm — MATH Test Sweep | Rates: ${ARRIVAL_RATES[*]} | Checkpoint: ${CHECKPOINT}"

set +e
OVERALL_EXIT=0
CONFIG_NUM=0

for arrival_rate in "${ARRIVAL_RATES[@]}"; do
    CONFIG_NUM=$((CONFIG_NUM + 1))
    [[ $CONFIG_NUM -gt 1 ]] && drain_vllm

    echo "[Config ${CONFIG_NUM}/${#ARRIVAL_RATES[@]}] arrival_rate=${arrival_rate} — $(date)"

    python Experiments/run_gptswarm_math.py \
        --test_limit "${TEST_LIMIT}" \
        --dataset-root Datasets/MATH \
        --checkpoint "${CHECKPOINT}" \
        --concurrency "${CONCURRENCY}" \
        --arrival-rate "${arrival_rate}" \
        --arrival-pattern "${ARRIVAL_PATTERN}" \
        --test-telemetry-csv "${OUTPUT_CSV}"

    EXIT_CODE=$?
    [[ $EXIT_CODE -ne 0 ]] && echo "WARNING: rate=${arrival_rate} exit=${EXIT_CODE}" && OVERALL_EXIT=1
done

set -e
echo "GPTSwarm MATH Test Sweep Complete | CSV: ${OUTPUT_CSV} | Exit: ${OVERALL_EXIT}"
exit $OVERALL_EXIT
