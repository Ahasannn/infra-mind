#!/bin/bash
#SBATCH --job-name=test_moa_gsm8k
#SBATCH --output=logs/test/gsm8k/moa/slurm-%j.out
#SBATCH --error=logs/test/gsm8k/moa/slurm-%j.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=18:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1

# =========================================================================
# MoA (Mixture of Agents) — Test Sweep for GSM8K
#   Arrival rates: 10, 30, 50, 100, 150, 200 req/min
#   Output CSV: logs/test/gsm8k/moa_gsm8k_test.csv
# =========================================================================

echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
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

mkdir -p logs/test/gsm8k/moa

TEST_LIMIT=500
CONCURRENCY=1000
ARRIVAL_RATES=(10 30 50 100 150 200)
ARRIVAL_PATTERN="poisson"
OUTPUT_CSV="logs/test/gsm8k/moa/moa_gsm8k_test.csv"

VLLM_PORTS=(8001 8002 8003 8004 8005)

drain_vllm() {
    echo "[Drain] Waiting for all vLLM queues to empty..."
    local start_time; start_time=$(date +%s); local last_log=0
    while true; do
        local total_active=0; local details=""
        for port in "${VLLM_PORTS[@]}"; do
            local metrics; metrics=$(curl -s -m 5 "http://127.0.0.1:${port}/metrics" 2>/dev/null || echo "")
            local running=0; local waiting=0
            if [[ -n "$metrics" ]]; then
                running=$(echo "$metrics" | grep -E "^vllm:num_requests_running" | head -1 | awk '{print int($2)}')
                waiting=$(echo "$metrics" | grep -E "^vllm:num_requests_waiting" | head -1 | awk '{print int($2)}')
                running=${running:-0}; waiting=${waiting:-0}
            fi
            local active=$((running + waiting)); total_active=$((total_active + active))
            if [[ $active -gt 0 ]]; then details="${details} port${port}:${running}r+${waiting}w"; fi
        done
        if [[ $total_active -eq 0 ]]; then
            local elapsed=$(( $(date +%s) - start_time ))
            [[ $elapsed -gt 5 ]] && echo "[Drain] All queues empty after ${elapsed}s"
            return 0
        fi
        local elapsed=$(( $(date +%s) - start_time ))
        if [[ $((elapsed - last_log)) -ge 30 ]]; then
            echo "[Drain] Waiting (${elapsed}s elapsed, ${total_active} active):${details}"; last_log=$elapsed
        fi
        sleep 5
    done
}

cleanup_vllm() {
    echo "Cleaning up vLLM servers..."
    local vllm_log_dir="logs/vllm/job_${SLURM_JOB_ID}"
    if ls "${vllm_log_dir}"/*.pid >/dev/null 2>&1; then
        for pidfile in "${vllm_log_dir}"/*.pid; do
            if [ -f "$pidfile" ]; then
                pid=$(cat "$pidfile"); name=$(basename "$pidfile" .pid)
                if kill -0 "$pid" 2>/dev/null; then
                    echo "Stopping $name (PID $pid)"; kill -TERM "$pid" 2>/dev/null; sleep 2
                    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null
                fi; rm -f "$pidfile"
            fi
        done
    fi; echo "Cleanup complete"
}
trap cleanup_vllm EXIT INT TERM

echo "Starting vLLM model pool..."
bash scripts/vllm/serve_full_pool.sh || { echo "ERROR: Failed to start vLLM"; exit 1; }
bash scripts/check_vllm_status.sh

echo "MoA GSM8K Test Sweep: rates=${ARRIVAL_RATES[*]} limit=${TEST_LIMIT}"

set +e
OVERALL_EXIT=0; CONFIG_NUM=0

for arrival_rate in "${ARRIVAL_RATES[@]}"; do
    CONFIG_NUM=$((CONFIG_NUM + 1))
    [[ $CONFIG_NUM -gt 1 ]] && { echo "Draining vLLM..."; drain_vllm; }

    echo "[${CONFIG_NUM}/${#ARRIVAL_RATES[@]}] rate=${arrival_rate} $(date)"

    python Experiments/run_moa_gsm8k.py \
        --test_limit "${TEST_LIMIT}" \
        --concurrency "${CONCURRENCY}" \
        --arrival-rate "${arrival_rate}" \
        --arrival-pattern "${ARRIVAL_PATTERN}" \
        --test-telemetry-csv "${OUTPUT_CSV}"

    EXIT_CODE=$?
    [[ $EXIT_CODE -ne 0 ]] && { echo "WARNING: rate=${arrival_rate} exit=${EXIT_CODE}"; OVERALL_EXIT=1; }
done

set -e
echo "MoA GSM8K Test Sweep Complete. CSV: ${OUTPUT_CSV} Exit: ${OVERALL_EXIT}"
exit $OVERALL_EXIT
