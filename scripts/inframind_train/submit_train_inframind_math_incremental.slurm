#!/bin/bash
#SBATCH --job-name=infra_incr
#SBATCH --output=logs/inframind_training/math/slurm-%j-incremental.out
#SBATCH --error=logs/inframind_training/math/slurm-%j-incremental.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1

# ===== Incremental InfraMind Training =====
# Resumes from existing checkpoint and trains only on NEW (rate, budget) pairs.
# Saves ~50% compute time by avoiding redundant training.
#
# Total new episodes: 12,456 (vs 15,570 for full retraining)
# Estimated time: ~10-12 hours

echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================="
echo ""

REPO_ROOT="/home/ah872032.ucf/system-aware-mas"
cd "$REPO_ROOT" || exit 1

# Load HPC environment
source scripts/setup_hpc_env.sh

# Activate virtual environment
source .venv/bin/activate || exit 1
echo "Python: $(which python)"
python --version
echo ""

export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"
export CODE_EXECUTION_TIMEOUT="10"

# Ensure log directory exists
mkdir -p logs/inframind_training/math

# ===== Cleanup function =====
cleanup_vllm() {
    echo ""
    echo "Cleaning up vLLM servers..."

    # Use job-specific log directory
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
                    if kill -0 "$pid" 2>/dev/null; then
                        kill -KILL "$pid" 2>/dev/null
                    fi
                fi
                rm -f "$pidfile"
            fi
        done
    else
        echo "No PID files found in ${vllm_log_dir}"
    fi

    # Extra safety: only kill processes owned by current user
    pkill -u $USER -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    echo "Cleanup complete"
}
trap cleanup_vllm EXIT INT TERM

# ===== Start vLLM =====
echo "Starting vLLM model pool..."
bash scripts/vllm/serve_full_pool.sh || { echo "ERROR: Failed to start vLLM"; exit 1; }
echo "vLLM servers ready!"
bash scripts/check_vllm_status.sh
echo ""

# ===== Run incremental training =====
set +e
bash scripts/inframind_train/train_inframind_math_incremental.sh
EXIT_CODE=$?
set -e

echo ""
echo "========================================="
echo "Incremental Training Complete"
echo "Exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "========================================="

exit $EXIT_CODE
