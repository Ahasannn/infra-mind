#!/bin/bash
# setup_hpc_env.sh
# Centralized HPC environment configuration for all SLURM jobs
# Source this at the beginning of any sbatch script:
#   source scripts/setup_hpc_env.sh

# ===== BLUE STORAGE CONFIGURATION =====
export BLUE_STORAGE="/blue/qi855292.ucf/ah872032.ucf"

# ===== HUGGING FACE CONFIGURATION =====
# 1. Set HF cache directory to blue storage
export HF_HOME="${BLUE_STORAGE}/huggingface_cache"

# 2. Read token from blue storage (primary) or fallback to home directory
if [[ -f "${HF_HOME}/token" ]]; then
    export HF_TOKEN=$(cat "${HF_HOME}/token")
elif [[ -f ~/.cache/huggingface/token ]]; then
    export HF_TOKEN=$(cat ~/.cache/huggingface/token)
elif [[ -f ~/.huggingface/token ]]; then
    export HF_TOKEN=$(cat ~/.huggingface/token)
fi

if [[ -z "${HF_TOKEN}" ]]; then
    echo "WARNING: HF_TOKEN not found in ${HF_HOME}/token or ~/.cache/huggingface/token"
fi

# ===== PYTORCH / TORCH CACHE CONFIGURATION =====
export TORCH_HOME="${BLUE_STORAGE}/torch_cache"
export TRITON_CACHE_DIR="${BLUE_STORAGE}/triton_cache"
export TRITON_HOME="${BLUE_STORAGE}/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="${BLUE_STORAGE}/torchinductor_cache"
export TORCH_EXTENSIONS_DIR="${BLUE_STORAGE}/torch_extensions"

# ===== TEMPORARY DIRECTORIES =====
# Critical: Override TMPDIR to prevent writes to /scratch/local
export TMPDIR="${BLUE_STORAGE}/tmp"
export TEMP="${BLUE_STORAGE}/tmp"
export TMP="${BLUE_STORAGE}/tmp"

# ===== GENERAL CACHE LOCATIONS =====
export XDG_CACHE_HOME="${BLUE_STORAGE}/cache"

# ===== CREATE DIRECTORIES =====
mkdir -p "${HF_HOME}" \
         "${TORCH_HOME}" \
         "${TRITON_CACHE_DIR}" \
         "${TORCHINDUCTOR_CACHE_DIR}" \
         "${TORCH_EXTENSIONS_DIR}" \
         "${TMPDIR}" \
         "${XDG_CACHE_HOME}"

# ===== HUGGING FACE SETTINGS =====
export TOKENIZERS_PARALLELISM="false"

# ===== DISPLAY CONFIGURATION =====
if [[ "${VERBOSE_ENV:-0}" == "1" ]]; then
    echo "========================================="
    echo "HPC Environment Configuration"
    echo "========================================="
    echo "Blue Storage:           ${BLUE_STORAGE}"
    echo "HF_HOME:                ${HF_HOME}"
    echo "HF_TOKEN:               $(if [[ -n "${HF_TOKEN}" ]]; then echo "✓ Set"; else echo "✗ Not found"; fi)"
    echo "TORCH_HOME:             ${TORCH_HOME}"
    echo "TRITON_CACHE_DIR:       ${TRITON_CACHE_DIR}"
    echo "TMPDIR:                 ${TMPDIR}"
    echo "XDG_CACHE_HOME:         ${XDG_CACHE_HOME}"
    echo "========================================="
fi
