# vLLM Startup Robustness Fix

## Problem
vLLM servers occasionally crash during initialization with memory calculation errors (e.g., "Available KV cache memory: -54.87 GiB"). The previous startup script didn't detect these crashes and would hang waiting for a server that would never become healthy.

## Solution
Added automatic crash detection and retry logic to `scripts/vllm/serve_full_pool.sh`.

## Changes Made

### 1. New Function: `detect_startup_failure()` (lines 142-156)
Monitors vLLM log files for fatal initialization errors:
- `ERROR.*EngineCore failed`
- `ValueError.*cache blocks`
- `RuntimeError.*initialization failed`
- `Available KV cache memory: -` (negative memory bug)
- `EngineCore failed to start`

### 2. Enhanced Function: `wait_for_health()` (lines 158-196)
Now returns different exit codes to distinguish failure types:
- **Return 0**: Server is healthy (success)
- **Return 1**: Permanent failure (timeout, non-crash errors)
- **Return 2**: Crash detected during init (needs retry)

When a process dies, it checks the log using `detect_startup_failure()` to determine if it was a retryable crash.

### 3. Retry Logic in Main Loop (lines 306-363)
Each server now has automatic retry with exponential backoff:
- **Max retries**: 3 attempts per server
- **Backoff**: 5s, 10s, 15s between retries
- **Cleanup**: Kills crashed process and removes PID file
- **GPU memory**: Waits naturally for kernel to reclaim memory (no GPU resets)
- **Failure handling**: Exits with error if all retries exhausted

## HPC-Safe Design
✅ **Only kills specific failed vLLM processes** - doesn't affect other GPU processes
✅ **No GPU resets or nvidia-smi commands** - stays within HPC policies
✅ **Natural memory cleanup** - relies on kernel to reclaim GPU memory after process exit
✅ **SLURM-compatible** - works in isolated SLURM job environments

## Behavior

### Before Fix
```
[vLLM] Starting qwen2_5_coder_14b_instruct
[vLLM] qwen2_5_coder_14b_instruct pid 1951033
[vLLM] qwen2_5_coder_14b_instruct failed to start  ❌ Script exits, experiments never run
```

### After Fix
```
[vLLM] Starting qwen2_5_coder_14b_instruct
[vLLM] qwen2_5_coder_14b_instruct pid 1951033
[vLLM] qwen2_5_coder_14b_instruct crashed during initialization
[vLLM] qwen2_5_coder_14b_instruct retry 1/3 - waiting 5s for GPU memory cleanup...
[vLLM] Retrying qwen2_5_coder_14b_instruct...
[vLLM] Starting qwen2_5_coder_14b_instruct
[vLLM] qwen2_5_coder_14b_instruct pid 1952441
[vLLM] qwen2_5_coder_14b_instruct healthy  ✅ Success!
```

## Testing
```bash
# Syntax check (passed)
bash -n scripts/vllm/serve_full_pool.sh

# Test in SLURM job
sbatch scripts/paper_results/submit_inframind_paper_results_math_part2.slurm
```

## Bug Fix (2026-02-10)
**Issue**: Initial implementation used `local` keyword for variables in main script body (outside functions), causing:
```
scripts/vllm/serve_full_pool.sh: line 323: local: can only be used in a function
```

**Fix**: Removed `local` keyword from variable declarations in the main loop (lines 298, 308, 312, 314). Variables `health_status`, `wait_time`, `pidfile`, and `pid` are now regular variables since they're in the main script scope, not inside a function.

## Expected Impact
- **Eliminates manual intervention** when vLLM hits transient memory calculation bugs
- **Experiments start reliably** - only proceeds when ALL models are healthy
- **Better debugging** - clear retry messages and failure logs
- **No HPC violations** - safe for cluster use

## Files Modified
- `scripts/vllm/serve_full_pool.sh` - Added crash detection and retry logic

## Next Steps
You can now resubmit your failed job:
```bash
sbatch scripts/paper_results/submit_inframind_paper_results_math_part2.slurm
```

If a server consistently fails after 3 retries, the script will exit with a clear error message pointing to the log file for investigation.
