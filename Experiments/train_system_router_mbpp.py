"""
Pilot training loop for the hierarchical System-Aware Router on a small MBPP slice.
Quality is measured by executing MBPP tests; latency is measured as sum of max LLM time per wave.
"""
import argparse
import math
import os
import sys
import random
import time
import threading
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Datasets.mbpp_dataset import MbppDataset  # noqa: E402
from MAR.SystemRouter.system_aware_router import SystemAwareRouter  # noqa: E402
from MAR.SystemRouter.env import SystemRouterEnv  # noqa: E402
from MAR.SystemRouter.trainer import SystemRouterTrainer  # noqa: E402
from MAR.Utils.telemetry import CsvTelemetryWriter  # noqa: E402
from MAR.Utils.request_patterns import RequestPattern  # noqa: E402
from MAR.Utils.request_shooter import RequestShooter, RequestResult  # noqa: E402


def sample_mbpp(limit: int, split: str = "train") -> List[dict]:
    ds = MbppDataset(split)
    indices = list(range(len(ds)))
    random.shuffle(indices)
    picked = indices[:limit]
    return [ds[i] for i in picked]

SYSTEM_ROUTER_CSV_FIELDS: Sequence[str] = (
    "run_id",
    "epoch",
    "episode_index",
    "record_type",
    "dataset",
    "split",
    "item_id",
    "query",
    "tests_json",
    "topology",
    "role_set",
    "budget_total",
    "arrival_rate",
    "arrival_pattern",
    "quality",
    "quality_is_solved",
    "quality_feedback",
    "workflow_latency_seconds",
    "final_response",
    "code_response",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "role_name",
    "step_index",
    "model_name",
    "strategy_name",
    "latency_seconds",
    "budget_remaining",
    "observed_ttft",
    "observed_tpot",
    "llm_running",
    "llm_waiting",
    "llm_kv_cache_usage",
    "llm_ttft_avg",
    "llm_itl_avg",
    "llm_e2e_avg",
    "prompt_base",
    "response_final",
)


def _default_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _resolve_item_id(row: Any, fallback: int) -> Any:
    keys = ("task_id", "id", "ID", "index", "idx")
    getter = row.get if hasattr(row, "get") else None
    for key in keys:
        value = None
        if getter is not None:
            value = getter(key)
        elif isinstance(row, dict):
            value = row.get(key)
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        if value == "":
            continue
        return value
    return fallback


def _save_checkpoint(
    checkpoint_path: Optional[str],
    checkpoint_dir: str,
    router: SystemAwareRouter,
    trainer: SystemRouterTrainer,
    epoch: int,
    run_id: str,
    args: Any,
) -> str:
    if checkpoint_path:
        path = checkpoint_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f"system_router_{run_id}_epoch_{epoch}.pt")
    payload = {
        "epoch": epoch,
        "run_id": run_id,
        "router_state_dict": router.state_dict(),
        "planner_optimizer_state_dict": trainer.planner_optimizer.state_dict(),
        "executor_optimizer_state_dict": trainer.executor_optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(payload, path)
    return path


def _parse_float_list(raw: str) -> List[float]:
    values: List[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values


def _parse_str_list(raw: str) -> List[str]:
    values: List[str] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(item)
    return values


def _episode_has_agent_errors(episode: Dict[str, Any]) -> bool:
    for step in episode.get("executor_transitions", []):
        if not step.get("success", True):
            return True
        error = step.get("error", "")
        if isinstance(error, str) and error.strip():
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot training for System-Aware Router on MBPP slice.")
    parser.add_argument("--limit", type=int, default=50, help="Number of MBPP items to train on.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs over the slice.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max generation tokens when calling LLMs.")
    parser.add_argument("--request-timeout", type=float, default=120.0, help="Per-request timeout in seconds.")
    parser.add_argument("--deterministic", action="store_true", help="Use argmax actions instead of sampling.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log-episodes", type=int, default=1, help="Episodes to log per epoch.")
    parser.add_argument("--arrival-rate", type=float, default=0.0, help="Arrival rate (req/sec). 0 disables shooting.")
    parser.add_argument("--arrival-rates", type=str, default="", help="Comma-separated arrival rates to sweep.")
    parser.add_argument("--arrival-pattern", type=str, default="poisson", help="Arrival pattern (poisson/microburst/sustained).")
    parser.add_argument("--arrival-patterns", type=str, default="", help="Comma-separated arrival patterns to sweep.")
    parser.add_argument("--concurrency", type=int, default=1, help="Max concurrent in-flight requests.")
    parser.add_argument("--burst-duration", type=float, default=3.0, help="Burst duration for microburst.")
    parser.add_argument("--spike-intensity", type=float, default=10.0, help="Spike intensity for microburst.")
    parser.add_argument("--spike-period", type=float, default=20.0, help="Spike period for microburst.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/system_router", help="Checkpoint output dir.")
    parser.add_argument("--checkpoint-path", type=str, default="", help="Fixed checkpoint path to overwrite.")
    parser.add_argument("--resume-checkpoint", action="store_true", help="Resume from checkpoint path if present.")
    parser.add_argument("--checkpoint-every", type=int, default=50, help="Save a checkpoint every N successful episodes (0 disables).")
    parser.add_argument("--telemetry-csv", type=str, default="", help="CSV path for per-episode telemetry.")
    parser.add_argument("--run-id", type=str, default="", help="Run id for telemetry/checkpoints.")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    router = SystemAwareRouter()
    env = SystemRouterEnv(router=router, max_tokens=args.max_tokens, request_timeout=args.request_timeout)
    trainer = SystemRouterTrainer(router)

    checkpoint_path = args.checkpoint_path.strip() or ""
    resume_checkpoint = args.resume_checkpoint
    if checkpoint_path and os.path.isfile(checkpoint_path):
        if resume_checkpoint or args.checkpoint_path:
            payload = torch.load(checkpoint_path, map_location=router.device)
            router.load_state_dict(payload.get("router_state_dict", payload))
            planner_state = payload.get("planner_optimizer_state_dict")
            executor_state = payload.get("executor_optimizer_state_dict")
            if planner_state:
                trainer.planner_optimizer.load_state_dict(planner_state)
            if executor_state:
                trainer.executor_optimizer.load_state_dict(executor_state)
            logger.info("Resumed checkpoint from {}", checkpoint_path)

    run_id = args.run_id or _default_run_id()
    telemetry_path = args.telemetry_csv or os.path.join("logs", f"system_router_mbpp_{run_id}.csv")
    telemetry_writer = CsvTelemetryWriter(telemetry_path, fieldnames=SYSTEM_ROUTER_CSV_FIELDS)
    logger.info("Telemetry CSV: {}", telemetry_path)

    data = sample_mbpp(limit=args.limit, split="train")
    logger.info("Loaded {} MBPP items for training slice", len(data))

    sweep_rates = _parse_float_list(args.arrival_rates) if args.arrival_rates else [args.arrival_rate]
    sweep_patterns = _parse_str_list(args.arrival_patterns) if args.arrival_patterns else [args.arrival_pattern]
    sweep_configs = [(rate, pattern) for rate in sweep_rates for pattern in sweep_patterns]
    global_epoch = 0
    episode_counter = 0
    last_checkpoint_episode = 0
    process_lock = threading.Lock()

    for sweep_idx, (arrival_rate, arrival_pattern) in enumerate(sweep_configs):
        for epoch in range(args.epochs):
            planner_transitions = []
            executor_transitions = []
            workflow_latencies = []
            use_shooter = arrival_rate > 0.0 or args.concurrency > 1

            def process_episode(row_idx: int, row: Any, episode: Dict[str, Any]) -> None:
                nonlocal episode_counter, last_checkpoint_episode
                query = str(row.get("task") or row["text"])
                tests = list(row["test_list"])
                item_id = _resolve_item_id(row, row_idx)

                if _episode_has_agent_errors(episode):
                    logger.warning("skip_episode_due_to_agent_error idx={} item_id={}", row_idx, item_id)
                    return

                with process_lock:
                    planner_transitions.append(episode["planner_transition"])
                    executor_transitions.extend(episode["executor_transitions"])
                    workflow_latencies.append(episode["workflow_latency_seconds"])

                    episode_index = episode_counter
                    episode_counter += 1
                    csv_rows: List[Dict[str, Any]] = [
                        {
                            "run_id": run_id,
                            "epoch": global_epoch,
                            "episode_index": episode_index,
                            "record_type": "episode",
                            "dataset": "mbpp",
                            "split": "train",
                            "item_id": item_id,
                            "query": query,
                            "tests_json": tests,
                            "topology": episode["topology"],
                            "role_set": episode["role_set"],
                            "budget_total": episode["budget_total"],
                            "arrival_rate": arrival_rate,
                            "arrival_pattern": arrival_pattern,
                            "quality": episode["quality"],
                            "quality_is_solved": episode["quality_is_solved"],
                            "quality_feedback": episode["quality_feedback"],
                            "workflow_latency_seconds": episode["workflow_latency_seconds"],
                            "final_response": episode["response"],
                            "code_response": episode.get("code_response", ""),
                            "prompt_tokens": episode["token_counts"].get("prompt_tokens", 0),
                            "completion_tokens": episode["token_counts"].get("completion_tokens", 0),
                            "total_tokens": episode["token_counts"].get("total_tokens", 0),
                        }
                    ]

                    for step in episode["executor_transitions"]:
                        token_counts = step.get("token_counts", {})
                        csv_rows.append(
                            {
                                "run_id": run_id,
                                "epoch": global_epoch,
                                "episode_index": episode_index,
                                "record_type": "role_step",
                                "dataset": "mbpp",
                                "split": "train",
                                "item_id": item_id,
                                "query": query,
                                "tests_json": tests,
                                "topology": episode["topology"],
                                "role_set": episode["role_set"],
                                "arrival_rate": arrival_rate,
                                "arrival_pattern": arrival_pattern,
                                "quality": step.get("quality", 0.0),
                                "workflow_latency_seconds": step.get("workflow_latency_seconds", 0.0),
                                "role_name": step.get("role", ""),
                                "step_index": step.get("step_index", 0),
                                "model_name": step.get("model", ""),
                                "strategy_name": step.get("strategy", ""),
                                "latency_seconds": step.get("latency_seconds", 0.0),
                                "budget_remaining": step.get("budget_remaining", 0.0),
                                "observed_ttft": step.get("observed_ttft", 0.0),
                                "observed_tpot": step.get("observed_tpot", 0.0),
                                "llm_running": step.get("llm_running", 0),
                                "llm_waiting": step.get("llm_waiting", 0),
                                "llm_kv_cache_usage": step.get("llm_kv_cache_usage", 0.0),
                                "llm_ttft_avg": step.get("llm_ttft_avg", 0.0),
                                "llm_itl_avg": step.get("llm_itl_avg", 0.0),
                                "llm_e2e_avg": step.get("llm_e2e_avg", 0.0),
                                "prompt_base": step.get("prompt_base", ""),
                                "response_final": step.get("response", ""),
                                "prompt_tokens": token_counts.get("prompt_tokens", 0),
                                "completion_tokens": token_counts.get("completion_tokens", 0),
                                "total_tokens": token_counts.get("total_tokens", 0),
                            }
                        )
                    telemetry_writer.append_rows(csv_rows)

                    if args.checkpoint_every > 0 and (episode_counter - last_checkpoint_episode) >= args.checkpoint_every:
                        saved_path = _save_checkpoint(
                            checkpoint_path or None,
                            args.checkpoint_dir,
                            router,
                            trainer,
                            global_epoch,
                            run_id,
                            args,
                        )
                        logger.info("Saved checkpoint: {}", saved_path)
                        last_checkpoint_episode = episode_counter

                    if len(planner_transitions) <= args.log_episodes:
                        logger.info(
                            "episode={} topology={} role_set={} quality={:.3f} latency={:.3f}s budget={:.2f}s",
                            len(planner_transitions) - 1,
                            episode["topology"],
                            episode["role_set"],
                            episode["quality"],
                            episode["workflow_latency_seconds"],
                            episode["budget_total"],
                        )
                        for step in episode["executor_transitions"]:
                            logger.info(
                                "  role={} model={} strategy={} call_latency={:.3f}s budget_rem={:.2f}s",
                                step["role"],
                                step["model"],
                                step["strategy"],
                                step["latency_seconds"],
                                step["budget_remaining"],
                            )

            if use_shooter:
                pattern = RequestPattern(
                    pattern=arrival_pattern,
                    rate=arrival_rate,
                    spike_intensity=args.spike_intensity,
                    spike_period=args.spike_period,
                    burst_duration=args.burst_duration,
                    seed=args.seed + sweep_idx * 1000 + epoch,
                )
                shooter = RequestShooter(
                    pattern,
                    max_concurrency=args.concurrency,
                    capture_output=True,
                    collect_results=True,
                    on_result=None,
                )

                def handler(row: Any) -> Dict[str, Any]:
                    query = str(row.get("task") or row["text"])
                    tests = list(row["test_list"])
                    query_id = _resolve_item_id(row, -1)
                    with torch.no_grad():
                        return env.step(
                            query=query,
                            tests=tests,
                            deterministic=args.deterministic,
                            latency_seed=query,
                            query_id=query_id,
                        )

                def _handle_result(res: RequestResult) -> None:
                    if not res.success:
                        logger.warning("request_failed idx={} item_id={} error={}", res.index, res.item_id, res.error)
                        return
                    row = data[res.index]
                    if res.output is None:
                        logger.warning("request_empty_output idx={} item_id={}", res.index, res.item_id)
                        return
                    process_episode(res.index, row, res.output)

                shooter.on_result = _handle_result
                shooter.run(
                    data,
                    handler=handler,
                    item_id_fn=lambda row, idx: str(_resolve_item_id(row, idx)),
                )
            else:
                for row_idx, row in enumerate(data):
                    query = str(row.get("task") or row["text"])
                    tests = list(row["test_list"])
                    item_id = _resolve_item_id(row, row_idx)
                    try:
                        episode = env.step(
                            query=query,
                            tests=tests,
                            deterministic=args.deterministic,
                            latency_seed=query,
                            query_id=item_id,
                        )
                    except Exception as exc:
                        logger.warning("request_failed idx={} item_id={} error={}", row_idx, item_id, exc)
                        continue
                    process_episode(row_idx, row, episode)

            metrics = trainer.train_batch(planner_transitions, executor_transitions)
            avg_quality = sum(t["quality"] for t in planner_transitions) / max(len(planner_transitions), 1)
            avg_latency = sum(workflow_latencies) / max(len(workflow_latencies), 1)
            logger.info(
                "epoch={} sweep={}/{} rate={} pattern={} planner_loss={:.4f} executor_loss={:.4f} avg_quality={:.4f} avg_latency={:.3f}s lambda={:.4f}",
                global_epoch,
                sweep_idx + 1,
                len(sweep_configs),
                arrival_rate,
                arrival_pattern,
                metrics.get("planner_loss", 0.0),
                metrics.get("executor_loss", 0.0),
                avg_quality,
                avg_latency,
                float(torch.nn.functional.softplus(router.lagrange_multiplier).detach().cpu().item()),
            )
            saved_path = _save_checkpoint(
                checkpoint_path or None,
                args.checkpoint_dir,
                router,
                trainer,
                global_epoch,
                run_id,
                args,
            )
            logger.info("Saved checkpoint: {}", saved_path)
            global_epoch += 1

    # Final sanity: route one sample deterministically
    if data:
        sample = data[0]
        try:
            result = env.step(
                query=str(sample.get("task") or sample["text"]),
                tests=list(sample["test_list"]),
                deterministic=True,
                latency_seed="final",
                query_id=_resolve_item_id(sample, 0),
            )
        except Exception as exc:
            logger.warning("final_sample_failed error={}", exc)
        else:
            logger.info(
                "final_sample topology={} role_set={} quality={:.4f} latency={:.3f}s",
                result["topology"],
                result["role_set"],
                result["quality"],
                result["workflow_latency_seconds"],
            )


if __name__ == "__main__":
    main()
