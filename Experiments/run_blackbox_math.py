"""
MAS Router baseline training/testing with blackbox (API) model pool.

This is the blackbox variant of ``run_math.py``. It loads the blackbox
model profile and configures per-provider API keys, then runs the same
MAS Router training and test pipeline.

Usage:
    python Experiments/run_blackbox_math.py --epochs 5 --train_limit 100 --test_limit 500

The blackbox model pool is defined in ``MAR/LLM/llm_profile_blackbox.json``.
API keys should be set in ``.env`` (GEMINI_API_KEY, OPENROUTER_API_KEY).
"""

import sys
import os
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load .env BEFORE any other imports so API keys are available
from dotenv import load_dotenv
load_dotenv()

import time
import argparse
import json
import torch
from loguru import logger
import torch.nn.functional as F
from tqdm import tqdm
import gc

from MAR.LLM.model_pool import ModelPool
from MAR.LLM.blackbox_setup import configure_blackbox_env
from MAR.MasRouter.mas_router import MasRouter
from MAR.Agent.reasoning_profile import reasoning_profile
from MAR.Prompts.tasks_profile import tasks_profile
from MAR.Utils.utils import fix_random_seed
from MAR.Utils.globals import Cost, PromptTokens, CompletionTokens
from MAR.Utils.log import configure_logging, ProgressTracker
from MAR.Utils.telemetry import CsvTelemetryWriter
from MAR.Utils.request_patterns import RequestPattern
from MAR.Utils.request_shooter import RequestShooter
from MAR.InfraMind.blackbox_metrics import BlackboxMetricsProvider
from MAR.InfraMind.metrics_watcher import model_metrics
from MAR.LLM.response_cache import ResponseCache, enable_cache
from Datasets.math_dataset import load_math_dataset, MATH_is_correct, MATH_get_predict

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Blackbox pool setup ─────────────────────────────────────────────
PROFILE_PATH = os.path.join(os.path.dirname(__file__), "..", "MAR", "LLM", "llm_profile_blackbox.json")


def dataloader(data_list, batch_size, i_batch):
    return data_list[i_batch * batch_size : i_batch * batch_size + batch_size]


# CSV fields — blackbox variant (queue + cache columns instead of vLLM metrics)
BASELINE_TRAIN_FIELDS = (
    "run_id", "epoch", "batch_id", "episode_index", "record_type",
    "dataset", "split", "item_id", "topology", "role_set",
    "workflow_latency_seconds", "llm_elapsed_seconds", "quality_is_solved",
    "step_index", "round_index", "dep_level",
    "role_name", "llm_name", "latency_seconds",
    "spatial_predecessors", "spatial_successors",
    "rpm_queue_running", "rpm_queue_waiting", "e2e_avg",
    "prompt_tokens", "completion_tokens", "cache_hit",
)

BASELINE_TEST_FIELDS = (
    "run_id", "dataset", "split", "batch_id", "item_id", "record_type",
    "task_name", "reasoning_name", "graph_id", "num_agents",
    "agent_roles_json", "agent_llms_json", "role_llm_map_json",
    "quality_is_correct", "quality_pred", "quality_gold",
    "eval_duration_sec", "utility",
    "workflow_latency_seconds", "llm_elapsed_seconds",
    "arrival_rate", "arrival_pattern",
    "step_index", "round_index", "dep_level",
    "role_name", "llm_name", "latency_seconds", "node_id",
    "spatial_predecessors", "spatial_successors",
    "rpm_queue_running", "rpm_queue_waiting", "e2e_avg",
    "prompt_tokens", "completion_tokens", "strategy_name", "query", "cost_rate",
    "cache_hit",
)


def parse_args():
    parser = argparse.ArgumentParser(description="MAS Router Baseline with Blackbox Models on MATH")
    parser.add_argument("--train_limit", type=int, default=100)
    parser.add_argument("--test_limit", type=int, default=500)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--domain", type=str, default="MATH")
    parser.add_argument("--prompt_file", type=str, default="MAR/Roles/FinalNode/math.json")
    parser.add_argument("--cost_rate", type=float, default=100.0)
    parser.add_argument("--max_agent", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=1800.0,
                        help="Per-workflow timeout in seconds. High because RPM=5 causes long queue waits.")
    parser.add_argument("--arrival-rate", type=float, nargs="+", default=[0.0])
    parser.add_argument("--arrival-pattern", type=str, default="poisson")
    parser.add_argument("--concurrency", type=int, default=1000)
    parser.add_argument("--train-telemetry-csv", type=str, default="")
    parser.add_argument("--test-telemetry-csv", type=str, default="")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--save-checkpoint", type=str, default="")
    parser.add_argument("--profile", type=str, default=PROFILE_PATH,
                        help="Path to blackbox model profile JSON.")
    parser.add_argument("--cache", type=str, default="cache/blackbox_math_mas.db",
                        help="Response cache DB path. Caches LLM responses across sweeps.")
    parser.add_argument("--no-cache", action="store_true", help="Disable response caching.")
    parser.add_argument("--dataset-root", type=str,
                        default=os.environ.get("MATH_DATASET_ROOT", "Datasets/MATH"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fix_random_seed(1234)

    # ── Setup blackbox pool ──────────────────────────────────────────
    pool = ModelPool(args.profile)
    configure_blackbox_env(pool)

    # Blackbox metrics provider (RPM tracking + client-side latency)
    blackbox_metrics = BlackboxMetricsProvider(rpm_limits=pool.get_rpm_limits())

    # Local request queue — RPM throttling + EDF scheduling
    from MAR.InfraMind.blackbox_queue import BlackboxRequestQueue
    BlackboxRequestQueue.configure(
        rpm_limits=pool.get_rpm_limits(),
        metrics_provider=blackbox_metrics,
    )

    # Response cache — deduplicates LLM calls across sweeps
    if not args.no_cache:
        cache = ResponseCache(args.cache)
        enable_cache(cache)
    else:
        cache = None

    # Use the pool's model profile for the MAS router
    llms = pool.get_llm_profile()
    reasonings = reasoning_profile
    tasks = tasks_profile

    logger.info("Blackbox model pool: {}", [m["Name"] for m in llms])

    # ── Load dataset ─────────────────────────────────────────────────
    dataset_root = args.dataset_root
    if not os.path.isabs(dataset_root):
        dataset_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", dataset_root))

    import random as _random
    train_dataset = load_math_dataset(dataset_root, split="train")
    if args.train_limit > 0:
        indices = list(range(len(train_dataset)))
        _random.Random(42).shuffle(indices)
        train_dataset = [train_dataset[i] for i in indices[: args.train_limit]]

    test_dataset = load_math_dataset(dataset_root, split="test")
    if args.test_limit > 0:
        indices = list(range(len(test_dataset)))
        _random.Random(42).shuffle(indices)
        test_dataset = [test_dataset[i] for i in indices[: args.test_limit]]

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_dir = "logs/blackbox/mas"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"MATH_blackbox_{current_time}.txt"
    configure_logging(log_name=log_file)
    run_id = current_time

    train_csv = args.train_telemetry_csv or f"{log_dir}/{args.domain}_blackbox_{current_time}_train.csv"
    os.makedirs(os.path.dirname(train_csv), exist_ok=True)
    train_writer = CsvTelemetryWriter(train_csv, fieldnames=BASELINE_TRAIN_FIELDS)
    logger.info(f"Train telemetry CSV: {train_csv}")

    # ── Router ───────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router = MasRouter(max_agent=args.max_agent, device=device).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        # Support both raw state_dict and InfraMind-wrapped checkpoints
        if isinstance(ckpt, dict) and "router_state_dict" in ckpt:
            router.load_state_dict(ckpt["router_state_dict"])
        else:
            router.load_state_dict(ckpt)
    optimizer = torch.optim.Adam(router.parameters(), lr=args.lr)

    logger.info("=" * 60)
    logger.info("Blackbox MAS Training — MATH")
    logger.info("=" * 60)
    logger.info("Models: {}", [m["Name"] for m in llms])
    logger.info("RPM limits: {}", pool.get_rpm_limits())
    logger.info("Train: {} items | Test: {} items | Epochs: {} | Batch: {}",
                len(train_dataset), len(test_dataset), args.epochs, args.batch_size)
    logger.info("Cost rate: {} | LR: {} | Max agents: {}", args.cost_rate, args.lr, args.max_agent)
    logger.info("Cache: {}", args.cache if not args.no_cache else "DISABLED")
    logger.info("Checkpoint save: {}", args.save_checkpoint or "NONE")
    logger.info("Train CSV: {}", train_csv)
    logger.info("=" * 60)

    episode_counter = 0
    checkpoint_dir = os.path.join("checkpoints", "mas_router_blackbox")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Training loop (same as run_math.py) ──────────────────────────
    for epoch in range(args.epochs):
        total_solved, total_executed = 0, 0
        train_batches = max(1, len(train_dataset) // args.batch_size)

        pbar = tqdm(range(train_batches), desc=f"Epoch {epoch}/{args.epochs}", unit="batch",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
        for i_batch in pbar:
            current_batch = dataloader(train_dataset, args.batch_size, i_batch)
            queries = [item["problem"] for item in current_batch]
            answers = [item["solution"] for item in current_batch]
            item_ids = [item.get("id", i_batch * args.batch_size + j) for j, item in enumerate(current_batch)]
            task_labels = [0 for _ in current_batch]
            tasks_y = torch.tensor(task_labels).to(device)

            optimizer.zero_grad()
            results, costs, log_probs, tasks_probs, vae_loss, agents_num = router.forward(
                queries, tasks, llms, reasonings, task_labels,
                prompt_file=args.prompt_file,
                item_ids=item_ids,
                dataset=args.domain,
                request_timeout=args.request_timeout,
            )

            skipped_indices = getattr(router, "last_skipped_indices", set())
            valid_indices = [idx for idx in range(len(queries)) if idx not in skipped_indices]
            if not valid_indices:
                logger.warning("All requests in batch {} timed out; skipping.", i_batch)
                continue

            valid_mask = torch.zeros(len(queries), dtype=torch.bool, device=device)
            valid_mask[valid_indices] = True
            tasks_probs_valid = tasks_probs[valid_mask]
            tasks_y_valid = tasks_y[valid_mask]
            task_loss = F.cross_entropy(tasks_probs_valid, tasks_y_valid)

            answers_loss = []
            is_solved_map = {}
            for idx in valid_indices:
                result = results[idx]
                true_answer = answers[idx]
                log_prob = log_probs[idx]
                cost = costs[idx]
                predict_answer = MATH_get_predict(result)
                is_solved = MATH_is_correct(predict_answer, true_answer)
                total_solved += is_solved
                total_executed += 1
                utility = is_solved - cost * args.cost_rate
                is_solved_map[idx] = is_solved
                answers_loss.append(-log_prob * utility)

            # Telemetry
            compact_workflows = getattr(router, "last_compact_workflows", [])
            if compact_workflows:
                csv_rows = []
                for idx, workflow in enumerate(compact_workflows):
                    if not workflow:
                        continue
                    ep_idx = episode_counter
                    episode_counter += 1
                    item_id = workflow.get("item_id", item_ids[idx] if idx < len(item_ids) else "")
                    base = {
                        "run_id": run_id, "epoch": epoch, "batch_id": i_batch,
                        "episode_index": ep_idx, "dataset": args.domain, "split": "train",
                        "item_id": item_id,
                        "topology": workflow.get("topology", ""),
                        "role_set": "-".join(workflow.get("role_set", [])),
                        "workflow_latency_seconds": workflow.get("workflow_latency_seconds", 0.0),
                        "llm_elapsed_seconds": workflow.get("llm_elapsed_seconds", 0.0),
                        "quality_is_solved": int(is_solved_map.get(idx, 0)),
                    }
                    csv_rows.append({**base, "record_type": "episode"})
                    for step in workflow.get("transitions", []):
                        _model = step.get("llm_name", "")
                        _snap = model_metrics.get(_model, {})
                        csv_rows.append({
                            **base, "record_type": "step",
                            "step_index": step.get("step_index", ""),
                            "round_index": step.get("round_index", ""),
                            "dep_level": step.get("dep_level", ""),
                            "role_name": step.get("role_name", ""),
                            "llm_name": _model,
                            "latency_seconds": step.get("latency_seconds", 0.0),
                            "rpm_queue_running": _snap.get("num_requests_running", 0),
                            "rpm_queue_waiting": _snap.get("num_requests_waiting", 0),
                            "e2e_avg": _snap.get("e2e_avg", 0.0),
                            "prompt_tokens": step.get("prompt_tokens", 0),
                            "completion_tokens": step.get("completion_tokens", 0),
                            "cache_hit": step.get("cache_hit", ""),
                        })
                train_writer.append_rows(csv_rows)

            answer_loss = torch.stack(answers_loss).sum() / len(answers_loss)
            if isinstance(vae_loss, torch.Tensor):
                vae_loss = vae_loss[valid_mask].mean() if vae_loss.ndim > 0 and vae_loss.shape[0] == len(queries) else vae_loss.mean()
            else:
                vae_loss = torch.tensor(vae_loss, device=device).mean()

            loss = task_loss + answer_loss + vae_loss * 0.001
            loss.backward()
            optimizer.step()

            accuracy = total_solved / total_executed if total_executed else 0.0
            # Per-model queue + cache stats for progress bar
            q_status = BlackboxRequestQueue.instance().get_status() if BlackboxRequestQueue.instance() else {}
            cache_stats = cache.get_model_stats() if cache else {}
            model_info = {}
            for m, s in q_status.items():
                short = m.split("/")[-1][:12]
                cs = cache_stats.get(m, {})
                h, miss = cs.get("hits", 0), cs.get("misses", 0)
                model_info[short] = f"{s.get('window_count', 0)}rq {h}h/{miss}m"
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{accuracy:.2%}",
                             solved=f"{total_solved}/{total_executed}", **model_info)

        logger.info("Epoch {} done: {}/{} solved ({:.1%})", epoch, total_solved, total_executed,
                     total_solved / total_executed if total_executed else 0.0)
        if BlackboxRequestQueue.instance():
            BlackboxRequestQueue.instance().log_stats()
        if cache:
            cache.log_stats()

        if args.save_checkpoint:
            ckpt_path = args.save_checkpoint
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            torch.save(router.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    logger.info("End blackbox training.")
    if cache:
        cache.log_stats()

    if args.skip_test:
        logger.info("Skipping test phase (--skip-test).")
        sys.exit(0)

    # ── Testing phase ────────────────────────────────────────────────
    logger.info("Start blackbox testing...")
    arrival_rates = args.arrival_rate if isinstance(args.arrival_rate, list) else [args.arrival_rate]
    test_csv = args.test_telemetry_csv or f"{log_dir}/{args.domain}_blackbox_{current_time}_test.csv"
    os.makedirs(os.path.dirname(test_csv), exist_ok=True)
    quality_writer = CsvTelemetryWriter(test_csv, fieldnames=BASELINE_TEST_FIELDS)

    for rate_idx, arrival_rate in enumerate(arrival_rates):
        # Drain queues between sweeps to prevent bleed-over
        if rate_idx > 0:
            bq = BlackboxRequestQueue.instance()
            if bq is not None:
                logger.info("Draining blackbox queues before next arrival rate sweep...")
                bq.wait_until_idle(poll_interval=2.0)
                logger.info("Queues drained.")

        logger.info("Testing with arrival_rate={} ({}/{})", arrival_rate, rate_idx + 1, len(arrival_rates))
        total_solved, total_executed, total_skipped, total_failed = 0, 0, 0, 0

        pattern_gen = RequestPattern(pattern=args.arrival_pattern, rate=arrival_rate, seed=1234)
        shooter = RequestShooter(pattern_gen, max_concurrency=args.concurrency,
                                 capture_output=True, collect_results=True)

        def handler(row):
            query = row["problem"]
            true_answer = row["solution"]
            item_id = row.get("id", "")
            with torch.no_grad():
                results, costs, log_probs, tasks_probs, vae_loss, agents_num = router.forward(
                    [query], tasks, llms, reasonings, [0],
                    prompt_file=args.prompt_file,
                    item_ids=[item_id],
                    dataset=args.domain,
                    split="test",
                    request_timeout=args.request_timeout,
                )
            workflows = getattr(router, "last_compact_workflows", [])
            wf = workflows[0] if workflows else {}
            skipped = getattr(router, "last_skipped_indices", set())
            if 0 in skipped:
                return None
            # Snapshot queue metrics NOW while other workflows are still in-flight
            metrics_snapshot = {m: dict(model_metrics.get(m, {})) for m in pool.models}
            return {
                "query": query, "true_answer": true_answer, "item_id": item_id,
                "result": results[0], "cost": costs[0],
                "workflow_latency_seconds": wf.get("workflow_latency_seconds", 0.0),
                "llm_elapsed_seconds": wf.get("llm_elapsed_seconds", 0.0),
                "transitions": wf.get("transitions", []),
                "task_name": wf.get("task_name", ""),
                "reasoning_name": wf.get("reasoning_name", ""),
                "graph_id": wf.get("graph_id", ""),
                "num_agents": wf.get("num_agents", 0),
                "agent_roles_json": wf.get("agent_roles_json", ""),
                "agent_llms_json": wf.get("agent_llms_json", ""),
                "role_llm_map_json": wf.get("role_llm_map_json", ""),
                "metrics_snapshot": metrics_snapshot,
            }

        results = shooter.run(test_dataset, handler=handler,
                              item_id_fn=lambda row, idx: str(row.get("id", idx)))

        for res in sorted(results, key=lambda r: r.index):
            if not res.success:
                total_failed += 1
                continue
            if res.output is None:
                total_skipped += 1
                continue
            payload = res.output
            predict_answer = MATH_get_predict(payload["result"])
            is_solved = MATH_is_correct(predict_answer, payload["true_answer"])
            total_solved += is_solved
            total_executed += 1

            csv_rows = [{
                "run_id": current_time, "dataset": args.domain, "split": "test",
                "item_id": str(payload["item_id"]), "record_type": "episode",
                "quality_is_correct": int(is_solved),
                "quality_pred": str(predict_answer),
                "quality_gold": str(MATH_get_predict(payload["true_answer"])),
                "workflow_latency_seconds": payload["workflow_latency_seconds"],
                "llm_elapsed_seconds": payload.get("llm_elapsed_seconds", 0.0),
                "arrival_rate": arrival_rate, "arrival_pattern": args.arrival_pattern,
                "cost_rate": args.cost_rate,
                "task_name": payload.get("task_name", ""),
                "reasoning_name": payload.get("reasoning_name", ""),
                "graph_id": payload.get("graph_id", ""),
                "num_agents": payload.get("num_agents", 0),
                "agent_roles_json": payload.get("agent_roles_json", ""),
                "agent_llms_json": payload.get("agent_llms_json", ""),
                "role_llm_map_json": payload.get("role_llm_map_json", ""),
            }]
            _metrics_snap = payload.get("metrics_snapshot", {})
            for si, step in enumerate(payload.get("transitions", [])):
                _model = step.get("llm_name", "")
                _snap = _metrics_snap.get(_model, {})
                csv_rows.append({
                    "run_id": current_time, "dataset": args.domain, "split": "test",
                    "item_id": str(payload["item_id"]), "record_type": "step",
                    "step_index": si,
                    "round_index": step.get("round_index", 0),
                    "dep_level": step.get("dep_level", 0),
                    "role_name": step.get("role_name", ""),
                    "llm_name": _model,
                    "latency_seconds": step.get("latency_seconds", 0.0),
                    "node_id": step.get("node_id", ""),
                    "spatial_predecessors": step.get("spatial_predecessors", ""),
                    "spatial_successors": step.get("spatial_successors", ""),
                    "rpm_queue_running": _snap.get("num_requests_running", 0),
                    "rpm_queue_waiting": _snap.get("num_requests_waiting", 0),
                    "e2e_avg": _snap.get("e2e_avg", 0.0),
                    "prompt_tokens": step.get("prompt_tokens", 0),
                    "completion_tokens": step.get("completion_tokens", 0),
                    "strategy_name": step.get("strategy_name", ""),
                    "arrival_rate": arrival_rate, "arrival_pattern": args.arrival_pattern,
                    "cost_rate": args.cost_rate,
                    "cache_hit": step.get("cache_hit", ""),
                })
            quality_writer.append_rows(csv_rows)

        accuracy = total_solved / total_executed if total_executed else 0.0

        # Collect per-sweep stats for detailed logging
        sweep_episodes = [r for r in results if r.success and r.output is not None]
        sweep_latencies = [r.output["workflow_latency_seconds"] for r in sweep_episodes]
        sweep_latencies.sort()

        # Per-model call counts and latencies
        model_stats = {}
        model_queue_stats = {}
        for r in sweep_episodes:
            snap = r.output.get("metrics_snapshot", {})
            for step in r.output.get("transitions", []):
                m = step.get("llm_name", "")
                if m not in model_stats:
                    model_stats[m] = []
                    model_queue_stats[m] = []
                model_stats[m].append(step.get("latency_seconds", 0.0))
                m_snap = snap.get(m, {})
                model_queue_stats[m].append(
                    m_snap.get("num_requests_running", 0) + m_snap.get("num_requests_waiting", 0)
                )

        # Topology counts
        topo_counts = {}
        for r in sweep_episodes:
            t = r.output.get("reasoning_name", "unknown")
            topo_counts[t] = topo_counts.get(t, 0) + 1

        logger.info("=" * 60)
        logger.info(
            "arrival_rate={}: {}/{} solved ({:.1%}) | skipped={} failed={} | total_items={}",
            arrival_rate, total_solved, total_executed, accuracy,
            total_skipped, total_failed, len(test_dataset),
        )
        if sweep_latencies:
            mid = len(sweep_latencies) // 2
            logger.info(
                "  Latency: mean={:.1f}s  median={:.1f}s  min={:.1f}s  max={:.1f}s",
                sum(sweep_latencies) / len(sweep_latencies),
                sweep_latencies[mid], sweep_latencies[0], sweep_latencies[-1],
            )
        logger.info("  Topologies: {}", topo_counts)
        for m in sorted(model_stats):
            lats = model_stats[m]
            qs = model_queue_stats[m]
            avg_q = sum(qs) / len(qs) if qs else 0
            max_q = max(qs) if qs else 0
            logger.info(
                "  {:30s}  calls={:3d}  avg_lat={:6.1f}s  avg_queue={:.1f}  max_queue={:.0f}",
                m, len(lats), sum(lats) / len(lats), avg_q, max_q,
            )
        logger.info("=" * 60)

    if cache:
        cache.log_stats()
    logger.info("Blackbox testing complete.")
