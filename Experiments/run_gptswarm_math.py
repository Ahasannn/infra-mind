"""GPTSwarm baseline — MATH dataset."""

import sys
import os
import io
import time
import argparse
import json
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from loguru import logger

from MAR.LLM.llm_profile_full import llm_profile, model_base_urls
from MAR.GPTSwarm.swarm_runner import SwarmRunner
from MAR.Utils.utils import fix_random_seed
from MAR.Utils.log import configure_logging, ProgressTracker
from MAR.Utils.telemetry import CsvTelemetryWriter
from MAR.Utils.request_patterns import RequestPattern
from MAR.Utils.request_shooter import RequestShooter
from MAR.InfraMind.metrics_watcher import start_metrics_watcher, model_metrics

from Datasets.math_dataset import load_math_dataset, MATH_is_correct, MATH_get_predict

os.environ["TOKENIZERS_PARALLELISM"] = "false"

GPTSWARM_TEST_FIELDS = (
    "run_id", "dataset", "item_id", "record_type",
    "quality_is_correct", "quality_feedback",
    "total_latency_seconds", "aggregator_latency_seconds",
    "num_active_edges", "num_nodes_succeeded", "num_nodes_failed",
    "node_latencies_json", "node_errors_json",
    "arrival_rate", "arrival_pattern",
    "aggregator_model", "metrics_snapshot_json",
)


def _build_metrics_url_map(base_urls: dict):
    urls = {}
    for name, base_url in base_urls.items():
        if not name or not base_url:
            continue
        base = base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3].rstrip("/")
        urls[name] = f"{base}/metrics"
    return urls


def _metrics_snapshot() -> str:
    try:
        return json.dumps({k: dict(v) for k, v in model_metrics.items()})
    except Exception:
        return "{}"


def parse_args():
    p = argparse.ArgumentParser(description="GPTSwarm baseline — MATH")
    p.add_argument("--test_limit", type=int, default=0)
    p.add_argument("--dataset-root", type=str, default="Datasets/MATH")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--aggregator-model", type=str,
                    default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--aggregator-temperature", type=float, default=0.3)
    p.add_argument("--request-timeout", type=float, default=600.0)
    p.add_argument("--arrival-rate", type=float, nargs="+", default=[0.0])
    p.add_argument("--arrival-pattern", type=str, default="poisson")
    p.add_argument("--concurrency", type=int, default=1000)
    p.add_argument("--spike-intensity", type=float, default=10.0)
    p.add_argument("--spike-period", type=float, default=20.0)
    p.add_argument("--burst-duration", type=float, default=3.0)
    p.add_argument("--test-telemetry-csv", type=str, default="")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fix_random_seed(1234)
    domain = "math"

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    configure_logging(log_name=f"gptswarm_{domain}_{current_time}.txt")
    run_id = current_time

    test_limit = args.test_limit if args.test_limit > 0 else 0
    test_data = load_math_dataset(args.dataset_root, "test", stratified_limit=test_limit)
    logger.info("GPTSwarm MATH: {} test items", len(test_data))

    model_names = [m["Name"] for m in llm_profile]
    runner = SwarmRunner(
        model_names=model_names,
        checkpoint_path=args.checkpoint or None,
        domain=domain,
        aggregator_model=args.aggregator_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        aggregator_temperature=args.aggregator_temperature,
        request_timeout=args.request_timeout,
    )

    metrics_url_map = _build_metrics_url_map(model_base_urls)
    if metrics_url_map:
        start_metrics_watcher(metrics_url_map, interval=1.0)

    telemetry_csv = args.test_telemetry_csv or f"logs/test/{domain}/gptswarm/gptswarm_{domain}_{current_time}.csv"
    os.makedirs(os.path.dirname(telemetry_csv), exist_ok=True)
    writer = CsvTelemetryWriter(telemetry_csv, fieldnames=GPTSWARM_TEST_FIELDS)
    logger.info("Telemetry CSV: {}", telemetry_csv)

    arrival_rates = args.arrival_rate if isinstance(args.arrival_rate, list) else [args.arrival_rate]

    for arrival_rate in arrival_rates:
        logger.info("Testing arrival_rate={}", arrival_rate)
        counters = {"solved": 0, "executed": 0}
        _ctr_lock = threading.Lock()

        progress = ProgressTracker(
            total=len(test_data),
            phase=f"GPTSwarm Test (rate={arrival_rate})",
            log_interval=10,
        )

        pattern_gen = RequestPattern(
            pattern=args.arrival_pattern, rate=arrival_rate,
            spike_intensity=args.spike_intensity,
            spike_period=args.spike_period,
            burst_duration=args.burst_duration, seed=1234,
        )

        def on_result(r):
            progress.update(success=r.success)
            if not r.success or r.output is None:
                return
            payload = r.output
            swarm = payload["swarm"]
            with _ctr_lock:
                counters["executed"] += 1
                counters["solved"] += int(payload["is_solved"])
                progress.total_quality += float(payload["is_solved"])
                progress.total_latency += swarm.total_latency
            writer.append_rows([{
                "run_id": run_id, "dataset": domain,
                "item_id": payload["item_id"], "record_type": "episode",
                "quality_is_correct": payload["is_solved"],
                "quality_feedback": payload["feedback"],
                "total_latency_seconds": swarm.total_latency,
                "aggregator_latency_seconds": swarm.aggregator_latency,
                "num_active_edges": swarm.num_active_edges,
                "num_nodes_succeeded": swarm.num_nodes_succeeded,
                "num_nodes_failed": swarm.num_nodes_failed,
                "node_latencies_json": json.dumps(swarm.node_latencies),
                "node_errors_json": json.dumps(swarm.node_errors) if swarm.node_errors else "",
                "arrival_rate": arrival_rate,
                "arrival_pattern": args.arrival_pattern,
                "aggregator_model": args.aggregator_model,
                "metrics_snapshot_json": _metrics_snapshot(),
            }])

        shooter = RequestShooter(
            pattern_gen, max_concurrency=args.concurrency,
            capture_output=True, collect_results=False,
            on_result=on_result,
        )

        def handler(item):
            query = item["problem"]
            true_answer = item["solution"]
            item_id = str(item.get("id", ""))

            swarm = runner.run_single(query=query, item_id=item_id)

            try:
                predict = MATH_get_predict(swarm.final_response)
                is_solved = MATH_is_correct(predict, true_answer)
            except Exception as e:
                logger.warning("Answer parsing failed for {}: {}", item_id, e)
                predict = "0"
                is_solved = False

            gold = MATH_get_predict(true_answer) if true_answer else "0"

            return {
                "item_id": item_id,
                "is_solved": is_solved,
                "feedback": f"pred={predict} gold={gold}",
                "swarm": swarm,
            }

        shooter.run(
            test_data, handler=handler,
            item_id_fn=lambda item, idx: str(item.get("id", idx)),
        )

        progress.log_final_summary()
        total_solved = counters["solved"]
        total_executed = counters["executed"]
        accuracy = total_solved / total_executed if total_executed else 0.0
        logger.info("GPTSwarm MATH rate={}: {}/{} ({:.1%})",
                     arrival_rate, total_solved, total_executed, accuracy)

    logger.info("GPTSwarm MATH complete.")
