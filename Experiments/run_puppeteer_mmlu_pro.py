"""Puppeteer baseline — MMLU-Pro dataset."""

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
from MAR.Puppeteer.puppeteer_runner import PuppeteerRunner
from MAR.Utils.utils import fix_random_seed
from MAR.Utils.log import configure_logging, ProgressTracker
from MAR.Utils.telemetry import CsvTelemetryWriter
from MAR.Utils.request_patterns import RequestPattern
from MAR.Utils.request_shooter import RequestShooter
from MAR.InfraMind.metrics_watcher import start_metrics_watcher, model_metrics

from Datasets.mmlu_pro_dataset import MmluProDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PUPPETEER_TEST_FIELDS = (
    "run_id", "dataset", "item_id", "record_type",
    "quality_is_correct", "quality_feedback",
    "total_latency_seconds", "num_steps",
    "num_steps_succeeded", "num_steps_failed",
    "assigned_model", "step_roles_json",
    "step_models_json", "step_latencies_json",
    "arrival_rate", "arrival_pattern",
    "metrics_snapshot_json",
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


def _postprocess_answer(text):
    """Extract letter answer (A-J) from model response."""
    import re
    import string
    valid_letters = set(string.ascii_uppercase[:10])

    if not text:
        return ""
    ans_pos = text.find("answer is")
    if ans_pos != -1:
        after = text[ans_pos + len("answer is"):].strip(":").strip()
        if after and after[0].upper() in valid_letters:
            return after[0].upper()
    option_match = re.search(r'(?:Option|option)\s+([A-J])', text)
    if option_match:
        return option_match.group(1).upper()
    paren_match = re.search(r'\(([A-J])\)', text)
    if paren_match:
        return paren_match.group(1).upper()
    stripped = text.strip().rstrip(".),:;")
    if len(stripped) <= 2 and stripped and stripped[0].upper() in valid_letters:
        return stripped[0].upper()
    matches = re.findall(r'(?<![a-zA-Z])([A-J])(?![a-zA-Z])', text)
    if matches:
        return matches[-1]
    return ""


def parse_args():
    p = argparse.ArgumentParser(description="Puppeteer baseline — MMLU-Pro")
    p.add_argument("--test_limit", type=int, default=0)
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--max-steps", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7)
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
    domain = "mmlu_pro"

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    configure_logging(log_name=f"puppeteer_{domain}_{current_time}.txt")
    run_id = current_time

    test_limit = args.test_limit if args.test_limit > 0 else 0
    dataset_test = MmluProDataset(
        "test",
        stratified_limit=test_limit,
    )

    test_items = []
    for i in range(len(dataset_test)):
        record = dataset_test[i]
        test_items.append({
            "item_id": i,
            "task": dataset_test.record_to_input(record)["task"],
            "answer": dataset_test.record_to_target_answer(record),
        })
    logger.info("Puppeteer MMLU-Pro: {} test items", len(test_items))

    model_names = [m["Name"] for m in llm_profile]
    runner = PuppeteerRunner(
        model_names=model_names,
        checkpoint_path=args.checkpoint or None,
        domain=domain,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        request_timeout=args.request_timeout,
    )

    metrics_url_map = _build_metrics_url_map(model_base_urls)
    if metrics_url_map:
        start_metrics_watcher(metrics_url_map, interval=1.0)

    telemetry_csv = args.test_telemetry_csv or f"logs/test/{domain}/puppeteer/puppeteer_{domain}_{current_time}.csv"
    os.makedirs(os.path.dirname(telemetry_csv), exist_ok=True)
    writer = CsvTelemetryWriter(telemetry_csv, fieldnames=PUPPETEER_TEST_FIELDS)
    logger.info("Telemetry CSV: {}", telemetry_csv)

    arrival_rates = args.arrival_rate if isinstance(args.arrival_rate, list) else [args.arrival_rate]

    for arrival_rate in arrival_rates:
        logger.info("Testing arrival_rate={}", arrival_rate)
        counters = {"solved": 0, "executed": 0}
        _ctr_lock = threading.Lock()

        progress = ProgressTracker(
            total=len(test_items),
            phase=f"Puppeteer Test (rate={arrival_rate})",
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
            pup = payload["puppeteer"]
            with _ctr_lock:
                counters["executed"] += 1
                counters["solved"] += int(payload["is_solved"])
                progress.total_quality += float(payload["is_solved"])
                progress.total_latency += pup.total_latency
            writer.append_rows([{
                "run_id": run_id, "dataset": domain,
                "item_id": payload["item_id"], "record_type": "episode",
                "quality_is_correct": payload["is_solved"],
                "quality_feedback": payload["feedback"],
                "total_latency_seconds": pup.total_latency,
                "num_steps": pup.num_steps,
                "num_steps_succeeded": pup.num_steps_succeeded,
                "num_steps_failed": pup.num_steps_failed,
                "assigned_model": pup.assigned_model,
                "step_roles_json": json.dumps(pup.step_roles),
                "step_models_json": json.dumps(pup.step_models),
                "step_latencies_json": json.dumps(pup.step_latencies),
                "arrival_rate": arrival_rate,
                "arrival_pattern": args.arrival_pattern,
                "metrics_snapshot_json": _metrics_snapshot(),
            }])

        shooter = RequestShooter(
            pattern_gen, max_concurrency=args.concurrency,
            capture_output=True, collect_results=False,
            on_result=on_result,
        )

        def handler(item):
            query = item["task"]
            true_answer = item["answer"]
            item_id = str(item.get("item_id", ""))

            pup = runner.run_single(query=query, item_id=item_id)

            predict = _postprocess_answer(pup.final_response)
            is_solved = str(predict).strip().upper() == str(true_answer).strip().upper()

            return {
                "item_id": item_id,
                "is_solved": is_solved,
                "feedback": f"pred={predict} gold={true_answer}",
                "puppeteer": pup,
            }

        shooter.run(
            test_items, handler=handler,
            item_id_fn=lambda item, idx: str(item.get("item_id", idx)),
        )

        progress.log_final_summary()
        total_solved = counters["solved"]
        total_executed = counters["executed"]
        accuracy = total_solved / total_executed if total_executed else 0.0
        logger.info("Puppeteer MMLU-Pro rate={}: {}/{} ({:.1%})",
                     arrival_rate, total_solved, total_executed, accuracy)

    logger.info("Puppeteer MMLU-Pro complete.")
