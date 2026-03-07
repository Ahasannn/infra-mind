"""Shared Puppeteer training loop across datasets.

Follows the GPTSwarm delegation pattern: each dataset provides
load/evaluate/query functions, and this module handles the training loop.

Usage:
    from MAR.Puppeteer.training import main
    main(
        default_dataset="mbpp",
        load_train_fn=...,
        load_val_fn=...,
        evaluate_fn=...,
        get_query_fn=...,
        get_item_id_fn=...,
    )
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Tuple

import torch
from loguru import logger

from MAR.LLM.llm_profile_full import llm_profile, model_base_urls
from MAR.LLM.llm_embedding import SentenceEncoder
from MAR.Puppeteer.puppeteer_policy import PuppeteerPolicy, PuppeteerResult
from MAR.Puppeteer.puppeteer_trainer import PuppeteerTrainer
from MAR.InfraMind.metrics_watcher import start_metrics_watcher, model_metrics
from MAR.Utils.utils import fix_random_seed
from MAR.Utils.log import configure_logging
from MAR.Utils.telemetry import CsvTelemetryWriter


TRAIN_TELEMETRY_FIELDS = (
    "run_id",
    "dataset",
    "record_type",
    "iteration",
    "phase",
    "item_id",
    "quality_is_correct",
    "quality_feedback",
    "total_latency_seconds",
    "num_steps",
    "num_steps_succeeded",
    "num_steps_failed",
    "step_models_json",
    "step_latencies_json",
    "reinforce_loss",
    "baseline",
    "metrics_snapshot_json",
)


def _build_metrics_url_map(base_urls: dict) -> dict:
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


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add Puppeteer-specific CLI arguments."""
    # Training
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Items per training iteration")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for policy network (Adam)")
    parser.add_argument("--baseline-decay", type=float, default=0.9,
                        help="EMA decay for REINFORCE baseline")
    parser.add_argument("--cost-penalty", type=float, default=0.1,
                        help="Weight for step-count cost in reward")
    parser.add_argument("--val-interval", type=int, default=10,
                        help="Validate every N iterations")

    # Chain params
    parser.add_argument("--max-steps", type=int, default=5,
                        help="Maximum reasoning chain length")

    # Dataset limits
    parser.add_argument("--train-limit", type=int, default=0,
                        help="Limit training examples (0 = all)")
    parser.add_argument("--val-limit", type=int, default=0,
                        help="Limit validation examples (0 = all)")

    # LLM params
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--request-timeout", type=float, default=600.0)

    # Checkpoints
    parser.add_argument("--checkpoint-dir", type=str, default="")
    parser.add_argument("--checkpoint-path", type=str, default="",
                        help="Resume from checkpoint")
    parser.add_argument("--telemetry-csv", type=str, default="")

    # Seed
    parser.add_argument("--seed", type=int, default=1234)


def main(
    default_dataset: str,
    load_train_fn: Callable[[int], list],
    load_val_fn: Callable[[int], list],
    evaluate_fn: Callable[[str, dict], Tuple[bool, str]],
    get_query_fn: Callable[[dict], str],
    get_item_id_fn: Callable[[dict], str],
) -> None:
    """Main Puppeteer training loop.

    Args:
        default_dataset: Dataset name (mbpp, gsm_hard, etc.).
        load_train_fn: (limit) -> list of training items.
        load_val_fn: (limit) -> list of validation items.
        evaluate_fn: (result_text, item) -> (is_correct, feedback).
        get_query_fn: (item) -> query string.
        get_item_id_fn: (item) -> item ID string.
    """
    parser = argparse.ArgumentParser(
        description=f"Puppeteer training — {default_dataset}"
    )
    add_common_args(parser)
    args, _ = parser.parse_known_args()

    fix_random_seed(args.seed)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    configure_logging(log_name=f"puppeteer_{default_dataset}_{current_time}.txt")
    run_id = current_time

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load data
    train_data = load_train_fn(args.train_limit)
    val_data = load_val_fn(args.val_limit)
    logger.info("Puppeteer {}: {} train, {} val", default_dataset, len(train_data), len(val_data))

    # Setup model names
    model_names = [m["Name"] for m in llm_profile]

    # Create sentence encoder (shared)
    encoder = SentenceEncoder()

    # Create policy + trainer
    policy = PuppeteerPolicy(
        num_models=len(model_names),
        model_names=model_names,
        domain=default_dataset,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        request_timeout=args.request_timeout,
    )
    trainer = PuppeteerTrainer(
        policy=policy,
        lr=args.lr,
        baseline_decay=args.baseline_decay,
        cost_penalty=args.cost_penalty,
    )

    # Load checkpoint if resuming
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        trainer.load_checkpoint(args.checkpoint_path)

    # Checkpoint directory
    checkpoint_dir = args.checkpoint_dir
    if not checkpoint_dir:
        checkpoint_dir = "checkpoints/puppeteer"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(
        checkpoint_dir, f"puppeteer_{default_dataset}_best.pt"
    )

    # Start metrics watcher
    metrics_url_map = _build_metrics_url_map(model_base_urls)
    if metrics_url_map:
        start_metrics_watcher(metrics_url_map, interval=1.0)

    # Telemetry
    telemetry_csv = args.telemetry_csv
    if not telemetry_csv:
        log_dir = f"logs/puppeteer_training/{default_dataset}"
        os.makedirs(log_dir, exist_ok=True)
        telemetry_csv = os.path.join(log_dir, f"puppeteer_{default_dataset}_{current_time}.csv")
    os.makedirs(os.path.dirname(telemetry_csv) or ".", exist_ok=True)
    writer = CsvTelemetryWriter(telemetry_csv, fieldnames=TRAIN_TELEMETRY_FIELDS)
    logger.info("Telemetry CSV: {}", telemetry_csv)

    # ── Training loop ─────────────────────────────────────────────────────────
    for iteration in range(1, args.iterations + 1):
        policy.train()

        # Sample batch
        batch = random.sample(train_data, min(args.batch_size, len(train_data)))

        # Pre-compute query embeddings
        queries = [get_query_fn(item) for item in batch]
        with torch.no_grad():
            query_embeddings = encoder(queries)

        # Process batch items sequentially (chain execution is already sequential)
        utilities = []
        all_log_probs = []
        step_counts = []

        for idx, item in enumerate(batch):
            query = queries[idx]
            item_id = get_item_id_fn(item)
            query_emb = query_embeddings[idx]

            result, log_probs = policy.execute_chain(
                query=query,
                item_id=item_id,
                query_embedding=query_emb,
                deterministic=False,
            )

            is_correct, feedback = evaluate_fn(result.final_response, item)
            utility = 1.0 if is_correct else 0.0
            utilities.append(utility)
            all_log_probs.append(log_probs)
            step_counts.append(result.num_steps)

            writer.append_rows([{
                "run_id": run_id,
                "dataset": default_dataset,
                "record_type": "train",
                "iteration": iteration,
                "phase": "train",
                "item_id": item_id,
                "quality_is_correct": is_correct,
                "quality_feedback": feedback,
                "total_latency_seconds": result.total_latency,
                "num_steps": result.num_steps,
                "num_steps_succeeded": result.num_steps_succeeded,
                "num_steps_failed": result.num_steps_failed,
                "step_models_json": json.dumps(result.step_models),
                "step_latencies_json": json.dumps(result.step_latencies),
                "reinforce_loss": 0.0,
                "baseline": trainer.baseline,
                "metrics_snapshot_json": _metrics_snapshot(),
            }])

        # REINFORCE update
        loss = trainer.train_step(utilities, all_log_probs, step_counts)
        train_acc = sum(utilities) / len(utilities) if utilities else 0.0

        logger.info(
            "[Iter {}/{}] train_acc={:.1%} loss={:.4f} baseline={:.3f} avg_steps={:.1f}",
            iteration, args.iterations, train_acc, loss, trainer.baseline,
            sum(step_counts) / len(step_counts) if step_counts else 0,
        )

        # ── Validation ────────────────────────────────────────────────────
        if iteration % args.val_interval == 0 or iteration == args.iterations:
            policy.eval()
            val_correct = 0
            val_total = 0
            val_steps_total = 0

            def _val_item(item):
                query = get_query_fn(item)
                item_id = get_item_id_fn(item)
                with torch.no_grad():
                    query_emb = encoder(query).squeeze(0)
                    result, _ = policy.execute_chain(
                        query=query,
                        item_id=item_id,
                        query_embedding=query_emb,
                        deterministic=True,
                    )
                is_correct, feedback = evaluate_fn(result.final_response, item)
                return item_id, result, is_correct, feedback

            val_concurrency = min(8, len(val_data))
            with ThreadPoolExecutor(max_workers=val_concurrency) as pool:
                val_futures = [pool.submit(_val_item, item) for item in val_data]
                for future in val_futures:
                    item_id, result, is_correct, feedback = future.result()
                    val_correct += int(is_correct)
                    val_total += 1
                    val_steps_total += result.num_steps

                    writer.append_rows([{
                        "run_id": run_id,
                        "dataset": default_dataset,
                        "record_type": "val",
                        "iteration": iteration,
                        "phase": "val",
                        "item_id": item_id,
                        "quality_is_correct": is_correct,
                        "quality_feedback": feedback,
                        "total_latency_seconds": result.total_latency,
                        "num_steps": result.num_steps,
                        "num_steps_succeeded": result.num_steps_succeeded,
                        "num_steps_failed": result.num_steps_failed,
                        "step_models_json": json.dumps(result.step_models),
                        "step_latencies_json": json.dumps(result.step_latencies),
                        "reinforce_loss": loss,
                        "baseline": trainer.baseline,
                        "metrics_snapshot_json": _metrics_snapshot(),
                    }])

            val_acc = val_correct / val_total if val_total else 0.0
            avg_val_steps = val_steps_total / val_total if val_total else 0.0
            logger.info(
                "[Iter {}/{}] val_acc={:.1%} ({}/{}) avg_steps={:.1f}",
                iteration, args.iterations, val_acc, val_correct, val_total, avg_val_steps,
            )

            # Save best checkpoint
            if val_acc > trainer.best_val_accuracy:
                trainer.best_val_accuracy = val_acc
                trainer.save_checkpoint(best_checkpoint_path)
                logger.info(
                    "[Iter {}] New best val accuracy: {:.1%} -> saved {}",
                    iteration, val_acc, best_checkpoint_path,
                )

    # ── Save last checkpoint ─────────────────────────────────────────────────
    last_checkpoint_path = os.path.join(
        checkpoint_dir, f"puppeteer_{default_dataset}_last.pt"
    )
    trainer.save_checkpoint(last_checkpoint_path)
    logger.info("Saved last checkpoint: {}", last_checkpoint_path)

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("Puppeteer {} training complete.", default_dataset)
    logger.info("Best val accuracy: {:.1%}", trainer.best_val_accuracy)
    logger.info("Best checkpoint: {}", best_checkpoint_path)
    logger.info("Last checkpoint: {}", last_checkpoint_path)
