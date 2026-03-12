#!/usr/bin/env python3
"""
Aggregate all experiment CSVs into a single results.json.

Reads test CSVs for MAS, InfraMind, MoA, GPTSwarm across all 5 datasets.
Produces episode-level and step-level aggregations grouped by method-specific parameters.

Usage:
    python results/aggregate_results.py
    python results/aggregate_results.py --logs-root /path/to/logs
"""

import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Increase CSV field size limit for large JSON fields in CSVs
csv.field_size_limit(sys.maxsize)

DATASETS = ["math", "mbpp", "gsm_hard", "humaneval", "mmlu_pro"]


def safe_float(v, default=None):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def safe_int(v, default=None):
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return default


def percentile(data, p):
    """Compute the p-th percentile of a list."""
    if not data:
        return None
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def compute_stats(values):
    """Compute summary statistics for a list of numeric values."""
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "median": None, "std": None, "p5": None, "p25": None, "p75": None, "p95": None, "min": None, "max": None}
    return {
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        "p5": round(percentile(values, 5), 4),
        "p25": round(percentile(values, 25), 4),
        "p75": round(percentile(values, 75), 4),
        "p95": round(percentile(values, 95), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def read_csv_rows(path):
    """Read all rows from a CSV file."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ─── MAS Baseline ───────────────────────────────────────────────────────────

def aggregate_mas(logs_root):
    """Aggregate MAS baseline test results."""
    results = {}
    for ds in DATASETS:
        csv_path = os.path.join(logs_root, "test", ds, "mas", f"mas_{ds}_test_5_10_15.csv")
        if not os.path.exists(csv_path):
            print(f"  [MAS] Missing: {csv_path}")
            continue

        print(f"  [MAS] Processing {ds}...")
        rows = read_csv_rows(csv_path)

        # Separate episodes and steps
        episodes = [r for r in rows if r["record_type"] == "episode"]
        steps = [r for r in rows if r["record_type"] == "step"]

        # Build step index: (item_id, arrival_rate, cost_rate) -> list of steps
        step_index = defaultdict(list)
        for s in steps:
            key = (s["item_id"], s["arrival_rate"], s.get("cost_rate", ""))
            step_index[key].append(s)

        # Group episodes by (arrival_rate, cost_rate)
        groups = defaultdict(list)
        for ep in episodes:
            ar = safe_float(ep["arrival_rate"])
            cr = safe_float(ep.get("cost_rate"))
            groups[(ar, cr)].append(ep)

        ds_results = []
        for (ar, cr), eps in sorted(groups.items()):
            accuracies = [safe_int(ep["quality_is_correct"], 0) for ep in eps]
            latencies = [safe_float(ep["workflow_latency_seconds"]) for ep in eps]
            llm_elapsed = [safe_float(ep["llm_elapsed_seconds"]) for ep in eps]

            # Aggregate step-level data for this group
            all_steps = []
            for ep in eps:
                key = (ep["item_id"], ep["arrival_rate"], ep.get("cost_rate", ""))
                all_steps.extend(step_index[key])

            # Token counts from steps
            prompt_tokens = [safe_int(s["prompt_tokens"]) for s in all_steps if safe_int(s.get("prompt_tokens")) is not None]
            completion_tokens = [safe_int(s["completion_tokens"]) for s in all_steps if safe_int(s.get("completion_tokens")) is not None]

            # Model distribution from steps
            model_counts = defaultdict(int)
            for s in all_steps:
                if s.get("llm_name"):
                    model_counts[s["llm_name"]] += 1
            total_steps = sum(model_counts.values()) or 1
            model_dist = {m: round(c / total_steps, 4) for m, c in sorted(model_counts.items())}

            # Topology distribution from episodes
            topo_counts = defaultdict(int)
            for ep in eps:
                tn = ep.get("task_name", "") + "/" + ep.get("reasoning_name", "")
                topo_counts[tn] += 1
            total_eps = len(eps) or 1
            topo_dist = {t: round(c / total_eps, 4) for t, c in sorted(topo_counts.items())}

            # Num agents
            num_agents_list = [safe_int(ep.get("num_agents")) for ep in eps]

            entry = {
                "arrival_rate": ar,
                "cost_rate": cr,
                "n_episodes": len(eps),
                "accuracy": round(sum(accuracies) / len(accuracies), 4) if accuracies else None,
                "n_correct": sum(accuracies),
                "latency": compute_stats(latencies),
                "llm_elapsed": compute_stats(llm_elapsed),
                "tokens_per_episode": {
                    "prompt": round(statistics.mean(prompt_tokens), 1) if prompt_tokens else None,
                    "completion": round(statistics.mean(completion_tokens), 1) if completion_tokens else None,
                    "steps_per_episode": round(len(all_steps) / len(eps), 2),
                },
                "model_distribution": model_dist,
                "topology_distribution": topo_dist,
                "num_agents": compute_stats(num_agents_list),
            }
            ds_results.append(entry)

        results[ds] = ds_results

    return results


# ─── InfraMind ───────────────────────────────────────────────────────────────

def aggregate_inframind(logs_root):
    """Aggregate InfraMind test results."""
    results = {}
    for ds in DATASETS:
        csv_path = os.path.join(logs_root, "test", ds, "inframind", f"inframind_{ds}_test.csv")
        if not os.path.exists(csv_path):
            print(f"  [InfraMind] Missing: {csv_path}")
            continue

        print(f"  [InfraMind] Processing {ds}...")
        rows = read_csv_rows(csv_path)

        episodes = [r for r in rows if r["record_type"] == "episode"]
        role_steps = [r for r in rows if r["record_type"] == "role_step"]

        # Build step index: (item_id, budget_total, arrival_rate) -> list of steps
        step_index = defaultdict(list)
        for s in role_steps:
            key = (s["item_id"], s["budget_total"], s["arrival_rate"])
            step_index[key].append(s)

        # Group episodes by (budget_total, arrival_rate)
        groups = defaultdict(list)
        for ep in episodes:
            bt = safe_float(ep["budget_total"])
            ar = safe_float(ep["arrival_rate"])
            groups[(bt, ar)].append(ep)

        ds_results = []
        for (bt, ar), eps in sorted(groups.items()):
            accuracies = [safe_int(ep["quality_is_solved"], 0) for ep in eps]
            latencies = [safe_float(ep["workflow_latency_seconds"]) for ep in eps]
            llm_elapsed = [safe_float(ep["llm_elapsed_seconds"]) for ep in eps]
            cost_ratios = [safe_float(ep.get("cost_ratio")) for ep in eps]
            lambdas = [safe_float(ep.get("lambda_lag")) for ep in eps]

            # Token counts from episodes (InfraMind has them at episode level)
            prompt_tokens = [safe_int(ep["prompt_tokens"]) for ep in eps if safe_int(ep.get("prompt_tokens")) is not None]
            completion_tokens = [safe_int(ep["completion_tokens"]) for ep in eps if safe_int(ep.get("completion_tokens")) is not None]
            total_tokens = [safe_int(ep["total_tokens"]) for ep in eps if safe_int(ep.get("total_tokens")) is not None]

            # Step-level aggregation
            all_steps = []
            for ep in eps:
                key = (ep["item_id"], ep["budget_total"], ep["arrival_rate"])
                all_steps.extend(step_index[key])

            # Model distribution
            model_counts = defaultdict(int)
            for s in all_steps:
                if s.get("model_name"):
                    model_counts[s["model_name"]] += 1
            total_s = sum(model_counts.values()) or 1
            model_dist = {m: round(c / total_s, 4) for m, c in sorted(model_counts.items())}

            # Strategy distribution
            strategy_counts = defaultdict(int)
            for s in all_steps:
                if s.get("strategy_name"):
                    strategy_counts[s["strategy_name"]] += 1
            total_strat = sum(strategy_counts.values()) or 1
            strategy_dist = {st: round(c / total_strat, 4) for st, c in sorted(strategy_counts.items())}

            # Model × Strategy joint distribution
            joint_counts = defaultdict(int)
            for s in all_steps:
                if s.get("model_name") and s.get("strategy_name"):
                    joint_counts[f"{s['model_name']}|{s['strategy_name']}"] += 1
            total_j = sum(joint_counts.values()) or 1
            joint_dist = {k: round(c / total_j, 4) for k, c in sorted(joint_counts.items())}

            # Topology distribution
            topo_counts = defaultdict(int)
            for ep in eps:
                topo_counts[ep.get("topology", "?")] += 1
            total_eps = len(eps) or 1
            topo_dist = {t: round(c / total_eps, 4) for t, c in sorted(topo_counts.items())}

            # Budget utilization
            budget_remaining = [safe_float(s.get("budget_remaining")) for s in all_steps if s.get("budget_remaining")]

            entry = {
                "budget": bt,
                "arrival_rate": ar,
                "n_episodes": len(eps),
                "accuracy": round(sum(accuracies) / len(accuracies), 4) if accuracies else None,
                "n_correct": sum(accuracies),
                "latency": compute_stats(latencies),
                "llm_elapsed": compute_stats(llm_elapsed),
                "cost_ratio": compute_stats(cost_ratios),
                "lambda": compute_stats(lambdas),
                "tokens_per_episode": {
                    "prompt": round(statistics.mean(prompt_tokens), 1) if prompt_tokens else None,
                    "completion": round(statistics.mean(completion_tokens), 1) if completion_tokens else None,
                    "total": round(statistics.mean(total_tokens), 1) if total_tokens else None,
                    "steps_per_episode": round(len(all_steps) / len(eps), 2),
                },
                "model_distribution": model_dist,
                "strategy_distribution": strategy_dist,
                "model_strategy_joint": joint_dist,
                "topology_distribution": topo_dist,
            }
            ds_results.append(entry)

        results[ds] = ds_results

    return results


# ─── MoA ─────────────────────────────────────────────────────────────────────

def aggregate_moa(logs_root):
    """Aggregate MoA test results."""
    results = {}
    for ds in DATASETS:
        csv_path = os.path.join(logs_root, "test", ds, "moa", f"moa_{ds}_test.csv")
        if not os.path.exists(csv_path):
            print(f"  [MoA] Missing: {csv_path}")
            continue

        print(f"  [MoA] Processing {ds}...")
        rows = read_csv_rows(csv_path)
        episodes = [r for r in rows if r["record_type"] == "episode"]

        groups = defaultdict(list)
        for ep in episodes:
            ar = safe_float(ep["arrival_rate"])
            groups[ar].append(ep)

        ds_results = []
        for ar, eps in sorted(groups.items()):
            accuracies = [safe_int(ep["quality_is_correct"], 0) for ep in eps]
            latencies = [safe_float(ep["total_latency_seconds"]) for ep in eps]
            agg_latencies = [safe_float(ep["aggregator_latency_seconds"]) for ep in eps]
            proposer_max = [safe_float(ep["proposer_latency_max_seconds"]) for ep in eps]

            entry = {
                "arrival_rate": ar,
                "n_episodes": len(eps),
                "accuracy": round(sum(accuracies) / len(accuracies), 4) if accuracies else None,
                "n_correct": sum(accuracies),
                "latency": compute_stats(latencies),
                "aggregator_latency": compute_stats(agg_latencies),
                "proposer_max_latency": compute_stats(proposer_max),
            }
            ds_results.append(entry)

        results[ds] = ds_results

    return results


# ─── GPTSwarm ────────────────────────────────────────────────────────────────

def aggregate_gptswarm(logs_root):
    """Aggregate GPTSwarm test results."""
    results = {}
    for ds in DATASETS:
        csv_path = os.path.join(logs_root, "test", ds, "gptswarm", f"gptswarm_{ds}_test.csv")
        if not os.path.exists(csv_path):
            print(f"  [GPTSwarm] Missing: {csv_path}")
            continue

        print(f"  [GPTSwarm] Processing {ds}...")
        rows = read_csv_rows(csv_path)
        episodes = [r for r in rows if r["record_type"] == "episode"]

        groups = defaultdict(list)
        for ep in episodes:
            ar = safe_float(ep["arrival_rate"])
            groups[ar].append(ep)

        ds_results = []
        for ar, eps in sorted(groups.items()):
            accuracies = [safe_int(ep["quality_is_correct"], 0) for ep in eps]
            latencies = [safe_float(ep["total_latency_seconds"]) for ep in eps]
            agg_latencies = [safe_float(ep["aggregator_latency_seconds"]) for ep in eps]
            active_edges = [safe_int(ep["num_active_edges"]) for ep in eps]
            nodes_succeeded = [safe_int(ep["num_nodes_succeeded"]) for ep in eps]

            entry = {
                "arrival_rate": ar,
                "n_episodes": len(eps),
                "accuracy": round(sum(accuracies) / len(accuracies), 4) if accuracies else None,
                "n_correct": sum(accuracies),
                "latency": compute_stats(latencies),
                "aggregator_latency": compute_stats(agg_latencies),
                "active_edges": compute_stats(active_edges),
                "nodes_succeeded": compute_stats(nodes_succeeded),
            }
            ds_results.append(entry)

        results[ds] = ds_results

    return results


# ─── Training summaries ─────────────────────────────────────────────────────

def aggregate_mas_training(logs_root):
    """Aggregate MAS baseline training results (best epoch per cost rate)."""
    results = {}
    for ds in DATASETS:
        train_dir = os.path.join(logs_root, "baseline_train", ds)
        if not os.path.isdir(train_dir):
            continue

        ds_results = []
        for fname in sorted(os.listdir(train_dir)):
            if not fname.endswith(".csv"):
                continue
            csv_path = os.path.join(train_dir, fname)
            print(f"  [MAS-train] Processing {fname}...")

            rows = read_csv_rows(csv_path)
            episodes = [r for r in rows if r["record_type"] == "episode"]

            # Group by (epoch, split)
            epoch_splits = defaultdict(list)
            for ep in episodes:
                epoch_splits[(ep.get("epoch", ""), ep.get("split", ""))].append(ep)

            # Find best validation epoch
            best_val_acc = -1
            best_val_epoch = None
            for (epoch, split), eps in epoch_splits.items():
                if split == "val":
                    acc = sum(safe_int(ep.get("quality_is_solved", ep.get("quality_is_correct", 0)), 0) for ep in eps) / max(len(eps), 1)
                    if acc > best_val_acc:
                        best_val_acc = acc
                        best_val_epoch = epoch

            # Summarize last epoch train and best val
            train_eps = epoch_splits.get((best_val_epoch, "train"), [])
            val_eps = epoch_splits.get((best_val_epoch, "val"), [])

            # Extract cost rate from filename
            cost_rate = None
            for part in fname.replace(".csv", "").split("_"):
                if part.startswith("cost"):
                    cost_rate = safe_float(part[4:])

            entry = {
                "filename": fname,
                "cost_rate": cost_rate,
                "best_val_epoch": safe_int(best_val_epoch),
                "best_val_accuracy": round(best_val_acc, 4) if best_val_acc >= 0 else None,
                "best_val_n": len(val_eps),
                "train_n": len(train_eps),
            }
            ds_results.append(entry)

        if ds_results:
            results[ds] = ds_results

    return results


def aggregate_inframind_training(logs_root):
    """Aggregate InfraMind training results (best epoch summary)."""
    results = {}
    for ds in DATASETS:
        train_dir = os.path.join(logs_root, "inframind_training", ds)
        if not os.path.isdir(train_dir):
            continue

        ds_results = []
        for fname in sorted(os.listdir(train_dir)):
            if not fname.endswith(".csv"):
                continue
            csv_path = os.path.join(train_dir, fname)
            print(f"  [InfraMind-train] Processing {fname}...")

            rows = read_csv_rows(csv_path)
            episodes = [r for r in rows if r["record_type"] == "episode"]

            if not episodes:
                continue

            # Group by (epoch, split)
            epoch_splits = defaultdict(list)
            for ep in episodes:
                epoch_splits[(ep.get("epoch", ""), ep.get("split", ""))].append(ep)

            # Find best validation epoch
            best_val_acc = -1
            best_val_epoch = None
            for (epoch, split), eps in epoch_splits.items():
                if split == "val":
                    acc = sum(safe_int(ep.get("quality_is_solved", 0), 0) for ep in eps) / max(len(eps), 1)
                    if acc > best_val_acc:
                        best_val_acc = acc
                        best_val_epoch = epoch

            # Get unique budget/arrival combos seen in training
            train_eps = [ep for ep in episodes if ep.get("split") == "train"]
            budgets = sorted(set(safe_float(ep["budget_total"]) for ep in train_eps if ep.get("budget_total")))
            rates = sorted(set(safe_float(ep["arrival_rate"]) for ep in train_eps if ep.get("arrival_rate")))

            # Lambda trajectory: mean lambda per epoch
            lambda_by_epoch = defaultdict(list)
            for ep in episodes:
                if ep.get("split") == "train" and ep.get("lambda_lag"):
                    lam = safe_float(ep["lambda_lag"])
                    if lam is not None:
                        lambda_by_epoch[ep["epoch"]].append(lam)
            lambda_trajectory = {k: round(statistics.mean(v), 4) for k, v in sorted(lambda_by_epoch.items())}

            n_epochs = len(set(ep.get("epoch", "") for ep in episodes))

            entry = {
                "filename": fname,
                "n_epochs": n_epochs,
                "best_val_epoch": safe_int(best_val_epoch),
                "best_val_accuracy": round(best_val_acc, 4) if best_val_acc >= 0 else None,
                "budget_tiers": budgets,
                "arrival_rates": rates,
                "lambda_trajectory": lambda_trajectory,
            }
            ds_results.append(entry)

        if ds_results:
            results[ds] = ds_results

    return results


def aggregate_gptswarm_training(logs_root):
    """Aggregate GPTSwarm training results."""
    results = {}
    for ds in DATASETS:
        train_dir = os.path.join(logs_root, "gptswarm_training", ds)
        if not os.path.isdir(train_dir):
            continue

        ds_results = []
        for fname in sorted(os.listdir(train_dir)):
            if not fname.endswith(".csv"):
                continue
            csv_path = os.path.join(train_dir, fname)
            print(f"  [GPTSwarm-train] Processing {fname}...")

            rows = read_csv_rows(csv_path)
            episodes = [r for r in rows if r["record_type"] == "episode"]

            if not episodes:
                continue

            # Group by (iteration, phase)
            iter_phases = defaultdict(list)
            for ep in episodes:
                iter_phases[(ep.get("iteration", ""), ep.get("phase", ""))].append(ep)

            # Best val iteration
            best_val_acc = -1
            best_val_iter = None
            for (iteration, phase), eps in iter_phases.items():
                if phase == "val":
                    acc = sum(safe_int(ep.get("quality_is_correct", 0), 0) for ep in eps) / max(len(eps), 1)
                    if acc > best_val_acc:
                        best_val_acc = acc
                        best_val_iter = iteration

            n_iterations = len(set(ep.get("iteration", "") for ep in episodes))

            entry = {
                "filename": fname,
                "n_iterations": n_iterations,
                "best_val_iteration": safe_int(best_val_iter),
                "best_val_accuracy": round(best_val_acc, 4) if best_val_acc >= 0 else None,
                "n_episodes": len(episodes),
            }
            ds_results.append(entry)

        if ds_results:
            results[ds] = ds_results

    return results


# ─── Cross-method comparison helpers ─────────────────────────────────────────

def build_cross_method_summary(mas, inframind, moa, gptswarm):
    """Build per-dataset, per-arrival-rate comparison across all methods."""
    summary = {}
    for ds in DATASETS:
        ds_summary = {}

        # Collect all arrival rates across methods
        all_rates = set()
        if ds in mas:
            all_rates.update(e["arrival_rate"] for e in mas[ds])
        if ds in inframind:
            all_rates.update(e["arrival_rate"] for e in inframind[ds])
        if ds in moa:
            all_rates.update(e["arrival_rate"] for e in moa[ds])
        if ds in gptswarm:
            all_rates.update(e["arrival_rate"] for e in gptswarm[ds])

        for rate in sorted(all_rates):
            if rate is None:
                continue
            rate_summary = {}

            # MoA at this rate
            if ds in moa:
                moa_entries = [e for e in moa[ds] if e["arrival_rate"] == rate]
                if moa_entries:
                    e = moa_entries[0]
                    rate_summary["moa"] = {
                        "accuracy": e["accuracy"],
                        "mean_latency": e["latency"]["mean"],
                        "median_latency": e["latency"]["median"],
                        "p95_latency": e["latency"]["p95"],
                        "n_episodes": e["n_episodes"],
                    }

            # GPTSwarm at this rate
            if ds in gptswarm:
                gs_entries = [e for e in gptswarm[ds] if e["arrival_rate"] == rate]
                if gs_entries:
                    e = gs_entries[0]
                    rate_summary["gptswarm"] = {
                        "accuracy": e["accuracy"],
                        "mean_latency": e["latency"]["mean"],
                        "median_latency": e["latency"]["median"],
                        "p95_latency": e["latency"]["p95"],
                        "n_episodes": e["n_episodes"],
                    }

            # MAS at this rate (multiple cost rates)
            if ds in mas:
                mas_entries = [e for e in mas[ds] if e["arrival_rate"] == rate]
                if mas_entries:
                    rate_summary["mas"] = {}
                    for e in sorted(mas_entries, key=lambda x: x["cost_rate"] or 0):
                        cr_key = f"cost_rate={e['cost_rate']}"
                        rate_summary["mas"][cr_key] = {
                            "accuracy": e["accuracy"],
                            "mean_latency": e["latency"]["mean"],
                            "median_latency": e["latency"]["median"],
                            "p95_latency": e["latency"]["p95"],
                            "n_episodes": e["n_episodes"],
                        }

            # InfraMind at this rate (multiple budgets)
            if ds in inframind:
                im_entries = [e for e in inframind[ds] if e["arrival_rate"] == rate]
                if im_entries:
                    rate_summary["inframind"] = {}
                    for e in sorted(im_entries, key=lambda x: x["budget"] or 0):
                        b_key = f"budget={e['budget']}"
                        rate_summary["inframind"][b_key] = {
                            "accuracy": e["accuracy"],
                            "mean_latency": e["latency"]["mean"],
                            "median_latency": e["latency"]["median"],
                            "p95_latency": e["latency"]["p95"],
                            "cost_ratio_mean": e["cost_ratio"]["mean"] if e.get("cost_ratio") else None,
                            "n_episodes": e["n_episodes"],
                        }

            if rate_summary:
                ds_summary[f"arrival_rate={rate}"] = rate_summary

        if ds_summary:
            summary[ds] = ds_summary

    return summary


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results into results.json")
    parser.add_argument("--logs-root", default=None, help="Path to logs root directory")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    # Resolve logs root
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    if args.logs_root:
        logs_root = args.logs_root
    else:
        # Try symlink first
        logs_link = repo_root / "logs"
        if logs_link.exists():
            logs_root = str(logs_link.resolve())
        else:
            logs_root = "/blue/qi855292.ucf/ah872032.ucf/logs"

    output_path = args.output or str(script_dir / "results.json")

    print(f"Logs root: {logs_root}")
    print(f"Output: {output_path}")
    print()

    # ── Test results ──
    print("=== Aggregating test results ===")
    print()

    print("[1/4] MAS Baseline...")
    mas_results = aggregate_mas(logs_root)
    print()

    print("[2/4] InfraMind...")
    inframind_results = aggregate_inframind(logs_root)
    print()

    print("[3/4] MoA...")
    moa_results = aggregate_moa(logs_root)
    print()

    print("[4/4] GPTSwarm...")
    gptswarm_results = aggregate_gptswarm(logs_root)
    print()

    # ── Training results ──
    print("=== Aggregating training results ===")
    print()

    print("[1/3] MAS Training...")
    mas_train = aggregate_mas_training(logs_root)
    print()

    print("[2/3] InfraMind Training...")
    inframind_train = aggregate_inframind_training(logs_root)
    print()

    print("[3/3] GPTSwarm Training...")
    gptswarm_train = aggregate_gptswarm_training(logs_root)
    print()

    # ── Cross-method comparison ──
    print("=== Building cross-method comparison ===")
    cross_method = build_cross_method_summary(mas_results, inframind_results, moa_results, gptswarm_results)
    print()

    # ── Assemble final JSON ──
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "logs_root": logs_root,
            "datasets": DATASETS,
            "methods": ["mas", "inframind", "moa", "gptswarm"],
            "description": "Aggregated experiment results for INFRAMIND project. "
                           "Test results contain per-parameter-combo stats. "
                           "Training results contain best-epoch summaries. "
                           "Cross-method comparison provides side-by-side accuracy/latency at each arrival rate.",
        },
        "test_results": {
            "mas": mas_results,
            "inframind": inframind_results,
            "moa": moa_results,
            "gptswarm": gptswarm_results,
        },
        "training_results": {
            "mas": mas_train,
            "inframind": inframind_train,
            "gptswarm": gptswarm_train,
        },
        "cross_method_comparison": cross_method,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary stats
    print("=== Summary ===")
    for method in ["mas", "inframind", "moa", "gptswarm"]:
        method_data = output["test_results"][method]
        for ds, entries in method_data.items():
            n_entries = len(entries)
            total_eps = sum(e["n_episodes"] for e in entries)
            print(f"  {method:12s} | {ds:12s} | {n_entries:3d} combos | {total_eps:5d} episodes")

    print(f"\nResults written to {output_path}")
    file_size = os.path.getsize(output_path)
    print(f"File size: {file_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
