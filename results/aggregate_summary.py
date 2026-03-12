#!/usr/bin/env python3
"""
Build results_summary.json — compact, analysis-ready summary of all test results.

Metrics per grouping:
  - accuracy, n_episodes, n_correct
  - latency: mean, median, p95
  - SLO compliance: % episodes completing within each budget tier
  - model_distribution: {model: count}
  - model_idle_pct: {model: % of steps where running+waiting==0}
  - queue_stats: {model: {mean_running, mean_waiting, mean_kv_cache}}
  - topology_distribution: {topology: count}
  - strategy_distribution: {strategy: count}  (InfraMind only)

Usage:
    python results/aggregate_summary.py
"""

import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

csv.field_size_limit(sys.maxsize)

DATASETS = ["math", "mbpp", "gsm_hard", "humaneval", "mmlu_pro"]
SLO_BUDGET_TIERS = [10, 30, 50, 100, 200, 300]


def sf(v, default=None):
    """Safe float."""
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def si(v, default=None):
    """Safe int."""
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return default


def pct(values, p):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return round(s[f] + (k - f) * (s[c] - s[f]), 4)


def lat_stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "median": None, "p95": None}
    return {
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "p95": round(pct(values, 95), 4),
    }


def slo_compliance(latencies, budget_tiers=SLO_BUDGET_TIERS):
    """Compute % of episodes with latency <= each budget tier."""
    lats = [v for v in latencies if v is not None]
    if not lats:
        return {}
    n = len(lats)
    return {str(bt): round(sum(1 for l in lats if l <= bt) / n * 100, 2) for bt in budget_tiers}


def model_idle_pct(steps, model_col):
    """Compute % of step observations where running+waiting==0, per model."""
    per_model = defaultdict(lambda: {"total": 0, "idle": 0})
    for s in steps:
        m = s.get(model_col)
        if not m:
            continue
        running = sf(s.get("llm_running"), 0)
        waiting = sf(s.get("llm_waiting"), 0)
        per_model[m]["total"] += 1
        if running + waiting == 0:
            per_model[m]["idle"] += 1
    result = {}
    for m, counts in sorted(per_model.items()):
        if counts["total"] > 0:
            result[m] = round(counts["idle"] / counts["total"] * 100, 2)
    return result


def queue_stats(steps, model_col):
    """Compute mean running, waiting, kv_cache per model."""
    per_model = defaultdict(lambda: {"running": [], "waiting": [], "kv_cache": []})
    for s in steps:
        m = s.get(model_col)
        if not m:
            continue
        r = sf(s.get("llm_running"))
        w = sf(s.get("llm_waiting"))
        kv = sf(s.get("llm_kv_cache_usage"))
        if r is not None:
            per_model[m]["running"].append(r)
        if w is not None:
            per_model[m]["waiting"].append(w)
        if kv is not None:
            per_model[m]["kv_cache"].append(kv)
    result = {}
    for m, vals in sorted(per_model.items()):
        result[m] = {
            "mean_running": round(statistics.mean(vals["running"]), 4) if vals["running"] else None,
            "mean_waiting": round(statistics.mean(vals["waiting"]), 4) if vals["waiting"] else None,
            "mean_kv_cache": round(statistics.mean(vals["kv_cache"]), 4) if vals["kv_cache"] else None,
            "n_observations": len(vals["running"]),
        }
    return result


def model_dist(steps, model_col):
    counts = defaultdict(int)
    for s in steps:
        m = s.get(model_col)
        if m:
            counts[m] += 1
    return dict(sorted(counts.items()))


def topo_dist_mas(episodes):
    counts = defaultdict(int)
    for ep in episodes:
        t = ep.get("task_name", "?") + "/" + ep.get("reasoning_name", "?")
        counts[t] += 1
    return dict(sorted(counts.items()))


def topo_dist_inframind(episodes):
    counts = defaultdict(int)
    for ep in episodes:
        counts[ep.get("topology", "?")] += 1
    return dict(sorted(counts.items()))


def strategy_dist(steps):
    counts = defaultdict(int)
    for s in steps:
        st = s.get("strategy_name")
        if st:
            counts[st] += 1
    return dict(sorted(counts.items()))


def read_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


# ─── MAS ─────────────────────────────────────────────────────────────────────

def summarize_mas(logs_root):
    results = {}
    for ds in DATASETS:
        csv_path = os.path.join(logs_root, "test", ds, "mas", f"mas_{ds}_test_5_10_15.csv")
        if not os.path.exists(csv_path):
            continue
        print(f"  [MAS] {ds}")
        rows = read_csv(csv_path)
        episodes = [r for r in rows if r["record_type"] == "episode"]
        steps = [r for r in rows if r["record_type"] == "step"]

        # Index steps by (item_id, arrival_rate, cost_rate)
        step_idx = defaultdict(list)
        for s in steps:
            step_idx[(s["item_id"], s["arrival_rate"], s.get("cost_rate", ""))].append(s)

        # Group by cost_rate
        by_cost = defaultdict(list)
        for ep in episodes:
            by_cost[sf(ep.get("cost_rate"))].append(ep)

        ds_result = {}
        for cr, cr_eps in sorted(by_cost.items()):
            # Sub-group by arrival_rate
            by_ar = defaultdict(list)
            for ep in cr_eps:
                by_ar[sf(ep["arrival_rate"])].append(ep)

            ar_results = {}
            for ar, ar_eps in sorted(by_ar.items()):
                accs = [si(ep["quality_is_correct"], 0) for ep in ar_eps]
                lats = [sf(ep["workflow_latency_seconds"]) for ep in ar_eps]

                # Gather steps for this group
                grp_steps = []
                for ep in ar_eps:
                    grp_steps.extend(step_idx[(ep["item_id"], ep["arrival_rate"], ep.get("cost_rate", ""))])

                ar_results[str(ar)] = {
                    "n_episodes": len(ar_eps),
                    "n_correct": sum(accs),
                    "accuracy": round(sum(accs) / len(accs), 4) if accs else None,
                    "latency": lat_stats(lats),
                    "slo_compliance": slo_compliance(lats),
                    "model_distribution": model_dist(grp_steps, "llm_name"),
                    "model_idle_pct": model_idle_pct(grp_steps, "llm_name"),
                    "queue_stats": queue_stats(grp_steps, "llm_name"),
                    "topology_distribution": topo_dist_mas(ar_eps),
                }

            # Also compute aggregate across all arrival rates for this cost_rate
            all_accs = [si(ep["quality_is_correct"], 0) for ep in cr_eps]
            all_lats = [sf(ep["workflow_latency_seconds"]) for ep in cr_eps]
            all_steps = []
            for ep in cr_eps:
                all_steps.extend(step_idx[(ep["item_id"], ep["arrival_rate"], ep.get("cost_rate", ""))])

            ds_result[str(cr)] = {
                "aggregate": {
                    "n_episodes": len(cr_eps),
                    "n_correct": sum(all_accs),
                    "accuracy": round(sum(all_accs) / len(all_accs), 4) if all_accs else None,
                    "latency": lat_stats(all_lats),
                    "slo_compliance": slo_compliance(all_lats),
                    "model_distribution": model_dist(all_steps, "llm_name"),
                    "model_idle_pct": model_idle_pct(all_steps, "llm_name"),
                    "queue_stats": queue_stats(all_steps, "llm_name"),
                    "topology_distribution": topo_dist_mas(cr_eps),
                },
                "by_arrival_rate": ar_results,
            }

        results[ds] = ds_result
    return results


# ─── InfraMind ───────────────────────────────────────────────────────────────

def summarize_inframind(logs_root):
    results = {}
    for ds in DATASETS:
        csv_path = os.path.join(logs_root, "test", ds, "inframind", f"inframind_{ds}_test.csv")
        if not os.path.exists(csv_path):
            continue
        print(f"  [InfraMind] {ds}")
        rows = read_csv(csv_path)
        episodes = [r for r in rows if r["record_type"] == "episode"]
        role_steps = [r for r in rows if r["record_type"] == "role_step"]

        # Index steps
        step_idx = defaultdict(list)
        for s in role_steps:
            step_idx[(s["item_id"], s["budget_total"], s["arrival_rate"])].append(s)

        def build_entry(eps, grp_steps):
            accs = [si(ep["quality_is_solved"], 0) for ep in eps]
            lats = [sf(ep["workflow_latency_seconds"]) for ep in eps]
            return {
                "n_episodes": len(eps),
                "n_correct": sum(accs),
                "accuracy": round(sum(accs) / len(accs), 4) if accs else None,
                "latency": lat_stats(lats),
                "slo_compliance": slo_compliance(lats),
                "model_distribution": model_dist(grp_steps, "model_name"),
                "model_idle_pct": model_idle_pct(grp_steps, "model_name"),
                "queue_stats": queue_stats(grp_steps, "model_name"),
                "topology_distribution": topo_dist_inframind(eps),
                "strategy_distribution": strategy_dist(grp_steps),
            }

        def get_steps(eps):
            grp = []
            for ep in eps:
                grp.extend(step_idx[(ep["item_id"], ep["budget_total"], ep["arrival_rate"])])
            return grp

        # By arrival_rate (across all budgets)
        by_ar = defaultdict(list)
        for ep in episodes:
            by_ar[sf(ep["arrival_rate"])].append(ep)

        ar_results = {}
        for ar, ar_eps in sorted(by_ar.items()):
            ar_results[str(ar)] = build_entry(ar_eps, get_steps(ar_eps))

        # By budget (across all arrival rates)
        by_budget = defaultdict(list)
        for ep in episodes:
            by_budget[sf(ep["budget_total"])].append(ep)

        budget_results = {}
        for bt, bt_eps in sorted(by_budget.items()):
            budget_results[str(bt)] = build_entry(bt_eps, get_steps(bt_eps))

        # By (budget, arrival_rate) — full grid
        by_combo = defaultdict(list)
        for ep in episodes:
            by_combo[(sf(ep["budget_total"]), sf(ep["arrival_rate"]))].append(ep)

        combo_results = {}
        for (bt, ar), combo_eps in sorted(by_combo.items()):
            combo_results[f"budget={bt},arrival_rate={ar}"] = build_entry(combo_eps, get_steps(combo_eps))

        # Global aggregate
        all_steps = []
        for ep in episodes:
            all_steps.extend(step_idx[(ep["item_id"], ep["budget_total"], ep["arrival_rate"])])

        results[ds] = {
            "aggregate": build_entry(episodes, all_steps),
            "by_arrival_rate": ar_results,
            "by_budget": budget_results,
            "by_budget_and_arrival_rate": combo_results,
        }

    return results


# ─── MoA ─────────────────────────────────────────────────────────────────────

def summarize_moa(logs_root):
    results = {}
    for ds in DATASETS:
        csv_path = os.path.join(logs_root, "test", ds, "moa", f"moa_{ds}_test.csv")
        if not os.path.exists(csv_path):
            continue
        print(f"  [MoA] {ds}")
        rows = read_csv(csv_path)
        episodes = [r for r in rows if r["record_type"] == "episode"]

        by_ar = defaultdict(list)
        for ep in episodes:
            by_ar[sf(ep["arrival_rate"])].append(ep)

        ar_results = {}
        for ar, ar_eps in sorted(by_ar.items()):
            accs = [si(ep["quality_is_correct"], 0) for ep in ar_eps]
            lats = [sf(ep["total_latency_seconds"]) for ep in ar_eps]
            agg_lats = [sf(ep["aggregator_latency_seconds"]) for ep in ar_eps]
            prop_max = [sf(ep["proposer_latency_max_seconds"]) for ep in ar_eps]

            ar_results[str(ar)] = {
                "n_episodes": len(ar_eps),
                "n_correct": sum(accs),
                "accuracy": round(sum(accs) / len(accs), 4) if accs else None,
                "latency": lat_stats(lats),
                "aggregator_latency": lat_stats(agg_lats),
                "proposer_max_latency": lat_stats(prop_max),
                "slo_compliance": slo_compliance(lats),
            }

        # Aggregate
        all_accs = [si(ep["quality_is_correct"], 0) for ep in episodes]
        all_lats = [sf(ep["total_latency_seconds"]) for ep in episodes]
        results[ds] = {
            "aggregate": {
                "n_episodes": len(episodes),
                "n_correct": sum(all_accs),
                "accuracy": round(sum(all_accs) / len(all_accs), 4) if all_accs else None,
                "latency": lat_stats(all_lats),
                "slo_compliance": slo_compliance(all_lats),
            },
            "by_arrival_rate": ar_results,
        }

    return results


# ─── GPTSwarm ────────────────────────────────────────────────────────────────

def summarize_gptswarm(logs_root):
    results = {}
    for ds in DATASETS:
        csv_path = os.path.join(logs_root, "test", ds, "gptswarm", f"gptswarm_{ds}_test.csv")
        if not os.path.exists(csv_path):
            continue
        print(f"  [GPTSwarm] {ds}")
        rows = read_csv(csv_path)
        episodes = [r for r in rows if r["record_type"] == "episode"]

        by_ar = defaultdict(list)
        for ep in episodes:
            by_ar[sf(ep["arrival_rate"])].append(ep)

        ar_results = {}
        for ar, ar_eps in sorted(by_ar.items()):
            accs = [si(ep["quality_is_correct"], 0) for ep in ar_eps]
            lats = [sf(ep["total_latency_seconds"]) for ep in ar_eps]
            edges = [si(ep["num_active_edges"]) for ep in ar_eps]
            nodes = [si(ep["num_nodes_succeeded"]) for ep in ar_eps]

            ar_results[str(ar)] = {
                "n_episodes": len(ar_eps),
                "n_correct": sum(accs),
                "accuracy": round(sum(accs) / len(accs), 4) if accs else None,
                "latency": lat_stats(lats),
                "slo_compliance": slo_compliance(lats),
                "mean_active_edges": round(statistics.mean([e for e in edges if e is not None]), 2) if edges else None,
                "mean_nodes_succeeded": round(statistics.mean([n for n in nodes if n is not None]), 2) if nodes else None,
            }

        all_accs = [si(ep["quality_is_correct"], 0) for ep in episodes]
        all_lats = [sf(ep["total_latency_seconds"]) for ep in episodes]
        results[ds] = {
            "aggregate": {
                "n_episodes": len(episodes),
                "n_correct": sum(all_accs),
                "accuracy": round(sum(all_accs) / len(all_accs), 4) if all_accs else None,
                "latency": lat_stats(all_lats),
                "slo_compliance": slo_compliance(all_lats),
            },
            "by_arrival_rate": ar_results,
        }

    return results


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    logs_link = repo_root / "logs"
    logs_root = str(logs_link.resolve()) if logs_link.exists() else "/blue/qi855292.ucf/ah872032.ucf/logs"
    output_path = str(script_dir / "results_summary.json")

    print(f"Logs: {logs_root}")
    print(f"Output: {output_path}\n")

    print("MAS Baseline:")
    mas = summarize_mas(logs_root)
    print()

    print("InfraMind:")
    inframind = summarize_inframind(logs_root)
    print()

    print("MoA:")
    moa = summarize_moa(logs_root)
    print()

    print("GPTSwarm:")
    gptswarm = summarize_gptswarm(logs_root)
    print()

    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "datasets": DATASETS,
            "methods": ["mas", "inframind", "moa", "gptswarm"],
            "slo_budget_tiers": SLO_BUDGET_TIERS,
            "notes": {
                "slo_compliance": "% of episodes where latency <= budget_tier (seconds)",
                "model_idle_pct": "% of step observations where llm_running + llm_waiting == 0",
                "queue_stats": "Mean running/waiting/kv_cache per model from step-level snapshots",
                "model_distribution": "Exact step counts per model",
                "topology_distribution": "Exact episode counts per topology",
                "strategy_distribution": "Exact step counts per strategy (InfraMind only)",
                "mas_grouping": "Grouped by cost_rate, then by arrival_rate within each",
                "inframind_grouping": "by_arrival_rate (across budgets), by_budget (across rates), by_budget_and_arrival_rate (full grid)",
            },
        },
        "mas": mas,
        "inframind": inframind,
        "moa": moa,
        "gptswarm": gptswarm,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"Written: {output_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
