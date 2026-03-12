#!/usr/bin/env python3
"""
Option B: Scatter — each method is ONE point (mean_accuracy, latency_at_rate).

Baselines use mean accuracy across all rates (since routing is identical,
accuracy variation is just LLM noise). Latency is rate-specific.
InfraMind uses actual per-rate accuracy (it adapts).

4 rows (rates) × 5 cols (datasets).
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import COLORS, MARKERS, LABELS, STROKE, DATASET_DISPLAY, apply_style, save_fig

apply_style()

RESULTS_PATH = Path(__file__).parent.parent.parent / "results_summary.json"
OUTPUT_DIR = Path(__file__).parent.parent

DATASETS = ["math", "mbpp", "gsm_hard", "humaneval", "mmlu_pro"]
RATES = [10.0, 50.0, 100.0, 200.0]
ALL_RATES = [10.0, 30.0, 50.0, 100.0, 200.0]
RATE_LABELS = {10.0: "10 req/min", 50.0: "50 req/min",
               100.0: "100 req/min", 200.0: "200 req/min"}
MAS_COST_RATE = "10.0"


def load_data():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def get_entry(data, method, ds, rate, cr=None):
    rate_key = str(rate)
    if method == "inframind":
        return data["inframind"].get(ds, {}).get("by_arrival_rate", {}).get(rate_key)
    elif method == "mas":
        return data["mas"].get(ds, {}).get(str(cr), {}).get("by_arrival_rate", {}).get(rate_key)
    elif method == "moa":
        return data["moa"].get(ds, {}).get("by_arrival_rate", {}).get(rate_key)
    elif method == "gptswarm":
        return data["gptswarm"].get(ds, {}).get("by_arrival_rate", {}).get(rate_key)
    return None


def get_mean_acc(data, method, ds, cr=None):
    """Mean accuracy across all arrival rates for a baseline (constant routing)."""
    accs = []
    for rate in ALL_RATES:
        e = get_entry(data, method, ds, rate, cr=cr)
        if e and e["accuracy"] is not None:
            accs.append(e["accuracy"])
    return np.mean(accs) if accs else None


def plot():
    data = load_data()
    fig, axes = plt.subplots(len(RATES), len(DATASETS),
                              figsize=(17, 3.2 * len(RATES)), squeeze=False)

    methods = [
        ("inframind", "INFRAMIND", None, True),
        ("mas",       "MAS",       float(MAS_COST_RATE), False),
        ("moa",       "MoA",       None, False),
        ("gptswarm",  "GPTSwarm",  None, False),
    ]

    for row, rate in enumerate(RATES):
        for col, ds in enumerate(DATASETS):
            ax = axes[row][col]

            for m_key, m_label, cr, adaptive in methods:
                e = get_entry(data, m_key, ds, rate, cr=cr)
                if e is None or e["latency"]["mean"] is None:
                    continue

                lat = e["latency"]["mean"]

                if adaptive:
                    acc = e["accuracy"]
                else:
                    acc = get_mean_acc(data, m_key, ds, cr=cr)

                if acc is None:
                    continue

                color = COLORS[m_label]
                marker = MARKERS[m_label]
                label_full = LABELS[m_label]
                size = 120 if m_key == "inframind" else 80

                lbl = label_full if (row == 0 and col == 0) else ""
                ax.scatter(lat, acc, color=color, marker=marker,
                           s=size, edgecolors="white", linewidth=0.8,
                           zorder=7 if m_key == "inframind" else 6,
                           label=lbl)

                # Label
                short = "Ours" if m_key == "inframind" else label_full.split("(")[0].strip()
                ha = "right" if m_key == "inframind" else "left"
                ox = -8 if m_key == "inframind" else 8
                ax.annotate(short, (lat, acc),
                            textcoords="offset points", xytext=(ox, -3),
                            fontsize=5.5, color=color, fontweight="bold",
                            path_effects=STROKE, ha=ha)

            # Formatting
            ax.set_xscale("log")
            ax.set_xlim(left=8, right=3000)
            ax.set_ylim(0.30, 1.05)  # Fixed scale so small accuracy gaps look small
            ax.set_xticks([10, 30, 100, 300, 1000])
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.tick_params(axis="both", labelsize=7.5)

            if row == 0:
                ax.set_title(DATASET_DISPLAY[ds], fontsize=11, fontweight="bold", pad=8)
            if col == 0:
                ax.set_ylabel(f"{RATE_LABELS[rate]}\nAccuracy", fontsize=8.5)
            if row == len(RATES) - 1:
                ax.set_xlabel("Mean Latency (s)", fontsize=9)

    handles, labels_list = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels_list, loc="lower center", ncol=4,
               fontsize=9, frameon=True, fancybox=False,
               edgecolor="#cccccc", bbox_to_anchor=(0.5, -0.01),
               handletextpad=0.5, columnspacing=1.5)

    fig.suptitle("Quality vs Latency (baselines at mean accuracy; latency increases with load)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.06, hspace=0.35, wspace=0.2)

    save_fig(fig, str(OUTPUT_DIR / "pareto_scatter"))
    plt.close(fig)


if __name__ == "__main__":
    plot()
    print("Done.")
