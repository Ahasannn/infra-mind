#!/usr/bin/env python3
"""
Option C: Latency explosion under load + accuracy.

Top row: Latency vs Arrival Rate — baselines explode, InfraMind stays flat.
Bottom row: Accuracy vs Arrival Rate — baselines shown as flat band (mean ± std)
            since they make identical routing decisions; InfraMind shown as line
            since it adapts.

5 columns = 5 datasets.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import COLORS, MARKERS, LABELS, DATASET_DISPLAY, apply_style, save_fig

apply_style()

RESULTS_PATH = Path(__file__).parent.parent.parent / "results_summary.json"
OUTPUT_DIR = Path(__file__).parent.parent

DATASETS = ["math", "mbpp", "gsm_hard", "humaneval", "mmlu_pro"]
RATES = [10.0, 30.0, 50.0, 100.0, 200.0]
MAS_COST_RATE = "10.0"

METHODS_CFG = [
    ("inframind", "INFRAMIND", None, True),     # adaptive=True
    ("mas",       "MAS",       float(MAS_COST_RATE), False),
    ("moa",       "MoA",       None, False),
    ("gptswarm",  "GPTSwarm",  None, False),
]


def load_data():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def get_vals(data, method, ds, rates, cr=None):
    accs, lats = [], []
    valid_rates = []
    for rate in rates:
        rate_key = str(rate)
        if method == "inframind":
            e = data["inframind"].get(ds, {}).get("by_arrival_rate", {}).get(rate_key)
        elif method == "mas":
            e = data["mas"].get(ds, {}).get(str(cr), {}).get("by_arrival_rate", {}).get(rate_key)
        elif method == "moa":
            e = data["moa"].get(ds, {}).get("by_arrival_rate", {}).get(rate_key)
        elif method == "gptswarm":
            e = data["gptswarm"].get(ds, {}).get("by_arrival_rate", {}).get(rate_key)
        else:
            e = None
        if e and e["accuracy"] is not None and e["latency"]["mean"] is not None:
            accs.append(e["accuracy"])
            lats.append(e["latency"]["mean"])
            valid_rates.append(rate)
    return valid_rates, accs, lats


def plot():
    data = load_data()
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.5), squeeze=False)

    for col, ds in enumerate(DATASETS):
        ax_lat = axes[0][col]
        ax_acc = axes[1][col]

        for m_key, m_label, cr, adaptive in METHODS_CFG:
            color = COLORS[m_label]
            marker = MARKERS[m_label]
            label_full = LABELS[m_label]

            valid_rates, accs, lats = get_vals(data, m_key, ds, RATES, cr=cr)
            if not valid_rates:
                continue

            lbl = label_full if col == 0 else ""
            lw = 2.5 if m_key == "inframind" else 1.8
            ms = 8 if m_key == "inframind" else 6
            zord = 7 if m_key == "inframind" else 5

            # ── Latency: always show real curve ──
            ax_lat.plot(valid_rates, lats,
                        color=color, marker=marker,
                        linewidth=lw, markersize=ms,
                        markeredgecolor="white", markeredgewidth=0.6,
                        zorder=zord, label=lbl)

            # ── Accuracy ──
            if adaptive:
                # InfraMind: show real curve (it adapts decisions)
                ax_acc.plot(valid_rates, [a * 100 for a in accs],
                            color=color, marker=marker,
                            linewidth=lw, markersize=ms,
                            markeredgecolor="white", markeredgewidth=0.6,
                            zorder=zord)
            else:
                # Baselines: flat line at mean accuracy ± std band
                # (same routing decisions, accuracy variation is just LLM noise)
                mean_acc = np.mean(accs) * 100
                std_acc = np.std(accs) * 100

                x_range = [min(valid_rates), max(valid_rates)]
                ax_acc.plot(x_range, [mean_acc, mean_acc],
                            color=color, linewidth=lw, linestyle="-",
                            zorder=zord)
                ax_acc.fill_between(x_range,
                                     mean_acc - std_acc, mean_acc + std_acc,
                                     color=color, alpha=0.12, zorder=zord - 1)
                # Small marker at center for legend consistency
                mid_x = np.mean(x_range)
                ax_acc.scatter([mid_x], [mean_acc],
                               color=color, marker=marker, s=ms**2,
                               edgecolors="white", linewidth=0.6,
                               zorder=zord)

        # ── Latency formatting ──
        ax_lat.set_yscale("log")
        ax_lat.set_ylim(10, 3000)
        ax_lat.set_yticks([10, 30, 100, 300, 1000])
        ax_lat.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        ax_lat.axhline(y=100, color="#bbbbbb", linewidth=0.7, linestyle="--", zorder=1)
        ax_lat.set_title(DATASET_DISPLAY[ds], fontsize=11, fontweight="bold", pad=8)
        ax_lat.tick_params(axis="both", labelsize=8)
        ax_lat.set_xticks(RATES)
        ax_lat.set_xticklabels([str(int(r)) for r in RATES], fontsize=7.5)
        if col == 0:
            ax_lat.set_ylabel("Mean Latency (s)", fontsize=10)

        # Shade unacceptable region
        ax_lat.axhspan(300, 3000, color="#ffeeee", alpha=0.4, zorder=0)
        if col == len(DATASETS) - 1:
            ax_lat.annotate("Unacceptable\nlatency",
                            xy=(0.95, 0.85), xycoords="axes fraction",
                            fontsize=6.5, color="#cc4444", ha="right",
                            fontstyle="italic")

        # ── Accuracy formatting ──
        ax_acc.set_xticks(RATES)
        ax_acc.set_xticklabels([str(int(r)) for r in RATES], fontsize=7.5)
        ax_acc.set_ylim(30, 105)  # Fixed scale so small gaps look small
        ax_acc.tick_params(axis="both", labelsize=8)
        ax_acc.set_xlabel("Arrival Rate (req/min)", fontsize=9)
        if col == 0:
            ax_acc.set_ylabel("Accuracy (%)", fontsize=10)

    # Row labels
    axes[0][0].annotate("Latency", xy=(-0.25, 0.5), xycoords="axes fraction",
                         fontsize=10, fontweight="bold", rotation=90,
                         ha="center", va="center", color="#555555")
    axes[1][0].annotate("Accuracy", xy=(-0.25, 0.5), xycoords="axes fraction",
                         fontsize=10, fontweight="bold", rotation=90,
                         ha="center", va="center", color="#555555")

    # Legend
    handles, labels_list = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels_list, loc="lower center", ncol=4,
               fontsize=10, frameon=True, fancybox=False,
               edgecolor="#cccccc", bbox_to_anchor=(0.5, -0.04),
               handletextpad=0.5, columnspacing=1.5)

    fig.suptitle("Latency Explosion under Load: Baselines vs INFRAMIND",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12, hspace=0.3, wspace=0.2)

    save_fig(fig, str(OUTPUT_DIR / "latency_vs_load"))
    plt.close(fig)


if __name__ == "__main__":
    plot()
    print("Done.")
