#!/usr/bin/env python3
"""
SLO Compliance: % of episodes completing within budget deadline.

Layout: 2 rows (Low=10, High=200 req/min) × 5 columns (datasets).
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
SHOW_RATES = [10.0, 200.0]
RATE_LABELS = {10.0: "Low Load (10 req/min)", 200.0: "High Load (200 req/min)"}
BUDGET_TIERS = [10, 30, 50, 100, 200, 300]
MAS_COST_RATE = "10.0"


def load_data():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def get_slo(slo_dict):
    return [slo_dict.get(str(bt), 0) for bt in BUDGET_TIERS]


def plot():
    data = load_data()
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), squeeze=False)
    x = np.arange(len(BUDGET_TIERS))

    for row, rate in enumerate(SHOW_RATES):
        for col, ds in enumerate(DATASETS):
            ax = axes[row][col]

            # ── InfraMind ──
            im = data["inframind"].get(ds, {}).get("by_arrival_rate", {}).get(str(rate))
            if im:
                slo = get_slo(im["slo_compliance"])
                ax.plot(x, slo, color=COLORS["INFRAMIND"], marker=MARKERS["INFRAMIND"],
                        linewidth=2.0, markersize=5,
                        markeredgecolor="white", markeredgewidth=0.5,
                        zorder=6, label=LABELS["INFRAMIND"] if row == 0 and col == 0 else "")
                ax.fill_between(x, 0, slo, color=COLORS["INFRAMIND"], alpha=0.08)

            # ── MAS ──
            mas = data["mas"].get(ds, {}).get(MAS_COST_RATE, {}).get("by_arrival_rate", {}).get(str(rate))
            if mas:
                slo = get_slo(mas["slo_compliance"])
                ax.plot(x, slo, color=COLORS["MAS"], marker=MARKERS["MAS"],
                        linewidth=1.8, markersize=5,
                        markeredgecolor="white", markeredgewidth=0.5,
                        zorder=5, label=LABELS["MAS"] if row == 0 and col == 0 else "")

            # ── MoA ──
            moa = data["moa"].get(ds, {}).get("by_arrival_rate", {}).get(str(rate))
            if moa:
                slo = get_slo(moa["slo_compliance"])
                ax.plot(x, slo, color=COLORS["MoA"], marker=MARKERS["MoA"],
                        linewidth=1.8, markersize=5,
                        markeredgecolor="white", markeredgewidth=0.5,
                        zorder=5, label=LABELS["MoA"] if row == 0 and col == 0 else "")

            # ── GPTSwarm ──
            gs = data["gptswarm"].get(ds, {}).get("by_arrival_rate", {}).get(str(rate))
            if gs:
                slo = get_slo(gs["slo_compliance"])
                ax.plot(x, slo, color=COLORS["GPTSwarm"], marker=MARKERS["GPTSwarm"],
                        linewidth=1.8, markersize=5,
                        markeredgecolor="white", markeredgewidth=0.5,
                        zorder=5, label=LABELS["GPTSwarm"] if row == 0 and col == 0 else "")

            # ── Reference ──
            ax.axhline(y=100, color="#cccccc", linewidth=0.6, linestyle="--", zorder=1)

            # ── Formatting ──
            ax.set_xticks(x)
            ax.set_xticklabels([str(bt) for bt in BUDGET_TIERS], fontsize=7.5)
            ax.set_ylim(-3, 108)
            ax.set_xlim(-0.3, len(BUDGET_TIERS) - 0.7)
            ax.tick_params(axis="both", labelsize=7.5)

            if row == 0:
                ax.set_title(DATASET_DISPLAY[ds], fontsize=11, fontweight="bold", pad=8)
            if col == 0:
                ax.set_ylabel("SLO Compliance (%)", fontsize=9)
                ax.annotate(RATE_LABELS[rate],
                            xy=(0, 0.5), xycoords="axes fraction",
                            xytext=(-55, 0), textcoords="offset points",
                            fontsize=8, fontweight="bold", rotation=90,
                            ha="center", va="center", color="#555555")
            if row == 1:
                ax.set_xlabel("Budget Deadline (s)", fontsize=9)

    # ── Legend ──
    handles, labels_list = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels_list, loc="lower center", ncol=4,
               fontsize=9, frameon=True, fancybox=False,
               edgecolor="#cccccc", bbox_to_anchor=(0.5, -0.01),
               handletextpad=0.4, columnspacing=1.5)

    fig.suptitle("SLO Compliance across Budget Deadlines",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.09, hspace=0.3, wspace=0.15)

    save_fig(fig, str(OUTPUT_DIR / "slo_compliance_all_datasets"))
    plt.close(fig)


if __name__ == "__main__":
    plot()
    print("Done.")
