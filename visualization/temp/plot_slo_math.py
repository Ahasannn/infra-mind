"""
SLO Compliance — MATH Dataset.
Publication-quality using SciencePlots.
"""

import scienceplots  # noqa: F401
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use(["science", "no-latex", "grid"])

# ── Shared palette ───────────────────────────────────────────────────────
COLORS = {
    "INFRAMIND": "#0d7a3e",
    "MAS":       "#d95f02",
    "MoA":       "#7570b3",
    "GPTSwarm":  "#e74c3c",
}
MARKERS = {"INFRAMIND": "o", "MAS": "s", "MoA": "D", "GPTSwarm": "p"}

# ── Load data ────────────────────────────────────────────────────────────
root = Path(__file__).resolve().parent.parent.parent / "logs" / "test" / "math"

df_infra = pd.read_csv(root / "inframind" / "inframind_math_test.csv",
                       usecols=["record_type", "arrival_rate", "budget_total",
                                "quality", "workflow_latency_seconds"])
df_infra = df_infra[df_infra["record_type"] == "episode"]

df_mas = pd.read_csv(root / "mas" / "mas_math_test_5_10_15.csv",
                     usecols=["record_type", "arrival_rate", "cost_rate",
                              "quality_is_correct", "workflow_latency_seconds"])
df_mas = df_mas[(df_mas["record_type"] == "episode") & (df_mas["cost_rate"] == 5)]

df_moa = pd.read_csv(root / "moa" / "moa_math_test.csv",
                     usecols=["record_type", "arrival_rate",
                              "quality_is_correct", "total_latency_seconds"])
df_moa = df_moa[df_moa["record_type"] == "episode"]

df_gpt = pd.read_csv(root / "gptswarm" / "gptswarm_math_test.csv",
                     usecols=["record_type", "arrival_rate",
                              "quality_is_correct", "total_latency_seconds"])
df_gpt = df_gpt[df_gpt["record_type"] == "episode"]

# ── Config ───────────────────────────────────────────────────────────────
arrival_rates = [10, 50, 100, 200]
budget_tiers  = [10, 30, 50, 100, 200, 300]  # skip 600, 1000 for SLO (trivially 100%)


# ── Compute SLO ──────────────────────────────────────────────────────────
def infra_slo(ar, bt):
    sub = df_infra[(df_infra["arrival_rate"] == ar) &
                   (df_infra["budget_total"] == bt)]
    if len(sub) == 0:
        return np.nan
    return (sub["workflow_latency_seconds"] <= bt).mean() * 100


def baseline_slo(df, ar, bt, lat_col):
    sub = df[df["arrival_rate"] == ar]
    if len(sub) == 0:
        return np.nan
    return (sub[lat_col] <= bt).mean() * 100


# ── Figure ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(12, 3.8), sharey=True)
fig.subplots_adjust(wspace=0.08, bottom=0.22)

x = np.arange(len(budget_tiers))
budget_labels = [f"{b}s" for b in budget_tiers]

for idx, ar in enumerate(arrival_rates):
    ax = axes[idx]

    series = {
        "INFRAMIND": [infra_slo(ar, bt) for bt in budget_tiers],
        "MAS":       [baseline_slo(df_mas, ar, bt, "workflow_latency_seconds")
                      for bt in budget_tiers],
        "MoA":       [baseline_slo(df_moa, ar, bt, "total_latency_seconds")
                      for bt in budget_tiers],
        "GPTSwarm":  [baseline_slo(df_gpt, ar, bt, "total_latency_seconds")
                      for bt in budget_tiers],
    }

    # Light fill under InfraMind
    ax.fill_between(x, 0, series["INFRAMIND"],
                    color=COLORS["INFRAMIND"], alpha=0.10, zorder=2)

    for name, vals in series.items():
        ax.plot(x, vals, marker=MARKERS[name], color=COLORS[name],
                linewidth=1.5, markersize=5,
                markeredgecolor="white", markeredgewidth=0.5,
                zorder=5, label=name if idx == 0 else None)

    ax.set_xticks(x)
    ax.set_xticklabels(budget_labels)
    ax.set_xlabel("Budget Deadline (s)")
    ax.set_title(f"{ar} req/min")
    ax.set_ylim(-3, 108)
    ax.set_xlim(-0.3, len(budget_tiers) - 0.7)

    if idx == 0:
        ax.set_ylabel("SLO Compliance (%)")

    ax.axhline(100, color="#cccccc", linewidth=0.6, linestyle="--", zorder=1)

fig.legend(*axes[0].get_legend_handles_labels(),
           loc="lower center", ncol=4, frameon=True,
           bbox_to_anchor=(0.5, 0.01), fontsize=8,
           handletextpad=0.3, columnspacing=1.2)

fig.suptitle("SLO Compliance | MATH", fontsize=12, y=1.02)

out_dir = Path(__file__).resolve().parent
for ext in ("png", "pdf", "svg"):
    out = out_dir / f"slo_math.{ext}"
    fig.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
    print(f"Saved -> {out}")
