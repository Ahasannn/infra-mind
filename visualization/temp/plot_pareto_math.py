"""
Quality vs Latency Pareto — MATH Dataset.
InfraMind shown as a budget-tier curve (controllable frontier).
Baselines as single points (no budget control).
"""

import scienceplots  # noqa: F401
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from pathlib import Path

plt.style.use(["science", "no-latex", "grid"])

stroke = [pe.withStroke(linewidth=2.5, foreground="white")]

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
budget_tiers  = [10, 50, 200]


# ── Helpers ──────────────────────────────────────────────────────────────
def agg_infra_curve(ar):
    rows = df_infra[(df_infra["arrival_rate"] == ar) &
                    (df_infra["budget_total"].isin(budget_tiers))]
    g = rows.groupby("budget_total").agg(
        acc=("quality", "mean"),
        lat=("workflow_latency_seconds", "mean"),
    ).reindex(budget_tiers).dropna()
    return g["lat"].values, g["acc"].values * 100, g.index.values


def agg_point(df, ar, acc_col, lat_col):
    rows = df[df["arrival_rate"] == ar]
    if len(rows) == 0:
        return None, None
    return rows[lat_col].mean(), rows[acc_col].mean() * 100


# ── Figure ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)
fig.subplots_adjust(wspace=0.08, bottom=0.22)

for idx, ar in enumerate(arrival_rates):
    ax = axes[idx]

    lats, accs, budgets = agg_infra_curve(ar)
    lat_mas, acc_mas = agg_point(df_mas, ar, "quality_is_correct",
                                  "workflow_latency_seconds")
    lat_moa, acc_moa = agg_point(df_moa, ar, "quality_is_correct",
                                  "total_latency_seconds")
    lat_gpt, acc_gpt = agg_point(df_gpt, ar, "quality_is_correct",
                                  "total_latency_seconds")

    # InfraMind: budget frontier curve
    ax.plot(lats, accs, "-o", color=COLORS["INFRAMIND"],
            linewidth=1.8, markersize=5,
            markeredgecolor="white", markeredgewidth=0.5,
            zorder=5, label="INFRAMIND (ours)" if idx == 0 else None)

    # Label budget tiers with smart offsets to avoid overlap
    n_pts = len(lats)
    for i, (lat_v, acc_v, bud) in enumerate(zip(lats, accs, budgets)):
        if i == 0:
            # First (lowest budget): label below
            xy_off = (0, -9)
            ha, va = "center", "top"
        elif i == n_pts - 1:
            # Last (highest budget): label above-left to avoid MAS overlap
            xy_off = (-14, 8)
            ha, va = "right", "bottom"
        else:
            # Middle point: label to the left
            xy_off = (-10, -2)
            ha, va = "right", "center"
        t = ax.annotate(f"b={int(bud)}s", xy=(lat_v, acc_v),
                        xytext=xy_off, textcoords="offset points",
                        ha=ha, va=va, fontsize=5.5,
                        color=COLORS["INFRAMIND"], fontstyle="italic")
        t.set_path_effects(stroke)

    # Baselines as single points with latency annotations
    for name, lat_v, acc_v in [("MAS", lat_mas, acc_mas),
                                ("MoA", lat_moa, acc_moa),
                                ("GPTSwarm", lat_gpt, acc_gpt)]:
        if lat_v is not None:
            ax.scatter(lat_v, acc_v, s=60, color=COLORS[name],
                       marker=MARKERS[name],
                       edgecolors="white", linewidths=0.6,
                       zorder=6, label=name if idx == 0 else None)
            # Annotate with actual latency for MoA/GPTSwarm (log hides severity)
            if name in ("MoA", "GPTSwarm") and lat_v > 80:
                lbl = f"{lat_v:.0f}s"
                if name == "GPTSwarm":
                    xy_off = (0, -8)
                    h, v = "center", "top"
                else:
                    xy_off = (0, 8)
                    h, v = "center", "bottom"
                t = ax.annotate(lbl, xy=(lat_v, acc_v),
                                xytext=xy_off, textcoords="offset points",
                                ha=h, va=v,
                                fontsize=5, color=COLORS[name], fontweight="bold")
                t.set_path_effects(stroke)

    # Log x-axis
    ax.set_xscale("log")
    ax.set_xlim(8, 3000)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:,.0f}" if x >= 1 else ""))
    ax.set_xticks([10, 30, 100, 300, 1000])

    ax.set_title(f"{ar} req/min")
    ax.set_xlabel("Avg. Latency (s)")
    if idx == 0:
        ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(48, 82)

fig.legend(*axes[0].get_legend_handles_labels(),
           loc="lower center", ncol=4, frameon=True,
           bbox_to_anchor=(0.5, 0.01), fontsize=8,
           handletextpad=0.3, columnspacing=1.2)

fig.suptitle("Quality vs Latency | MATH", fontsize=12, y=1.02)

out_dir = Path(__file__).resolve().parent
for ext in ("png", "pdf", "svg"):
    out = out_dir / f"pareto_math.{ext}"
    fig.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
    print(f"Saved -> {out}")
