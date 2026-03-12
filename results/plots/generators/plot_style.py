"""Shared plotting style for INFRAMIND paper figures."""

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex", "grid"])
except ImportError:
    pass

# ─── Colors & Markers ────────────────────────────────────────────────────────
COLORS = {
    "INFRAMIND": "#0d7a3e",   # Dark green
    "MAS":       "#d95f02",   # Orange
    "MoA":       "#7570b3",   # Purple
    "GPTSwarm":  "#e74c3c",   # Red
}

MARKERS = {
    "INFRAMIND": "o",
    "MAS":       "s",
    "MoA":       "D",
    "GPTSwarm":  "^",
}

LABELS = {
    "INFRAMIND": "INFRAMIND (Ours)",
    "MAS":       "MASRouter",
    "MoA":       "MoA",
    "GPTSwarm":  "GPTSwarm",
}

# White outline for text on plots
STROKE = [pe.withStroke(linewidth=2.5, foreground="white")]

DATASET_DISPLAY = {
    "math": "MATH",
    "mbpp": "MBPP",
    "gsm_hard": "GSM-Hard",
    "humaneval": "HumanEval",
    "mmlu_pro": "MMLU-Pro",
}

ARRIVAL_RATE_LABELS = {
    10.0:  "10 req/min",
    30.0:  "30 req/min",
    50.0:  "50 req/min",
    100.0: "100 req/min",
    200.0: "200 req/min",
}

# ─── Shared rcParams ─────────────────────────────────────────────────────────
def apply_style():
    plt.rcParams.update({
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save_fig(fig, path_stem):
    """Save figure as PNG, PDF, and SVG."""
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(f"{path_stem}.{ext}", facecolor="white")
    print(f"  Saved: {path_stem}.{{png,pdf,svg}}")
