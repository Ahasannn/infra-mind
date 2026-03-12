#!/usr/bin/env python3
"""
Generate results tables for the paper.

Baselines use MEAN accuracy across all arrival rates (routing is identical,
per-rate variation is just LLM noise). InfraMind uses actual per-rate accuracy.

Tables:
1. Main results: Accuracy / Latency / SLO@100s at Low/Mid/High load
2. InfraMind budget tier breakdown
3. Load degradation
4. Model & strategy adaptation patterns
5. LaTeX versions
"""

import json
import statistics
from pathlib import Path

RESULTS_PATH = Path(__file__).parent.parent.parent / "results_summary.json"
OUTPUT_DIR = Path(__file__).parent.parent.parent

DATASETS = ["math", "mbpp", "gsm_hard", "humaneval", "mmlu_pro"]
DS_DISPLAY = {"math": "MATH", "mbpp": "MBPP", "gsm_hard": "GSM-Hard",
              "humaneval": "HumanEval", "mmlu_pro": "MMLU-Pro"}

LOW_RATE = 10.0
MID_RATE = 100.0
HIGH_RATE = 200.0
LOAD_RATES = {"Low (10)": LOW_RATE, "Mid (100)": MID_RATE, "High (200)": HIGH_RATE}
ALL_RATES = [10.0, 30.0, 50.0, 100.0, 200.0]

MAS_COST_RATE = "10.0"


def load_data():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def get_raw_entry(data, method, ds, rate):
    rate_key = str(rate)
    if method == "inframind":
        return data["inframind"].get(ds, {}).get("by_arrival_rate", {}).get(rate_key)
    elif method == "mas":
        return data["mas"].get(ds, {}).get(MAS_COST_RATE, {}).get("by_arrival_rate", {}).get(rate_key)
    elif method == "moa":
        return data["moa"].get(ds, {}).get("by_arrival_rate", {}).get(rate_key)
    elif method == "gptswarm":
        return data["gptswarm"].get(ds, {}).get("by_arrival_rate", {}).get(rate_key)
    return None


def get_mean_acc(data, method, ds):
    """Mean accuracy across all rates for load-unaware baselines."""
    accs = []
    for rate in ALL_RATES:
        e = get_raw_entry(data, method, ds, rate)
        if e and e.get("accuracy") is not None:
            accs.append(e["accuracy"])
    return statistics.mean(accs) if accs else None


def get_entry(data, method, ds, rate):
    """Get (accuracy, latency, slo@100) — accuracy is mean for baselines."""
    e = get_raw_entry(data, method, ds, rate)
    if e is None:
        return None

    adaptive = (method == "inframind")
    if adaptive:
        acc = e.get("accuracy")
    else:
        acc = get_mean_acc(data, method, ds)

    lat = e.get("latency", {}).get("mean")
    slo = e.get("slo_compliance", {}).get("100")

    return {"accuracy": acc, "latency": lat, "slo_100": slo, "n_episodes": e.get("n_episodes")}


def fmt_acc(v): return f"{v*100:.1f}" if v is not None else "—"
def fmt_lat(v): return f"{v:.1f}" if v is not None else "—"
def fmt_slo(v): return f"{v:.1f}" if v is not None else "—"


def generate_main_table(data):
    methods = ["inframind", "mas", "moa", "gptswarm"]
    method_display = {"inframind": "INFRAMIND", "mas": "MASRouter", "moa": "MoA", "gptswarm": "GPTSwarm"}

    lines_md = []
    lines_latex = []

    # ─── Markdown ───
    lines_md.append("# Main Results\n")
    lines_md.append("Accuracy (%) / Mean Latency (s) / SLO Compliance at 100s budget (%)\n")
    lines_md.append(f"MAS uses cost_rate={MAS_COST_RATE}. "
                    "Baseline accuracy is mean across all arrival rates (routing is identical). "
                    "InfraMind accuracy is per-rate (it adapts).\n")

    for load_name, rate in LOAD_RATES.items():
        lines_md.append(f"\n## {load_name} req/min\n")
        header = "| Dataset |"
        sep = "|---------|"
        for m in methods:
            header += f" {method_display[m]} Acc | {method_display[m]} Lat | {method_display[m]} SLO |"
            sep += "----:|----:|----:|"
        lines_md.append(header)
        lines_md.append(sep)

        for ds in DATASETS:
            entries = {m: get_entry(data, m, ds, rate) for m in methods}
            accs = [entries[m]["accuracy"] if entries[m] else None for m in methods]
            lats = [entries[m]["latency"] if entries[m] else None for m in methods]
            slos = [entries[m]["slo_100"] if entries[m] else None for m in methods]

            best_acc = max((v for v in accs if v is not None), default=None)
            best_lat = min((v for v in lats if v is not None), default=None)
            best_slo = max((v for v in slos if v is not None), default=None)

            row = f"| {DS_DISPLAY[ds]} |"
            for m in methods:
                e = entries[m]
                if e is None:
                    row += " — | — | — |"
                    continue
                a = fmt_acc(e["accuracy"])
                l = fmt_lat(e["latency"])
                s = fmt_slo(e["slo_100"])
                if e["accuracy"] is not None and best_acc and abs(e["accuracy"] - best_acc) < 0.001:
                    a = f"**{a}**"
                if e["latency"] is not None and best_lat and abs(e["latency"] - best_lat) < 0.5:
                    l = f"**{l}**"
                if e["slo_100"] is not None and best_slo and abs(e["slo_100"] - best_slo) < 0.1:
                    s = f"**{s}**"
                row += f" {a} | {l} | {s} |"
            lines_md.append(row)

    # ─── LaTeX ───
    lines_latex.append(r"\begin{table*}[t]")
    lines_latex.append(r"\centering")
    lines_latex.append(r"\caption{Main results across five benchmarks under varying load. "
                       r"Accuracy (\%), Mean Latency (s), and SLO Compliance at 100s budget (\%). "
                       r"Baseline accuracy is averaged across all arrival rates (routing decisions are load-independent). "
                       r"Best values per metric are \textbf{bolded}.}")
    lines_latex.append(r"\label{tab:main_results}")
    lines_latex.append(r"\resizebox{\textwidth}{!}{%")
    lines_latex.append(r"\begin{tabular}{l|ccc|ccc|ccc|ccc}")
    lines_latex.append(r"\toprule")
    lines_latex.append(r"& \multicolumn{3}{c|}{\textbf{INFRAMIND (Ours)}} "
                       r"& \multicolumn{3}{c|}{\textbf{MASRouter}} "
                       r"& \multicolumn{3}{c|}{\textbf{MoA}} "
                       r"& \multicolumn{3}{c}{\textbf{GPTSwarm}} \\")
    lines_latex.append(r"\textbf{Dataset} & Acc & Lat & SLO "
                       r"& Acc & Lat & SLO "
                       r"& Acc & Lat & SLO "
                       r"& Acc & Lat & SLO \\")

    for load_name, rate in LOAD_RATES.items():
        lines_latex.append(r"\midrule")
        lines_latex.append(r"\multicolumn{13}{c}{\textit{" + load_name + r" req/min}} \\")
        lines_latex.append(r"\midrule")

        for ds in DATASETS:
            entries = {m: get_entry(data, m, ds, rate) for m in methods}
            accs = [entries[m]["accuracy"] if entries[m] else None for m in methods]
            lats = [entries[m]["latency"] if entries[m] else None for m in methods]
            slos = [entries[m]["slo_100"] if entries[m] else None for m in methods]

            best_acc = max((v for v in accs if v is not None), default=None)
            best_lat = min((v for v in lats if v is not None), default=None)
            best_slo = max((v for v in slos if v is not None), default=None)

            row_parts = [DS_DISPLAY[ds]]
            for m in methods:
                e = entries[m]
                if e is None:
                    row_parts.extend(["--", "--", "--"])
                    continue
                a = f"{e['accuracy']*100:.1f}" if e["accuracy"] is not None else "--"
                l = f"{e['latency']:.1f}" if e["latency"] is not None else "--"
                s = f"{e['slo_100']:.1f}" if e["slo_100"] is not None else "--"
                if e["accuracy"] is not None and best_acc and abs(e["accuracy"] - best_acc) < 0.001:
                    a = r"\textbf{" + a + "}"
                if e["latency"] is not None and best_lat and abs(e["latency"] - best_lat) < 0.5:
                    l = r"\textbf{" + l + "}"
                if e["slo_100"] is not None and best_slo and abs(e["slo_100"] - best_slo) < 0.1:
                    s = r"\textbf{" + s + "}"
                row_parts.extend([a, l, s])
            lines_latex.append(" & ".join(row_parts) + r" \\")

    lines_latex.append(r"\bottomrule")
    lines_latex.append(r"\end{tabular}}")
    lines_latex.append(r"\end{table*}")

    return "\n".join(lines_md), "\n".join(lines_latex)


def generate_inframind_budget_table(data):
    lines = ["\n\n# InfraMind: Performance by Budget Tier (averaged across arrival rates)\n"]
    header = "| Dataset | Budget | Acc (%) | Lat (s) | SLO@100 (%) | SLO@300 (%) |"
    sep = "|---------|--------|--------:|--------:|------------:|------------:|"
    lines.append(header)
    lines.append(sep)

    for ds in DATASETS:
        for bt in ["30.0", "50.0", "100.0", "200.0", "300.0"]:
            e = data["inframind"].get(ds, {}).get("by_budget", {}).get(bt)
            if not e:
                continue
            lines.append(f"| {DS_DISPLAY[ds]} | {bt} | {fmt_acc(e['accuracy'])} | "
                         f"{fmt_lat(e['latency']['mean'])} | "
                         f"{fmt_slo(e['slo_compliance'].get('100'))} | "
                         f"{fmt_slo(e['slo_compliance'].get('300'))} |")
    return "\n".join(lines)


def generate_load_degradation_table(data):
    lines = ["\n\n# Load Degradation: Low (10) → Mid (100) → High (200) req/min\n"]
    lines.append("Latency increase factor from low to high load. "
                 "Baseline accuracy is constant (mean across rates).\n")

    methods = ["inframind", "mas", "moa", "gptswarm"]
    method_display = {"inframind": "INFRAMIND", "mas": "MASRouter", "moa": "MoA", "gptswarm": "GPTSwarm"}

    header = "| Method | Dataset | Accuracy (%) | Lat Low (s) | Lat Mid (s) | Lat High (s) | Lat ×increase |"
    sep = "|--------|---------|-------------:|------------:|------------:|--------------:|--------------:|"
    lines.append(header)
    lines.append(sep)

    for m in methods:
        for ds in DATASETS:
            e_low = get_entry(data, m, ds, LOW_RATE)
            e_mid = get_entry(data, m, ds, MID_RATE)
            e_high = get_entry(data, m, ds, HIGH_RATE)
            if not all([e_low, e_high]):
                continue

            # For baselines, accuracy is already mean. For InfraMind use low-load acc.
            if m == "inframind":
                acc = e_low["accuracy"]
            else:
                acc = get_mean_acc(data, m, ds)

            lat_low = e_low["latency"] or 1
            lat_mid = e_mid["latency"] if e_mid else 0
            lat_high = e_high["latency"] or 0
            lat_increase = lat_high / lat_low if lat_low > 0 else 0

            lines.append(f"| {method_display[m]} | {DS_DISPLAY[ds]} | "
                         f"{fmt_acc(acc)} | {lat_low:.1f} | {lat_mid:.1f} | {lat_high:.1f} | "
                         f"{lat_increase:.1f}× |")
    return "\n".join(lines)


def generate_model_strategy_table(data):
    lines = ["\n\n# InfraMind: Model & Strategy Adaptation Patterns\n"]
    lines.append("Shows how InfraMind shifts model/strategy selection under different loads.\n")

    for ds in DATASETS:
        lines.append(f"\n## {DS_DISPLAY[ds]}\n")

        # Strategy distribution
        lines.append("### Strategy Distribution (% of steps)\n")
        lines.append("| Load | Flash | Concise | DeepThink |")
        lines.append("|------|------:|--------:|----------:|")

        for load_name, rate in LOAD_RATES.items():
            e = data["inframind"].get(ds, {}).get("by_arrival_rate", {}).get(str(rate))
            if not e or not e.get("strategy_distribution"):
                continue
            sd = e["strategy_distribution"]
            total = sum(sd.values()) or 1
            lines.append(f"| {load_name} | {sd.get('Flash',0)/total*100:.1f} | "
                         f"{sd.get('Concise',0)/total*100:.1f} | "
                         f"{sd.get('DeepThink',0)/total*100:.1f} |")

        # Model distribution
        lines.append("\n### Model Distribution (% of steps)\n")
        sample = data["inframind"].get(ds, {}).get("by_arrival_rate", {})
        if not sample:
            continue
        first_entry = next(iter(sample.values()))
        model_names = sorted(first_entry.get("model_distribution", {}).keys())
        short_names = {m: m.split("/")[-1][:20] for m in model_names}

        lines.append("| Load |" + " | ".join(short_names[m] for m in model_names) + " |")
        lines.append("|------|" + " | ".join("----:" for _ in model_names) + " |")

        for load_name, rate in LOAD_RATES.items():
            e = data["inframind"].get(ds, {}).get("by_arrival_rate", {}).get(str(rate))
            if not e:
                continue
            md = e.get("model_distribution", {})
            total = sum(md.values()) or 1
            vals = [f"{md.get(m, 0)/total*100:.1f}" for m in model_names]
            lines.append(f"| {load_name} | " + " | ".join(vals) + " |")

    return "\n".join(lines)


def generate_paper_table(data):
    """
    Clean paper table: datasets as columns, methods as rows.
    Two sections: Low Load and High Load, each with Acc + Lat per dataset.
    Baselines show mean accuracy (same in both sections). InfraMind shows per-rate accuracy.
    """
    methods = [
        ("inframind", r"\textsc{InfraMind} (Ours)", "InfraMind"),
        ("mas", r"MASRouter", "MASRouter"),
        ("gptswarm", r"GPTSwarm", "GPTSwarm"),
        ("moa", r"MoA", "MoA"),
    ]

    # Collect all data: {(method, dataset, rate) -> {acc, lat}}
    all_data = {}
    for m_key, _, _ in methods:
        for ds in DATASETS:
            for rate in [LOW_RATE, HIGH_RATE]:
                e = get_entry(data, m_key, ds, rate)
                if not e:
                    continue
                if m_key == "inframind":
                    acc = e["accuracy"]
                else:
                    acc = get_mean_acc(data, m_key, ds)
                all_data[(m_key, ds, rate)] = {
                    "acc": acc, "lat": e["latency"] or 0
                }

    # Find best per dataset per rate
    best = {}
    for ds in DATASETS:
        for rate in [LOW_RATE, HIGH_RATE]:
            accs = [all_data[(m, ds, rate)]["acc"] for m, _, _ in methods if (m, ds, rate) in all_data and all_data[(m, ds, rate)]["acc"]]
            lats = [all_data[(m, ds, rate)]["lat"] for m, _, _ in methods if (m, ds, rate) in all_data]
            if accs and lats:
                best[(ds, rate)] = {"acc": max(accs), "lat": min(lats)}

    def bf(val, best_val, fmt, higher_better=True):
        s = fmt(val)
        if val is None or best_val is None:
            return s
        is_best = abs(val - best_val) < (0.001 if higher_better else 0.5)
        return r"\textbf{" + s + "}" if is_best else s

    # --- LaTeX ---
    latex = []
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Accuracy (\%) and mean latency (s) under low and high load. "
                 r"Baseline accuracy is load-independent (mean across rates); only latency degrades. "
                 r"\textbf{Bold} = best in column. "
                 r"\sys{} trades a small accuracy margin for dramatically lower latency that remains stable under load.}")
    latex.append(r"\label{tab:main_results}")
    latex.append(r"\small")
    latex.append(r"\begin{tabular}{l cc cc cc cc cc}")
    latex.append(r"\toprule")
    # Dataset headers
    ds_headers = " & ".join(r"\multicolumn{2}{c}{\textbf{" + DS_DISPLAY[ds] + r"}}" for ds in DATASETS)
    latex.append(r"& " + ds_headers + r" \\")
    # cmidrule
    cmidrules = " ".join(r"\cmidrule(lr){" + f"{2+i*2}-{3+i*2}" + r"}" for i in range(5))
    latex.append(cmidrules)
    # Sub-headers
    sub = " & ".join(["Acc & Lat"] * 5)
    latex.append(r"\textbf{Method} & " + sub + r" \\")
    latex.append(r"\midrule")

    for rate, rate_label, color in [(LOW_RATE, "Low Load (10 req/min)", "green"),
                                     (HIGH_RATE, "High Load (200 req/min)", "red")]:
        latex.append(r"\multicolumn{11}{c}{\textit{" + rate_label + r"}} \\")
        latex.append(r"\midrule")
        for m_key, m_display, _ in methods:
            cols = []
            for ds in DATASETS:
                v = all_data.get((m_key, ds, rate))
                b = best.get((ds, rate), {})
                if v:
                    a = bf(v["acc"], b.get("acc"), lambda x: f"{x*100:.1f}", True)
                    l = bf(v["lat"], b.get("lat"), lambda x: f"{x:.0f}", False)
                    cols.append(f"{a} & {l}")
                else:
                    cols.append("-- & --")
            latex.append(f"{m_display} & " + " & ".join(cols) + r" \\")
        latex.append(r"\midrule")

    # Replace last \midrule with \bottomrule
    latex[-1] = r"\bottomrule"
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table*}")

    # --- Markdown ---
    md = []
    md.append("\n\n# Paper Table: Main Results\n")
    for rate, rate_label in [(LOW_RATE, "Low Load (10 req/min)"), (HIGH_RATE, "High Load (200 req/min)")]:
        md.append(f"\n## {rate_label}\n")
        header = "| Method | " + " | ".join(f"{DS_DISPLAY[ds]} Acc | {DS_DISPLAY[ds]} Lat" for ds in DATASETS) + " |"
        sep = "|--------|" + "|".join(["----:|----:"] * 5) + "|"
        md.append(header)
        md.append(sep)
        for m_key, _, m_clean in methods:
            row = f"| {m_clean} |"
            for ds in DATASETS:
                v = all_data.get((m_key, ds, rate))
                if v:
                    row += f" {v['acc']*100:.1f} | {v['lat']:.0f} |"
                else:
                    row += " -- | -- |"
            md.append(row)

    return "\n".join(latex), "\n".join(md)


def main():
    data = load_data()

    main_md, main_latex = generate_main_table(data)
    budget_table = generate_inframind_budget_table(data)
    degradation_table = generate_load_degradation_table(data)
    model_strategy_table = generate_model_strategy_table(data)
    paper_latex, paper_md = generate_paper_table(data)

    md_path = OUTPUT_DIR / "tables.md"
    with open(md_path, "w") as f:
        f.write(main_md + paper_md + budget_table + degradation_table + model_strategy_table)
    print(f"Written: {md_path}")

    latex_path = OUTPUT_DIR / "tables.tex"
    with open(latex_path, "w") as f:
        f.write("% ─── Compact paper table ───\n")
        f.write(paper_latex)
        f.write("\n\n% ─── Full wide table (supplementary) ───\n")
        f.write(main_latex)
    print(f"Written: {latex_path}")


if __name__ == "__main__":
    main()
