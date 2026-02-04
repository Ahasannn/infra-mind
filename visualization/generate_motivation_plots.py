#!/usr/bin/env python3
"""
Generate motivation plots from MATH baseline arrival rate sweep data.
Saves plots as PNG files instead of showing interactively.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
})

# Output directory
OUTPUT_DIR = Path(__file__).parent / 'output_plots'
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Output directory: {OUTPUT_DIR}")

# ── Load data ──
DATA_PATH = Path('../logs/motivation_plot_generator_data/baseline_motivation_sweep_math_test_300_poisson.csv')
if not DATA_PATH.exists():
    DATA_PATH = Path('../logs/motivation_plot_generator_data/baseline_motivation_sweep_math_test_temp.csv')

if not DATA_PATH.exists():
    # Try relative to script location
    DATA_PATH = Path(__file__).parent.parent / 'logs/motivation_plot_generator_data/baseline_motivation_sweep_math_test_300_poisson.csv'
    if not DATA_PATH.exists():
        DATA_PATH = Path(__file__).parent.parent / 'logs/motivation_plot_generator_data/baseline_motivation_sweep_math_test_temp.csv'

print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# ── Filter to step-level records (per-LLM-call metrics) ──
step_df = df[df.record_type == 'step'].copy()
ep_df = df[df.record_type == 'episode'].copy()

# ── Numeric coercion ──
for col in ['latency_seconds', 'arrival_rate', 'llm_running', 'llm_waiting',
            'llm_kv_cache_usage', 'observed_ttft', 'observed_tpot',
            'llm_ttft_avg', 'llm_itl_avg', 'llm_e2e_avg', 'llm_queue_avg', 'llm_inference_avg']:
    if col in step_df.columns:
        step_df[col] = pd.to_numeric(step_df[col], errors='coerce')

# ── Queue depth = running + waiting ──
step_df['queue_depth'] = step_df['llm_running'].fillna(0) + step_df['llm_waiting'].fillna(0)

# ── Short model names ──
model_name_map = {
    'Qwen/Qwen2.5-Coder-14B-Instruct': 'Qwen-14B',
    'mistralai/Mistral-Small-24B-Instruct-2501': 'Mistral-24B',
    'meta-llama/Llama-3.2-3B-Instruct': 'Llama-3B',
    'meta-llama/Llama-3.1-8B-Instruct': 'Llama-8B',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B': 'DeepSeek-32B',
}
step_df['model_short'] = step_df['llm_name'].map(model_name_map)
step_df = step_df[step_df['model_short'].notna()].copy()

arrival_rates = sorted(step_df['arrival_rate'].dropna().unique())
model_order = ['Llama-3B', 'Llama-8B', 'Qwen-14B', 'Mistral-24B', 'DeepSeek-32B']
models_in_data = [m for m in model_order if m in step_df['model_short'].unique()]

print(f'Step-level LLM calls:  {len(step_df)}')
print(f'Episode records:       {len(ep_df)}')
print(f'Arrival rates:         {arrival_rates}')
print(f'Models:                {models_in_data}')
print()

# ── Figure 1: Average Queue Depth per Model vs Arrival Rate ──
print("Generating Figure 1: Queue Depth vs Arrival Rate...")
queue_stats = (
    step_df.groupby(['arrival_rate', 'model_short'])['queue_depth']
    .mean()
    .reset_index()
)
queue_pivot = (
    queue_stats.pivot(index='arrival_rate', columns='model_short', values='queue_depth')
    .reindex(columns=models_in_data)
)

colors = sns.color_palette('Set2', n_colors=len(models_in_data))
x = np.arange(len(arrival_rates))
bar_width = 0.8 / max(len(models_in_data), 1)

fig, ax = plt.subplots(figsize=(10, 6))
for i, model in enumerate(models_in_data):
    offset = (i - (len(models_in_data) - 1) / 2) * bar_width
    ax.bar(
        x + offset,
        queue_pivot[model].values,
        width=bar_width,
        label=model,
        color=colors[i],
        edgecolor='black',
        linewidth=0.3,
    )

ax.set_xlabel('Arrival Rate (requests/minute)')
ax.set_ylabel('Avg Queue Depth (running + waiting)')
ax.set_title('Average Queue Depth per Model vs Arrival Rate')
ax.set_xticks(x)
ax.set_xticklabels([str(int(r)) for r in arrival_rates])
ax.legend(title='Model', loc='upper left')
ax.grid(True, axis='y', alpha=0.25)
plt.tight_layout()
output_file = OUTPUT_DIR / 'fig1_queue_depth_vs_arrival_rate.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"  Saved: {output_file}")
plt.close()

# ── Figure 2: Average Latency per Model vs Arrival Rate ──
print("Generating Figure 2: Latency vs Arrival Rate...")
latency_stats = (
    step_df.groupby(['arrival_rate', 'model_short'])['latency_seconds']
    .mean()
    .reset_index()
)
latency_pivot = (
    latency_stats.pivot(index='arrival_rate', columns='model_short', values='latency_seconds')
    .reindex(columns=models_in_data)
)

colors = sns.color_palette('Set2', n_colors=len(models_in_data))
x = np.arange(len(arrival_rates))
bar_width = 0.8 / max(len(models_in_data), 1)

fig, ax = plt.subplots(figsize=(10, 6))
for i, model in enumerate(models_in_data):
    offset = (i - (len(models_in_data) - 1) / 2) * bar_width
    vals = latency_pivot[model].values
    ax.bar(
        x + offset,
        vals,
        width=bar_width,
        label=model,
        color=colors[i],
        edgecolor='black',
        linewidth=0.3,
    )

ax.set_xlabel('Arrival Rate (requests/minute)')
ax.set_ylabel('Avg Latency (seconds)')
ax.set_title('Average Latency per Model vs Arrival Rate')
ax.set_xticks(x)
ax.set_xticklabels([str(int(r)) for r in arrival_rates])
ax.legend(title='Model', loc='upper left')
ax.grid(True, axis='y', alpha=0.25)
plt.tight_layout()
output_file = OUTPUT_DIR / 'fig2_latency_vs_arrival_rate.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"  Saved: {output_file}")
plt.close()

# ── Figure 3+: Average Latency vs Arrival Rate per Role ──
roles = sorted([r for r in step_df['role_name'].dropna().unique() if pd.notna(r)])
print(f'\nGenerating per-role plots for {len(roles)} roles...')

for role in roles:
    print(f"  Processing role: {role}")
    role_df = step_df[step_df['role_name'] == role].copy()

    if len(role_df) == 0:
        print(f'    No data for role: {role}')
        continue

    role_models = sorted([m for m in role_df['model_short'].dropna().unique()])

    if len(role_models) == 0:
        print(f'    No models found for role: {role}')
        continue

    latency_stats = (
        role_df.groupby(['arrival_rate', 'model_short'])['latency_seconds']
        .mean()
        .reset_index()
    )

    latency_pivot = (
        latency_stats.pivot(index='arrival_rate', columns='model_short', values='latency_seconds')
    )

    role_arrival_rates = sorted(latency_pivot.index.dropna().unique())

    colors = sns.color_palette('Set2', n_colors=len(role_models))
    x = np.arange(len(role_arrival_rates))
    bar_width = 0.8 / max(len(role_models), 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model in enumerate(role_models):
        if model not in latency_pivot.columns:
            continue
        offset = (i - (len(role_models) - 1) / 2) * bar_width
        vals = latency_pivot[model].values
        ax.bar(
            x + offset,
            vals,
            width=bar_width,
            label=model,
            color=colors[i % len(colors)],
            edgecolor='black',
            linewidth=0.3,
        )

    ax.set_xlabel('Arrival Rate (requests/minute)', fontsize=14)
    ax.set_ylabel('Avg Latency (seconds)', fontsize=14)
    ax.set_title(f'Latency vs Arrival Rate - {role}', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(r)}' for r in role_arrival_rates])
    ax.legend(title='Model', loc='upper left', fontsize=11)
    ax.grid(True, axis='y', alpha=0.25)
    plt.tight_layout()

    # Sanitize role name for filename
    safe_role_name = role.replace('/', '_').replace(' ', '_')
    output_file = OUTPUT_DIR / f'fig3_{safe_role_name}_latency.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'    Saved: {output_file}')
    plt.close()

print(f"\n✓ All plots generated in: {OUTPUT_DIR}")
print(f"  Total files: {len(list(OUTPUT_DIR.glob('*.png')))}")
