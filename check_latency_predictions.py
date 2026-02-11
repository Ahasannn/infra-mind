#!/usr/bin/env python3
"""Check latency predictor outputs for different models."""

import pandas as pd
import sys

csv_path = "logs/paper_results/math/inframind_math_test_1000_poisson_part2.csv"

print(f"Loading CSV: {csv_path}")
df = pd.read_csv(csv_path)

# Filter to role_step records only (these have model selection data)
df_steps = df[df['record_type'] == 'role_step'].copy()

print(f"\nTotal role_step records: {len(df_steps)}")

# Check columns with system metrics
metric_cols = [col for col in df.columns if 'llm_' in col or 'observed' in col]
print(f"\nSystem metric columns: {metric_cols}")

# Group by model and analyze latency metrics
if 'model_name' in df_steps.columns:
    print("\n=== Latency Metrics by Model ===")

    # For each model, show average observed latencies
    for model in df_steps['model_name'].dropna().unique():
        model_df = df_steps[df_steps['model_name'] == model]
        print(f"\n{model}:")
        print(f"  Count: {len(model_df)}")

        if 'observed_ttft' in model_df.columns:
            ttft = model_df['observed_ttft'].dropna()
            if len(ttft) > 0:
                print(f"  TTFT: mean={ttft.mean():.3f}s, std={ttft.std():.3f}s, min={ttft.min():.3f}s, max={ttft.max():.3f}s")

        if 'observed_tpot' in model_df.columns:
            tpot = model_df['observed_tpot'].dropna()
            if len(tpot) > 0:
                print(f"  TPOT: mean={tpot.mean():.4f}s, std={tpot.std():.4f}s")

        if 'latency_seconds' in model_df.columns:
            lat = model_df['latency_seconds'].dropna()
            if len(lat) > 0:
                print(f"  Total latency: mean={lat.mean():.3f}s, std={lat.std():.3f}s")

        # Check queue metrics at time of selection
        if 'llm_running' in model_df.columns:
            running = model_df['llm_running'].dropna()
            if len(running) > 0:
                print(f"  Queue running: mean={running.mean():.1f}, max={running.max():.0f}")

        if 'llm_waiting' in model_df.columns:
            waiting = model_df['llm_waiting'].dropna()
            if len(waiting) > 0:
                print(f"  Queue waiting: mean={waiting.mean():.1f}, max={waiting.max():.0f}")

# Check if there are episodes where DeepSeek was available but not selected
print("\n\n=== Checking Episode-Level Model Usage ===")
episode_df = df[df['record_type'] == 'episode'].copy()
print(f"Total episodes: {len(episode_df)}")

# Sample a few episodes to see what models were considered
print("\n=== Sample Episode Model Selections ===")
sample_episodes = episode_df.head(10)
for idx, row in sample_episodes.iterrows():
    episode_idx = row['episode_index']
    # Get all steps for this episode
    ep_steps = df_steps[df_steps['episode_index'] == episode_idx]
    models_used = ep_steps['model_name'].dropna().unique()
    print(f"Episode {episode_idx}: {list(models_used)}")
