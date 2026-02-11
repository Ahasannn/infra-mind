#!/usr/bin/env python3
"""
Deep dive into why DeepThink and Concise have similar latencies
"""

import pandas as pd
import numpy as np

# Read the CSV
csv_path = "logs/generate_data_for_latency_length_predictor/baseline_mas_inference_math_train_519_poisson.csv"
df = pd.read_csv(csv_path)

print("=" * 80)
print("PROMPT STRATEGY ANALYSIS - WHY ARE THEY SIMILAR?")
print("=" * 80)
print()

# Analyze step-level data
step_df = df[df['record_type'] == 'step'].copy()
strategy_df = step_df[step_df['strategy_name'].notna()].copy()

strategies = ['Flash', 'Concise', 'DeepThink']

print("STRATEGY PROMPTS:")
print("-" * 80)
print("Flash:     'Answer quickly with minimal steps. 1-2 sentences.'")
print("Concise:   'Short but careful. 2-3 bullet points before answer.'")
print("DeepThink: 'Step by step with brief plan. Explicit assumptions.'")
print()
print("=" * 80)

# Compare token generation across strategies
print("TOKEN GENERATION BY STRATEGY (at 200 req/min)")
print("=" * 80)
print()

for strategy in strategies:
    strat_data = strategy_df[
        (strategy_df['strategy_name'] == strategy) &
        (strategy_df['arrival_rate'] == 200.0) &
        (strategy_df['completion_tokens'].notna())
    ]

    if len(strat_data) == 0:
        continue

    print(f"{strategy}:")
    print(f"  Count:              {len(strat_data):,}")
    print(f"  Completion Tokens:")
    print(f"    Mean:             {strat_data['completion_tokens'].mean():.1f} tokens")
    print(f"    Median:           {strat_data['completion_tokens'].median():.1f} tokens")
    print(f"    Min/Max:          {strat_data['completion_tokens'].min():.0f} / {strat_data['completion_tokens'].max():.0f} tokens")
    print(f"  Latency:")
    print(f"    Mean:             {strat_data['latency_seconds'].mean():.3f} sec")
    print(f"    Median:           {strat_data['latency_seconds'].median():.3f} sec")
    print(f"  Time per token:     {(strat_data['latency_seconds'] / strat_data['completion_tokens']).mean():.4f} sec/token")
    print()

print("=" * 80)
print("TOKEN GENERATION BY STRATEGY (ALL ARRIVAL RATES)")
print("=" * 80)
print()

arrival_rates = [10.0, 50.0, 100.0, 200.0]

for rate in arrival_rates:
    print(f"Arrival Rate: {rate} req/min")
    print("-" * 40)

    for strategy in strategies:
        strat_data = strategy_df[
            (strategy_df['strategy_name'] == strategy) &
            (strategy_df['arrival_rate'] == rate) &
            (strategy_df['completion_tokens'].notna())
        ]

        if len(strat_data) == 0:
            continue

        avg_tokens = strat_data['completion_tokens'].mean()
        avg_latency = strat_data['latency_seconds'].mean()

        print(f"  {strategy:10s}: {avg_tokens:6.1f} tokens, {avg_latency:6.3f}s latency")

    print()

# Check if models are actually following the strategy instructions
print("=" * 80)
print("TOKEN OUTPUT DISTRIBUTION BY STRATEGY (200 req/min)")
print("=" * 80)
print()

for strategy in strategies:
    strat_data = strategy_df[
        (strategy_df['strategy_name'] == strategy) &
        (strategy_df['arrival_rate'] == 200.0) &
        (strategy_df['completion_tokens'].notna())
    ]['completion_tokens']

    if len(strat_data) == 0:
        continue

    print(f"{strategy}:")
    print(f"  0-50 tokens:     {(strat_data <= 50).sum():>4} ({(strat_data <= 50).sum() / len(strat_data) * 100:>5.1f}%)")
    print(f"  51-100 tokens:   {((strat_data > 50) & (strat_data <= 100)).sum():>4} ({((strat_data > 50) & (strat_data <= 100)).sum() / len(strat_data) * 100:>5.1f}%)")
    print(f"  101-200 tokens:  {((strat_data > 100) & (strat_data <= 200)).sum():>4} ({((strat_data > 100) & (strat_data <= 200)).sum() / len(strat_data) * 100:>5.1f}%)")
    print(f"  201-400 tokens:  {((strat_data > 200) & (strat_data <= 400)).sum():>4} ({((strat_data > 200) & (strat_data <= 400)).sum() / len(strat_data) * 100:>5.1f}%)")
    print(f"  401+ tokens:     {(strat_data > 400).sum():>4} ({(strat_data > 400).sum() / len(strat_data) * 100:>5.1f}%)")
    print()

print("=" * 80)
print("WORKFLOW LATENCY DISTRIBUTION (HISTOGRAM)")
print("=" * 80)
print()

episode_df = df[df['record_type'] == 'episode'].copy()

# Define latency buckets
buckets = [
    (0, 5, "0-5s"),
    (5, 10, "5-10s"),
    (10, 20, "10-20s"),
    (20, 30, "20-30s"),
    (30, 40, "30-40s"),
    (40, 60, "40-60s"),
    (60, 90, "60-90s"),
    (90, 120, "90-120s"),
    (120, 180, "120-180s"),
    (180, float('inf'), "180s+"),
]

for rate in arrival_rates:
    rate_data = episode_df[episode_df['arrival_rate'] == rate]['workflow_latency_seconds']

    print(f"Arrival Rate: {rate} req/min (n={len(rate_data)})")
    print("-" * 60)

    for low, high, label in buckets:
        if high == float('inf'):
            count = (rate_data >= low).sum()
        else:
            count = ((rate_data >= low) & (rate_data < high)).sum()

        pct = count / len(rate_data) * 100

        # Create a visual bar
        bar_length = int(pct / 2)  # Scale to fit terminal
        bar = "█" * bar_length

        print(f"  {label:10s}: {count:>4} ({pct:>5.1f}%) {bar}")

    print(f"  Mean: {rate_data.mean():.2f}s, Median: {rate_data.median():.2f}s, Std: {rate_data.std():.2f}s")
    print()

print("=" * 80)
print("LATENCY vs TOKENS CORRELATION")
print("=" * 80)
print()

for rate in arrival_rates:
    print(f"Arrival Rate: {rate} req/min")
    print("-" * 40)

    rate_data = strategy_df[
        (strategy_df['arrival_rate'] == rate) &
        (strategy_df['completion_tokens'].notna())
    ].copy()

    correlation = rate_data['completion_tokens'].corr(rate_data['latency_seconds'])
    print(f"  Correlation (tokens vs latency): {correlation:.3f}")

    # Break down by token ranges
    for token_range_tuple in [((0, 100), "0-100"), ((100, 200), "100-200"), ((200, 500), "200-500"), ((500, 10000), "500+")]:
        token_range, label = token_range_tuple
        token_subset = rate_data[
            (rate_data['completion_tokens'] >= token_range[0]) &
            (rate_data['completion_tokens'] < token_range[1])
        ]

        if len(token_subset) > 0:
            avg_tokens = token_subset['completion_tokens'].mean()
            avg_latency = token_subset['latency_seconds'].mean()
            print(f"    {label} tokens: n={len(token_subset):>4}, avg_latency={avg_latency:>6.2f}s")

    print()

print("=" * 80)
print("THE VERDICT: Why are DeepThink and Concise similar?")
print("=" * 80)
print()
print("Checking if models actually respect the prompt strategies...")
print()

# Compare actual output lengths
for rate in [200.0]:  # Focus on high load
    print(f"At {rate} req/min:")
    print("-" * 40)

    for strategy in strategies:
        strat_data = strategy_df[
            (strategy_df['strategy_name'] == strategy) &
            (strategy_df['arrival_rate'] == rate) &
            (strategy_df['completion_tokens'].notna())
        ]

        if len(strat_data) == 0:
            continue

        avg_tokens = strat_data['completion_tokens'].mean()
        median_tokens = strat_data['completion_tokens'].median()

        print(f"  {strategy:10s}: {avg_tokens:6.1f} tokens (median: {median_tokens:.0f})")

    print()
    print("ANALYSIS:")
    print("  If models were following instructions:")
    print("    - Flash should have ~20-50 tokens (1-2 sentences)")
    print("    - Concise should have ~50-100 tokens (2-3 bullet points)")
    print("    - DeepThink should have ~100-300 tokens (step-by-step)")
    print()

print("=" * 80)
