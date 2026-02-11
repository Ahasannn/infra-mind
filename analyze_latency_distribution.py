#!/usr/bin/env python3
"""
Analyze latency distributions from predictor training data CSV
"""

import pandas as pd
import numpy as np

# Read the CSV
csv_path = "logs/generate_data_for_latency_length_predictor/baseline_mas_inference_math_train_519_poisson.csv"
df = pd.read_csv(csv_path)

print("=" * 80)
print("LATENCY DISTRIBUTION ANALYSIS")
print("=" * 80)
print()

# Get unique arrival rates
arrival_rates = sorted(df['arrival_rate'].unique())
print(f"Arrival Rates Found: {arrival_rates}")
print()

# Analyze step-level latencies
step_df = df[df['record_type'] == 'step'].copy()

print("=" * 80)
print("STEP-LEVEL LATENCY (latency_seconds)")
print("=" * 80)
print()

for rate in arrival_rates:
    rate_data = step_df[step_df['arrival_rate'] == rate]['latency_seconds']

    print(f"Arrival Rate: {rate} req/min")
    print(f"  Count:       {len(rate_data):,}")
    print(f"  Min:         {rate_data.min():.4f} sec")
    print(f"  Max:         {rate_data.max():.4f} sec")
    print(f"  Mean:        {rate_data.mean():.4f} sec")
    print(f"  Median:      {rate_data.median():.4f} sec")
    print(f"  Std Dev:     {rate_data.std():.4f} sec")
    print(f"  25th %ile:   {rate_data.quantile(0.25):.4f} sec")
    print(f"  75th %ile:   {rate_data.quantile(0.75):.4f} sec")
    print(f"  95th %ile:   {rate_data.quantile(0.95):.4f} sec")
    print(f"  99th %ile:   {rate_data.quantile(0.99):.4f} sec")
    print()

# Analyze TTFT (Time To First Token)
print("=" * 80)
print("TIME TO FIRST TOKEN (observed_ttft)")
print("=" * 80)
print()

ttft_df = step_df[step_df['observed_ttft'].notna()].copy()

for rate in arrival_rates:
    rate_data = ttft_df[ttft_df['arrival_rate'] == rate]['observed_ttft']

    if len(rate_data) == 0:
        print(f"Arrival Rate: {rate} req/min - No TTFT data")
        print()
        continue

    print(f"Arrival Rate: {rate} req/min")
    print(f"  Count:       {len(rate_data):,}")
    print(f"  Min:         {rate_data.min():.4f} sec")
    print(f"  Max:         {rate_data.max():.4f} sec")
    print(f"  Mean:        {rate_data.mean():.4f} sec")
    print(f"  Median:      {rate_data.median():.4f} sec")
    print(f"  Std Dev:     {rate_data.std():.4f} sec")
    print(f"  95th %ile:   {rate_data.quantile(0.95):.4f} sec")
    print()

# Analyze TPOT (Time Per Output Token)
print("=" * 80)
print("TIME PER OUTPUT TOKEN (observed_tpot)")
print("=" * 80)
print()

tpot_df = step_df[step_df['observed_tpot'].notna()].copy()

for rate in arrival_rates:
    rate_data = tpot_df[tpot_df['arrival_rate'] == rate]['observed_tpot']

    if len(rate_data) == 0:
        print(f"Arrival Rate: {rate} req/min - No TPOT data")
        print()
        continue

    print(f"Arrival Rate: {rate} req/min")
    print(f"  Count:       {len(rate_data):,}")
    print(f"  Min:         {rate_data.min():.6f} sec")
    print(f"  Max:         {rate_data.max():.6f} sec")
    print(f"  Mean:        {rate_data.mean():.6f} sec")
    print(f"  Median:      {rate_data.median():.6f} sec")
    print(f"  Std Dev:     {rate_data.std():.6f} sec")
    print()

# Analyze workflow-level latencies
print("=" * 80)
print("WORKFLOW-LEVEL LATENCY (workflow_latency_seconds)")
print("=" * 80)
print()

episode_df = df[df['record_type'] == 'episode'].copy()

for rate in arrival_rates:
    rate_data = episode_df[episode_df['arrival_rate'] == rate]['workflow_latency_seconds']

    print(f"Arrival Rate: {rate} req/min")
    print(f"  Count:       {len(rate_data):,}")
    print(f"  Min:         {rate_data.min():.4f} sec")
    print(f"  Max:         {rate_data.max():.4f} sec")
    print(f"  Mean:        {rate_data.mean():.4f} sec")
    print(f"  Median:      {rate_data.median():.4f} sec")
    print(f"  Std Dev:     {rate_data.std():.4f} sec")
    print(f"  95th %ile:   {rate_data.quantile(0.95):.4f} sec")
    print()

# Strategy breakdown
print("=" * 80)
print("LATENCY BY STRATEGY")
print("=" * 80)
print()

strategy_df = step_df[step_df['strategy_name'].notna()].copy()
strategies = sorted(strategy_df['strategy_name'].unique())

for strategy in strategies:
    print(f"Strategy: {strategy}")
    print("-" * 40)

    strat_data = strategy_df[strategy_df['strategy_name'] == strategy]

    for rate in arrival_rates:
        rate_data = strat_data[strat_data['arrival_rate'] == rate]['latency_seconds']

        if len(rate_data) == 0:
            continue

        print(f"  Arrival Rate {rate} req/min: "
              f"n={len(rate_data):>4}, "
              f"mean={rate_data.mean():>6.3f}s, "
              f"median={rate_data.median():>6.3f}s, "
              f"max={rate_data.max():>6.3f}s")

    print()

# LLM breakdown
print("=" * 80)
print("LATENCY BY LLM MODEL")
print("=" * 80)
print()

llm_df = step_df[step_df['llm_name'].notna()].copy()
llms = sorted(llm_df['llm_name'].unique())

for llm in llms:
    llm_short = llm.split('/')[-1]  # Get just the model name
    print(f"LLM: {llm_short}")
    print("-" * 40)

    llm_data = llm_df[llm_df['llm_name'] == llm]

    for rate in arrival_rates:
        rate_data = llm_data[llm_data['arrival_rate'] == rate]['latency_seconds']

        if len(rate_data) == 0:
            continue

        print(f"  Arrival Rate {rate} req/min: "
              f"n={len(rate_data):>4}, "
              f"mean={rate_data.mean():>6.3f}s, "
              f"median={rate_data.median():>6.3f}s")

    print()

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
