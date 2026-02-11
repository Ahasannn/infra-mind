#!/usr/bin/env python3
"""
Sample actual model outputs to show how they ignore strategy prompts
"""

import pandas as pd

csv_path = "logs/generate_data_for_latency_length_predictor/baseline_mas_inference_math_train_519_poisson.csv"
df = pd.read_csv(csv_path)

step_df = df[df['record_type'] == 'step'].copy()
strategy_df = step_df[step_df['strategy_name'].notna()].copy()

print("=" * 100)
print("SAMPLE ACTUAL OUTPUTS FROM MODELS")
print("=" * 100)
print()

strategies = ['Flash', 'Concise', 'DeepThink']

for strategy in strategies:
    print(f"\n{'=' * 100}")
    print(f"STRATEGY: {strategy}")
    print("=" * 100)

    if strategy == 'Flash':
        print("INSTRUCTION: 'Respond in one or two sentences' (expected: ~20-50 tokens)")
    elif strategy == 'Concise':
        print("INSTRUCTION: '2-3 bullet points before answer' (expected: ~50-100 tokens)")
    else:
        print("INSTRUCTION: 'Step by step with plan' (expected: ~100-300 tokens)")

    print()

    # Get samples at different token ranges
    strat_data = strategy_df[strategy_df['strategy_name'] == strategy].copy()

    # Sample from low tokens (following instructions)
    low_tokens = strat_data[strat_data['completion_tokens'] <= 50].head(1)
    if len(low_tokens) > 0:
        row = low_tokens.iloc[0]
        print(f"✅ GOOD EXAMPLE (Following Instructions): {row['completion_tokens']:.0f} tokens")
        print(f"   Role: {row['role_name']}")
        print(f"   LLM: {row['llm_name'].split('/')[-1]}")
        print(f"   Latency: {row['latency_seconds']:.2f}s")
        print()

    # Sample from medium tokens
    med_tokens = strat_data[
        (strat_data['completion_tokens'] > 200) &
        (strat_data['completion_tokens'] <= 400)
    ].head(1)
    if len(med_tokens) > 0:
        row = med_tokens.iloc[0]
        print(f"⚠️  MEDIUM VIOLATION: {row['completion_tokens']:.0f} tokens (should be less!)")
        print(f"   Role: {row['role_name']}")
        print(f"   LLM: {row['llm_name'].split('/')[-1]}")
        print(f"   Latency: {row['latency_seconds']:.2f}s")
        print()

    # Sample from high tokens (ignoring instructions)
    high_tokens = strat_data[strat_data['completion_tokens'] > 500].head(1)
    if len(high_tokens) > 0:
        row = high_tokens.iloc[0]
        print(f"❌ SEVERE VIOLATION: {row['completion_tokens']:.0f} tokens (WAY over limit!)")
        print(f"   Role: {row['role_name']}")
        print(f"   LLM: {row['llm_name'].split('/')[-1]}")
        print(f"   Latency: {row['latency_seconds']:.2f}s")
        print()

    # Statistics
    print(f"STATISTICS for {strategy}:")
    print(f"  Mean tokens: {strat_data['completion_tokens'].mean():.1f}")
    print(f"  Median tokens: {strat_data['completion_tokens'].median():.1f}")
    print(f"  % following instructions (<= expected range):")
    if strategy == 'Flash':
        pct = (strat_data['completion_tokens'] <= 50).sum() / len(strat_data) * 100
        print(f"    {pct:.1f}% generated <= 50 tokens")
    elif strategy == 'Concise':
        pct = (strat_data['completion_tokens'] <= 100).sum() / len(strat_data) * 100
        print(f"    {pct:.1f}% generated <= 100 tokens")
    else:
        pct = (strat_data['completion_tokens'] <= 300).sum() / len(strat_data) * 100
        print(f"    {pct:.1f}% generated <= 300 tokens")

    print()

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
print()
print("The models are largely IGNORING the strategy instructions and generating")
print("similar-length outputs regardless of whether they're told to be brief (Flash)")
print("or detailed (DeepThink).")
print()
print("This explains why Concise and DeepThink have similar latencies - they generate")
print("similar token counts (~380 vs ~410 tokens).")
print()
print("=" * 100)
