#!/usr/bin/env python3
"""Check if DeepSeek metrics exist in the CSV."""

import pandas as pd

csv_path = "logs/paper_results/math/inframind_math_test_1000_poisson_part2.csv"
df = pd.read_csv(csv_path)

# Check if there are any system state columns that might have DeepSeek data
print("=== Checking for DeepSeek in system metrics ===\n")

# Get a sample row to see what the system state looks like
sample = df[df['record_type'] == 'role_step'].iloc[0]

# Check running/waiting columns
print("Sample row columns with 'llm_' prefix:")
for col in df.columns:
    if 'llm_' in col:
        print(f"  {col}: {sample[col]}")

# The system state might be encoded in the latency predictions
# Let's check if DeepSeek is being FILTERED OUT due to unavailability

# Check the training script to see if models are being filtered
import sys
sys.path.insert(0, '/home/ah872032.ucf/system-aware-mas')

from MAR.InfraMind.metrics_watcher import model_metrics

print("\n\n=== Current model_metrics state ===")
print(f"Keys: {list(model_metrics.keys())}")
for model_name, metrics in model_metrics.items():
    print(f"\n{model_name}:")
    print(f"  {metrics}")
