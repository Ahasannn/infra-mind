#!/usr/bin/env python3
"""Analyze InfraMind model selection patterns from CSV."""

import pandas as pd
from collections import Counter
import sys

def analyze_csv(csv_path):
    """Analyze model selection patterns."""
    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"\nTotal rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    # Check if we have model selection columns
    if 'model_name' in df.columns:
        print("\n=== Model Selection Distribution ===")
        model_counts = df['model_name'].value_counts()
        print(model_counts)
        print(f"\nUnique models: {df['model_name'].nunique()}")
        print(f"Models: {df['model_name'].unique().tolist()}")

    # Check topology distribution
    if 'topology' in df.columns:
        print("\n=== Topology Distribution ===")
        topology_counts = df['topology'].value_counts()
        print(topology_counts)
        print(f"\nUnique topologies: {df['topology'].nunique()}")

    # Check strategy distribution
    if 'strategy' in df.columns:
        print("\n=== Strategy Distribution ===")
        strategy_counts = df['strategy'].value_counts()
        print(strategy_counts)

    # Cross-tabulation: topology vs model
    if 'topology' in df.columns and 'model_name' in df.columns:
        print("\n=== Topology × Model Cross-Tab ===")
        crosstab = pd.crosstab(df['topology'], df['model_name'])
        print(crosstab)

    # Check if there's role information
    if 'role' in df.columns:
        print("\n=== Role Distribution ===")
        role_counts = df['role'].value_counts()
        print(role_counts)

        if 'model_name' in df.columns:
            print("\n=== Role × Model Cross-Tab ===")
            role_model_crosstab = pd.crosstab(df['role'], df['model_name'])
            print(role_model_crosstab)

    # Check budget-related columns
    budget_cols = [col for col in df.columns if 'budget' in col.lower() or 'cost' in col.lower()]
    if budget_cols:
        print(f"\n=== Budget/Cost Columns ===")
        print(f"Found columns: {budget_cols}")
        for col in budget_cols:
            print(f"\n{col} statistics:")
            print(df[col].describe())

    # Check latency-related columns
    latency_cols = [col for col in df.columns if 'latency' in col.lower() or 'time' in col.lower()]
    if latency_cols:
        print(f"\n=== Latency/Time Columns ===")
        print(f"Found columns: {latency_cols}")
        for col in latency_cols[:3]:  # Show first 3 only
            print(f"\n{col} statistics:")
            print(df[col].describe())

    # Sample a few rows to see the data structure
    print("\n=== Sample Rows ===")
    print(df.head(10).to_string())

    # Check if all rows have the same model
    if 'model_name' in df.columns:
        unique_models = df['model_name'].unique()
        if len(unique_models) == 1:
            print(f"\n⚠️  WARNING: ALL ROWS USE THE SAME MODEL: {unique_models[0]}")
            print("This is likely a bug in the router selection logic!")

if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "logs/paper_results/math/inframind_math_test_1000_poisson_part2.csv"
    analyze_csv(csv_path)
