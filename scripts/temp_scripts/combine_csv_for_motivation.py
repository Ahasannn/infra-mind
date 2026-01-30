#!/usr/bin/env python3
"""
Combine multiple CSV files into one output file.
Edit INPUT_CSVS and OUTPUT_CSV before running.
"""

from pathlib import Path

import pandas as pd

# Paths relative to the project root.
INPUT_CSVS = [
    "logs/motivation_plot_generator_data/baseline_motivation_sweep_with_slack_10.csv",
    "logs/motivation_plot_generator_data/baseline_motivation_sweep_with_slack_100.csv",
    "logs/motivation_plot_generator_data/baseline_motivation_sweep_with_slack_400.csv"
]
OUTPUT_CSV = "logs/motivation_plot_generator_data/baseline_motivation_sweep_with_slack_10_100_400.csv"

def main():
    project_root = Path(__file__).parent.parent.parent
    input_paths = [project_root / Path(p) for p in INPUT_CSVS]
    output_path = project_root / Path(OUTPUT_CSV)

    print("Input CSVs:")
    for path in input_paths:
        print(f"  - {path}")
    print(f"Output CSV: {output_path}")

    combined_frames = []
    for csv_path in input_paths:
        df = pd.read_csv(csv_path)
        print(f"{csv_path}: {len(df)} rows")
        combined_frames.append(df)

    combined_df = pd.concat(combined_frames, ignore_index=True)
    print(f"Combined rows: {len(combined_df)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    print(f"Saved combined CSV to: {output_path}")

    if "arrival_rate" in combined_df.columns:
        rates = sorted(combined_df["arrival_rate"].unique())
        print(f"Arrival rates: {rates}")


if __name__ == "__main__":
    main()
