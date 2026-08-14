"""Analyze logged predictions from predict_camera_log.py.

Reads a CSV of predictions and generates summary statistics and plots.

Usage:
    python analyze_predictions.py predictions.csv
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def analyze_csv(csv_path):
    """Parse CSV and print summary statistics."""
    if not Path(csv_path).exists():
        print(f"File not found: {csv_path}")
        return

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No data in CSV")
        return

    # Extract affect names from fieldnames
    affects = []
    for key in rows[0].keys():
        if key.endswith('_level'):
            affect = key.replace('_level', '')
            affects.append(affect)

    print(f"\n=== Prediction Summary ===")
    print(f"Total predictions: {len(rows)}")
    print(f"Affects tracked: {', '.join(affects)}")
    print(f"Time range: {rows[0]['timestamp']} to {rows[-1]['timestamp']}\n")

    # Per-affect statistics
    for affect in affects:
        level_key = f"{affect}_level"
        conf_key = f"{affect}_confidence"

        levels = [row[level_key] for row in rows]
        confs = [float(row[conf_key]) for row in rows]

        level_counts = defaultdict(int)
        for lv in levels:
            level_counts[lv] += 1

        avg_conf = sum(confs) / len(confs)
        min_conf = min(confs)
        max_conf = max(confs)

        print(f"{affect}:")
        print(f"  Average confidence: {avg_conf:.3f}")
        print(f"  Confidence range: {min_conf:.3f} – {max_conf:.3f}")
        print(f"  Level distribution:")
        for lv in ["Very Low", "Low", "High", "Very High"]:
            count = level_counts.get(lv, 0)
            pct = 100 * count / len(levels)
            print(f"    {lv:10} {count:3} ({pct:5.1f}%)")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_predictions.py <csv_file>")
        sys.exit(1)

    analyze_csv(sys.argv[1])
