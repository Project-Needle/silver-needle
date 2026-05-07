"""
Generate a comparison report from two pre-existing result JSON files
produced by accuracy_probe.py — without making any new API calls.

Usage:
  python compare_results.py results_a.json results_b.json
  python compare_results.py results_a.json results_b.json --output-dir reports/
  python compare_results.py results_a.json results_b.json --output report.md
"""

import json
import argparse
from datetime import datetime
from pathlib import Path

from accuracy_probe import generate_report_comparison


def load_results(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    required = {"name", "model", "n_questions", "correct_count", "accuracy",
                "avg_response_length", "median_response_length", "avg_latency",
                "per_type_accuracy", "responses"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"{path.name} is missing fields: {missing}")
    return data


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare two accuracy_probe result JSONs and produce a Markdown report"
    )
    p.add_argument("results_a", help="JSON file for vendor A (reference)")
    p.add_argument("results_b", help="JSON file for vendor B (audit target)")
    p.add_argument("--output-dir", default=None,
                   help="Directory for the output report (default: same dir as results_a)")
    p.add_argument("--output", default=None,
                   help="Explicit output path for the report (overrides --output-dir)")
    return p.parse_args()


def main():
    args = parse_args()

    path_a = Path(args.results_a)
    path_b = Path(args.results_b)

    for p in (path_a, path_b):
        if not p.is_file():
            print(f"Error: file not found: {p}")
            return

    stats_a = load_results(path_a)
    stats_b = load_results(path_b)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = generate_report_comparison(stats_a, stats_b, timestamp)

    if args.output:
        report_path = Path(args.output)
    else:
        out_dir = Path(args.output_dir) if args.output_dir else path_a.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / f"comparison_report_{timestamp}.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report saved to {report_path}")
    print(f"  Vendor A ({stats_a['name']}): {stats_a['accuracy']:.1%}")
    print(f"  Vendor B ({stats_b['name']}): {stats_b['accuracy']:.1%}")
    delta = stats_b["accuracy"] - stats_a["accuracy"]
    print(f"  Delta (B−A): {delta:+.1%}")


if __name__ == "__main__":
    main()
