#!/usr/bin/env python3
"""Export Saved Results as LaTeX Tables.

Reads a benchmark_results.json file produced by any experiment script and
regenerates all LaTeX tables and PNG figures.

Usage
-----
    python scripts/export_results_tables.py \\
        --results experiments/results/paper_experiment_*/benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aairm.evaluation.benchmarker import BenchmarkResult
from aairm.evaluation.reporter import Reporter


def main() -> None:
    p = argparse.ArgumentParser(description="Export results to LaTeX tables.")
    p.add_argument("--results", required=True,
                   help="Path to benchmark_results.json file.")
    p.add_argument("--output", default=None,
                   help="Output directory (default: same as results file).")
    args = p.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: {results_path} not found.")
        sys.exit(1)

    raw = json.loads(results_path.read_text())
    results = {
        name: BenchmarkResult(
            policy_name=data["policy_name"],
            overall=data.get("overall", {}),
            per_category=data.get("per_category", {}),
            rl_curve=data.get("rl_curve"),
        )
        for name, data in raw.items()
    }

    output_dir = Path(args.output) if args.output else results_path.parent
    reporter = Reporter(results, output_dir=output_dir)
    reporter.generate_all()
    print(f"✓ Tables and figures exported to: {output_dir}")


if __name__ == "__main__":
    main()
