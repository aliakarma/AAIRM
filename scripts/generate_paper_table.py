#!/usr/bin/env python3
"""Generate formatted markdown table for paper Table 2 from experiment results.

Reads summary.json from the results directory, computes statistics across seeds,
and outputs a markdown table matching the paper format.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np


def load_summary(results_dir: Path) -> Dict[str, Any]:
    """Load summary.json from results directory."""
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found in {results_dir}")
    
    with open(summary_path, 'r') as f:
        return json.load(f)


def compute_stats(data: list) -> str:
    """Compute mean ± std formatted string."""
    if not data:
        return "N/A"
    mean = np.mean(data)
    std = np.std(data)
    return ".3f"


def format_table(results: Dict[str, Any]) -> str:
    """Format results into markdown table."""
    policies = ["AAIRM", "Baseline1", "Baseline2"]
    
    # Extract metrics
    metrics = {}
    for policy in policies:
        if policy not in results:
            continue
        overall = results[policy].get("overall", {})
        metrics[policy] = {
            "Stockout%": overall.get("stockout_rate", 0) * 100,
            "FillRate%": (1 - overall.get("stockout_rate", 0)) * 100,
            "TotalCost": overall.get("total_cost", 0),
            "AvgInv": overall.get("avg_inventory", 0),
            "Spoil%": overall.get("spoilage_rate", 0) * 100,
            "DivIdx": overall.get("diversification_index", 0),
        }
    
    # For AAIRM, collect per-seed for std
    aairm_stockouts = []
    if "AAIRM" in results and "per_seed" in results["AAIRM"]:
        for seed_data in results["AAIRM"]["per_seed"].values():
            aairm_stockouts.append(seed_data.get("stockout_rate", 0) * 100)
    
    std_stockout = np.std(aairm_stockouts) if aairm_stockouts else 0
    
    # Build table
    table = "| Policy | Stockout% | FillRate% | TotalCost | AvgInv | Spoil% | DivIdx |\n"
    table += "|--------|----------|-----------|-----------|--------|--------|--------|\n"
    
    for policy in policies:
        if policy not in metrics:
            continue
        m = metrics[policy]
        row = f"| {policy} | {m['Stockout%']:.1f} | {m['FillRate%']:.1f} | {m['TotalCost']:.3f} | {m['AvgInv']:.1f} | {m['Spoil%']:.1f} | {m['DivIdx']:.2f} |\n"
        table += row
    
    # Footnote
    table += "\n*Results over 10 seeds, 200 episodes, 100 SKUs, 45-day warm-up, real retail data*\n"
    
    return table


def main():
    parser = argparse.ArgumentParser(description="Generate paper table from results")
    parser.add_argument("--results-dir", required=True, help="Results directory path")
    parser.add_argument("--output", required=True, help="Output markdown file path")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    
    summary = load_summary(results_dir)
    table = format_table(summary)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(table)
    
    print(f"Table written to {output_path}")


if __name__ == "__main__":
    main()