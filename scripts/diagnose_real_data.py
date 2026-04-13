#!/usr/bin/env python3
"""Diagnose Real-World Dataset Statistics vs Synthetic Parameters.

Loads real-world dataset, computes per-category demand statistics,
compares against synthetic simulator parameters, and outputs mismatch report.

Usage:
    python scripts/diagnose_real_data.py --dataset m5
    python scripts/diagnose_real_data.py --dataset favorita

Outputs:
    experiments/diagnostics/data_mismatch_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from aairm.data.loader import DataLoader
from aairm.utils.config import AAIRMConfig

# Synthetic parameters from data/synthetic/demand_params_seed.json
SYNTHETIC_PARAMS = {
    "base_demand_range": {"min": 20.0, "max": 200.0},
    "demand_noise": {"sigma_range": {"min": 0.05, "max": 0.25}},
}

# Computed synthetic expectations
SYNTHETIC_MEAN = (SYNTHETIC_PARAMS["base_demand_range"]["min"] + SYNTHETIC_PARAMS["base_demand_range"]["max"]) / 2
SYNTHETIC_SIGMA_MEAN = (SYNTHETIC_PARAMS["demand_noise"]["sigma_range"]["min"] + SYNTHETIC_PARAMS["demand_noise"]["sigma_range"]["max"]) / 2
SYNTHETIC_CV = np.sqrt(np.exp(SYNTHETIC_SIGMA_MEAN**2) - 1)


def compute_category_stats(demand_history: dict[str, np.ndarray], sku_catalog: pd.DataFrame, category: str) -> dict:
    """Compute demand statistics for a category."""
    skus = sku_catalog[sku_catalog["category"] == category]["sku_id"].tolist()
    all_demands = []
    autocorr_lag1 = []
    autocorr_lag7 = []

    for sku in skus:
        if sku in demand_history:
            demands = demand_history[sku]
            all_demands.extend(demands)
            series = pd.Series(demands)
            autocorr_lag1.append(series.autocorr(lag=1) if len(series) > 1 else 0)
            autocorr_lag7.append(series.autocorr(lag=7) if len(series) > 7 else 0)

    if not all_demands:
        return {}

    all_demands = np.array(all_demands)
    mean_d = np.mean(all_demands)
    std_d = np.std(all_demands)
    zero_days = np.sum(all_demands == 0)
    cv = std_d / mean_d if mean_d > 0 else 0
    perc95 = np.percentile(all_demands, 95)

    return {
        "mean_daily_demand": float(mean_d),
        "std_daily_demand": float(std_d),
        "min_demand": float(np.min(all_demands)),
        "max_demand": float(np.max(all_demands)),
        "95th_percentile": float(perc95),
        "zero_demand_days": int(zero_days),
        "intermittent_rate": float(zero_days / len(all_demands)),
        "coefficient_of_variation": float(cv),
        "autocorr_lag1": float(np.mean(autocorr_lag1)),
        "autocorr_lag7": float(np.mean(autocorr_lag7)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnose real-world data statistics.")
    p.add_argument("--dataset", choices=["m5", "favorita"], default="m5", help="Dataset to diagnose.")
    args = p.parse_args()

    # Load config and data
    config = AAIRMConfig()
    loader = DataLoader(config, mode=args.dataset)
    data = loader.load()

    demand_history = data["demand_history"]
    sku_catalog = data["sku_catalog"]

    categories = ["grocery", "frozen_food", "apparel", "cosmetics", "dry_fruits"]
    report = {"dataset": args.dataset, "categories": {}, "mismatches": []}

    for cat in categories:
        stats = compute_category_stats(demand_history, sku_catalog, cat)
        if stats:
            report["categories"][cat] = stats

            # Check mismatches
            real_cv = stats["coefficient_of_variation"]
            real_mean = stats["mean_daily_demand"]
            if real_cv > 1.5 * SYNTHETIC_CV:
                report["mismatches"].append(f"{cat}: CV {real_cv:.3f} > 1.5x synthetic {SYNTHETIC_CV:.3f}")
            if real_mean > 2 * SYNTHETIC_MEAN:
                report["mismatches"].append(f"{cat}: mean {real_mean:.1f} > 2x synthetic {SYNTHETIC_MEAN:.1f}")

    # Save report
    output_dir = Path("experiments/diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "data_mismatch_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()</content>
<parameter name="filePath">c:\Users\babur\OneDrive\Dokumenty\GitHub\AAIRM\scripts\diagnose_real_data.py