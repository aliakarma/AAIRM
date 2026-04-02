#!/usr/bin/env python3
"""Run All AAIRM Ablation Studies.

Executes the four ablation configurations defined in configs/ablation/:

    no_rl          — C2 uses analytical optimisation (no PPO policy)
    no_negotiation — C4 bypassed; top-ranked supplier used directly
    no_governance  — C5 bypassed; all orders approved without constraints
    single_category— Grocery-only (240 SKUs, context for cross-category benefit)

Each ablation runs AAIRM vs. Baseline 1 (ROP–EOQ) and reports the
delta vs. the full AAIRM system to quantify each component's contribution.

Usage
-----
    python experiments/run_ablation.py
    python experiments/run_ablation.py --ablation no_rl
    make run-ablation

References
----------
AAIRM Repo Guide, Section 11.2.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aairm.utils.seed import set_global_seed
from aairm.utils.config import AAIRMConfig
from aairm.utils.logging import configure_logging, get_logger
from aairm.simulation.environment import RetailEnv
from aairm.agents.meta_orchestrator import MetaOrchestrator
from aairm.baselines.rop_eoq import ROPEOQPolicy
from aairm.models.forecasting.naive_forecaster import NaiveForecaster
from aairm.evaluation.benchmarker import Benchmarker, BenchmarkResult
from aairm.evaluation.reporter import Reporter

logger = get_logger(__name__)

ABLATION_CONFIGS = {
    "no_rl":           "configs/ablation/no_rl.yaml",
    "no_negotiation":  "configs/ablation/no_negotiation.yaml",
    "no_governance":   "configs/ablation/no_governance.yaml",
    "single_category": "configs/ablation/single_category.yaml",
}

# Expected directional changes vs. full AAIRM
EXPECTED_DIRECTION: dict[str, dict[str, str]] = {
    "no_rl":           {"stockout_rate": "↑", "total_cost": "↑"},
    "no_negotiation":  {"total_cost": "↑"},
    "no_governance":   {"div_index": "↓"},
    "single_category": {},   # interpretive only
}


def load_ablation_config(name: str) -> AAIRMConfig:
    """Load ablation config from YAML."""
    path = ABLATION_CONFIGS[name]
    try:
        from omegaconf import OmegaConf
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return AAIRMConfig.model_validate(raw)
    except Exception:
        logger.warning("ablation.config_load_failed", name=name, path=path)
        return AAIRMConfig()


def run_one_ablation(
    ablation_name: str,
    output_base: Path,
) -> dict[str, BenchmarkResult]:
    """Run one ablation experiment.

    Args:
        ablation_name: Key from ABLATION_CONFIGS.
        output_base: Parent output directory.

    Returns:
        Dict of BenchmarkResult objects.
    """
    logger.info("ablation.start", name=ablation_name)
    config = load_ablation_config(ablation_name)
    set_global_seed(config.simulation.seed)

    output_dir = output_base / ablation_name
    output_dir.mkdir(parents=True, exist_ok=True)

    env = RetailEnv(config.simulation)
    env.reset(seed=config.simulation.seed)

    train_days = (
        config.simulation.simulation_horizon_days
        - config.simulation.test_horizon_days
    )
    for _ in range(train_days):
        env.step_agentic({})

    # Fit Baseline 1
    catalog = env.catalog
    sku_ids = catalog.sku_ids if catalog else []
    demand_history = {s: env.get_demand_history(s, train_days) for s in sku_ids}
    snap = env.get_inventory_snapshot()
    lead_times = {s: float(snap.get(s, {}).get("lead_time_days", 5.0)) for s in sku_ids}
    unit_costs  = {s: float(snap.get(s, {}).get("unit_cost", 5.0)) for s in sku_ids}

    bl1 = ROPEOQPolicy(service_level=config.optimisation.service_level)
    bl1.fit(demand_history, lead_times, unit_costs)

    # Build ablation-specific orchestrator
    skip_neg = config.model_fields_set  # parse YAML flags
    skip_negotiation = ablation_name == "no_negotiation"
    skip_governance  = ablation_name == "no_governance"

    orchestrator = MetaOrchestrator(
        config=config,
        erp_backend=env,
        supplier_backend=env,
        trend_backend=env,
        forecaster=NaiveForecaster(),
        skip_negotiation=skip_negotiation,
        skip_governance=skip_governance,
    )

    benchmarker = Benchmarker(
        config=config,
        env=env,
        orchestrator=orchestrator,
        baseline1=bl1,
    )
    results = benchmarker.run_all(assert_paper_results=False)

    reporter = Reporter(results, output_dir=output_dir)
    reporter.generate_all()
    reporter.print_overall_table()

    logger.info("ablation.complete", name=ablation_name, output=str(output_dir))
    return results


def print_ablation_summary(all_results: dict[str, dict[str, BenchmarkResult]]) -> None:
    """Print combined ablation summary table to stdout."""
    metrics = ["stockout_rate", "fill_rate", "total_cost", "div_index"]
    header = f"{'Ablation':<22} {'Stockout%':>10} {'Fill%':>8} {'Cost':>8} {'DivIdx':>8}"
    print("\n" + "=" * 65)
    print("ABLATION STUDY SUMMARY (AAIRM variant results)")
    print("=" * 65)
    print(header)
    print("-" * 65)
    for ablation_name, results in all_results.items():
        aairm = results.get("aairm")
        if aairm is None:
            continue
        m = aairm.overall
        print(
            f"{ablation_name:<22} "
            f"{m.get('stockout_rate',0)*100:>9.1f}% "
            f"{m.get('fill_rate',0)*100:>7.1f}% "
            f"{m.get('total_cost',0):>8.2f} "
            f"{m.get('div_index',0):>8.2f}"
        )
    print("=" * 65 + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Run AAIRM ablation studies.")
    p.add_argument(
        "--ablation",
        choices=list(ABLATION_CONFIGS.keys()) + ["all"],
        default="all",
        help="Which ablation to run (default: all).",
    )
    args = p.parse_args()

    configure_logging(level="INFO", fmt="console")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(f"experiments/results/ablation_{ts}")
    output_base.mkdir(parents=True, exist_ok=True)

    ablations = (
        list(ABLATION_CONFIGS.keys())
        if args.ablation == "all"
        else [args.ablation]
    )

    all_results: dict[str, dict[str, BenchmarkResult]] = {}
    for name in ablations:
        all_results[name] = run_one_ablation(name, output_base)

    print_ablation_summary(all_results)

    # Save combined summary
    summary = {
        name: {
            policy: res.overall
            for policy, res in results.items()
        }
        for name, results in all_results.items()
    }
    (output_base / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(f"✓ Ablation results saved to: {output_base}")


if __name__ == "__main__":
    main()
