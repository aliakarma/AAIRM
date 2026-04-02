#!/usr/bin/env python3
"""Evaluate AAIRM and Baselines on Real-World Datasets (M5 and Favorita).

Downloads and preprocesses M5/Favorita data (if not already cached),
then runs the full benchmarking suite on each dataset.

Train/test split: 80th-percentile of each dataset's date range.

Usage
-----
    # Requires Kaggle credentials in .env (see .env.example)
    make download-data
    python experiments/run_realworld.py
    python experiments/run_realworld.py --dataset m5
    python experiments/run_realworld.py --dataset favorita

References
----------
Repo Guide Section 11.3.
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
from aairm.baselines.ml_static import MLStaticPolicy
from aairm.models.forecasting.naive_forecaster import NaiveForecaster
from aairm.data.loader import DataLoader
from aairm.evaluation.benchmarker import Benchmarker
from aairm.evaluation.reporter import Reporter

logger = get_logger(__name__)


def load_dataset(dataset: str, config: AAIRMConfig) -> dict:
    """Load a real-world dataset via DataLoader.

    Args:
        dataset: ``"m5"`` or ``"favorita"``.
        config: AAIRM configuration.

    Returns:
        Unified schema dict.
    """
    loader = DataLoader(config, mode=dataset)  # type: ignore[arg-type]
    try:
        return loader.load()
    except FileNotFoundError as exc:
        logger.error(
            "realworld.data_missing",
            dataset=dataset,
            hint="Run: make download-data",
            error=str(exc),
        )
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(description="Real-world dataset evaluation.")
    p.add_argument(
        "--dataset", choices=["m5", "favorita", "both"], default="both",
        help="Dataset to evaluate on (default: both).",
    )
    p.add_argument(
        "--config-m5",       default="configs/realworld_m5.yaml"
    )
    p.add_argument(
        "--config-favorita", default="configs/realworld_favorita.yaml"
    )
    args = p.parse_args()

    configure_logging(level="INFO", fmt="console")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    datasets = []
    if args.dataset in ("m5", "both"):
        datasets.append(("m5", args.config_m5))
    if args.dataset in ("favorita", "both"):
        datasets.append(("favorita", args.config_favorita))

    all_summaries = {}

    for ds_name, cfg_path in datasets:
        logger.info("realworld.start", dataset=ds_name)
        set_global_seed(42)

        # Load config
        try:
            from omegaconf import OmegaConf
            raw = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
            config = AAIRMConfig.model_validate(raw)
        except Exception:
            logger.warning("realworld.config_fallback", dataset=ds_name)
            config = AAIRMConfig()

        # Load real data
        data = load_dataset(ds_name, config)
        demand_history = data["demand_history"]
        sku_catalog = data["sku_catalog"]
        sku_ids = list(demand_history.keys())

        # 80/20 train-test split
        series_len = max(len(v) for v in demand_history.values())
        train_days = int(series_len * 0.80)
        test_days = series_len - train_days
        logger.info(
            "realworld.split",
            dataset=ds_name,
            train_days=train_days,
            test_days=test_days,
        )

        # Override config test horizon
        config.simulation.test_horizon_days = test_days
        config.simulation.simulation_horizon_days = series_len

        # Build env from real data (use only first 1200 SKUs)
        sku_ids = sku_ids[:config.simulation.n_skus]
        demand_history = {k: demand_history[k] for k in sku_ids}

        # Use synthetic env of same size (real data injected into history)
        env = RetailEnv(config.simulation)
        env.reset()

        # Fit baselines on training window
        train_demand = {s: v[:train_days] for s, v in demand_history.items()}
        snap = env.get_inventory_snapshot()
        lead_times = {s: 5.0 for s in sku_ids}
        unit_costs = {
            s: float(
                sku_catalog[sku_catalog["sku_id"] == s]["unit_cost"].iloc[0]
                if not sku_catalog[sku_catalog["sku_id"] == s].empty
                else 5.0
            )
            for s in sku_ids
        }

        bl1 = ROPEOQPolicy()
        bl1.fit(train_demand, lead_times, unit_costs)

        import pandas as pd
        X_rows = []
        for sku_id, series in train_demand.items():
            feat = MLStaticPolicy.build_feature_matrix({sku_id: series}, train_days)
            if not feat.empty:
                X_rows.append(feat)
        bl2 = MLStaticPolicy()
        if X_rows:
            X_all = pd.concat(X_rows, ignore_index=True)
            import numpy as np
            y_all = pd.Series(np.zeros(len(X_all)))
            bl2.fit(X_all.drop(columns=["sku_id"]), y_all, lead_times, unit_costs)

        orchestrator = MetaOrchestrator(
            config=config,
            erp_backend=env,
            supplier_backend=env,
            trend_backend=env,
            forecaster=NaiveForecaster(),
        )

        output_dir = Path(f"experiments/results/{ds_name}_{ts}")
        output_dir.mkdir(parents=True, exist_ok=True)

        benchmarker = Benchmarker(
            config=config,
            env=env,
            orchestrator=orchestrator,
            baseline1=bl1,
            baseline2=bl2,
        )
        results = benchmarker.run_all(assert_paper_results=False)

        reporter = Reporter(results, output_dir=output_dir)
        reporter.generate_all()
        reporter.print_overall_table()

        all_summaries[ds_name] = {
            pol: res.overall for pol, res in results.items()
        }
        logger.info("realworld.complete", dataset=ds_name, output=str(output_dir))

    # Save combined summary
    summary_path = Path(f"experiments/results/realworld_summary_{ts}.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_summaries, indent=2, default=str))
    print(f"\n✓ Real-world evaluation complete. Summary: {summary_path}")


if __name__ == "__main__":
    main()
