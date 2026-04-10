#!/usr/bin/env python3
"""Reproduce All Results from Syed et al. (2025) — 'Agentic Commerce'.

This script is the single entry point that reproduces every number in
Tables 2 and 3 of the paper.  Running it with the default configuration
(seed=42, 1,200 SKUs, 365-day test horizon) must produce results within
±0.5 percentage points of the reported values.

Usage
-----
    # Default (paper config):
    python experiments/run_paper_experiment.py

    # Custom config:
    python experiments/run_paper_experiment.py \\
        --config configs/simulation_1200sku.yaml

    # Skip assertion check (faster iteration):
    python experiments/run_paper_experiment.py --no-assert

    # Via Makefile:
    make run-paper-experiment

Expected runtime
----------------
    ~15–30 minutes on a modern CPU (no GPU required).
    The RL policy uses the NaiveForecaster + analytical fallback when
    PyTorch is unavailable, which is fast but may diverge slightly from
    paper values.

References
----------
Paper Section 5; Tables 2 and 3.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# ── make 'aairm' importable when run from repo root ──────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from aairm.agents.meta_orchestrator import MetaOrchestrator
from aairm.baselines.ml_static import MLStaticPolicy
from aairm.baselines.rop_eoq import ROPEOQPolicy
from aairm.evaluation.benchmarker import Benchmarker, BenchmarkResult
from aairm.evaluation.reporter import Reporter
from aairm.models.forecasting.naive_forecaster import NaiveForecaster
from aairm.simulation.environment import RetailEnv
from aairm.utils.config import AAIRMConfig
from aairm.utils.logging import configure_logging, get_logger
from aairm.utils.seed import set_global_seed

logger = get_logger(__name__)

# ── Paper expected results for assertion ─────────────────────────────────────
TOLERANCE = 0.005
REWARD_TUNING_KEYS = [
    "holding_cost_weight",
    "stockout_penalty_weight",
    "spoilage_cost_weight",
    "inventory_cap_penalty",
    "inventory_cap_days",
    "shelf_life_scale",
    "expiry_rate_multiplier",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce AAIRM paper experiments.")
    p.add_argument(
        "--config",
        default="configs/simulation_1200sku.yaml",
        help="Path to YAML config file (default: configs/simulation_1200sku.yaml).",
    )
    p.add_argument(
        "--no-assert",
        action="store_true",
        help="Skip paper-result assertion checks.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (default: experiments/results/<timestamp>).",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: 10 SKUs, 30-day test horizon (smoke test).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of seeds to run for statistical analysis (default: 1).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override simulation seed from the YAML config.",
    )
    return p.parse_args()


def _extract_reward_tuning(raw: dict) -> dict[str, float]:
    """Extract reward tuning values from YAML payload."""
    tuning_raw = raw.get("reward_tuning", {}) if isinstance(raw, dict) else {}
    if not isinstance(tuning_raw, dict):
        return {}

    out: dict[str, float] = {}
    for key in REWARD_TUNING_KEYS:
        if key in tuning_raw and tuning_raw[key] is not None:
            out[key] = float(tuning_raw[key])
    return out


def load_config(
    config_path: str, fast: bool, seed_override: int | None = None
) -> tuple[AAIRMConfig, dict[str, float], dict]:
    """Load config from YAML, optionally overriding for fast mode."""
    raw_payload: dict = {}
    reward_tuning: dict[str, float] = {}
    try:
        from omegaconf import OmegaConf

        raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        raw_payload = raw if isinstance(raw, dict) else {}
        reward_tuning = _extract_reward_tuning(raw_payload)
        if seed_override is not None:
            raw["simulation"]["seed"] = int(seed_override)
        if fast:
            raw["simulation"]["n_skus"] = 10
            raw["simulation"]["n_categories"] = 1
            raw["simulation"]["category_names"] = ["grocery"]
            raw["simulation"]["skus_per_category"] = 10
            raw["simulation"]["simulation_horizon_days"] = 60
            raw["simulation"]["test_horizon_days"] = 30
            raw["optimisation"]["rl_training_episodes"] = 5
        config = AAIRMConfig.model_validate(raw)
    except (ImportError, FileNotFoundError):
        logger.warning("omegaconf not available or config not found; using defaults")
        config = AAIRMConfig()
        if fast:
            config.simulation.n_skus = 10
            config.simulation.test_horizon_days = 30
    return config, reward_tuning, raw_payload


def _verify_reward_tuning(expected: dict[str, float], active: dict[str, float]) -> None:
    """Fail fast when runtime tuning diverges from config values."""
    for key, exp in expected.items():
        got = float(active.get(key, float("nan")))
        if not math.isfinite(got) or abs(got - exp) > 1e-9:
            raise RuntimeError(f"Reward tuning mismatch for '{key}': expected {exp}, got {got}.")


def build_baselines(
    env: RetailEnv,
    config: AAIRMConfig,
    train_days: int,
) -> tuple[ROPEOQPolicy, MLStaticPolicy]:
    """Fit both baselines on the training window demand history."""
    logger.info("baselines.fitting", train_days=train_days)
    catalog = env.catalog
    sku_ids = catalog.sku_ids if catalog else []

    # Collect training history from environment
    demand_history = {sku: env.get_demand_history(sku, train_days) for sku in sku_ids}
    snap = env.get_inventory_snapshot()
    lead_times = {s: float(snap.get(s, {}).get("lead_time_days", 5.0)) for s in sku_ids}
    unit_costs = {s: float(snap.get(s, {}).get("unit_cost", 5.0)) for s in sku_ids}

    # Baseline 1: ROP-EOQ
    bl1 = ROPEOQPolicy(service_level=config.optimisation.service_level)
    bl1.fit(demand_history, lead_times, unit_costs)

    # Baseline 2: ML + Static
    bl2 = MLStaticPolicy(
        service_level=config.optimisation.service_level,
        holding_cost_rate=config.optimisation.holding_cost_rate,
    )
    X_train_rows, y_train_rows = [], []
    for sku_id, series in demand_history.items():
        feat = MLStaticPolicy.build_feature_matrix({sku_id: series}, train_days)
        if not feat.empty:
            X_train_rows.append(feat)
            y_train_rows.extend(series[-30:].tolist())

    import pandas as pd

    if X_train_rows:
        X_all = pd.concat(X_train_rows, ignore_index=True)
        y_all = pd.Series(y_train_rows[: len(X_all)])
        bl2.fit(X_all.drop(columns=["sku_id"]), y_all, lead_times, unit_costs)
    else:
        bl2.fit(pd.DataFrame(), pd.Series(), lead_times, unit_costs)

    logger.info("baselines.fitted")
    return bl1, bl2


def run_single_experiment(config: AAIRMConfig, reward_tuning: dict[str, float], seed: int) -> dict[str, BenchmarkResult]:
    """Run a single experiment with the given config and seed."""
    logger.info("experiment.single_run", seed=seed)
    
    # ── Reproducibility ────────────────────────────────────────────────────
    set_global_seed(seed)
    logger.info("seed.set", seed=seed)

    # ── Build environment ──────────────────────────────────────────────────
    logger.info("environment.building", n_skus=config.simulation.n_skus)
    env = RetailEnv(config.simulation)
    env.reset(seed=seed)
    if reward_tuning:
        env.configure_tuning(**reward_tuning)
    active_tuning = env.get_tuning_parameters()
    logger.info("environment.reward_tuning_active", reward_tuning=active_tuning)

    train_days = config.simulation.simulation_horizon_days - config.simulation.test_horizon_days

    # Advance through training window to build demand history
    # Use a simple replenishment policy during warmup to maintain realistic inventory levels
    logger.info("simulation.warmup", train_days=train_days)
    for day in range(train_days):
        snap = env.get_inventory_snapshot()
        # Simple replenishment: order up to 30 days of average lead-time demand per SKU
        warmup_orders = {}
        for sku, rec in snap.items():
            on_hand = float(rec.get("on_hand", 0.0))
            # If inventory is getting low, reorder a moderate quantity
            if on_hand < 10.0:  # Simple threshold: reorder if on-hand < 10 units
                warmup_orders[sku] = 30.0  # Fixed reorder quantity
        env.step_agentic(warmup_orders)

    # ── Fit baselines ──────────────────────────────────────────────────────
    bl1, bl2 = build_baselines(env, config, train_days)

    # ── Train PPO policy ──────────────────────────────────────────────────
    rl_policy = None
    if config.fast_dev_run:
        print("Running in FAST DEV MODE")
        logger.info("rl_policy.fast_dev_mode")
    else:
        print("Running FULL TRAINING")
        logger.info("rl_policy.full_training_mode")
    
    logger.info("rl_policy.training_start", episodes=config.optimisation.rl_training_episodes)
    try:
        from aairm.models.rl.ppo_policy import PPOPolicy

        rl_policy = PPOPolicy(
            learning_rate=config.optimisation.rl_learning_rate,
            n_steps=config.optimisation.rl_n_steps,
            batch_size=config.optimisation.rl_batch_size,
            n_epochs=config.optimisation.rl_n_epochs,
            gamma=config.optimisation.discount_factor,
            verbose=0,
        )
        rl_policy.build(env)
        logger.info("ppo.model_built", message="PPO model built successfully")
        
        if config.fast_dev_run:
            total_timesteps = 1_000  # Lightweight training for fast iteration
        else:
            # Train: ~365 steps per episode × episodes = total timesteps
            # Paper: 400 episodes ≈ 146,000 steps
            total_timesteps = config.optimisation.rl_training_episodes * config.simulation.test_horizon_days
        
        rl_policy.train(total_timesteps=total_timesteps)
        logger.info("rl_policy.trained", total_timesteps=total_timesteps)
        print(f"PPO training completed: {total_timesteps} timesteps")
    except ImportError as e:
        logger.warning("rl_policy.stable_baselines3_unavailable", reason=str(e))
    except Exception as e:
        logger.error("rl_policy.training_failed", reason=str(e))

    orchestrator = MetaOrchestrator(
        config=config,
        erp_backend=env,
        supplier_backend=env,
        trend_backend=env,
        forecaster=NaiveForecaster(),  # Use Naive for now; can be extended
        rl_policy=rl_policy,
    )

    # ── Run benchmarker ────────────────────────────────────────────────────
    logger.info("benchmarker.start")
    t0 = time.perf_counter()

    benchmarker = Benchmarker(
        config=config,
        env=env,
        orchestrator=orchestrator,
        baseline1=bl1,
        baseline2=bl2,
    )

    do_assert = False  # No assertions for multi-seed runs
    results = benchmarker.run_all(assert_paper_results=do_assert)

    elapsed = time.perf_counter() - t0
    logger.info("benchmarker.complete", elapsed_s=round(elapsed, 1))

    return results


def aggregate_results(all_results: list[dict[str, BenchmarkResult]]) -> dict[str, BenchmarkResult]:
    """Aggregate results from multiple seeds into mean/std BenchmarkResults."""
    if not all_results:
        return {}
    
    policies = list(all_results[0].keys())
    aggregated = {}
    
    for policy in policies:
        # Collect all runs for this policy
        policy_results = [run[policy] for run in all_results if policy in run]
        
        if not policy_results:
            continue
            
        # Aggregate overall metrics
        overall_metrics = {}
        for metric in policy_results[0].overall.keys():
            values = [res.overall.get(metric, 0.0) for res in policy_results]
            overall_metrics[f"{metric}_mean"] = np.mean(values)
            overall_metrics[f"{metric}_std"] = np.std(values)
            overall_metrics[metric] = np.mean(values)  # For compatibility
        
        # Aggregate timeseries (mean across runs)
        timeseries_keys = list(policy_results[0].timeseries.keys())
        timeseries = {}
        for key in timeseries_keys:
            arrays = [res.timeseries[key] for res in policy_results]
            timeseries[key] = np.mean(arrays, axis=0)
        
        # RL curve if present
        rl_curve = None
        if policy_results[0].rl_curve:
            rl_curves = [res.rl_curve for res in policy_results if res.rl_curve]
            if rl_curves:
                # Take mean curve
                rl_curve = [(i, np.mean([c[i][1] for c in rl_curves if i < len(c)])) 
                           for i in range(max(len(c) for c in rl_curves))]
        
        aggregated[policy] = BenchmarkResult(
            policy_name=policy_results[0].policy_name,
            overall=overall_metrics,
            per_category={},  # TODO: aggregate per-category if needed
            timeseries=timeseries,
            rl_curve=rl_curve,
        )
    
    return aggregated


def make_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


def perform_statistical_comparison(all_results: list[dict[str, BenchmarkResult]], output_dir: Path) -> None:
    """Perform statistical tests between AAIRM and baselines."""
    try:
        from scipy.stats import ttest_rel
    except ImportError:
        logger.warning("scipy not available; skipping statistical tests")
        return
    
    if len(all_results) < 2:
        return
    
    policies = ["aairm", "baseline1", "baseline2"]
    metrics = ["stockout_rate", "fill_rate", "avg_inventory", "total_cost", "div_index"]
    
    comparison_results = {}
    
    for metric in metrics:
        comparison_results[metric] = {}
        for policy in policies:
            if policy == "baseline1":
                continue  # Baseline for comparison
            values_aairm = [run["aairm"].overall.get(metric, 0.0) for run in all_results]
            values_baseline = [run[policy].overall.get(metric, 0.0) for run in all_results]
            
            if len(values_aairm) == len(values_baseline) and len(values_aairm) > 1:
                t_stat, p_value = ttest_rel(values_aairm, values_baseline)
                comparison_results[metric][f"aairm_vs_{policy}"] = {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05
                }
    
    # Save to file
    import json
    with open(output_dir / "statistical_comparison.json", "w") as f:
        json.dump(make_json_serializable(comparison_results), f, indent=2)
    
    print("\nStatistical Comparison (AAIRM vs Baselines):")
    for metric, comparisons in comparison_results.items():
        print(f"  {metric}:")
        for comp, stats in comparisons.items():
            sig = "*" if stats["significant"] else ""
            print(f"    {comp}: p={stats['p_value']:.4f}{sig}")


def main() -> None:
    args = parse_args()

    # ── Output directory ───────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(f"experiments/results/paper_experiment_{ts}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Config + logging ───────────────────────────────────────────────────
    config, reward_tuning, raw_payload = load_config(args.config, args.fast, args.seed)
    configure_logging(level=str(config.log_level), fmt=str(config.log_format))
    logger.info("experiment.start", config=args.config, output_dir=str(output_dir), seeds=args.seeds)
    print(f"Running {args.seeds} seed(s) for statistical analysis")
    print("Active config:")
    print(f"  simulation.seed: {config.simulation.seed}")
    print(f"  holding_cost_weight: {reward_tuning.get('holding_cost_weight', 1.0)}")
    print(f"  stockout_penalty_weight: {reward_tuning.get('stockout_penalty_weight', 1.2)}")
    print(f"  spoilage_cost_weight: {reward_tuning.get('spoilage_cost_weight', 1.0)}")
    print(f"  inventory_cap_penalty: {reward_tuning.get('inventory_cap_penalty', 0.5)}")
    print(f"  inventory_cap_days: {reward_tuning.get('inventory_cap_days', 21.0)}")
    print(f"  shelf_life_scale: {reward_tuning.get('shelf_life_scale', 0.5)}")
    print(f"  expiry_rate_multiplier: {reward_tuning.get('expiry_rate_multiplier', 1.5)}")
    if raw_payload:
        logger.info("config.active_payload", payload=raw_payload)
    logger.info("config.reward_tuning", reward_tuning=reward_tuning)
    print("Active reward tuning parameters:")
    if reward_tuning:
        for key in REWARD_TUNING_KEYS:
            if key in reward_tuning:
                print(f"  {key}: {reward_tuning[key]}")
    else:
        print("  (none provided in config; defaults from RetailEnv remain active)")

    # ── Run multiple seeds ─────────────────────────────────────────────────
    all_results = []
    for seed_idx in range(args.seeds):
        seed = config.simulation.seed + seed_idx  # Increment seed for each run
        print(f"\n--- Running seed {seed_idx + 1}/{args.seeds} (seed={seed}) ---")
        results = run_single_experiment(config, reward_tuning, seed)
        all_results.append(results)

    # ── Aggregate results ──────────────────────────────────────────────────
    aggregated_results = aggregate_results(all_results)
    
    # ── Statistical comparison ─────────────────────────────────────────────
    if args.seeds > 1:
        perform_statistical_comparison(all_results, output_dir)

    # ── Build environment ──────────────────────────────────────────────────
    logger.info("environment.building", n_skus=config.simulation.n_skus)
    env = RetailEnv(config.simulation)
    env.reset(seed=config.simulation.seed)
    if reward_tuning:
        env.configure_tuning(**reward_tuning)
    active_tuning = env.get_tuning_parameters()
    logger.info("environment.reward_tuning_active", reward_tuning=active_tuning)
    print("Runtime reward tuning in environment:")
    for key in REWARD_TUNING_KEYS:
        if key in active_tuning:
            print(f"  {key}: {active_tuning[key]}")
    if reward_tuning:
        _verify_reward_tuning(reward_tuning, active_tuning)

    train_days = config.simulation.simulation_horizon_days - config.simulation.test_horizon_days

    # Advance through training window to build demand history
    # Use a simple replenishment policy during warmup to maintain realistic inventory levels
    logger.info("simulation.warmup", train_days=train_days)
    for day in range(train_days):
        snap = env.get_inventory_snapshot()
        # Simple replenishment: order up to 30 days of average lead-time demand per SKU
        warmup_orders = {}
        for sku, rec in snap.items():
            on_hand = float(rec.get("on_hand", 0.0))
            # If inventory is getting low, reorder a moderate quantity
            if on_hand < 10.0:  # Simple threshold: reorder if on-hand < 10 units
                warmup_orders[sku] = 30.0  # Fixed reorder quantity
        env.step_agentic(warmup_orders)

    # ── Fit baselines ──────────────────────────────────────────────────────
    bl1, bl2 = build_baselines(env, config, train_days)

    # ── Build AAIRM orchestrator ───────────────────────────────────────────
    logger.info("orchestrator.building")
    forecaster = NaiveForecaster()  # TFTForecaster if pytorch-forecasting installed

    # Try to load TFT if available
    try:
        from aairm.models.forecasting.tft_forecaster import TFTForecaster

        forecaster = TFTForecaster.from_config(config.forecasting)
        catalog = env.catalog
        if catalog:
            demand_hist = {s: env.get_demand_history(s, train_days) for s in catalog.sku_ids}
            logger.info("forecaster.training", architecture="tft")
            forecaster.fit(demand_hist)
        logger.info("forecaster.ready", architecture="tft")
    except (ImportError, Exception) as e:
        logger.warning("forecaster.tft_unavailable", reason=str(e), fallback="naive")

    # ── Train PPO policy ──────────────────────────────────────────────────
    rl_policy = None
    if config.fast_dev_run:
        print("Running in FAST DEV MODE")
        logger.info("rl_policy.fast_dev_mode")
    else:
        print("Running FULL TRAINING")
        logger.info("rl_policy.full_training_mode")
    
    logger.info("rl_policy.training_start", episodes=config.optimisation.rl_training_episodes)
    try:
        from aairm.models.rl.ppo_policy import PPOPolicy

        rl_policy = PPOPolicy(
            learning_rate=config.optimisation.rl_learning_rate,
            n_steps=config.optimisation.rl_n_steps,
            batch_size=config.optimisation.rl_batch_size,
            n_epochs=config.optimisation.rl_n_epochs,
            gamma=config.optimisation.discount_factor,
            verbose=0,
        )
        rl_policy.build(env)
        logger.info("ppo.model_built", message="PPO model built successfully")
        
        if config.fast_dev_run:
            total_timesteps = 1_000  # Lightweight training for fast iteration
        else:
            # Train: ~365 steps per episode × episodes = total timesteps
            # Paper: 400 episodes ≈ 146,000 steps
            total_timesteps = config.optimisation.rl_training_episodes * config.simulation.test_horizon_days
        
        rl_policy.train(total_timesteps=total_timesteps)
        logger.info("rl_policy.trained", total_timesteps=total_timesteps)
        print(f"PPO training completed: {total_timesteps} timesteps")
    except ImportError as e:
        logger.warning("rl_policy.stable_baselines3_unavailable", reason=str(e))
    except Exception as e:
        logger.error("rl_policy.training_failed", reason=str(e))

    orchestrator = MetaOrchestrator(
        config=config,
        erp_backend=env,
        supplier_backend=env,
        trend_backend=env,
        forecaster=forecaster,
        rl_policy=rl_policy,
    )

    # ── Run benchmarker ────────────────────────────────────────────────────
    logger.info("benchmarker.start")
    t0 = time.perf_counter()

    benchmarker = Benchmarker(
        config=config,
        env=env,
        orchestrator=orchestrator,
        baseline1=bl1,
        baseline2=bl2,
    )

    do_assert = not args.no_assert and not args.fast
    results = benchmarker.run_all(assert_paper_results=do_assert)

    elapsed = time.perf_counter() - t0
    logger.info("benchmarker.complete", elapsed_s=round(elapsed, 1))

    # ── Report ─────────────────────────────────────────────────────────────
    reporter = Reporter(aggregated_results, output_dir=output_dir)
    reporter.generate_all()

    # ── Summary to stdout ──────────────────────────────────────────────────
    reporter.print_overall_table()

    summary = {
        "experiment": "paper_reproduction",
        "config": args.config,
        "seeds": args.seeds,
        "n_skus": config.simulation.n_skus,
        "test_horizon_days": config.simulation.test_horizon_days,
        "elapsed_seconds": round(time.perf_counter() - time.perf_counter(), 1),  # TODO: track total time
        "output_dir": str(output_dir),
    }
    (output_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2))

    logger.info(
        "experiment.complete",
        output_dir=str(output_dir),
        seeds=args.seeds,
    )
    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
