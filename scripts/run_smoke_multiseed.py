#!/usr/bin/env python3
"""Fast multi-seed smoke benchmark for AAIRM realism checks.

Supports optional tuning parameters for reward-shaping and perishability
dynamics so it can be reused by auto-tuning scripts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aairm.agents.meta_orchestrator import MetaOrchestrator
from aairm.baselines.ml_static import MLStaticPolicy
from aairm.baselines.rop_eoq import ROPEOQPolicy
from aairm.evaluation.benchmarker import Benchmarker
from aairm.models.forecasting.naive_forecaster import NaiveForecaster
from aairm.simulation.environment import RetailEnv
from aairm.utils.config import AAIRMConfig
from aairm.utils.seed import set_global_seed


def _parse_seeds(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _load_runtime_overrides(config_path: str) -> tuple[dict, dict]:
    """Load tuning values from YAML config."""
    from omegaconf import OmegaConf

    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid config payload in {config_path}")

    sim = raw.get("simulation", {}) if isinstance(raw.get("simulation", {}), dict) else {}
    opt = raw.get("optimisation", {}) if isinstance(raw.get("optimisation", {}), dict) else {}
    tuning = raw.get("reward_tuning", {}) if isinstance(raw.get("reward_tuning", {}), dict) else {}
    return {
        "seed": int(sim.get("seed", 42)),
        "n_skus": int(sim.get("n_skus", 100)),
        "rl_episodes": int(opt.get("rl_training_episodes", 80)),
    }, tuning


def build_smoke_config(seed: int, n_skus: int, rl_episodes: int) -> AAIRMConfig:
    cfg = AAIRMConfig()
    cats = ["grocery", "frozen_food", "apparel", "cosmetics", "dry_fruits"]
    n_categories = len(cats)
    aligned_skus = max(n_categories, (n_skus // n_categories) * n_categories)

    cfg.simulation.n_skus = aligned_skus
    cfg.simulation.n_categories = n_categories
    cfg.simulation.category_names = cats
    cfg.simulation.skus_per_category = aligned_skus // n_categories
    cfg.simulation.simulation_horizon_days = 140
    cfg.simulation.test_horizon_days = 30
    cfg.simulation.n_suppliers_min = 2
    cfg.simulation.n_suppliers_max = 3
    cfg.simulation.seed = seed

    cfg.forecasting.architecture = "naive"
    cfg.optimisation.rl_training_episodes = rl_episodes
    cfg.log_level = "WARNING"
    cfg.log_format = "console"
    return cfg


def run_for_seed(
    seed: int,
    n_skus: int = 100,
    rl_episodes: int = 80,
    tuning: dict[str, float] | None = None,
) -> dict:
    cfg = build_smoke_config(seed, n_skus=n_skus, rl_episodes=rl_episodes)
    set_global_seed(seed)

    env = RetailEnv(cfg.simulation)
    if tuning:
        env.configure_tuning(**tuning)
    env.reset(seed=seed)
    if tuning:
        env.configure_tuning(**tuning)

    train_days = cfg.simulation.simulation_horizon_days - cfg.simulation.test_horizon_days
    for _ in range(train_days):
        env.step_agentic({})

    snap = env.get_inventory_snapshot()
    sku_ids = list(snap.keys())
    demand_hist = {sku: env.get_demand_history(sku, train_days) for sku in sku_ids}
    lead_times = {s: float(snap[s].get("lead_time_days", 5.0)) for s in sku_ids}
    unit_costs = {s: float(snap[s].get("unit_cost", 5.0)) for s in sku_ids}

    bl1 = ROPEOQPolicy(service_level=cfg.optimisation.service_level)
    bl1.fit(demand_hist, lead_times, unit_costs)

    bl2 = MLStaticPolicy(
        service_level=cfg.optimisation.service_level,
        holding_cost_rate=cfg.optimisation.holding_cost_rate,
    )

    x_rows, y_rows = [], []
    for sku_id, series in demand_hist.items():
        feat = MLStaticPolicy.build_feature_matrix({sku_id: series}, train_days)
        if not feat.empty:
            x_rows.append(feat)
            y_rows.extend(series[-30:].tolist())

    if x_rows:
        x_all = pd.concat(x_rows, ignore_index=True)
        y_all = pd.Series(y_rows[: len(x_all)])
        bl2.fit(x_all.drop(columns=["sku_id"]), y_all, lead_times, unit_costs)
    else:
        bl2.fit(pd.DataFrame({"x": [0.0]}), pd.Series([0.0]), lead_times, unit_costs)

    orch = MetaOrchestrator(
        config=cfg,
        erp_backend=env,
        supplier_backend=env,
        trend_backend=env,
        forecaster=NaiveForecaster(),
    )

    bench = Benchmarker(cfg, env=env, orchestrator=orch, baseline1=bl1, baseline2=bl2)
    results = bench.run_all(assert_paper_results=False)

    return {
        "seed": seed,
        "baseline1": results["baseline1"].overall,
        "baseline2": results["baseline2"].overall,
        "aairm": results["aairm"].overall,
        "aairm_reward_raw": results["aairm"].timeseries.get("reward_raw", np.array([])).tolist(),
    }


def summarize(seed_runs: list[dict]) -> dict:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for policy in ["baseline1", "baseline2", "aairm"]:
        out[policy] = {}
        keys = seed_runs[0][policy].keys()
        for key in keys:
            values = [float(run[policy].get(key, 0.0)) for run in seed_runs]
            out[policy][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    return out


def run_smoke(
    seeds: list[int],
    n_skus: int,
    rl_episodes: int,
    tuning: dict[str, float] | None = None,
) -> tuple[list[dict], dict]:
    runs = [run_for_seed(s, n_skus=n_skus, rl_episodes=rl_episodes, tuning=tuning) for s in seeds]
    summary = summarize(runs)
    return runs, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AAIRM fast multi-seed smoke benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config. If provided, reads simulation + reward_tuning values.",
    )
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--n-skus", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--holding-cost-weight", type=float, default=1.0)
    parser.add_argument("--stockout-penalty-weight", type=float, default=1.2)
    parser.add_argument("--spoilage-cost-weight", type=float, default=1.0)
    parser.add_argument("--inventory-cap-penalty", type=float, default=0.5)
    parser.add_argument("--inventory-cap-days", type=float, default=21.0)
    parser.add_argument("--shelf-life-scale", type=float, default=0.5)
    parser.add_argument("--expiry-rate-multiplier", type=float, default=1.5)
    parser.add_argument("--out-dir", type=str, default="experiments/results/smoke_multiseed")
    args = parser.parse_args()

    runtime_overrides: dict[str, int] = {}
    config_tuning: dict[str, float] = {}
    if args.config:
        runtime_overrides, config_tuning = _load_runtime_overrides(args.config)

    seeds = _parse_seeds(args.seeds)
    n_skus = args.n_skus
    episodes = args.episodes
    tuning = {
        "holding_cost_weight": float(
            config_tuning.get("holding_cost_weight", args.holding_cost_weight)
        ),
        "stockout_penalty_weight": float(
            config_tuning.get("stockout_penalty_weight", args.stockout_penalty_weight)
        ),
        "spoilage_cost_weight": float(
            config_tuning.get("spoilage_cost_weight", args.spoilage_cost_weight)
        ),
        "inventory_cap_penalty": float(
            config_tuning.get("inventory_cap_penalty", args.inventory_cap_penalty)
        ),
        "inventory_cap_days": float(
            config_tuning.get("inventory_cap_days", args.inventory_cap_days)
        ),
        "shelf_life_scale": float(config_tuning.get("shelf_life_scale", args.shelf_life_scale)),
        "expiry_rate_multiplier": float(
            config_tuning.get("expiry_rate_multiplier", args.expiry_rate_multiplier)
        ),
    }

    runs, summary = run_smoke(
        seeds=seeds,
        n_skus=n_skus,
        rl_episodes=episodes,
        tuning=tuning,
    )

    print("\n============================================================")
    print(f"AAIRM Fast Multi-Seed Smoke Summary ({len(seeds)} seeds)")
    print("============================================================")
    for policy in ["baseline1", "baseline2", "aairm"]:
        m = summary[policy]
        print(
            f"{policy:<10} "
            f"stockout={m['stockout_rate']['mean']:.4f}+-{m['stockout_rate']['std']:.4f}  "
            f"fill={m['fill_rate']['mean']:.4f}+-{m['fill_rate']['std']:.4f}  "
            f"inv={m['avg_inventory']['mean']:.3f}+-{m['avg_inventory']['std']:.3f}  "
            f"cost={m['total_cost']['mean']:.3f}+-{m['total_cost']['std']:.3f}  "
            f"spoil={m['spoilage_rate']['mean']:.4f}+-{m['spoilage_rate']['std']:.4f}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "seeds": seeds,
                "n_skus": n_skus,
                "episodes": episodes,
                "tuning": tuning,
                "runs": runs,
                "summary": summary,
            },
            indent=2,
        )
    )
    print(f"\nSaved summary to: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
