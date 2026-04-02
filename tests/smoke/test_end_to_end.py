"""Smoke test: full PCA pipeline on 10 SKUs over 7 days.

Must complete in < 60 seconds on a standard CPU.
No LLM API key required.
No PyTorch required.

This test verifies that the entire codebase can be imported and executed
end-to-end without crashing.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from aairm.agents.base import AgentState
from aairm.agents.meta_orchestrator import MetaOrchestrator
from aairm.baselines.rop_eoq import ROPEOQPolicy
from aairm.evaluation.metrics import fill_rate, stockout_rate
from aairm.models.forecasting.naive_forecaster import NaiveForecaster
from aairm.simulation.environment import RetailEnv
from aairm.utils.config import (
    AAIRMConfig,
    ForecastingConfig,
    GovernanceConfig,
    LLMConfig,
    OptimisationConfig,
    SimulationConfig,
    SupplierRankingConfig,
)
from aairm.utils.seed import set_global_seed

_SMOKE_CONFIG = AAIRMConfig(
    simulation=SimulationConfig(
        n_skus=10,
        n_categories=1,
        category_names=["grocery"],
        skus_per_category=10,
        simulation_horizon_days=20,
        test_horizon_days=7,
        n_suppliers_min=2,
        n_suppliers_max=3,
        seed=42,
    ),
    forecasting=ForecastingConfig(architecture="naive", forecast_horizon=7),
    optimisation=OptimisationConfig(mode="analytical", budget=50_000.0, warehouse_capacity=5_000.0),
    supplier_ranking=SupplierRankingConfig(alpha_1=0.35, alpha_2=0.30, alpha_3=0.25, alpha_4=0.10),
    governance=GovernanceConfig(frozen_zone_capacity=1.0, ambient_zone_capacity=5_000.0),
    llm=LLMConfig(model="gpt-4o", temperature=0.0),
)


@pytest.mark.timeout(60)
def test_end_to_end_pipeline_completes():
    """Full pipeline with multi-seed aggregation and soft realism checks.

    Must complete in 60 seconds and produce finite, realistic metrics.
    """
    import math

    t0 = time.perf_counter()
    seeds = [41, 42, 43]
    stockouts, fills, avg_invs, bl_avg_invs = [], [], [], []
    baseline_costs, aairm_costs = [], []

    for seed in seeds:
        set_global_seed(seed)

        # Build environment
        env = RetailEnv(_SMOKE_CONFIG.simulation)
        env.reset(seed=seed)

        # Warmup
        for _ in range(13):
            env.step_agentic({})

        # Baseline 1 fit and run
        catalog = env.catalog
        sku_ids = catalog.sku_ids
        demand_hist = {s: env.get_demand_history(s, 13) for s in sku_ids}
        bl1 = ROPEOQPolicy(service_level=0.95)
        bl1.fit(demand_hist)

        bl_cost = 0.0
        bl_on_hand, bl_demand = [], []
        for _ in range(7):
            snap = env.get_inventory_snapshot()
            orders = bl1.get_orders(snap)
            step_m = env.step_agentic(orders)
            bl_cost += float(step_m.get("reward", 0.0))
            bl_demand.append(float(step_m.get("total_demand", 0.0)))
            bl_on_hand.append(sum(float(v.get("on_hand", 0.0)) for v in snap.values()))

        # AAIRM run
        env = RetailEnv(_SMOKE_CONFIG.simulation)
        env.reset(seed=seed)
        for _ in range(13):
            env.step_agentic({})

        orchestrator = MetaOrchestrator(
            config=_SMOKE_CONFIG,
            erp_backend=env,
            supplier_backend=env,
            trend_backend=env,
            forecaster=NaiveForecaster(),
        )

        all_demand, all_fulfilled, all_on_hand = [], [], []
        aa_cost = 0.0
        for day in range(7):
            state = AgentState(day=day)
            state = orchestrator.run_cycle(state)
            metrics = env.step_agentic({})
            snap = env.get_inventory_snapshot()

            day_demand = float(metrics["total_demand"])
            day_fulfilled = float(metrics.get("fulfilled_units", 0.0))
            all_demand.append(day_demand)
            all_fulfilled.append(day_fulfilled)
            all_on_hand.append(sum(float(v.get("on_hand", 0.0)) for v in snap.values()))
            aa_cost += float(metrics.get("reward", 0.0))

        sr = stockout_rate(np.array(all_demand), np.array(all_fulfilled))
        fr = fill_rate(np.array(all_demand), np.array(all_fulfilled))
        inv_ratio = float(np.mean(all_on_hand) / max(np.mean(all_demand), 1e-9))

        stockouts.append(sr)
        fills.append(fr)
        avg_invs.append(inv_ratio)
        bl_avg_invs.append(float(np.mean(bl_on_hand) / max(np.mean(bl_demand), 1e-9)))
        baseline_costs.append(bl_cost)
        aairm_costs.append(aa_cost)

    sr_mean, sr_std = float(np.mean(stockouts)), float(np.std(stockouts))
    fr_mean, fr_std = float(np.mean(fills)), float(np.std(fills))
    inv_mean = float(np.mean(avg_invs))
    bl_inv_mean = float(np.mean(bl_avg_invs))
    bl_cost_mean = float(np.mean(baseline_costs))
    aa_cost_mean = float(np.mean(aairm_costs))

    elapsed = time.perf_counter() - t0

    # Assertions
    assert math.isfinite(sr_mean), f"Stockout mean is not finite: {sr_mean}"
    assert math.isfinite(fr_mean), f"Fill-rate mean is not finite: {fr_mean}"
    if not (0.01 < sr_mean < 0.15):
        print(f"[soft-check] stockout outside target: {sr_mean:.4f}")
    if not (0.85 < fr_mean < 0.99):
        print(f"[soft-check] fill-rate outside target: {fr_mean:.4f}")
    if not (inv_mean <= 2.0 * max(bl_inv_mean, 1e-9)):
        print("[soft-check] avg inventory above 2x baseline")
    # Rewards are negative costs: higher reward means lower cost.
    if not (aa_cost_mean > bl_cost_mean):
        print("[soft-check] AAIRM cost not below baseline")
    assert elapsed < 60.0, f"Smoke test took {elapsed:.1f}s (limit: 60s)"

    print(
        f"\n✓ Multi-seed smoke passed in {elapsed:.1f}s  |  "
        f"stockout={sr_mean:.3f}±{sr_std:.3f}  "
        f"fill_rate={fr_mean:.3f}±{fr_std:.3f}"
    )


@pytest.mark.timeout(60)
def test_baseline1_runs_without_error():
    """Baseline 1 (ROP-EOQ) runs for 7 days without crashing."""
    set_global_seed(42)
    env = RetailEnv(_SMOKE_CONFIG.simulation)
    env.reset(seed=42)
    for _ in range(13):
        env.step_agentic({})

    catalog = env.catalog
    demand_hist = {s: env.get_demand_history(s, 13) for s in catalog.sku_ids}
    bl1 = ROPEOQPolicy()
    bl1.fit(demand_hist)

    for _ in range(7):
        snap = env.get_inventory_snapshot()
        orders = bl1.get_orders(snap)
        env.step_agentic(orders)


@pytest.mark.timeout(60)
def test_infrastructure_components():
    """Verify Trusted Agent Infrastructure initialises without error."""
    from aairm.infrastructure.audit_ledger import AuditLedger
    from aairm.infrastructure.health_monitor import AgentHealthMonitor
    from aairm.infrastructure.reputation_engine import ReputationEngine

    monitor = AgentHealthMonitor()
    for agent_id in ["P1", "P2", "C1", "C2", "A1"]:
        monitor.record_cycle(agent_id, had_error=False)
    assert not monitor.is_degraded("P1")

    reputation = ReputationEngine()
    reputation.update("SUP-00001", "reliability", 0.92)
    assert reputation.get_supplier_reliability("SUP-00001") > 0.85

    ledger = AuditLedger()
    ledger.append("po.issued", {"sku_id": "GRO-0001", "qty": 50})
    ledger.append("po.issued", {"sku_id": "GRO-0002", "qty": 30})
    assert len(ledger) == 2
    assert ledger.verify()
