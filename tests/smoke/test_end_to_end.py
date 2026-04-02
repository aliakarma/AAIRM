"""Smoke test: full PCA pipeline on 10 SKUs over 7 days.

Must complete in < 60 seconds on a standard CPU.
No LLM API key required.
No PyTorch required.

This test verifies that the entire codebase can be imported and executed
end-to-end without crashing.
"""

from __future__ import annotations

import time

import pytest

from aairm.utils.seed import set_global_seed
from aairm.utils.config import (
    AAIRMConfig,
    ForecastingConfig,
    GovernanceConfig,
    LLMConfig,
    OptimisationConfig,
    SimulationConfig,
    SupplierRankingConfig,
)
from aairm.simulation.environment import RetailEnv
from aairm.agents.base import AgentState
from aairm.agents.meta_orchestrator import MetaOrchestrator
from aairm.baselines.rop_eoq import ROPEOQPolicy
from aairm.models.forecasting.naive_forecaster import NaiveForecaster
from aairm.evaluation.metrics import stockout_rate, fill_rate


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
    optimisation=OptimisationConfig(mode="analytical", budget=50_000.0,
                                    warehouse_capacity=5_000.0),
    supplier_ranking=SupplierRankingConfig(
        alpha_1=0.35, alpha_2=0.30, alpha_3=0.25, alpha_4=0.10
    ),
    governance=GovernanceConfig(frozen_zone_capacity=0.0,
                                ambient_zone_capacity=5_000.0),
    llm=LLMConfig(model="gpt-4o", temperature=0.0),
)


@pytest.mark.timeout(60)
def test_end_to_end_pipeline_completes():
    """Full pipeline: setup → warmup → AAIRM cycles → metrics.

    Must complete in 60 seconds and produce valid, finite metrics.
    """
    import math

    t0 = time.perf_counter()
    set_global_seed(42)

    # Build environment
    env = RetailEnv(_SMOKE_CONFIG.simulation)
    env.reset(seed=42)

    # Warmup (13 days)
    for _ in range(13):
        env.step_agentic({})

    # Build baselines (Baseline 1)
    catalog = env.catalog
    sku_ids = catalog.sku_ids
    demand_hist = {s: env.get_demand_history(s, 13) for s in sku_ids}
    bl1 = ROPEOQPolicy(service_level=0.95)
    bl1.fit(demand_hist)

    # Build AAIRM
    orchestrator = MetaOrchestrator(
        config=_SMOKE_CONFIG,
        erp_backend=env,
        supplier_backend=env,
        trend_backend=env,
        forecaster=NaiveForecaster(),
    )

    # Run 7-day test horizon
    all_demand, all_fulfilled = [], []
    for day in range(7):
        state = AgentState(day=day)
        state = orchestrator.run_cycle(state)

        metrics = env.step_agentic(
            {sku: info.get("quantity", 0.0)
             for sku, info in state.approved_orders.items()}
        )
        snap = env.get_inventory_snapshot()
        day_demand = metrics["total_demand"]
        day_stockout = metrics["stockout_units"]
        all_demand.append(day_demand)
        all_fulfilled.append(max(0.0, day_demand - day_stockout))

    import numpy as np
    sr = stockout_rate(np.array(all_demand), np.array(all_fulfilled))
    fr = fill_rate(np.array(all_demand), np.array(all_fulfilled))

    elapsed = time.perf_counter() - t0

    # Assertions
    assert math.isfinite(sr), f"Stockout rate is not finite: {sr}"
    assert math.isfinite(fr), f"Fill rate is not finite: {fr}"
    assert 0.0 <= sr <= 1.0, f"Stockout rate out of range: {sr}"
    assert 0.0 <= fr <= 1.0, f"Fill rate out of range: {fr}"
    assert elapsed < 60.0, f"Smoke test took {elapsed:.1f}s (limit: 60s)"

    print(
        f"\n✓ Smoke test passed in {elapsed:.1f}s  |  "
        f"stockout_rate={sr:.3f}  fill_rate={fr:.3f}"
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
    from aairm.infrastructure.health_monitor import AgentHealthMonitor
    from aairm.infrastructure.reputation_engine import ReputationEngine
    from aairm.infrastructure.audit_ledger import AuditLedger

    monitor = AgentHealthMonitor()
    for agent_id in ["P1", "P2", "C1", "C2", "A1"]:
        monitor.record_cycle(agent_id, had_error=False)
    assert not monitor.is_degraded("P1")

    reputation = ReputationEngine()
    reputation.update("SUP-00001", "reliability", 0.92)
    assert reputation.get_supplier_reliability("SUP-00001") > 0.85

    ledger = AuditLedger()
    h1 = ledger.append("po.issued", {"sku_id": "GRO-0001", "qty": 50})
    h2 = ledger.append("po.issued", {"sku_id": "GRO-0002", "qty": 30})
    assert len(ledger) == 2
    assert ledger.verify()
