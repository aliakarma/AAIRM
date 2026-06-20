"""Integration tests for inter-agent communication and state propagation."""

from __future__ import annotations

import pytest

from aairm.agents.base import AgentState
from aairm.agents.perception.inventory_monitor import InventoryMonitorAgent
from aairm.agents.conceptualization.demand_forecasting import DemandForecastingAgent
from aairm.agents.conceptualization.reorder_optimisation import ReorderOptimisationAgent
from aairm.models.forecasting.naive_forecaster import NaiveForecaster
from aairm.utils.config import ForecastingConfig, OptimisationConfig, SimulationConfig


@pytest.mark.integration
def test_p1_to_c1_state_flow(mock_erp_stub):
    """P1 output (low_stock_skus) feeds correctly into C1."""
    sim_cfg = SimulationConfig(n_skus=10, n_categories=1,
                               category_names=["grocery"], seed=42)
    p1 = InventoryMonitorAgent(sim_cfg, erp_backend=mock_erp_stub)
    state = AgentState(day=0)
    state = p1.run(state)

    # P1 should have identified some low-stock SKUs
    # (all effective=55 vs ROP~57 from mock values)
    assert isinstance(state.low_stock_skus, list)
    assert isinstance(state.sku_inventory_snapshot, dict)


@pytest.mark.integration
def test_c1_to_c2_state_flow(mock_erp_stub):
    """C1 demand forecasts flow into C2 order proposals."""
    from aairm.utils.config import AAIRMConfig
    sim_cfg = SimulationConfig(n_skus=10, n_categories=1,
                               category_names=["grocery"], seed=42)
    fc_cfg = ForecastingConfig(architecture="naive", forecast_horizon=7)
    opt_cfg = OptimisationConfig(mode="analytical", budget=100_000.0)
    config = AAIRMConfig(
        simulation=sim_cfg,
        forecasting=fc_cfg,
        optimisation=opt_cfg,
    )

    forecaster = NaiveForecaster()
    p1 = InventoryMonitorAgent(sim_cfg, erp_backend=mock_erp_stub)
    c1 = DemandForecastingAgent(fc_cfg, forecaster=forecaster)
    c2 = ReorderOptimisationAgent(config, forecaster=forecaster, erp_backend=mock_erp_stub)

    state = AgentState(day=0)
    state = p1.run(state)

    # Manually add context features for C1
    for sku_id in state.low_stock_skus:
        state.context_features[sku_id] = {
            "rolling_7d_mean": 10.0,
            "rolling_7d_std": 2.0,
            "history": [10.0] * 14,
            "unit_cost": 5.0,
            "lead_time_days": 5.0,
            "days_to_expiry": 9999.0,
        }

    state = c1.run(state)
    assert len(state.demand_forecasts) == len(state.low_stock_skus)

    state = c2.run(state)
    assert isinstance(state.order_proposals, dict)
    for qty in state.order_proposals.values():
        assert qty >= 0.0


@pytest.mark.integration
def test_governance_rejects_anomalous_sku(mock_erp_stub):
    """C5 should hold orders for SKUs flagged with anomalies."""
    from aairm.agents.conceptualization.governance import GovernanceAgent
    from aairm.utils.config import GovernanceConfig

    gov_cfg = GovernanceConfig(human_approval_threshold=999_999.0)
    agent = GovernanceAgent(gov_cfg, total_budget=100_000.0)

    state = AgentState(day=0)
    state.sku_inventory_snapshot = {
        "GRO-0001": {"category": "grocery", "unit_volume": 0.05,
                     "unit_cost": 5.0, "days_to_expiry": 9999.0,
                     "demand_mean_daily": 10.0}
    }
    state.negotiated_terms = {
        "GRO-0001": {
            "supplier_id": "SUP-1", "unit_price": 5.0,
            "quantity": 100.0, "delivery_window_days": 5.0,
            "payment_terms": "Net-30", "discount_applied": 0.0
        }
    }
    state.anomaly_alerts = [
        {"type": "inventory_discrepancy", "sku_id": "GRO-0001",
         "detail": "Large on-hand delta detected"}
    ]

    state = agent.run(state)
    # GRO-0001 should be held due to anomaly_hold
    assert "GRO-0001" not in state.approved_orders
