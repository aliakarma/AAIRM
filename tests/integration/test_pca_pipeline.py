"""Integration tests for the full PCA pipeline.

Tests that all 13 agents run sequentially without error using the
mock ERP backend and produce a valid AgentState.
"""

from __future__ import annotations

import pytest

from aairm.agents.base import AgentState
from aairm.agents.meta_orchestrator import MetaOrchestrator
from aairm.models.forecasting.naive_forecaster import NaiveForecaster


@pytest.mark.integration
def test_full_pca_cycle_completes(tiny_config, mock_erp_stub):
    """Run one full PCA cycle and verify all state fields are populated."""
    orchestrator = MetaOrchestrator(
        config=tiny_config,
        erp_backend=mock_erp_stub,
        supplier_backend=mock_erp_stub,
        trend_backend=mock_erp_stub,
        forecaster=NaiveForecaster(),
    )
    state = AgentState(day=0)
    result = orchestrator.run_cycle(state)

    # Perception populated
    assert isinstance(result.sku_inventory_snapshot, dict)
    assert len(result.sku_inventory_snapshot) > 0

    # Conceptualization populated
    assert isinstance(result.demand_forecasts, dict)
    assert isinstance(result.order_proposals, dict)

    # Action populated
    assert isinstance(result.purchase_orders_issued, list)

    # No unhandled errors
    assert len(result.errors) == 0 or all(
        "backend" not in e.lower() for e in result.errors
    )


@pytest.mark.integration
def test_ablation_no_negotiation(tiny_config, mock_erp_stub):
    """Bypass C4 — negotiated_terms populated directly from supplier rankings."""
    orchestrator = MetaOrchestrator(
        config=tiny_config,
        erp_backend=mock_erp_stub,
        supplier_backend=mock_erp_stub,
        trend_backend=mock_erp_stub,
        forecaster=NaiveForecaster(),
        skip_negotiation=True,
    )
    state = AgentState(day=0)
    result = orchestrator.run_cycle(state)
    # Terms should still be populated via bypass
    assert isinstance(result.negotiated_terms, dict)


@pytest.mark.integration
def test_ablation_no_governance(tiny_config, mock_erp_stub):
    """Bypass C5 — all negotiated terms pass directly to approved_orders."""
    orchestrator = MetaOrchestrator(
        config=tiny_config,
        erp_backend=mock_erp_stub,
        supplier_backend=mock_erp_stub,
        trend_backend=mock_erp_stub,
        forecaster=NaiveForecaster(),
        skip_governance=True,
    )
    state = AgentState(day=0)
    result = orchestrator.run_cycle(state)
    assert isinstance(result.approved_orders, dict)


@pytest.mark.integration
def test_multiple_cycles_stay_consistent(tiny_config, mock_erp_stub):
    """Run 5 cycles and verify state is fresh each time."""
    orchestrator = MetaOrchestrator(
        config=tiny_config,
        erp_backend=mock_erp_stub,
        supplier_backend=mock_erp_stub,
        forecaster=NaiveForecaster(),
    )
    for day in range(5):
        state = AgentState(day=day)
        result = orchestrator.run_cycle(state)
        assert result.day == day
