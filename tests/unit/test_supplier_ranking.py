"""Unit tests for the Supplier Ranking Agent (C3)."""

from __future__ import annotations

import pytest

from aairm.agents.base import AgentState
from aairm.agents.conceptualization.supplier_ranking import SupplierRankingAgent
from aairm.utils.config import SupplierRankingConfig


@pytest.fixture
def config():
    return SupplierRankingConfig(alpha_1=0.35, alpha_2=0.30, alpha_3=0.25, alpha_4=0.10)


@pytest.fixture
def mock_supplier_backend():
    class MockBackend:
        def query_catalogue(self, sku_id):
            return [
                {"supplier_id": "S1", "unit_cost": 5.0, "lead_time_mean": 3.0,
                 "lead_time_std": 0.5, "reliability": 0.95, "moq": 10},
                {"supplier_id": "S2", "unit_cost": 4.0, "lead_time_mean": 7.0,
                 "lead_time_std": 1.5, "reliability": 0.70, "moq": 5},
                {"supplier_id": "S3", "unit_cost": 6.0, "lead_time_mean": 2.0,
                 "lead_time_std": 0.2, "reliability": 0.98, "moq": 20},
            ]
    return MockBackend()


def test_ranking_returns_top3(config, mock_supplier_backend):
    agent = SupplierRankingAgent(config, supplier_backend=mock_supplier_backend, top_k=3)
    state = AgentState()
    state.order_proposals = {"GRO-0001": 50.0}
    state = agent.run(state)
    assert "GRO-0001" in state.supplier_rankings
    assert len(state.supplier_rankings["GRO-0001"]) <= 3


def test_ranking_sorted_ascending(config, mock_supplier_backend):
    agent = SupplierRankingAgent(config, supplier_backend=mock_supplier_backend)
    state = AgentState()
    state.order_proposals = {"GRO-0001": 50.0}
    state = agent.run(state)
    suppliers = state.supplier_rankings["GRO-0001"]
    scores = [s["composite_score"] for s in suppliers]
    assert scores == sorted(scores)


def test_moq_violation_flagged(config, mock_supplier_backend):
    agent = SupplierRankingAgent(config, supplier_backend=mock_supplier_backend)
    state = AgentState()
    state.order_proposals = {"GRO-0001": 3.0}   # below all MOQs
    state = agent.run(state)
    suppliers = state.supplier_rankings["GRO-0001"]
    assert all(s.get("moq_violation") is True for s in suppliers)


def test_no_backend_appends_error(config):
    agent = SupplierRankingAgent(config, supplier_backend=None)
    state = AgentState()
    state.order_proposals = {"GRO-0001": 50.0}
    state = agent.run(state)
    assert len(state.errors) > 0


def test_empty_proposals_produces_empty_rankings(config, mock_supplier_backend):
    agent = SupplierRankingAgent(config, supplier_backend=mock_supplier_backend)
    state = AgentState()
    state.order_proposals = {}
    state = agent.run(state)
    assert state.supplier_rankings == {}
