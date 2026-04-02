"""Shared pytest fixtures for the AAIRM test suite.

All fixtures are session-scoped where safe to reuse, and function-scoped
where state mutation could cause test interference.

Fixtures
--------
tiny_config        — AAIRMConfig with 10 SKUs, 1 category, 30-day horizon
tiny_catalog       — SKUCatalog with 10 SKUs
tiny_gen           — DemandGenerator for 10 SKUs over 30 days
tiny_supplier_sim  — SupplierSimulator for tiny catalog
tiny_erp           — ERPStub backed by tiny components
tiny_env           — RetailEnv with 10 SKUs, 1 category, seed=42
sample_demand_array— Deterministic 365-day demand array (seed=42)
mock_erp_stub      — Fully mocked ERPStub returning deterministic values
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

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


# ---------------------------------------------------------------------------
# Tiny config (10 SKUs, fast for unit and integration tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tiny_sim_config() -> SimulationConfig:
    """Minimal simulation configuration for fast tests."""
    return SimulationConfig(
        n_skus=10,
        n_categories=1,
        category_names=["grocery"],
        skus_per_category=10,
        simulation_horizon_days=60,
        test_horizon_days=30,
        n_suppliers_min=2,
        n_suppliers_max=3,
        seed=42,
    )


@pytest.fixture(scope="session")
def tiny_config(tiny_sim_config: SimulationConfig) -> AAIRMConfig:
    """Full AAIRMConfig wired to the tiny simulation config."""
    return AAIRMConfig(
        simulation=tiny_sim_config,
        forecasting=ForecastingConfig(
            architecture="naive",
            forecast_horizon=7,
            context_length=14,
        ),
        optimisation=OptimisationConfig(
            mode="analytical",
            budget=100_000.0,
            warehouse_capacity=10_000.0,
            service_level=0.95,
            rl_training_episodes=5,
        ),
        supplier_ranking=SupplierRankingConfig(
            alpha_1=0.35, alpha_2=0.30, alpha_3=0.25, alpha_4=0.10,
        ),
        governance=GovernanceConfig(
            frozen_zone_capacity=0.0,
            ambient_zone_capacity=10_000.0,
        ),
        llm=LLMConfig(model="gpt-4o", temperature=0.0),
    )


# ---------------------------------------------------------------------------
# Simulation components
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tiny_catalog(tiny_sim_config: SimulationConfig):
    """10-SKU catalog (session-scoped — read-only)."""
    from aairm.simulation.sku_catalog import SKUCatalog
    set_global_seed(42)
    return SKUCatalog(
        n_skus=tiny_sim_config.n_skus,
        category_names=tiny_sim_config.category_names,
        seed=42,
    )


@pytest.fixture(scope="session")
def tiny_gen(tiny_catalog):
    """DemandGenerator for 10 SKUs over 60 days."""
    from aairm.simulation.demand_generator import DemandGenerator
    return DemandGenerator(tiny_catalog, n_days=60, seed=42)


@pytest.fixture(scope="session")
def tiny_supplier_sim(tiny_catalog):
    """SupplierSimulator for the tiny catalog."""
    from aairm.simulation.supplier_simulator import SupplierSimulator
    return SupplierSimulator(tiny_catalog, n_suppliers_min=2, n_suppliers_max=3, seed=42)


@pytest.fixture(scope="function")
def tiny_erp(tiny_catalog, tiny_gen, tiny_supplier_sim):
    """Fresh ERPStub for each test (function-scoped to avoid state leakage)."""
    from aairm.simulation.erp_stub import ERPStub
    return ERPStub(tiny_catalog, tiny_gen, tiny_supplier_sim)


@pytest.fixture(scope="function")
def tiny_env(tiny_sim_config):
    """RetailEnv with 10 SKUs — function-scoped for clean state per test."""
    from aairm.simulation.environment import RetailEnv
    env = RetailEnv(tiny_sim_config)
    env.reset(seed=42)
    return env


# ---------------------------------------------------------------------------
# Demand arrays for metrics tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_demand_array() -> np.ndarray:
    """Deterministic 365-day demand array (seed=42, mean≈100 units/day)."""
    rng = np.random.default_rng(42)
    return np.maximum(0.0, rng.normal(loc=100.0, scale=20.0, size=365))


@pytest.fixture(scope="session")
def sample_fulfilled_array(sample_demand_array: np.ndarray) -> np.ndarray:
    """Fulfilled array with ~4% stockout rate (≈paper AAIRM value)."""
    rng = np.random.default_rng(123)
    stockout_mask = rng.random(len(sample_demand_array)) < 0.04
    fulfilled = sample_demand_array.copy()
    fulfilled[stockout_mask] *= rng.uniform(0.7, 0.95, size=stockout_mask.sum())
    return fulfilled


# ---------------------------------------------------------------------------
# Mock ERP stub (no real simulation needed)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def mock_erp_stub(tiny_catalog) -> MagicMock:
    """Fully mocked ERPStub returning deterministic values.

    Returns on-hand=50, reserved=5, in_transit=10 for every SKU.
    Lead time = 5 days; demand mean = 10 units/day.
    """
    mock = MagicMock()
    sku_ids = tiny_catalog.sku_ids

    def _snapshot():
        return {
            sku: {
                "on_hand": 50.0,
                "reserved": 5.0,
                "in_transit": 10.0,
                "effective_available": 55.0,
                "unit_cost": 5.0,
                "unit_volume": 0.05,
                "category": "grocery",
                "is_perishable": False,
                "days_to_expiry": 9999.0,
                "demand_mean_daily": 10.0,
                "demand_std_daily": 2.0,
                "lead_time_days": 5.0,
            }
            for sku in sku_ids
        }

    def _history(sku_id: str, n_days: int) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(sku_id)) % 2**31)
        return np.maximum(0.0, rng.normal(10.0, 2.0, size=n_days))

    mock.get_inventory_snapshot.side_effect = _snapshot
    mock.get_demand_history.side_effect = _history
    mock.get_trend_signals.return_value = []
    mock.get_pending_receipts.return_value = []
    mock.create_purchase_order.return_value = None
    mock.update_inbound_schedule.return_value = None
    mock.process_goods_receipt.return_value = None
    mock.query_catalogue.return_value = [
        {
            "supplier_id": "SUP-00001",
            "sku_id": "GRO-0001",
            "unit_cost": 5.0,
            "lead_time_mean": 5.0,
            "lead_time_std": 1.0,
            "reliability": 0.90,
            "moq": 20,
            "country": "SA",
        }
    ]
    mock.submit_purchase_order.return_value = {
        "po_id": "PO-TEST001",
        "confirmed": True,
        "eta_days": 5,
        "fulfilled_quantity": 100.0,
        "partial_fulfilment": False,
        "noisy_ack": False,
        "delay_days": 0,
    }
    return mock
