"""Integration tests for the RetailEnv simulation environment."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.integration
def test_env_reset_returns_obs(tiny_env):
    obs, info = tiny_env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (5,)
    assert "day" in info


@pytest.mark.integration
def test_env_step_agentic_returns_metrics(tiny_env):
    metrics = tiny_env.step_agentic({})
    required = {"day", "total_demand", "stockout_units", "reward"}
    assert required.issubset(metrics.keys())
    assert metrics["total_demand"] >= 0
    assert metrics["stockout_units"] >= 0


@pytest.mark.integration
def test_env_inventory_snapshot_populated(tiny_env):
    snap = tiny_env.get_inventory_snapshot()
    assert len(snap) == tiny_env._config.n_skus
    first = next(iter(snap.values()))
    required_keys = {"on_hand", "effective_available", "unit_cost",
                     "demand_mean_daily", "lead_time_days"}
    assert required_keys.issubset(first.keys())


@pytest.mark.integration
def test_env_demand_history_correct_length(tiny_env):
    catalog = tiny_env.catalog
    sku_id = catalog.sku_ids[0]
    hist = tiny_env.get_demand_history(sku_id, 7)
    assert len(hist) == 7
    assert np.all(hist >= 0)


@pytest.mark.integration
def test_env_catalogue_query_returns_offers(tiny_env):
    catalog = tiny_env.catalog
    sku_id = catalog.sku_ids[0]
    offers = tiny_env.query_catalogue(sku_id)
    assert len(offers) >= 2
    assert "supplier_id" in offers[0]


@pytest.mark.integration
def test_env_accumulated_metrics_after_steps(tiny_env):
    for _ in range(5):
        tiny_env.step_agentic({})
    m = tiny_env.get_accumulated_metrics()
    assert "stockout_rate" in m
    assert "fill_rate" in m
    assert 0.0 <= m["stockout_rate"] <= 1.0
    assert 0.0 <= m["fill_rate"] <= 1.0


@pytest.mark.integration
def test_env_reproducible_with_same_seed():
    from aairm.simulation.environment import RetailEnv
    from aairm.utils.config import SimulationConfig
    cfg = SimulationConfig(n_skus=5, n_categories=1,
                           category_names=["grocery"], seed=42)
    e1 = RetailEnv(cfg); e1.reset(seed=42)
    e2 = RetailEnv(cfg); e2.reset(seed=42)
    m1 = e1.step_agentic({})
    m2 = e2.step_agentic({})
    assert m1["total_demand"] == m2["total_demand"]
