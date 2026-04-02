"""Unit tests for aairm/simulation/demand_generator.py."""

from __future__ import annotations

import numpy as np
import pytest

from aairm.simulation.sku_catalog import SKUCatalog
from aairm.simulation.demand_generator import DemandGenerator


@pytest.fixture(scope="module")
def small_gen():
    catalog = SKUCatalog(n_skus=10, category_names=["grocery"], seed=42)
    return DemandGenerator(catalog, n_days=30, seed=42)


def test_demand_matrix_shape(small_gen):
    assert small_gen._demand_matrix.shape == (10, 30)


def test_demand_non_negative(small_gen):
    assert np.all(small_gen._demand_matrix >= 0.0)


def test_get_demand_returns_float(small_gen):
    catalog = SKUCatalog(n_skus=10, category_names=["grocery"], seed=42)
    sku = catalog.sku_ids[0]
    d = small_gen.get_demand(sku, day=0)
    assert isinstance(d, float)
    assert d >= 0


def test_get_history_correct_length(small_gen):
    catalog = SKUCatalog(n_skus=10, category_names=["grocery"], seed=42)
    sku = catalog.sku_ids[0]
    hist = small_gen.get_history(sku, n_days=7, up_to_day=15)
    assert len(hist) == 7


def test_get_history_pads_if_short(small_gen):
    catalog = SKUCatalog(n_skus=10, category_names=["grocery"], seed=42)
    sku = catalog.sku_ids[0]
    hist = small_gen.get_history(sku, n_days=30, up_to_day=5)
    assert len(hist) == 30


def test_get_demand_stats(small_gen):
    catalog = SKUCatalog(n_skus=10, category_names=["grocery"], seed=42)
    sku = catalog.sku_ids[0]
    stats = small_gen.get_demand_stats(sku, up_to_day=20)
    assert "mu_d" in stats and "sigma_d" in stats
    assert stats["mu_d"] > 0
    assert stats["sigma_d"] > 0


def test_trend_signals_returns_list(small_gen):
    signals = small_gen.get_trend_signals(day=10)
    assert isinstance(signals, list)


def test_reproducibility():
    """Same seed → same demand matrix."""
    catalog = SKUCatalog(n_skus=5, category_names=["grocery"], seed=42)
    gen1 = DemandGenerator(catalog, n_days=10, seed=42)
    gen2 = DemandGenerator(catalog, n_days=10, seed=42)
    np.testing.assert_array_equal(gen1._demand_matrix, gen2._demand_matrix)


def test_different_seeds_differ():
    catalog = SKUCatalog(n_skus=5, category_names=["grocery"], seed=42)
    gen1 = DemandGenerator(catalog, n_days=10, seed=42)
    gen2 = DemandGenerator(catalog, n_days=10, seed=99)
    assert not np.array_equal(gen1._demand_matrix, gen2._demand_matrix)
