"""Unit tests for aairm/baselines/rop_eoq.py."""

from __future__ import annotations

import numpy as np
import pytest

from aairm.baselines.rop_eoq import ROPEOQPolicy


@pytest.fixture
def fitted_policy():
    demand = {"GRO-0001": np.random.default_rng(42).normal(50, 10, 365).clip(0)}
    policy = ROPEOQPolicy(service_level=0.95)
    policy.fit(demand)
    return policy


def test_fit_sets_parameters(fitted_policy):
    assert "GRO-0001" in fitted_policy._mu
    assert fitted_policy._mu["GRO-0001"] > 0


def test_get_orders_returns_dict(fitted_policy):
    snapshot = {
        "GRO-0001": {
            "effective_available": 10.0,  # below ROP
            "unit_cost": 5.0,
            "unit_volume": 0.05,
            "category": "grocery",
        }
    }
    orders = fitted_policy.get_orders(snapshot)
    assert isinstance(orders, dict)
    # Should trigger an order since stock (10) is likely below ROP
    assert "GRO-0001" in orders
    assert orders["GRO-0001"] > 0


def test_no_order_when_stock_high(fitted_policy):
    snapshot = {
        "GRO-0001": {
            "effective_available": 99999.0,  # way above ROP
            "unit_cost": 5.0,
        }
    }
    orders = fitted_policy.get_orders(snapshot)
    assert "GRO-0001" not in orders


def test_get_orders_before_fit_raises():
    policy = ROPEOQPolicy()
    with pytest.raises(RuntimeError):
        policy.get_orders({})


def test_top_supplier_returns_cheapest():
    policy = ROPEOQPolicy()
    offers = [
        {"supplier_id": "A", "unit_cost": 10.0},
        {"supplier_id": "B", "unit_cost": 7.5},
        {"supplier_id": "C", "unit_cost": 12.0},
    ]
    best = policy.get_top_supplier("GRO-0001", offers)
    assert best["supplier_id"] == "B"


def test_top_supplier_empty_returns_none():
    policy = ROPEOQPolicy()
    assert policy.get_top_supplier("SKU", []) is None
