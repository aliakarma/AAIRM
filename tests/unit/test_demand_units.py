"""Unit tests for demand unit handling in safety stock calculations."""

import pytest
from aairm.models.safety_stock import SafetyStockCalculator


def test_safety_stock_real_units():
    """Test that safety stock operates in real units, not normalized."""
    calc = SafetyStockCalculator(
        default_service_level=0.95,
        service_level_targets={"grocery": 0.95}
    )

    sku_id = "TEST-SKU"
    calc.set_category(sku_id, "grocery")

    # Warm up with 30 days of 100 units/day demand
    for _ in range(30):
        calc.update(sku_id, 100.0)

    # Compute safety stock for lead time of 3 days
    ss = calc.compute_safety_stock(sku_id, lead_time_days=3)

    # With mean 100, std should be small, ss ≈ 1.645 * std * sqrt(3)
    # Should be > 10 (reasonable lower bound)
    assert ss > 10.0, f"Safety stock too low: {ss}, indicates normalized units"

    # Upper bound sanity check (< 500)
    assert ss < 500.0, f"Safety stock too high: {ss}, indicates wrong scale"