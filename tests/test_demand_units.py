"""Unit tests for demand unit handling and safety stock scaling."""

from __future__ import annotations

from aairm.models.safety_stock import SafetyStockCalculator


def test_safety_stock_uses_real_units_after_warmup() -> None:
    sku_id = "TEST-SKU-001"
    calc = SafetyStockCalculator(default_service_level=0.95)

    # Warm up with 100 units/day for 30 days.
    for _ in range(30):
        calc.update(sku_id, 100.0)

    ss = calc.compute_safety_stock(sku_id, lead_time_days=3)
    assert ss > 10.0, "Safety stock should exceed 10 units for real-unit demand history"
    assert ss < 500.0, "Safety stock should remain below 500 units for stable demand"
