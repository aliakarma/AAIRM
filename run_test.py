from tests.test_integration_replenishment import TestSafetyStockIntegration

t = TestSafetyStockIntegration()

try:
    t.test_safety_stock_uses_real_units()
    print("test_safety_stock_uses_real_units: PASSED")
except Exception as e:
    print(f"test_safety_stock_uses_real_units: FAILED - {e}")

try:
    t.test_order_quantity_nonzero_when_below_rop()
    print("test_order_quantity_nonzero_when_below_rop: PASSED")
except Exception as e:
    print(f"test_order_quantity_nonzero_when_below_rop: FAILED - {e}")

try:
    t.test_update_called_during_simulation()
    print("test_update_called_during_simulation: PASSED")
except Exception as e:
    print(f"test_update_called_during_simulation: FAILED - {e}")

try:
    t.test_on_order_decrements_on_arrival()
    print("test_on_order_decrements_on_arrival: PASSED")
except Exception as e:
    print(f"test_on_order_decrements_on_arrival: FAILED - {e}")