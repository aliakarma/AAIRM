import pytest
import numpy as np
from omegaconf import OmegaConf
from aairm.models.safety_stock import SafetyStockCalculator
from aairm.agents.replenishment import ReplenishmentAgent
from aairm.simulation.environment import InventoryEnvironment

class TestSafetyStockIntegration:

    def test_safety_stock_uses_real_units(self):
        """Safety stock must be >> 1.0 when demand is ~100 units/day."""
        cfg = OmegaConf.load('configs/replenishment.yaml')
        calc = SafetyStockCalculator(
            service_level_targets=dict(cfg.service_level_targets)
        )
        calc.set_category('sku_001', 'grocery')
        # Feed 30 days of realistic demand
        for _ in range(30):
            calc.update('sku_001', float(np.random.normal(100, 15)))
        ss = calc.compute_safety_stock('sku_001', lead_time_days=3)
        # With mean=100, std~15, lead=3: SS = 1.881*15*sqrt(3) ~ 49 units
        assert ss > 10.0, f"Safety stock too small: {ss} — check units"
        assert ss < 500.0, f"Safety stock too large: {ss} — check units"

    def test_order_quantity_nonzero_when_below_rop(self):
        """Agent must order when inventory is below reorder point."""
        cfg = OmegaConf.load('configs/replenishment.yaml')
        agent = ReplenishmentAgent(config=cfg)
        # Prime the safety stock calculator with demand history
        for _ in range(30):
            agent.safety_stock_calc.update('sku_001', 100.0)
        agent.safety_stock_calc.set_category('sku_001', 'grocery')
        qty = agent.compute_order_quantity(
            sku_id='sku_001',
            current_inventory=50.0,   # below 7-day demand of 700
            on_order_qty=0.0,
            lead_time_days=3,
            forecasted_total_demand=700.0,  # 7-day forecast
            category='grocery',
        )
        assert qty > 0, f"Agent ordered 0 units with inventory=50 and 7d demand=700"
        assert qty > 100, f"Order qty {qty} too small — expect ~650+ units"

    def test_update_called_during_simulation(self):
        """Verify safety_stock_calc.update is called in environment step."""
        cfg = OmegaConf.load('configs/replenishment.yaml')
        env = InventoryEnvironment(n_skus=5, config=cfg)
        env.reset()
        initial_history_len = len(
            env.replenishment_agent.safety_stock_calc
               ._demand_history.get('sku_000', [])
        )
        # Run one step
        env.step()
        after_history_len = len(
            env.replenishment_agent.safety_stock_calc
               ._demand_history.get('sku_000', [])
        )
        assert after_history_len > initial_history_len, \
            "safety_stock_calc.update() was NOT called during env.step(). " \
            "This is the wiring bug — fix environment.py step() method."

    def test_on_order_decrements_on_arrival(self):
        """on_order must decrease when an order arrives."""
        cfg = OmegaConf.load('configs/replenishment.yaml')
        env = InventoryEnvironment(n_skus=5, config=cfg)
        env.reset()
        # Manually place an order
        env.on_order['sku_000'] = 200.0
        lead_time = env.lead_times.get('sku_000', 3)
        # Advance past lead time
        for _ in range(lead_time + 1):
            env.step()
        assert env.on_order.get('sku_000', 0) < 200.0, \
            "on_order was never decremented — orders accumulate forever " \
            "causing inventory_position to be inflated and suppressing future orders."