"""Replenishment Logic — Newsvendor-Style Order Quantity Calculation."""

from __future__ import annotations

from typing import Any
import os

import numpy as np
from scipy import stats
import structlog

from aairm.models.safety_stock import SafetyStockCalculator

logger = structlog.get_logger(__name__)

DEBUG_ORDERS = os.getenv("AAIRM_DEBUG_ORDERS", "0") == "1"


class ReplenishmentAgent:
    """Agent for computing order quantities."""

    def __init__(self, config: AAIRMConfig):
        self.config = config
        self.safety_stock_calc = SafetyStockCalculator(
            default_service_level=config.replenishment.service_level,
            service_level_targets=config.replenishment.service_level_targets,
        )

    def update_demand(self, sku_id: str, actual_demand: float):
        """Update safety stock calculator with actual demand."""
        self.safety_stock_calc.update(sku_id, actual_demand)

    def compute_reorder_point(self, sku_id: str, lead_time_days: int, forecast_next_n_days: float) -> float:
        """Compute reorder point."""
        return self.safety_stock_calc.compute_reorder_point(sku_id, lead_time_days, forecast_next_n_days)

    def compute_order_quantity(
        self,
        sku_id: str,
        current_inventory: float,
        on_order_qty: float,
        lead_time_days: int,
        forecaster: BaseForecaster,
        category: str,
    ) -> int:
        """Compute order quantity."""
        self.safety_stock_calc.set_category(sku_id, category)
        # Step 1: forecast demand over review period + lead time
        review_period = self.config.replenishment.review_period_days.get(category, 7)
        total_horizon = lead_time_days + review_period
        forecast_total = forecaster.predict_total_demand(sku_id, total_horizon)

        # Step 2: compute reorder point
        forecast_over_lead_time = forecaster.predict_total_demand(sku_id, lead_time_days)
        rop = self.compute_reorder_point(sku_id, lead_time_days, forecast_over_lead_time)

        # Step 3: check if we need to order
        inventory_position = current_inventory + on_order_qty
        if inventory_position > rop:
            order_qty = 0
        else:
            # Step 4: target stock = cover demand over full horizon + safety stock
            ss = self.safety_stock_calc.compute_safety_stock(sku_id, lead_time_days)
            target_stock = forecast_total + ss

            # Step 5: order = gap between target and current position
            raw_order = target_stock - inventory_position
            order_qty = max(0, raw_order)

            # Step 6: apply MOQ
            min_order = self.config.replenishment.min_order_qty.get(category, 1)
            if 0 < order_qty < min_order:
                order_qty = min_order

            # Step 7: cap at max order
            max_order = self.config.replenishment.max_order_days_supply.get(category, 30) * (forecast_total / total_horizon)
            order_qty = min(order_qty, max_order)

        order_qty = int(round(order_qty))

        if DEBUG_ORDERS:
            print(f"[ORDER_DEBUG] sku={sku_id} cat={category} inv={current_inventory:.1f} "
                  f"rop={rop:.1f} forecast_7d={forecast_over_lead_time:.1f} safety={ss:.1f} "
                  f"demand_std=0.0 lead={lead_time_days} computed_qty={order_qty} emergency=False")

        return order_qty

    def emergency_reorder(self, sku_id: str, current_inventory: float, demand_forecast: np.ndarray, forecaster: BaseForecaster) -> int:
        """Emergency reorder."""
        if current_inventory >= demand_forecast[0]:
            return 0
        
        # Order 7-day supply
        demand_7d = forecaster.predict("dummy", np.array([]), {}, horizon=7)
        order_qty = int(round(np.sum(demand_7d)))
        
        logger.info("emergency_reorder", order_qty=order_qty)
        
        if DEBUG_ORDERS:
            print(f"[ORDER_DEBUG] sku={sku_id} emergency=True computed_qty={order_qty}")
        
        return order_qty

    def save_state(self):
        """Save state."""
        self.safety_stock_calc.save_state()


# Legacy functions for compatibility
def compute_reorder_point(
    forecaster: BaseForecaster,
    lead_time_days: int,
    service_level: float = 0.95,
) -> float:
    """Legacy function."""
    demand_forecast = forecaster.predict("dummy", np.array([]), {}, horizon=lead_time_days)
    demand_mean_lead = np.sum(demand_forecast)
    
    low, high = forecaster.forecast_uncertainty()
    demand_std_lead = (high - low) / (2 * 1.96)
    
    z = stats.norm.ppf(service_level)
    safety_stock = z * demand_std_lead * np.sqrt(lead_time_days)
    
    return demand_mean_lead + safety_stock


def compute_order_quantity(
    current_inventory: float,
    reorder_point: float,
    forecaster: BaseForecaster,
    target_inventory_days: int,
    config: AAIRMConfig,
    category: str,
) -> int:
    """Legacy function."""
    if current_inventory > reorder_point:
        order_qty = 0
    else:
        demand_forecast = forecaster.predict("dummy", np.array([]), {}, horizon=target_inventory_days)
        demand_mean_target = np.sum(demand_forecast)
        
        low, high = forecaster.forecast_uncertainty()
        demand_std_target = (high - low) / (2 * 1.96)
        z = stats.norm.ppf(0.95)
        safety_stock = z * demand_std_target * np.sqrt(target_inventory_days)
        
        target_stock = demand_mean_target + safety_stock
        order_qty = max(0, target_stock - current_inventory)
    
    min_order = config.replenishment.min_order_qty.get(category, 1)
    order_qty = max(order_qty, min_order)
    
    if DEBUG_ORDERS:
        print(f"[ORDER_DEBUG] sku=dummy cat={category} inv={current_inventory:.1f} "
              f"rop={reorder_point:.1f} forecast_7d={demand_mean_target:.1f if 'demand_mean_target' in locals() else 0:.1f} "
              f"safety={safety_stock:.1f if 'safety_stock' in locals() else 0:.1f} "
              f"demand_std={demand_std_target:.1f if 'demand_std_target' in locals() else 0:.1f} "
              f"lead=0 computed_qty={order_qty} emergency=False")
    
    return int(round(order_qty))


def emergency_reorder(
    current_inventory: float,
    demand_forecast: np.ndarray,
    forecaster: BaseForecaster,
    config: AAIRMConfig,
) -> int:
    """Legacy function."""
    if current_inventory >= demand_forecast[0]:
        return 0
    
    demand_7d = forecaster.predict("dummy", np.array([]), {}, horizon=7)
    order_qty = int(round(np.sum(demand_7d)))
    
    logger.info("emergency_reorder", order_qty=order_qty)
    
    return order_qty