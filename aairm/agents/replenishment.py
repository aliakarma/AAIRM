"""Replenishment Logic — Newsvendor-Style Order Quantity Calculation."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats
import structlog

from aairm.models.forecasting.base_forecaster import BaseForecaster
from aairm.utils.config import AAIRMConfig

logger = structlog.get_logger(__name__)


def compute_reorder_point(
    forecaster: BaseForecaster,
    lead_time_days: int,
    service_level: float = 0.95,
) -> float:
    """Compute reorder point using newsvendor formula.
    
    Args:
        forecaster: Demand forecaster.
        lead_time_days: Lead time in days.
        service_level: Service level (e.g., 0.95).
        
    Returns:
        Reorder point (demand during lead time + safety stock).
    """
    # Assume forecaster has predict method with horizon
    demand_forecast = forecaster.predict("dummy", np.array([]), {}, horizon=lead_time_days)
    demand_mean_lead = np.sum(demand_forecast)
    
    # For uncertainty, assume forecast_uncertainty returns (low, high)
    low, high = forecaster.forecast_uncertainty()
    demand_std_lead = (high - low) / (2 * 1.96)  # approx std from 95% CI
    
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
    """Compute order quantity using newsvendor formula.
    
    Args:
        current_inventory: Current inventory level.
        reorder_point: Reorder point.
        forecaster: Demand forecaster.
        target_inventory_days: Target inventory days.
        config: Configuration.
        category: SKU category.
        
    Returns:
        Order quantity (integer).
    """
    if current_inventory > reorder_point:
        return 0
    
    demand_forecast = forecaster.predict("dummy", np.array([]), {}, horizon=target_inventory_days)
    demand_mean_target = np.sum(demand_forecast)
    
    # Safety stock same as in reorder_point
    low, high = forecaster.forecast_uncertainty()
    demand_std_target = (high - low) / (2 * 1.96)
    z = stats.norm.ppf(0.95)  # hardcoded for now
    safety_stock = z * demand_std_target * np.sqrt(target_inventory_days)
    
    target_stock = demand_mean_target + safety_stock
    order_qty = max(0, target_stock - current_inventory)
    
    # Apply MOQ
    min_order = config.replenishment.min_order_qty.get(category, 1)
    order_qty = max(order_qty, min_order)
    
    return int(round(order_qty))


def emergency_reorder(
    current_inventory: float,
    demand_forecast: np.ndarray,
    forecaster: BaseForecaster,
    config: AAIRMConfig,
) -> int:
    """Emergency reorder when about to stock out.
    
    Args:
        current_inventory: Current inventory.
        demand_forecast: Next day's demand forecast.
        forecaster: Forecaster.
        config: Configuration.
        
    Returns:
        Emergency order quantity.
    """
    if current_inventory >= demand_forecast[0]:
        return 0
    
    # Order 7-day supply
    demand_7d = forecaster.predict("dummy", np.array([]), {}, horizon=7)
    order_qty = int(round(np.sum(demand_7d)))
    
    logger.info("emergency_reorder", order_qty=order_qty)
    
    return order_qty