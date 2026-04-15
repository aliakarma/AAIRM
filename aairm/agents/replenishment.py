"""Replenishment Agent — Handles order quantity decisions.

Integrates safety stock calculations with forecasting for replenishment.
"""

from __future__ import annotations

from aairm.agents.base import BaseAgent
from aairm.models.safety_stock import SafetyStockCalculator
from aairm.utils.config import ReplenishmentConfig
from aairm.utils.logging import get_logger

logger = get_logger(__name__)


class ReplenishmentAgent(BaseAgent):
    """Agent for computing replenishment order quantities.

    Args:
        config: Replenishment configuration.
    """

    def __init__(self, config: ReplenishmentConfig) -> None:
        super().__init__("Replenishment", config)
        self.config = config
        self.safety_stock_calc = SafetyStockCalculator(
            service_level_targets=config.service_level_targets
        )

    def compute_order_quantity(
        self,
        sku_id: str,
        current_inventory: float,
        on_order_qty: float,
        lead_time_days: int,
        forecasted_total_demand: float,
        category: str,
    ) -> float:
        """Compute the order quantity for an SKU.

        Formula: order_qty = max(0, target_inventory - current_inventory - on_order_qty)
        Where target_inventory = forecasted_demand + safety_stock

        Args:
            sku_id: SKU identifier.
            current_inventory: Current inventory level.
            on_order_qty: Quantity already on order.
            lead_time_days: Supplier lead time in days.
            forecasted_total_demand: Total forecasted demand over the review period.
            category: SKU category.

        Returns:
            Order quantity to place.
        """
        # Set category for safety stock calculation
        self.safety_stock_calc.set_category(sku_id, category)

        # Compute safety stock
        safety_stock = self.safety_stock_calc.compute_safety_stock(
            sku_id, lead_time_days
        )

        # Target inventory = forecasted demand + safety stock
        target_inventory = forecasted_total_demand + safety_stock

        # Inventory position = current + on_order
        inventory_position = current_inventory + on_order_qty

        # Order quantity = max(0, target - position)
        order_qty = max(0.0, target_inventory - inventory_position)

        # Apply minimum order quantity
        if order_qty > 0:
            order_qty = max(order_qty, self.config.min_order_qty)

        logger.debug(
            f"SKU {sku_id}: target={target_inventory:.1f}, "
            f"position={inventory_position:.1f}, "
            f"order_qty={order_qty:.1f}"
        )

        return order_qty