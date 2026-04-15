"""Safety Stock Calculator — Category-aware safety stock computation.

Implements the safety stock calculation for replenishment decisions.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict

import numpy as np
from scipy.stats import norm

from aairm.utils.logging import get_logger

logger = get_logger(__name__)


class SafetyStockCalculator:
    """Calculates safety stock levels based on demand history and service levels.

    Args:
        default_service_level: Default service level target (0-1).
        service_level_targets: Category-specific service level overrides.
    """

    def __init__(
        self,
        default_service_level: float = 0.95,
        service_level_targets: Dict[str, float] | None = None,
    ) -> None:
        self.default_service_level = default_service_level
        self.service_level_targets = service_level_targets or {}
        self._demand_history: Dict[str, list[float]] = defaultdict(list)
        self._sku_categories: Dict[str, str] = {}

    def set_category(self, sku_id: str, category: str) -> None:
        """Assign a category to an SKU for category-specific service levels."""
        self._sku_categories[sku_id] = category

    def update(self, sku_id: str, demand: float) -> None:
        """Update demand history for an SKU."""
        self._demand_history[sku_id].append(demand)

    def compute_safety_stock(self, sku_id: str, lead_time_days: int) -> float:
        """Compute safety stock for an SKU given lead time.

        Uses the formula: SS = z * std_demand * sqrt(lead_time)

        Where z is the service level factor from normal distribution.
        """
        if sku_id not in self._demand_history or not self._demand_history[sku_id]:
            logger.warning(f"No demand history for {sku_id}, using 0 safety stock")
            return 0.0

        demand_history = np.array(self._demand_history[sku_id])
        if len(demand_history) < 7:  # Minimum for std calculation
            logger.warning(f"Insufficient demand history for {sku_id}")
            return 0.0

        # Get service level for this SKU's category
        category = self._sku_categories.get(sku_id)
        service_level = self.service_level_targets.get(
            category, self.default_service_level
        )

        # Compute z-factor
        z = norm.ppf(service_level)

        # Compute demand std
        std_demand = np.std(demand_history, ddof=1)

        # Safety stock = z * std * sqrt(lead_time)
        safety_stock = z * std_demand * np.sqrt(lead_time_days)

        return float(safety_stock)