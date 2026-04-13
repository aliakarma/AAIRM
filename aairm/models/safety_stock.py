"""Safety Stock Calculator."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy import stats


class SafetyStockCalculator:
    """Calculates safety stock using rolling demand history."""

    def __init__(self, default_service_level: float = 0.95, service_level_targets: dict[str, float] | None = None):
        self.default_service_level = default_service_level
        self.default_z = stats.norm.ppf(default_service_level)
        self.service_level_targets = service_level_targets or {
            "grocery": 0.97,
            "frozen_food": 0.95,
            "apparel": 0.90,
            "cosmetics": 0.92,
            "dry_fruits": 0.93,
        }
        self.z_values = {sl: stats.norm.ppf(sl) for sl in self.service_level_targets.values()}
        self.z_values.update({default_service_level: self.default_z})  # ensure default is included
        self._demand_history: dict[str, list[float]] = {}
        self._categories: dict[str, str] = {}

    def update(self, sku_id: str, actual_demand: float):
        """Update demand history with actual real demand."""
        assert actual_demand > 0.01, 'Demand looks normalized, not real units'
        if sku_id not in self._demand_history:
            self._demand_history[sku_id] = []
        self._demand_history[sku_id].append(actual_demand)
        # Keep rolling window of last 28 days only
        self._demand_history[sku_id] = self._demand_history[sku_id][-28:]

    def set_category(self, sku_id: str, category: str):
        """Set category for SKU."""
        self._categories[sku_id] = category

    def compute_safety_stock(self, sku_id: str, lead_time_days: int) -> float:
        """Compute safety stock for SKU."""
        category = self._categories.get(sku_id, "grocery")
        service_level = self.service_level_targets.get(category, self.default_service_level)
        z = self.z_values.get(service_level, self.default_z)
        
        history = self._demand_history.get(sku_id, [])
        if len(history) < 7:
            # Not enough data: use conservative fallback = 3 * sqrt(lead_time) * mean
            mean_d = np.mean(history) if history else 1.0
            return 3.0 * np.sqrt(lead_time_days) * mean_d
        demand_std = np.std(history, ddof=1)
        demand_mean = np.mean(history)
        # Standard formula: SS = z * sigma_d * sqrt(L)
        ss = z * demand_std * np.sqrt(lead_time_days)
        # Floor at 20% of mean * lead_time to prevent under-stocking
        min_ss = 0.20 * demand_mean * lead_time_days
        return max(ss, min_ss)

    def compute_reorder_point(self, sku_id: str, lead_time_days: int, forecast_next_n_days: float) -> float:
        """Compute reorder point."""
        ss = self.compute_safety_stock(sku_id, lead_time_days)
        return forecast_next_n_days + ss

    def save_state(self):
        """Save calculator state."""
        output_dir = Path("experiments/diagnostics")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "safety_stock_state.pkl", "wb") as f:
            pickle.dump(self, f)