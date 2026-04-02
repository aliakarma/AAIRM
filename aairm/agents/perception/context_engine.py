"""Context Engine (P4) — Perception Layer.

Assembles feature-rich contextual representations for the Demand
Forecasting Agent (C1) by combining:

  - Historical sales trajectories per SKU.
  - Calendar and seasonal effects (day-of-week, holidays, promotions).
  - Price history.
  - Category-level metadata.
  - Rolling statistics (7-day mean, 28-day mean, 7-day std).

The output ``state.context_features`` is the primary input to C1.

References
----------
Paper Section 4.1 (agent P4); Eq. 2 feature vector x_{i,t}.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aairm.agents.base import AgentState, BaseAgent
from aairm.utils.config import SimulationConfig

# Holiday lookup for the simulation's Saudi retail context
# (day-of-year → uplift name)
_HOLIDAY_DAYS: dict[int, str] = {
    # Approximate DOYs; full implementation uses a proper calendar
    100: "Eid_Al_Fitr",
    101: "Eid_Al_Fitr",
    102: "Eid_Al_Fitr",
    175: "Eid_Al_Adha",
    176: "Eid_Al_Adha",
    177: "Eid_Al_Adha",
    1:   "New_Year",
    272: "National_Day",
    273: "National_Day",
}


def _is_holiday(day_of_year: int) -> bool:
    return day_of_year in _HOLIDAY_DAYS


def _compute_rolling(arr: np.ndarray, window: int) -> tuple[float, float]:
    """Return (mean, std) over the last ``window`` observations."""
    recent = arr[-window:] if len(arr) >= window else arr
    return float(np.mean(recent)), float(np.std(recent) + 1e-8)


class ContextEngine(BaseAgent):
    """P4 — Context Engine.

    Args:
        config: Simulation configuration.
        history_backend: Object implementing
            ``get_demand_history(sku_id, n_days) -> np.ndarray``.
        context_length: Number of history days to include.
    """

    def __init__(
        self,
        config: SimulationConfig,
        history_backend: Any = None,
        context_length: int = 60,
    ) -> None:
        super().__init__("P4", config)
        self._backend = history_backend
        self._context_length = context_length

    def run(self, state: AgentState) -> AgentState:
        """Assemble context feature vectors for all low-stock SKUs.

        Args:
            state: Pipeline state.  Reads ``state.low_stock_skus`` and
                ``state.sku_inventory_snapshot``.

        Returns:
            Updated state with ``state.context_features`` populated.
            Structure: ``{sku_id: {history, rolling_7d_mean,
            rolling_7d_std, rolling_28d_mean, day_of_week,
            day_of_year, is_holiday, is_weekend, unit_cost}}``.
        """
        t0 = self._log_start(state)
        features: dict[str, Any] = {}

        day_of_week = state.day % 7
        day_of_year = (state.day % 365) + 1
        is_weekend = day_of_week >= 5
        is_hol = _is_holiday(day_of_year)

        for sku_id in state.low_stock_skus:
            rec = state.sku_inventory_snapshot.get(sku_id, {})

            # Retrieve demand history
            if self._backend is not None:
                try:
                    history: np.ndarray = self._backend.get_demand_history(
                        sku_id, self._context_length
                    )
                except Exception as exc:  # noqa: BLE001
                    self._append_error(state, f"History fetch failed for {sku_id}: {exc}")
                    history = np.zeros(self._context_length)
            else:
                history = np.zeros(self._context_length)

            r7_mean, r7_std = _compute_rolling(history, 7)
            r28_mean, _ = _compute_rolling(history, 28)

            features[sku_id] = {
                "history": history.tolist(),
                "rolling_7d_mean": r7_mean,
                "rolling_7d_std": r7_std,
                "rolling_28d_mean": r28_mean,
                "day_of_week": day_of_week,
                "day_of_year": day_of_year,
                "is_holiday": is_hol,
                "is_weekend": is_weekend,
                "unit_cost": float(rec.get("unit_cost", 1.0)),
                "lead_time_days": float(rec.get("lead_time_days", 5.0)),
                "days_to_expiry": float(rec.get("days_to_expiry", 9999.0)),
                "category": rec.get("category", "unknown"),
            }

        state.context_features = features
        self._record_event(state, "context.assembled", n_skus=len(features))
        self._log_end(state, t0, n_skus=len(features))
        return state
