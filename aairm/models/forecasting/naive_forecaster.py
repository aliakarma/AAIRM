"""Seasonal Naive Forecaster — fast baseline for C1.

Uses the mean of the most recent 7-day window as the point forecast
for all future horizons.  Uncertainty is estimated from the rolling
standard deviation.

This is the fastest forecaster in AAIRM and is used:
  - As the fallback when TFT / LSTM training has not completed.
  - In smoke tests (no PyTorch dependency required).
  - As the ``architecture="naive"`` option in ForecastingConfig.

References
----------
Paper Section 5.2 (Baseline 2 uses a richer ML model; naive is a
sub-baseline used for ablation and smoke testing).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from aairm.models.forecasting.base_forecaster import BaseForecaster


class NaiveForecaster(BaseForecaster):
    """Seasonal naive forecaster (7-day rolling mean).

    Args:
        window: Rolling window for mean/std computation (default 7).
    """

    def __init__(self, window: int = 7) -> None:
        self._window = window
        self._fitted = False

    def fit(
        self,
        demand_history: dict[str, np.ndarray],
        context: dict[str, Any] | None = None,
    ) -> "NaiveForecaster":
        """No training required; marks the forecaster as fitted.

        Args:
            demand_history: Historical demand (not used by naive model).
            context: Ignored.

        Returns:
            Self.
        """
        self._fitted = True
        return self

    def predict(
        self,
        sku_id: str,
        history: np.ndarray,
        context: dict[str, Any],
        horizon: int = 7,
    ) -> dict[str, Any]:
        """Forecast using the 7-day rolling mean.

        Args:
            sku_id: SKU identifier (unused; kept for interface consistency).
            history: Demand history array.
            context: Feature context (rolling_7d_mean / rolling_7d_std
                used if present to avoid recomputing).
            horizon: Forecast horizon in days.

        Returns:
            Forecast dict with ``mean``, ``variance``, ``p10``, ``p50``,
            ``p90``, ``horizon_days``, ``model``.
        """
        # Use pre-computed rolling stats from P4 if available
        mean_daily = float(context.get("rolling_7d_mean", 0.0))
        std_daily = float(context.get("rolling_7d_std", 1.0))

        if mean_daily == 0.0 and len(history) > 0:
            recent = history[-self._window:]
            mean_daily = float(np.mean(recent))
            std_daily = float(np.std(recent) + 1e-8)

        # Scale to horizon
        mean_horizon = mean_daily * horizon
        std_horizon = std_daily * np.sqrt(horizon)

        p10 = max(0.0, float(stats.norm.ppf(0.10, mean_horizon, std_horizon)))
        p50 = max(0.0, mean_horizon)
        p90 = max(0.0, float(stats.norm.ppf(0.90, mean_horizon, std_horizon)))

        return {
            "mean": max(0.0, mean_horizon),
            "variance": float(std_horizon ** 2),
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "horizon_days": horizon,
            "model": "naive",
        }
