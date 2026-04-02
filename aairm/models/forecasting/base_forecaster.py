"""Abstract base class for all demand forecasting models (C1).

All forecasting architectures (TFT, LSTM, Naive) must implement this
interface so the Demand Forecasting Agent (C1) can swap them without
code changes.

References
----------
Paper Section 4.2.1; Eq. 2 (ŷ_{i,t+h} = f_θ(x_{i,t}, h)).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseForecaster(ABC):
    """Abstract demand forecaster.

    Subclasses implement :meth:`fit` (offline training) and
    :meth:`predict` (online inference used by C1).
    """

    @abstractmethod
    def fit(
        self,
        demand_history: dict[str, np.ndarray],
        context: dict[str, Any] | None = None,
    ) -> "BaseForecaster":
        """Train the forecasting model.

        Args:
            demand_history: ``{sku_id: np.ndarray of shape (n_days,)}``.
            context: Optional additional training context features.

        Returns:
            Self (for method chaining).
        """

    @abstractmethod
    def predict(
        self,
        sku_id: str,
        history: np.ndarray,
        context: dict[str, Any],
        horizon: int = 7,
    ) -> dict[str, Any]:
        """Generate a demand forecast for one SKU.

        Args:
            sku_id: SKU identifier.
            history: Demand history array of shape ``(context_length,)``.
            context: Feature dict from the Context Engine (P4).
            horizon: Forecast horizon in days.

        Returns:
            Dict with keys:
            ``{mean, variance, p10, p50, p90, horizon_days, model}``.
        """

    def save(self, path: str | Path) -> None:
        """Persist model weights to disk.

        Args:
            path: File path (format is implementation-specific).
        """

    def load(self, path: str | Path) -> "BaseForecaster":
        """Load model weights from disk.

        Args:
            path: File path written by :meth:`save`.

        Returns:
            Self with loaded weights.
        """
        return self
