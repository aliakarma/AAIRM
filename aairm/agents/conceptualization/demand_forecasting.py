"""Demand Forecasting Agent (C1) — Conceptualization Layer.

Implements Eq. 2 of the paper:

    ŷ_{i,t+h} = f_θ(x_{i,t}, h)

where ``f_θ`` is a learned Temporal Fusion Transformer (or LSTM / naive
fallback) parameterised by θ, and ``x_{i,t}`` is the feature vector
assembled by the Context Engine (P4).

Training minimises Eq. 3 (MSE or pinball loss).

The agent outputs per-SKU point forecasts plus uncertainty summaries
(mean, variance, p10, p50, p90) for use by C2.

References
----------
Paper Section 4.2.1; Eqs. 2–3.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aairm.agents.base import AgentState, BaseAgent
from aairm.utils.config import ForecastingConfig


class DemandForecastingAgent(BaseAgent):
    """C1 — Demand Forecasting Agent.

    Args:
        config: :class:`~aairm.utils.config.ForecastingConfig`.
        forecaster: An object implementing the
            :class:`~aairm.models.forecasting.base_forecaster.BaseForecaster`
            interface.  Injected by the MetaOrchestrator.
    """

    def __init__(
        self,
        config: ForecastingConfig,
        forecaster: Any = None,
    ) -> None:
        super().__init__("C1", config)
        self._forecaster = forecaster
        self._horizon: int = config.forecast_horizon

    def run(self, state: AgentState) -> AgentState:
        """Compute demand forecasts for all low-stock and candidate SKUs.

        Ensures forecasts are generated proactively, even when low_stock_skus
        is empty.  Combines low_stock (priority) and replenishment_candidates
        (secondary priority) for a complete replenishment view.

        Reads
        -----
        state.low_stock_skus
            High-priority SKUs requiring replenishment.
        state.replenishment_candidates
            Secondary-priority candidates identified by soft thresholds.
        state.context_features
            Feature vectors assembled by P4.

        Writes
        ------
        state.demand_forecasts
            ``{sku_id: {mean, variance, p10, p50, p90, horizon_days}}``

        Args:
            state: Current pipeline state.

        Returns:
            Updated state.
        """
        t0 = self._log_start(state, n_low_stock=len(state.low_stock_skus),
                           n_candidates=len(state.replenishment_candidates))

        # Combine low_stock (high priority) and candidates
        all_skus = state.low_stock_skus + state.replenishment_candidates
        # Deduplicate while preserving order
        seen = set()
        unique_skus = []
        for sku_id in all_skus:
            if sku_id not in seen:
                seen.add(sku_id)
                unique_skus.append(sku_id)

        if not unique_skus:
            self._log.warning(
                "forecasting.no_candidates",
                day=state.day,
                note="No low-stock or candidate SKUs available for forecasting.",
            )

        forecasts: dict[str, dict[str, Any]] = {}

        for sku_id in unique_skus:
            ctx = state.context_features.get(sku_id)
            if ctx is None:
                self._append_error(
                    state, f"No context features for {sku_id}; using naive fallback."
                )
                ctx = {"history": [10.0] * 60, "rolling_7d_mean": 10.0,
                       "rolling_7d_std": 2.0}

            history = np.array(ctx.get("history", [10.0] * 60), dtype=float)

            if self._forecaster is not None:
                try:
                    result = self._forecaster.predict(
                        sku_id=sku_id,
                        history=history,
                        context=ctx,
                        horizon=self._horizon,
                    )
                except Exception as exc:  # noqa: BLE001
                    self._append_error(
                        state, f"Forecaster failed for {sku_id}: {exc}. Using naive."
                    )
                    result = self._naive_forecast(history)
            else:
                result = self._naive_forecast(history)

            forecasts[sku_id] = result
            self._record_event(
                state,
                "forecast.computed",
                sku_id=sku_id,
                mean=result["mean"],
                variance=result["variance"],
            )

        state.demand_forecasts = forecasts
        self._log.info(
            "forecasting.output",
            n_forecasts=len(forecasts),
            sample_skus=list(forecasts.keys())[:5],
        )
        self._log_end(state, t0, n_forecasts=len(forecasts))
        return state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _naive_forecast(self, history: np.ndarray) -> dict[str, Any]:
        """Seasonal naive fallback: use last 7-day mean as forecast.

        Args:
            history: Demand history array.

        Returns:
            Forecast dict with ``mean``, ``variance``, ``p10``, ``p50``, ``p90``.
        """
        recent = history[-7:] if len(history) >= 7 else history
        mean = float(np.mean(recent)) * self._horizon
        std = float(np.std(recent) + 1e-8) * np.sqrt(self._horizon)
        # Approximate quantiles via Gaussian assumption
        from scipy import stats  # local import to keep top-level clean

        p10 = float(stats.norm.ppf(0.10, loc=mean, scale=std))
        p50 = mean
        p90 = float(stats.norm.ppf(0.90, loc=mean, scale=std))
        return {
            "mean": max(0.0, mean),
            "variance": float(std**2),
            "p10": max(0.0, p10),
            "p50": max(0.0, p50),
            "p90": max(0.0, p90),
            "horizon_days": self._horizon,
            "model": "naive",
        }
