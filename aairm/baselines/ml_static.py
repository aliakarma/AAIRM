"""ML + Static Policy Baseline (Baseline 2).

A non-agentic ML-augmented reorder policy as described in Section 5.2:

  - A global LightGBM gradient-boosted tree provides daily demand forecasts.
  - Order logic remains rule-based (fixed ROP applied to the forecast mean).
  - Forecasts are recomputed daily from the rolling window.
  - No supplier negotiation; no cross-category coordination.

Expected paper performance:
    stockout_rate   = 6.2%
    fill_rate       = 95.4%
    avg_inventory   = 1.32  (normalised)
    total_cost      = 0.93  (normalised)
    div_index       = 0.47

References
----------
Paper Section 5.2.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from aairm.utils.math_utils import rop, safety_stock
from aairm.utils.logging import get_logger

logger = get_logger(__name__)


class MLStaticPolicy:
    """Non-agentic ML-augmented reorder policy.

    Args:
        service_level: Target service level (default 0.95).
        holding_cost_rate: Annual holding cost rate (default 0.25).
        penalty_cost_multiplier: Stockout penalty relative to unit cost
            (default 3.0).
    """

    def __init__(
        self,
        service_level: float = 0.95,
        holding_cost_rate: float = 0.25,
        penalty_cost_multiplier: float = 3.0,
    ) -> None:
        self._sl = service_level
        self._h = holding_cost_rate
        self._p_mult = penalty_cost_multiplier
        self._model: Any = None
        self._fitted = False
        self._lead_times: dict[str, float] = {}
        self._unit_costs: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        lead_times: dict[str, float] | None = None,
        unit_costs: dict[str, float] | None = None,
    ) -> "MLStaticPolicy":
        """Train the LightGBM forecasting model.

        Args:
            X_train: Feature DataFrame with columns:
                ``[rolling_7d_mean, rolling_28d_mean, rolling_7d_std,
                   day_of_week, day_of_year, is_holiday, is_weekend]``.
            y_train: Target series of daily demand values.
            lead_times: Per-SKU average lead times.
            unit_costs: Per-SKU unit costs.

        Returns:
            Self.
        """
        self._lead_times = lead_times or {}
        self._unit_costs = unit_costs or {}

        try:
            import lightgbm as lgb  # type: ignore

            self._model = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                verbose=-1,
            )
            self._model.fit(X_train, y_train)
            self._fitted = True
            logger.info("ml_static.trained", n_samples=len(X_train))
        except ImportError:
            logger.warning("ml_static.lightgbm_not_available; using naive fallback")
            self._model = None
            self._fitted = True   # fallback is always "fitted"
        return self

    def predict_demand(self, X_today: pd.DataFrame) -> dict[str, float]:
        """Return per-SKU next-day demand forecasts.

        Args:
            X_today: Feature DataFrame with one row per SKU.
                Must include a ``sku_id`` column.

        Returns:
            ``{sku_id: forecast_demand}``
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_demand().")

        if "sku_id" not in X_today.columns:
            raise ValueError("X_today must include a 'sku_id' column.")

        sku_ids = X_today["sku_id"].tolist()
        feature_cols = [c for c in X_today.columns if c != "sku_id"]
        X = X_today[feature_cols]

        if self._model is not None:
            try:
                preds = self._model.predict(X)
            except Exception as exc:  # noqa: BLE001
                logger.error("ml_static.predict.failed", error=str(exc))
                preds = X["rolling_7d_mean"].values if "rolling_7d_mean" in X else np.ones(len(X)) * 10.0
        else:
            # Naive fallback: use rolling 7-day mean
            preds = (
                X["rolling_7d_mean"].values
                if "rolling_7d_mean" in X.columns
                else np.ones(len(X)) * 10.0
            )

        return {
            sku_id: max(0.0, float(pred))
            for sku_id, pred in zip(sku_ids, preds)
        }

    def get_orders(
        self,
        inventory_snapshot: dict[str, dict[str, Any]],
        demand_forecasts: dict[str, float],
    ) -> dict[str, float]:
        """Compute order quantities using fixed ROP applied to forecast mean.

        No negotiation or cross-category coordination is performed.
        Uses the newsvendor critical ratio to derive the order quantity:

            Q* = argmin_Q E[C(Q)] with forecast-based μ̂ and σ̂.

        Args:
            inventory_snapshot: Per-SKU inventory dict from ERPStub.
            demand_forecasts: Per-SKU one-step-ahead forecasts from predict_demand.

        Returns:
            ``{sku_id: Q*}`` for triggered SKUs.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before get_orders().")

        orders: dict[str, float] = {}

        for sku_id, rec in inventory_snapshot.items():
            effective = float(rec.get("effective_available", 0.0))
            mu_forecast = demand_forecasts.get(sku_id, 10.0)
            sigma_hist = float(rec.get("demand_std_daily", 2.0))
            lead_time = self._lead_times.get(sku_id, 5.0)
            unit_cost = self._unit_costs.get(sku_id, 10.0)

            # ROP using ML forecast mean
            reorder_pt = rop(mu_forecast, sigma_hist, lead_time, self._sl)

            if effective <= reorder_pt:
                # Order to bring stock up to lead-time demand + safety stock
                ss = safety_stock(sigma_hist, lead_time, self._sl)
                target = mu_forecast * lead_time + ss
                q = max(0.0, target - effective)
                orders[sku_id] = round(q, 2)

        return orders

    def get_top_supplier(
        self, sku_id: str, supplier_offers: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Return cheapest supplier (no negotiation in Baseline 2).

        Args:
            sku_id: SKU identifier.
            supplier_offers: List of supplier offer dicts.

        Returns:
            The offer with the lowest unit cost.
        """
        if not supplier_offers:
            return None
        return min(supplier_offers, key=lambda o: float(o.get("unit_cost", 9999.0)))

    @staticmethod
    def build_feature_matrix(
        demand_history: dict[str, np.ndarray],
        current_day: int,
    ) -> pd.DataFrame:
        """Build the feature matrix X consumed by fit() and predict_demand().

        Args:
            demand_history: ``{sku_id: np.ndarray}``.
            current_day: Current simulation day (for calendar features).

        Returns:
            DataFrame with one row per SKU and feature columns.
        """
        rows = []
        day_of_week = current_day % 7
        day_of_year = (current_day % 365) + 1
        is_weekend = day_of_week >= 5
        is_holiday = day_of_year in {100, 101, 102, 175, 176, 177}

        for sku_id, series in demand_history.items():
            recent7 = series[-7:] if len(series) >= 7 else series
            recent28 = series[-28:] if len(series) >= 28 else series
            rows.append(
                {
                    "sku_id": sku_id,
                    "rolling_7d_mean": float(np.mean(recent7)),
                    "rolling_7d_std": float(np.std(recent7) + 1e-8),
                    "rolling_28d_mean": float(np.mean(recent28)),
                    "day_of_week": day_of_week,
                    "day_of_year": day_of_year,
                    "is_holiday": int(is_holiday),
                    "is_weekend": int(is_weekend),
                }
            )
        return pd.DataFrame(rows)
