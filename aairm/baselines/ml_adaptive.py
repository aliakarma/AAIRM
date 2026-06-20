"""ML + Adaptive Policy Baseline (Baseline 3).

The strongest *non-agentic* comparator in the paper (Section 5.2). It
strengthens Baseline 2 (ML + Static) by dynamically updating the safety
stock from the gradient-boosted forecaster's residuals instead of holding a
fixed safety factor:

    SS_{i,t} = z * sigma_hat_{i,t}

where ``sigma_hat_{i,t}`` is the rolling standard deviation of the most
recent forecast errors for SKU ``i`` and ``z`` is the service-level safety
factor. This demand-adaptive update removes the systematic-overstocking
pathology of BL2 and is therefore a fairer comparator.

Canonical paper performance (Table 3, 100-SKU, 10 seeds):
    stockout_rate = 2.84 %
    fill_rate     = 97.16 %
    avg_inventory = 6.43
    total_cost    = 0.962  (normalized to BL1)
    spoilage_rate = 5.41 %

References
----------
product.tex Section "Baselines and Evaluation Metrics" (Baseline 3).
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np
import pandas as pd

from aairm.utils.logging import get_logger
from aairm.utils.math_utils import rop, safety_stock

logger = get_logger(__name__)


class MLAdaptivePolicy:
    """Non-agentic ML forecaster with demand-adaptive safety stock (Baseline 3).

    Args:
        service_level: Target cycle service level (default 0.95).
        holding_cost_rate: Annual holding cost rate (default 0.25).
        residual_window: Number of recent periods over which the forecast-error
            standard deviation ``sigma_hat`` is estimated (default 30).
        n_estimators: XGBoost tree count (paper: 200 at 100-SKU, 300 at 500-SKU).
    """

    def __init__(
        self,
        service_level: float = 0.95,
        holding_cost_rate: float = 0.25,
        residual_window: int = 30,
        n_estimators: int = 200,
    ) -> None:
        self._sl = service_level
        self._h = holding_cost_rate
        self._window = int(residual_window)
        self._n_estimators = int(n_estimators)

        self._model: Any = None
        self._fitted = False
        self._lead_times: dict[str, float] = {}
        self._unit_costs: dict[str, float] = {}
        # Rolling forecast-error buffers per SKU (the adaptive component).
        self._residuals: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._window)
        )

    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        lead_times: dict[str, float] | None = None,
        unit_costs: dict[str, float] | None = None,
    ) -> "MLAdaptivePolicy":
        """Train the gradient-boosted forecaster (XGBoost, LightGBM fallback)."""
        self._lead_times = lead_times or {}
        self._unit_costs = unit_costs or {}

        self._model = self._build_regressor()
        if self._model is not None and not X_train.empty:
            try:
                self._model.fit(X_train, y_train)
                logger.info("ml_adaptive.trained", n_samples=len(X_train),
                            backend=type(self._model).__name__)
            except Exception as exc:  # noqa: BLE001
                logger.error("ml_adaptive.fit_failed", error=str(exc))
                self._model = None
        self._fitted = True
        return self

    def _build_regressor(self) -> Any:
        """Prefer XGBoost (paper), fall back to LightGBM, then naive."""
        try:
            import xgboost as xgb  # type: ignore

            return xgb.XGBRegressor(
                n_estimators=self._n_estimators,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                random_state=42,
                verbosity=0,
            )
        except ImportError:
            pass
        try:
            import lightgbm as lgb  # type: ignore

            return lgb.LGBMRegressor(
                n_estimators=self._n_estimators,
                learning_rate=0.05,
                random_state=42,
                verbose=-1,
            )
        except ImportError:
            logger.warning("ml_adaptive.no_gbm_backend; using rolling-mean fallback")
            return None

    # ------------------------------------------------------------------
    def predict_demand(self, X_today: pd.DataFrame) -> dict[str, float]:
        """Per-SKU next-period demand forecasts."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_demand().")
        if "sku_id" not in X_today.columns:
            raise ValueError("X_today must include a 'sku_id' column.")

        sku_ids = X_today["sku_id"].tolist()
        feats = X_today[[c for c in X_today.columns if c != "sku_id"]]

        if self._model is not None:
            try:
                preds = self._model.predict(feats)
            except Exception:  # noqa: BLE001
                preds = feats.get("rolling_7d_mean", pd.Series(np.ones(len(feats)) * 10.0)).values
        else:
            preds = feats.get("rolling_7d_mean", pd.Series(np.ones(len(feats)) * 10.0)).values

        return {s: max(0.0, float(p)) for s, p in zip(sku_ids, preds)}

    def observe_actuals(self, forecasts: dict[str, float], actuals: dict[str, float]) -> None:
        """Update rolling forecast-error buffers — the adaptive mechanism.

        Call once per period with the realized demand so that ``sigma_hat``
        tracks recent forecast volatility per SKU.
        """
        for sku, fc in forecasts.items():
            if sku in actuals:
                self._residuals[sku].append(float(actuals[sku]) - float(fc))

    def _sigma_hat(self, sku_id: str, fallback: float) -> float:
        """Rolling std of recent forecast errors; ``fallback`` until warmed up."""
        buf = self._residuals.get(sku_id)
        if buf is not None and len(buf) >= 2:
            return float(np.std(buf) + 1e-8)
        return float(fallback)

    # ------------------------------------------------------------------
    def get_orders(
        self,
        inventory_snapshot: dict[str, dict[str, Any]],
        demand_forecasts: dict[str, float],
    ) -> dict[str, float]:
        """Order quantities using ROP with *adaptive* safety stock SS = z*sigma_hat."""
        if not self._fitted:
            raise RuntimeError("Call fit() before get_orders().")

        orders: dict[str, float] = {}
        for sku_id, rec in inventory_snapshot.items():
            effective = float(rec.get("effective_available", 0.0))
            mu_forecast = demand_forecasts.get(sku_id, 10.0)
            sigma_hist = float(rec.get("demand_std_daily", 2.0))
            sigma_hat = self._sigma_hat(sku_id, fallback=sigma_hist)
            lead_time = self._lead_times.get(sku_id, 5.0)

            reorder_pt = rop(mu_forecast, sigma_hat, lead_time, self._sl)
            if effective <= reorder_pt:
                ss = safety_stock(sigma_hat, lead_time, self._sl)
                target = mu_forecast * lead_time + ss
                q = max(0.0, target - effective)
                orders[sku_id] = round(q, 2)
        return orders

    def get_top_supplier(
        self, sku_id: str, supplier_offers: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Cheapest supplier (no negotiation in Baseline 3)."""
        if not supplier_offers:
            return None
        return min(supplier_offers, key=lambda o: float(o.get("unit_cost", 9999.0)))

    @staticmethod
    def build_feature_matrix(
        demand_history: dict[str, np.ndarray], current_day: int
    ) -> pd.DataFrame:
        """Reuse the BL2 feature schema for parity of comparison."""
        from aairm.baselines.ml_static import MLStaticPolicy

        return MLStaticPolicy.build_feature_matrix(demand_history, current_day)
