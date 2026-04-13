"""LightGBM Demand Forecaster.

Uses LightGBM for demand forecasting with lag and calendar features.
Falls back to empirical forecast if calibration fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import structlog
from scipy import stats

from aairm.models.forecasting.base_forecaster import BaseForecaster

logger = structlog.get_logger(__name__)


class ForecastCalibrationError(Exception):
    """Raised when forecasted mean is not within 50% of actual mean."""


class LightGBMForecaster(BaseForecaster):
    """LightGBM-based demand forecaster with empirical fallback."""

    def __init__(self) -> None:
        self.model: lgb.Booster | None = None
        self.fitted = False
        self.fallback_active = False

    def pre_train(
        self,
        real_demand_history: pd.DataFrame,
        warm_up_days: int = 60,
    ) -> "LightGBMForecaster":
        """Pre-train on real demand history with validation.

        Args:
            real_demand_history: DataFrame with columns [sku_id, date, demand].
            warm_up_days: Number of days for training.

        Returns:
            Self.

        Raises:
            ForecastCalibrationError: If validation fails.
        """
        # Create features
        df = real_demand_history.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["sku_id", "date"])
        
        # Lag features
        df["lag_1"] = df.groupby("sku_id")["demand"].shift(1)
        df["lag_7"] = df.groupby("sku_id")["demand"].shift(7)
        df["lag_14"] = df.groupby("sku_id")["demand"].shift(14)
        df["lag_28"] = df.groupby("sku_id")["demand"].shift(28)
        
        # Calendar features
        df["day_of_week"] = df["date"].dt.day_of_week
        df["week_of_year"] = df["date"].dt.isocalendar().week
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        
        # Rolling features
        df["rolling_mean_7"] = df.groupby("sku_id")["demand"].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        df["rolling_mean_14"] = df.groupby("sku_id")["demand"].transform(
            lambda x: x.rolling(14, min_periods=1).mean()
        )
        df["rolling_std_7"] = df.groupby("sku_id")["demand"].transform(
            lambda x: x.rolling(7, min_periods=1).std()
        )
        
        # Drop NaN
        df = df.dropna()
        
        # Train on first warm_up_days - 10
        train_end = warm_up_days - 10
        train_df = df[df["date"].dt.day <= train_end]
        
        # Validation on days 51-60
        val_df = df[(df["date"].dt.day >= 51) & (df["date"].dt.day <= 60)]
        
        if train_df.empty or val_df.empty:
            raise ValueError("Insufficient data for training/validation")
        
        feature_cols = [
            "lag_1", "lag_7", "lag_14", "lag_28",
            "day_of_week", "week_of_year", "is_weekend",
            "rolling_mean_7", "rolling_mean_14", "rolling_std_7"
        ]
        
        X_train = train_df[feature_cols]
        y_train = train_df["demand"]
        X_val = val_df[feature_cols]
        y_val = val_df["demand"]
        
        # Train LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
        }
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)],
        )
        
        # Validate
        val_pred = self.model.predict(X_val)
        pred_mean = np.mean(val_pred)
        actual_mean = np.mean(y_val)
        
        if not (0.5 * actual_mean <= pred_mean <= 1.5 * actual_mean):
            raise ForecastCalibrationError(
                f"Predicted mean {pred_mean:.2f} not within 50% of actual {actual_mean:.2f}"
            )
        
        self.fitted = True
        self.fallback_active = False
        logger.info("lightgbm.pre_train.success", pred_mean=pred_mean, actual_mean=actual_mean)
        return self

    def fit(
        self,
        demand_history: dict[str, np.ndarray],
        context: dict[str, Any] | None = None,
    ) -> "LightGBMForecaster":
        """Fit method for compatibility (calls pre_train if DataFrame provided)."""
        if context and "demand_df" in context:
            return self.pre_train(context["demand_df"])
        # If no DataFrame, use empirical
        self.fitted = False
        self.fallback_active = True
        return self

    def predict(
        self,
        sku_id: str,
        history: np.ndarray,
        context: dict[str, Any],
        horizon: int = 7,
    ) -> np.ndarray:
        """Predict demand for horizon days."""
        if not self.fitted or self.fallback_active:
            return self._fallback_forecast(history, horizon)
        
        # Use LightGBM
        forecasts = []
        current_history = history.copy()
        
        for _ in range(horizon):
            # Create features
            lag_1 = current_history[-1] if len(current_history) >= 1 else 0
            lag_7 = current_history[-7] if len(current_history) >= 7 else np.mean(current_history[-len(current_history):])
            lag_14 = current_history[-14] if len(current_history) >= 14 else np.mean(current_history[-len(current_history):])
            lag_28 = current_history[-28] if len(current_history) >= 28 else np.mean(current_history[-len(current_history):])
            
            rolling_mean_7 = np.mean(current_history[-7:]) if len(current_history) >= 7 else np.mean(current_history)
            rolling_mean_14 = np.mean(current_history[-14:]) if len(current_history) >= 14 else np.mean(current_history)
            rolling_std_7 = np.std(current_history[-7:]) if len(current_history) >= 7 else np.std(current_history)
            
            # Assume context has date or day
            day_of_week = context.get("day_of_week", 0)
            week_of_year = context.get("week_of_year", 1)
            is_weekend = 1 if day_of_week in [5, 6] else 0
            
            features = np.array([
                lag_1, lag_7, lag_14, lag_28,
                day_of_week, week_of_year, is_weekend,
                rolling_mean_7, rolling_mean_14, rolling_std_7
            ]).reshape(1, -1)
            
            pred = self.model.predict(features)[0]
            forecasts.append(max(0, pred))
            current_history = np.append(current_history, pred)
        
        logger.info("forecast.mode", mode="lgbm", sku_id=sku_id, horizon=horizon)
        return np.array(forecasts)

    def _fallback_forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Empirical forecast using last 14 days."""
        if len(history) == 0:
            return np.zeros(horizon)
        
        last_14 = history[-14:] if len(history) >= 14 else history
        mean_14 = np.mean(last_14)
        std_14 = np.std(last_14)
        
        # Mean + 1.96 * std as upper bound
        forecast = mean_14 + 1.96 * std_14
        forecasts = np.full(horizon, max(0, forecast))
        
        logger.info("forecast.mode", mode="empirical_fallback", horizon=horizon)
        return forecasts

    def forecast_uncertainty(self) -> tuple[float, float]:
        """Return confidence interval [low, high] for forecast."""
        if not self.fitted or self.fallback_active:
            # For empirical, use mean ± std
            return (-1.96, 1.96)  # standardized
        
        # For LightGBM, could use quantile regression, but for now placeholder
        return (-1.0, 1.0)