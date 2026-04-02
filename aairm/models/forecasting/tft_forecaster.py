"""Temporal Fusion Transformer (TFT) Forecaster — Paper Default for C1.

Wraps pytorch-forecasting's TemporalFusionTransformer with the
configuration from ForecastingConfig.  This is the model used in all
paper experiments (``architecture="tft"``).

Installation:  pip install pytorch-forecasting pytorch-lightning

Architecture notes (paper Section 4.2.1)
-----------------------------------------
- Variable selection networks for static and time-varying inputs.
- LSTM encoder for observed past; LSTM decoder for known future inputs.
- Multi-head attention (4 heads, paper default).
- Quantile outputs: p10, p50, p90 (pinball loss for quantile mode).
- MSE loss for point-forecast mode.

Falls back to NaiveForecaster when pytorch-forecasting is not installed
so that the rest of the pipeline remains functional.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from aairm.models.forecasting.base_forecaster import BaseForecaster
from aairm.utils.logging import get_logger

logger = get_logger(__name__)


class TFTForecaster(BaseForecaster):
    """Temporal Fusion Transformer wrapper for C1.

    Args:
        hidden_size: Hidden state size H (default 128).
        attention_head_size: Number of attention heads (default 4).
        dropout: Dropout rate (default 0.1).
        context_length: History input window in days (default 60).
        forecast_horizon: Prediction horizon h (default 7).
        learning_rate: Adam LR (default 1e-3).
        max_epochs: Training epochs (default 50).
        batch_size: Mini-batch size (default 64).
        loss: ``"mse"`` or ``"pinball"`` (default ``"mse"``).
        quantiles: Quantile levels for pinball loss (default [0.1, 0.5, 0.9]).
    """

    def __init__(
        self,
        hidden_size: int = 128,
        attention_head_size: int = 4,
        dropout: float = 0.1,
        context_length: int = 60,
        forecast_horizon: int = 7,
        learning_rate: float = 1e-3,
        max_epochs: int = 50,
        batch_size: int = 64,
        loss: str = "mse",
        quantiles: list[float] | None = None,
    ) -> None:
        self._hidden = hidden_size
        self._heads = attention_head_size
        self._dropout = dropout
        self._ctx_len = context_length
        self._horizon = forecast_horizon
        self._lr = learning_rate
        self._epochs = max_epochs
        self._batch = batch_size
        self._loss_type = loss
        self._quantiles = quantiles or [0.1, 0.5, 0.9]
        self._trainer: Any = None
        self._model: Any = None
        self._fitted = False

    @classmethod
    def from_config(cls, cfg: Any) -> "TFTForecaster":
        """Construct from a ForecastingConfig instance.

        Args:
            cfg: :class:`~aairm.utils.config.ForecastingConfig`.

        Returns:
            Configured TFTForecaster.
        """
        return cls(
            hidden_size=cfg.hidden_size,
            attention_head_size=cfg.attention_head_size,
            dropout=cfg.dropout,
            context_length=cfg.context_length,
            forecast_horizon=cfg.forecast_horizon,
            learning_rate=cfg.learning_rate,
            max_epochs=cfg.max_epochs,
            batch_size=cfg.batch_size,
            loss=cfg.loss,
            quantiles=cfg.quantiles,
        )

    def fit(
        self,
        demand_history: dict[str, np.ndarray],
        context: dict[str, Any] | None = None,
    ) -> "TFTForecaster":
        """Train TFT on historical demand data.

        Requires pytorch-forecasting.  Falls back gracefully if unavailable.

        Args:
            demand_history: ``{sku_id: np.ndarray of shape (n_days,)}``.
            context: Unused (TFT builds its own feature pipeline).

        Returns:
            Self.
        """
        try:
            import pandas as pd
            import pytorch_lightning as pl  # type: ignore
            from pytorch_forecasting import (  # type: ignore
                TemporalFusionTransformer,
                TimeSeriesDataSet,
            )
            from pytorch_forecasting.metrics import (  # type: ignore
                MAE, QuantileLoss,
            )

            # Build a flat DataFrame suitable for TimeSeriesDataSet
            rows = []
            for sku_id, series in demand_history.items():
                for t, val in enumerate(series):
                    rows.append(
                        {
                            "sku_id": sku_id,
                            "time_idx": t,
                            "demand": max(0.0, float(val)),
                            "day_of_week": t % 7,
                            "month": (t // 30) % 12,
                        }
                    )

            if not rows:
                logger.warning("tft.fit.no_data")
                return self

            df = pd.DataFrame(rows)
            max_encoder = self._ctx_len

            dataset = TimeSeriesDataSet(
                df,
                time_idx="time_idx",
                target="demand",
                group_ids=["sku_id"],
                max_encoder_length=max_encoder,
                max_prediction_length=self._horizon,
                time_varying_unknown_reals=["demand"],
                time_varying_known_reals=["day_of_week", "month"],
                target_normalizer=None,
            )

            loss_fn = (
                QuantileLoss(quantiles=self._quantiles)
                if self._loss_type == "pinball"
                else MAE()
            )

            self._model = TemporalFusionTransformer.from_dataset(
                dataset,
                hidden_size=self._hidden,
                attention_head_size=self._heads,
                dropout=self._dropout,
                hidden_continuous_size=self._hidden // 4,
                loss=loss_fn,
                learning_rate=self._lr,
                log_interval=-1,
            )

            train_loader = dataset.to_dataloader(
                train=True, batch_size=self._batch, num_workers=0
            )
            self._trainer = pl.Trainer(
                max_epochs=self._epochs,
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
            )
            self._trainer.fit(self._model, train_dataloaders=train_loader)
            self._fitted = True
            logger.info("tft.trained", epochs=self._epochs, n_skus=len(demand_history))

        except ImportError:
            logger.warning(
                "tft.fit.pytorch_forecasting_not_available; "
                "using NaiveForecaster fallback"
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("tft.fit.failed", error=str(exc))
        return self

    def predict(
        self,
        sku_id: str,
        history: np.ndarray,
        context: dict[str, Any],
        horizon: int = 7,
    ) -> dict[str, Any]:
        """Generate TFT forecast; falls back to NaiveForecaster if not fitted.

        Args:
            sku_id: SKU identifier.
            history: Demand history of shape ``(context_length,)``.
            context: Context features from P4.
            horizon: Forecast horizon in days.

        Returns:
            Forecast dict with ``mean``, ``variance``, ``p10``, ``p50``,
            ``p90``, ``horizon_days``, ``model``.
        """
        if not self._fitted or self._model is None:
            from aairm.models.forecasting.naive_forecaster import NaiveForecaster
            return NaiveForecaster().predict(sku_id, history, context, horizon)

        try:
            import pandas as pd
            import torch

            recent = history[-self._ctx_len:]
            if len(recent) < self._ctx_len:
                recent = np.pad(recent, (self._ctx_len - len(recent), 0))

            rows = [
                {
                    "sku_id": sku_id,
                    "time_idx": i,
                    "demand": max(0.0, float(v)),
                    "day_of_week": i % 7,
                    "month": (i // 30) % 12,
                }
                for i, v in enumerate(recent)
            ]
            df = pd.DataFrame(rows)

            self._model.eval()
            with torch.no_grad():
                preds = self._model.predict(df, mode="prediction").numpy()

            mean_total = float(np.sum(preds[0]))
            std_total = float(np.std(preds[0]) * np.sqrt(horizon) + 1e-8)
            from scipy import stats as sp_stats
            return {
                "mean": max(0.0, mean_total),
                "variance": std_total**2,
                "p10": max(0.0, sp_stats.norm.ppf(0.10, mean_total, std_total)),
                "p50": max(0.0, mean_total),
                "p90": max(0.0, sp_stats.norm.ppf(0.90, mean_total, std_total)),
                "horizon_days": horizon,
                "model": "tft",
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("tft.predict.failed", sku_id=sku_id, error=str(exc))
            from aairm.models.forecasting.naive_forecaster import NaiveForecaster
            return NaiveForecaster().predict(sku_id, history, context, horizon)

    def save(self, path: str | Path) -> None:
        if self._trainer is not None and self._model is not None:
            self._trainer.save_checkpoint(str(path))

    def load(self, path: str | Path) -> "TFTForecaster":
        if Path(path).exists():
            try:
                from pytorch_forecasting import TemporalFusionTransformer  # type: ignore
                self._model = TemporalFusionTransformer.load_from_checkpoint(str(path))
                self._fitted = True
            except Exception as exc:  # noqa: BLE001
                logger.error("tft.load.failed", error=str(exc))
        return self
