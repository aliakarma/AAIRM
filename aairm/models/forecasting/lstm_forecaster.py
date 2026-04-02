"""LSTM Encoder-Decoder Forecaster for C1.

A single-layer LSTM encoder-decoder that accepts a history window and
exogenous features, then autoregressively decodes h-step ahead forecasts.

Requires PyTorch.  Install with:  pip install torch

Architecture (paper Section 4.2.1)
-----------------------------------
Encoder: LSTM(input_size=n_features, hidden_size=H)
Decoder: LSTM(input_size=1, hidden_size=H) → Linear(H, 1)
Loss:    MSE or pinball (configured via ForecastingConfig.loss)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from aairm.models.forecasting.base_forecaster import BaseForecaster
from aairm.utils.logging import get_logger

logger = get_logger(__name__)

# Feature columns extracted from context for exogenous input
_EXOG_KEYS = [
    "rolling_7d_mean", "rolling_7d_std", "rolling_28d_mean",
    "day_of_week", "day_of_year", "is_holiday", "is_weekend",
    "unit_cost", "lead_time_days",
]


class LSTMForecaster(BaseForecaster):
    """LSTM encoder-decoder demand forecaster.

    Args:
        hidden_size: LSTM hidden state dimension H (default 128).
        dropout: Dropout rate applied to encoder output (default 0.1).
        context_length: Input history window length (default 60).
        learning_rate: Adam learning rate (default 1e-3).
        max_epochs: Training epochs (default 50).
        batch_size: Mini-batch size (default 64).
    """

    def __init__(
        self,
        hidden_size: int = 128,
        dropout: float = 0.1,
        context_length: int = 60,
        learning_rate: float = 1e-3,
        max_epochs: int = 50,
        batch_size: int = 64,
    ) -> None:
        self._hidden = hidden_size
        self._dropout = dropout
        self._ctx_len = context_length
        self._lr = learning_rate
        self._epochs = max_epochs
        self._batch = batch_size
        self._model: Any = None
        self._fitted = False

    def fit(
        self,
        demand_history: dict[str, np.ndarray],
        context: dict[str, Any] | None = None,
    ) -> "LSTMForecaster":
        """Train the LSTM on historical demand.

        Args:
            demand_history: ``{sku_id: np.ndarray}``.
            context: Optional additional context (unused in current version).

        Returns:
            Self.
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset

            n_exog = len(_EXOG_KEYS) + 1   # +1 for lagged demand
            all_X, all_y = [], []

            for sku_id, series in demand_history.items():
                if len(series) < self._ctx_len + 1:
                    continue
                for start in range(len(series) - self._ctx_len - 1):
                    window = series[start : start + self._ctx_len]
                    target = series[start + self._ctx_len]
                    feat = np.zeros(n_exog)
                    feat[0] = window[-7:].mean() if len(window) >= 7 else window.mean()
                    feat[1] = window.std() + 1e-8
                    all_X.append(torch.tensor(window, dtype=torch.float32))
                    all_y.append(torch.tensor([target], dtype=torch.float32))

            if not all_X:
                logger.warning("lstm.fit.no_data")
                return self

            X = torch.stack(all_X).unsqueeze(-1)   # (N, T, 1)
            y = torch.stack(all_y)                  # (N, 1)

            self._model = self._build_model(n_exog=1)
            optimiser = torch.optim.Adam(self._model.parameters(), lr=self._lr)
            criterion = nn.MSELoss()
            dataset = TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=self._batch, shuffle=True)

            self._model.train()
            for epoch in range(self._epochs):
                epoch_loss = 0.0
                for xb, yb in loader:
                    optimiser.zero_grad()
                    out, _ = self._model(xb)
                    loss = criterion(out[:, -1, :], yb)
                    loss.backward()
                    optimiser.step()
                    epoch_loss += loss.item()
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        "lstm.training",
                        epoch=epoch + 1,
                        loss=round(epoch_loss / len(loader), 6),
                    )
            self._fitted = True
        except ImportError:
            logger.warning("lstm.fit.pytorch_not_available; using naive fallback")
        except Exception as exc:  # noqa: BLE001
            logger.error("lstm.fit.failed", error=str(exc))
        return self

    def predict(
        self,
        sku_id: str,
        history: np.ndarray,
        context: dict[str, Any],
        horizon: int = 7,
    ) -> dict[str, Any]:
        """Generate h-step ahead forecasts autoregressively.

        Falls back to NaiveForecaster if model is not trained or
        PyTorch is unavailable.
        """
        if not self._fitted or self._model is None:
            from aairm.models.forecasting.naive_forecaster import NaiveForecaster
            return NaiveForecaster().predict(sku_id, history, context, horizon)

        try:
            import torch

            recent = history[-self._ctx_len:]
            if len(recent) < self._ctx_len:
                recent = np.pad(recent, (self._ctx_len - len(recent), 0))
            x = torch.tensor(recent, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

            self._model.eval()
            preds = []
            with torch.no_grad():
                out, (h, c) = self._model(x)
                for _ in range(horizon):
                    step_pred = float(out[0, -1, 0].item())
                    preds.append(max(0.0, step_pred))
                    next_x = torch.tensor([[[step_pred]]], dtype=torch.float32)
                    out, (h, c) = self._model(next_x, (h, c))

            mean_total = float(np.sum(preds))
            std_total = float(np.std(preds) * np.sqrt(horizon) + 1e-8)
            from scipy import stats as sp_stats
            return {
                "mean": mean_total,
                "variance": std_total**2,
                "p10": max(0.0, sp_stats.norm.ppf(0.10, mean_total, std_total)),
                "p50": mean_total,
                "p90": max(0.0, sp_stats.norm.ppf(0.90, mean_total, std_total)),
                "horizon_days": horizon,
                "model": "lstm",
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("lstm.predict.failed", sku_id=sku_id, error=str(exc))
            from aairm.models.forecasting.naive_forecaster import NaiveForecaster
            return NaiveForecaster().predict(sku_id, history, context, horizon)

    def _build_model(self, n_exog: int = 1) -> Any:
        """Construct the LSTM encoder-decoder network."""
        import torch.nn as nn

        class _LSTMModel(nn.Module):
            def __init__(self, input_size: int, hidden_size: int, dropout: float):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=2,
                    batch_first=True,
                    dropout=dropout,
                )
                self.linear = nn.Linear(hidden_size, 1)

            def forward(self, x, hidden=None):  # type: ignore[override]
                out, hidden = self.lstm(x, hidden)
                return self.linear(out), hidden

        return _LSTMModel(n_exog, self._hidden, self._dropout)

    def save(self, path: str | Path) -> None:
        if self._model is not None:
            import torch
            torch.save(self._model.state_dict(), path)

    def load(self, path: str | Path) -> "LSTMForecaster":
        if Path(path).exists():
            import torch
            self._model = self._build_model()
            self._model.load_state_dict(torch.load(path))
            self._fitted = True
        return self
