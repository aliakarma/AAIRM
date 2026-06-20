"""WRMSSE — Weighted Root Mean Squared Scaled Error (official M5 metric).

Used in the external validation on the M5 Forecasting Competition data
(product.tex Table 11), where AAIRM's C1 forecaster attains a centralized
WRMSSE of 0.66. RMSSE scales each series' forecast error by the in-sample
one-step naive error, and series are weighted by their cumulative dollar
sales over the trailing evaluation window.

References
----------
Makridakis et al. (2022) M5 Accuracy competition; product.tex Section
"External Validation on Public Retail Data".
"""

from __future__ import annotations

import numpy as np


def rmsse(
    y_train: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """Root Mean Squared Scaled Error for one series.

    .. math::
        \\text{RMSSE} = \\sqrt{\\frac{\\frac{1}{h}\\sum_t (y_t - \\hat{y}_t)^2}
        {\\frac{1}{n-1}\\sum_{t=2}^{n} (y_t - y_{t-1})^2}}

    Args:
        y_train: In-sample history used for the naive-error denominator.
        y_true: Out-of-sample actuals over the forecast horizon.
        y_pred: Forecasts aligned with ``y_true``.

    Returns:
        RMSSE (>= 0). Returns 0.0 for degenerate (flat) training series.
    """
    y_train = np.asarray(y_train, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_train.size < 2:
        return 0.0
    denom = np.mean(np.diff(y_train) ** 2)
    if denom <= 0:
        return 0.0
    num = np.mean((y_true - y_pred) ** 2)
    return float(np.sqrt(num / denom))


def wrmsse(
    series_train: dict[str, np.ndarray],
    series_true: dict[str, np.ndarray],
    series_pred: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted RMSSE across all series (the official M5 aggregate).

    Args:
        series_train: ``{series_id: in-sample history}``.
        series_true: ``{series_id: out-of-sample actuals}``.
        series_pred: ``{series_id: forecasts}``.
        weights: ``{series_id: weight}``; defaults to equal weights. In M5
            these are cumulative-dollar-sales shares (sum to 1).

    Returns:
        WRMSSE (lower is better; the paper reports 0.66 on M5).
    """
    ids = list(series_true.keys())
    if not ids:
        return 0.0
    if weights is None:
        weights = {i: 1.0 / len(ids) for i in ids}
    total_w = sum(weights.get(i, 0.0) for i in ids) or 1.0

    acc = 0.0
    for i in ids:
        w = weights.get(i, 0.0) / total_w
        acc += w * rmsse(
            series_train.get(i, np.array([])),
            series_true[i],
            series_pred.get(i, np.zeros_like(series_true[i])),
        )
    return float(acc)


def sales_weights(series_true: dict[str, np.ndarray],
                  prices: dict[str, float] | None = None) -> dict[str, float]:
    """Cumulative-dollar-sales weights for WRMSSE aggregation."""
    prices = prices or {}
    raw = {i: float(np.sum(v)) * prices.get(i, 1.0) for i, v in series_true.items()}
    total = sum(raw.values()) or 1.0
    return {i: w / total for i, w in raw.items()}
