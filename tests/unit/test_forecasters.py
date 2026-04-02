"""Unit tests for demand forecasting models."""

from __future__ import annotations

import numpy as np
import pytest

from aairm.models.forecasting.naive_forecaster import NaiveForecaster


@pytest.fixture
def sample_history():
    return np.random.default_rng(42).normal(50.0, 5.0, 60).clip(0)


def test_naive_predict_returns_dict(sample_history):
    f = NaiveForecaster()
    result = f.predict("SKU-1", sample_history, {}, horizon=7)
    required = {"mean", "variance", "p10", "p50", "p90", "horizon_days", "model"}
    assert required.issubset(result.keys())


def test_naive_predict_nonnegative(sample_history):
    f = NaiveForecaster()
    result = f.predict("SKU-1", sample_history, {}, horizon=7)
    assert result["mean"] >= 0
    assert result["p10"] >= 0
    assert result["p90"] >= result["p50"] >= result["p10"]


def test_naive_uses_context_if_provided(sample_history):
    f = NaiveForecaster()
    ctx = {"rolling_7d_mean": 60.0, "rolling_7d_std": 3.0}
    result = f.predict("SKU-1", sample_history, ctx, horizon=7)
    # Mean should be approximately 60 * 7 = 420
    assert abs(result["mean"] - 420.0) < 30.0


def test_naive_fit_is_noop(sample_history):
    f = NaiveForecaster()
    f2 = f.fit({"SKU-1": sample_history})
    assert f2 is f   # returns self


def test_naive_horizon_scaling(sample_history):
    f = NaiveForecaster()
    r7  = f.predict("SKU-1", sample_history, {}, horizon=7)
    r14 = f.predict("SKU-1", sample_history, {}, horizon=14)
    # 14-day mean should be approximately 2× the 7-day mean
    assert abs(r14["mean"] - 2.0 * r7["mean"]) < r7["mean"] * 0.05


def test_empty_history_does_not_crash():
    f = NaiveForecaster()
    result = f.predict("SKU-1", np.array([]), {}, horizon=7)
    assert result["mean"] >= 0
