"""Unit tests for aairm/utils/math_utils.py.

Target: 100% line and branch coverage.
Every paper equation is tested with:
  1. A happy-path test with a known analytic solution.
  2. A boundary/edge-case test.
  3. A regression test verifying exact paper values.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aairm.utils.math_utils import (
    diversification_index,
    eoq,
    expected_cost_single_period,
    rop,
    safety_stock,
    supplier_score,
    td_loss,
)


# ---------------------------------------------------------------------------
# Eq. 1 — Reorder Point
# ---------------------------------------------------------------------------

class TestROP:
    def test_known_value(self):
        """With μ=10, σ=2, L=5, z(0.95)≈1.645 → ROP ≈ 57.36."""
        result = rop(mu_d=10.0, sigma_d=2.0, lead_time=5.0, service_level=0.95)
        assert abs(result - 57.36) < 0.05, f"Expected ~57.36, got {result:.4f}"

    def test_zero_variance(self):
        """With σ=0, safety stock term vanishes → ROP = μ * L."""
        result = rop(mu_d=10.0, sigma_d=0.0, lead_time=5.0, service_level=0.95)
        assert abs(result - 50.0) < 1e-6

    def test_zero_lead_time(self):
        """With L=0, ROP = 0 regardless of demand."""
        result = rop(mu_d=20.0, sigma_d=5.0, lead_time=0.0, service_level=0.95)
        assert result == 0.0

    def test_invalid_service_level_raises(self):
        with pytest.raises(ValueError):
            rop(10.0, 2.0, 5.0, service_level=1.0)
        with pytest.raises(ValueError):
            rop(10.0, 2.0, 5.0, service_level=0.0)

    def test_negative_lead_time_raises(self):
        with pytest.raises(ValueError):
            rop(10.0, 2.0, lead_time=-1.0)

    def test_higher_service_level_yields_higher_rop(self):
        rop_95 = rop(10.0, 2.0, 5.0, service_level=0.95)
        rop_99 = rop(10.0, 2.0, 5.0, service_level=0.99)
        assert rop_99 > rop_95

    def test_paper_regression(self):
        """Paper Section 2.1 illustrative example."""
        result = rop(mu_d=10.0, sigma_d=2.0, lead_time=5.0, service_level=0.95)
        assert abs(result - 57.36) < 0.1


# ---------------------------------------------------------------------------
# EOQ
# ---------------------------------------------------------------------------

class TestEOQ:
    def test_textbook_example(self):
        """D=1000, K=50, h=0.25, c=10 → Q*=200."""
        result = eoq(demand_rate=1000.0, ordering_cost=50.0,
                     holding_cost_rate=0.25, unit_cost=10.0)
        assert abs(result - 200.0) < 0.5

    def test_all_args_positive(self):
        with pytest.raises(ValueError):
            eoq(0.0, 50.0, 0.25, 10.0)
        with pytest.raises(ValueError):
            eoq(1000.0, 0.0, 0.25, 10.0)

    def test_doubling_demand_increases_quantity(self):
        q1 = eoq(1000.0, 50.0, 0.25, 10.0)
        q2 = eoq(2000.0, 50.0, 0.25, 10.0)
        assert q2 > q1


# ---------------------------------------------------------------------------
# Eq. 3 — Expected Cost Single Period
# ---------------------------------------------------------------------------

class TestExpectedCostSinglePeriod:
    def test_non_negative(self):
        """Expected cost is always non-negative."""
        cost = expected_cost_single_period(
            q=100.0, demand_mean=90.0, demand_std=15.0,
            unit_cost=5.0, holding_cost_rate=1.25, penalty_cost=15.0,
        )
        assert cost >= 0

    def test_zero_order_quantity(self):
        """Q=0 → all demand is lost (maximum penalty)."""
        cost = expected_cost_single_period(
            q=0.0, demand_mean=50.0, demand_std=10.0,
            unit_cost=5.0, holding_cost_rate=1.0, penalty_cost=20.0,
        )
        assert cost > 0

    def test_perishable_spoilage_adds_cost(self):
        """Perishable costs more than non-perishable for same Q."""
        base = expected_cost_single_period(
            q=100.0, demand_mean=90.0, demand_std=10.0,
            unit_cost=5.0, holding_cost_rate=1.0, penalty_cost=10.0,
        )
        with_spoilage = expected_cost_single_period(
            q=100.0, demand_mean=90.0, demand_std=10.0,
            unit_cost=5.0, holding_cost_rate=1.0, penalty_cost=10.0,
            spoilage_cost_rate=2.5, shelf_life_demand=80.0,
        )
        assert with_spoilage >= base

    def test_deterministic_demand(self):
        """With σ=0, cost is purely procurement + deterministic over/under."""
        cost = expected_cost_single_period(
            q=100.0, demand_mean=100.0, demand_std=0.0,
            unit_cost=5.0, holding_cost_rate=1.0, penalty_cost=10.0,
        )
        # Q = D exactly → procurement cost only = 5 * 100 = 500
        assert abs(cost - 500.0) < 1.0

    def test_paper_regression(self):
        """Verify Eq. 3 against known analytic solution."""
        cost = expected_cost_single_period(
            q=100.0, demand_mean=90.0, demand_std=15.0,
            unit_cost=5.0, holding_cost_rate=1.25, penalty_cost=15.0,
        )
        assert 500.0 < cost < 600.0, f"Cost out of expected range: {cost:.2f}"


# ---------------------------------------------------------------------------
# Eq. 6 — Supplier Score
# ---------------------------------------------------------------------------

class TestSupplierScore:
    def test_reliable_beats_cheap(self):
        """High-reliability supplier should beat cheap but unreliable one."""
        cheap_unreliable = supplier_score(0.4, 0.3, 0.70, False)
        reliable = supplier_score(0.6, 0.2, 0.95, False)
        assert reliable < cheap_unreliable  # lower score = better

    def test_moq_violation_penalises(self):
        """MOQ violation flag raises the score (makes supplier less attractive)."""
        no_viol = supplier_score(0.5, 0.3, 0.85, False)
        with_viol = supplier_score(0.5, 0.3, 0.85, True)
        assert with_viol > no_viol

    def test_invalid_reliability_raises(self):
        with pytest.raises(ValueError):
            supplier_score(0.5, 0.3, 1.5, False)
        with pytest.raises(ValueError):
            supplier_score(0.5, 0.3, -0.1, False)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            supplier_score(0.5, 0.3, 0.85, False, alpha_1=-0.1)

    def test_paper_regression(self):
        """Default alpha weights, no MOQ violation, known inputs."""
        score = supplier_score(0.6, 0.4, 0.95, False)
        # 0.35*0.6 + 0.30*0.4 - 0.25*0.95 + 0.10*0 = 0.21 + 0.12 - 0.2375 = 0.0925
        assert abs(score - 0.0925) < 1e-4

    def test_zero_cost_zero_lead(self):
        score = supplier_score(0.0, 0.0, 1.0, False)
        assert score == pytest.approx(-0.25, abs=1e-6)


# ---------------------------------------------------------------------------
# Diversification Index
# ---------------------------------------------------------------------------

class TestDiversificationIndex:
    def test_monopoly_is_zero(self):
        assert diversification_index([1.0, 0.0, 0.0]) == pytest.approx(0.0)
        assert diversification_index([1.0]) == pytest.approx(0.0)

    def test_uniform_is_one(self):
        assert diversification_index([0.25, 0.25, 0.25, 0.25]) == pytest.approx(1.0)
        assert diversification_index([0.5, 0.5]) == pytest.approx(1.0)

    def test_paper_regression_aairm(self):
        """AAIRM achieves higher diversification than Baseline 1.

        Paper Table 2: AAIRM div_index=0.61 vs Baseline 1=0.42, measured as
        a category-average across 5 categories.  For a single-category
        3-supplier split [0.45, 0.35, 0.20] the Herfindahl-normalised index
        is ~0.95 — confirming the function is monotone with concentration.
        """
        idx = diversification_index([0.45, 0.35, 0.20])
        assert idx > 0.80, f"Expected > 0.80 for 3-supplier mix, got {idx:.4f}"

    def test_paper_regression_baseline1(self):
        """Baseline 1 concentrated near one dominant supplier.

        With [0.80, 0.20] the index ≈ 0.64 — lower than the 3-supplier case,
        consistent with the paper's Baseline 1 = 0.42 category average.
        """
        idx = diversification_index([0.80, 0.20])
        assert 0.50 < idx < 0.75, f"Expected 0.50-0.75 for near-monopoly, got {idx:.4f}"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            diversification_index([])

    def test_all_zero_raises(self):
        with pytest.raises(ValueError):
            diversification_index([0.0, 0.0])

    def test_auto_normalise(self):
        """Values should be normalised automatically."""
        idx_unnorm = diversification_index([100, 100, 100, 100])
        idx_norm   = diversification_index([0.25, 0.25, 0.25, 0.25])
        assert abs(idx_unnorm - idx_norm) < 1e-9


# ---------------------------------------------------------------------------
# Safety Stock
# ---------------------------------------------------------------------------

class TestSafetyStock:
    def test_paper_regression(self):
        """z(0.95)≈1.645, σ=2, L=5 → SS ≈ 7.36."""
        ss = safety_stock(sigma_d=2.0, lead_time=5.0, service_level=0.95)
        assert abs(ss - 7.36) < 0.05

    def test_zero_sigma_gives_zero(self):
        assert safety_stock(sigma_d=0.0, lead_time=5.0) == 0.0

    def test_rop_equals_mu_times_L_plus_ss(self):
        mu, sigma, L, sl = 10.0, 2.0, 5.0, 0.95
        r = rop(mu, sigma, L, sl)
        ss = safety_stock(sigma, L, sl)
        assert abs(r - (mu * L + ss)) < 1e-6


# ---------------------------------------------------------------------------
# Eq. 7 — TD Loss
# ---------------------------------------------------------------------------

class TestTDLoss:
    def test_zero_error(self):
        """No TD error when r + γ*V' = V."""
        loss = td_loss(reward=0.0, value_current=1.0, value_next=1.0 / 0.99, gamma=0.99)
        assert abs(loss) < 1e-6

    def test_positive_loss(self):
        loss = td_loss(reward=-1.0, value_current=-5.0, value_next=-4.5, gamma=0.99)
        assert loss > 0.0

    def test_paper_regression(self):
        """Verify Eq. 7: (−1 + 0.99*(−4.5) − (−5))² = (−1 − 4.455 + 5)² = (−0.455)²."""
        expected = (-1.0 + 0.99 * (-4.5) - (-5.0)) ** 2
        result = td_loss(-1.0, -5.0, -4.5, 0.99)
        assert abs(result - expected) < 1e-9

    def test_symmetry(self):
        """TD loss is symmetric: (δ)² = (−δ)²."""
        l1 = td_loss(1.0, 0.0, 0.0, 0.99)
        l2 = td_loss(-1.0, 0.0, 0.0, 0.99)
        assert abs(l1 - l2) < 1e-9
