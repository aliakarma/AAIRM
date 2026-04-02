"""Unit tests for aairm/utils/cost_functions.py."""

from __future__ import annotations

import pytest

from aairm.utils.cost_functions import single_period_cost, supplier_score, td_loss


def test_single_period_cost_non_negative():
    cost = single_period_cost(100.0, 90.0, 15.0, 5.0, 1.25, 15.0)
    assert cost >= 0.0


def test_single_period_cost_is_alias():
    from aairm.utils.math_utils import expected_cost_single_period
    c1 = single_period_cost(100.0, 90.0, 15.0, 5.0, 1.0, 10.0)
    c2 = expected_cost_single_period(100.0, 90.0, 15.0, 5.0, 1.0, 10.0)
    assert abs(c1 - c2) < 1e-9


def test_supplier_score_is_alias():
    from aairm.utils.math_utils import supplier_score as ss
    s1 = supplier_score(0.5, 0.3, 0.85, False)
    s2 = ss(0.5, 0.3, 0.85, False)
    assert abs(s1 - s2) < 1e-9


def test_td_loss_is_alias():
    from aairm.utils.math_utils import td_loss as tl
    l1 = td_loss(-1.0, -5.0, -4.5, 0.99)
    l2 = tl(-1.0, -5.0, -4.5, 0.99)
    assert abs(l1 - l2) < 1e-9
