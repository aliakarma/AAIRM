"""Convenience module that re-exports cost-related functions from math_utils.

Import from here when you only need cost computation, to keep call sites
readable without importing the full math_utils namespace.

Examples:
    >>> from aairm.utils.cost_functions import single_period_cost
    >>> cost = single_period_cost(100.0, 90.0, 15.0, 5.0, 1.25, 15.0)
"""

from aairm.utils.math_utils import (
    expected_cost_single_period as single_period_cost,
    supplier_score,
    td_loss,
)

__all__ = ["single_period_cost", "supplier_score", "td_loss"]
