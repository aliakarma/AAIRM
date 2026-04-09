"""Evaluation Metrics for AAIRM Experiments.

Implements all five metrics reported in Tables 2 and 3 of the paper.
Every function includes the expected paper value in its docstring for
regression testing.

Paper results (Table 2):
    Policy         stockout  fill    avg_inv  total_cost  div_index
    Baseline 1     8.7%      93.1%   1.45     1.00        0.42
    Baseline 2     6.2%      95.4%   1.32     0.93        0.47
    AAIRM          3.9%      97.8%   1.19     0.84        0.61

References
----------
Paper Section 5.2 (Evaluation Metrics); Tables 2 and 3.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from aairm.utils.math_utils import diversification_index

# ---------------------------------------------------------------------------
# Metric 1 — Stockout Rate
# ---------------------------------------------------------------------------


def stockout_rate(
    demand: ArrayLike,
    fulfilled: ArrayLike,
) -> float:
    """Fraction of demand not satisfied from on-hand inventory.

    .. math::

        \\text{stockout\\_rate} =
        \\frac{\\sum_t \\max(0,\\, d_t - f_t)}{\\sum_t d_t}

    Args:
        demand: Array of demand realisations ``d_t`` (non-negative).
        fulfilled: Array of fulfilled units ``f_t`` (non-negative,
            ``f_t ≤ d_t``).

    Returns:
        Stockout rate in :math:`[0, 1]`.

    References:
        Paper Table 2: AAIRM = 0.039, Baseline 1 = 0.087.

    Examples:
        >>> stockout_rate([10, 10, 10], [10, 8, 10])
        0.06666666666666667
        >>> stockout_rate([100], [100])
        0.0
    """
    d = np.asarray(demand, dtype=float)
    f = np.asarray(fulfilled, dtype=float)
    total_demand = float(d.sum())
    if total_demand == 0:
        return 0.0
    total_stockout = float(np.maximum(0.0, d - f).sum())
    return total_stockout / total_demand


# ---------------------------------------------------------------------------
# Metric 2 — Fill Rate
# ---------------------------------------------------------------------------


def fill_rate(
    demand: ArrayLike,
    fulfilled: ArrayLike,
) -> float:
    """Fraction of demand fully satisfied from on-hand inventory.

    .. math::

        \\text{fill\\_rate} = 1 - \\text{stockout\\_rate}

    Args:
        demand: Array of demand realisations.
        fulfilled: Array of fulfilled units.

    Returns:
        Fill rate in :math:`[0, 1]`.

    References:
        Paper Table 2: AAIRM = 0.978, Baseline 1 = 0.931.

    Examples:
        >>> round(fill_rate([10, 10, 10], [10, 8, 10]), 4)
        0.9333
    """
    return 1.0 - stockout_rate(demand, fulfilled)


# ---------------------------------------------------------------------------
# Metric 3 — Average Inventory Ratio
# ---------------------------------------------------------------------------


def average_inventory_ratio(
    on_hand: ArrayLike,
    demand: ArrayLike,
) -> float:
    """Mean on-hand inventory normalised by mean daily demand.

    .. math::

        \\text{avg\\_inv} = \\frac{\\bar{I}}{\\bar{D}}

    where :math:`\\bar{I}` is the time-average on-hand stock and
    :math:`\\bar{D}` is the time-average demand.

    Args:
        on_hand: Array of end-of-day on-hand stock levels.
        demand: Array of daily demand realisations.

    Returns:
        Average inventory ratio (dimensionless, normalised).
        A value of 1.0 means stock equals one day of demand on average.

    References:
        Paper Table 2 (normalised to Baseline 1):
        AAIRM = 1.19, Baseline 1 = 1.45.

    Examples:
        >>> round(average_inventory_ratio([15, 20, 10], [10, 10, 10]), 2)
        1.5
    """
    i_arr = np.asarray(on_hand, dtype=float)
    d_arr = np.asarray(demand, dtype=float)
    mean_demand = float(d_arr.mean())
    if mean_demand == 0:
        return 0.0
    return float(i_arr.mean() / mean_demand)


# ---------------------------------------------------------------------------
# Metric 4 — Total Cost (normalised)
# ---------------------------------------------------------------------------


def total_cost_normalised(
    procurement_costs: ArrayLike,
    holding_costs: ArrayLike,
    penalty_costs: ArrayLike,
    spoilage_costs: ArrayLike,
    baseline_total: float,
) -> float:
    """Sum of all cost components, normalised to Baseline 1's total cost.

    .. math::

        \\text{total\\_cost} =
        \\frac{\\sum(C_{proc} + C_{hold} + C_{pen} + C_{spoil})}{C_{baseline}}

    Args:
        procurement_costs: Per-period procurement cost array.
        holding_costs: Per-period holding cost array.
        penalty_costs: Per-period stockout penalty array.
        spoilage_costs: Per-period spoilage cost array.
        baseline_total: The reference (Baseline 1) total cost for normalisation.

    Returns:
        Normalised total cost.  Baseline 1 = 1.00 by definition.
        Values < 1.0 indicate lower cost than Baseline 1.

    References:
        Paper Table 2: AAIRM = 0.84, Baseline 2 = 0.93.

    Examples:
        >>> total_cost_normalised([800], [100], [50], [50], 1000.0)
        1.0
        >>> total_cost_normalised([700], [80], [40], [20], 1000.0)
        0.84
    """
    total = (
        float(np.asarray(procurement_costs, dtype=float).sum())
        + float(np.asarray(holding_costs, dtype=float).sum())
        + float(np.asarray(penalty_costs, dtype=float).sum())
        + float(np.asarray(spoilage_costs, dtype=float).sum())
    )
    if baseline_total <= 0:
        return 0.0
    return total / baseline_total


# ---------------------------------------------------------------------------
# Metric 5 — Supplier Diversification Index
# ---------------------------------------------------------------------------


def supplier_diversification_index(
    procurement_volumes: dict[str, dict[str, float]],
) -> float:
    """Category-averaged Herfindahl-normalised diversification index.

    Computes the diversification index per category (using
    :func:`~aairm.utils.math_utils.diversification_index`) then
    averages across all categories.

    Args:
        procurement_volumes: Nested dict:
            ``{category: {supplier_id: total_procurement_volume}}``.

    Returns:
        Average diversification index across categories, in :math:`[0, 1]`.
        Higher values indicate lower single-source concentration.

    References:
        Paper Table 2: AAIRM = 0.61, Baseline 1 = 0.42.

    Examples:
        >>> vols = {"grocery": {"SUP-1": 100, "SUP-2": 100}}
        >>> round(supplier_diversification_index(vols), 1)
        1.0
        >>> vols = {"grocery": {"SUP-1": 1000, "SUP-2": 0}}
        >>> round(supplier_diversification_index(vols), 1)
        0.0
    """
    if not procurement_volumes:
        return 0.0

    indices = []
    for _category, vol_by_supplier in procurement_volumes.items():
        total = sum(vol_by_supplier.values())
        if total <= 0:
            continue
        shares = list(vol_by_supplier.values())
        idx = diversification_index(shares)
        indices.append(idx)

    if not indices:
        return 0.0
    return float(np.mean(indices))


def spoilage_rate(
    demand: ArrayLike,
    spoilage_units: ArrayLike,
) -> float:
    """Fraction of demand-equivalent volume lost due to spoilage.

    Args:
        demand: Daily demand array.
        spoilage_units: Daily expired/spoiled units array.

    Returns:
        Spoilage rate in [0, 1] relative to total demand volume.
    """
    d = np.asarray(demand, dtype=float)
    s = np.asarray(spoilage_units, dtype=float)
    total_demand = float(d.sum())
    if total_demand <= 0:
        return 0.0
    return float(np.clip(s.sum() / total_demand, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Convenience: compute all metrics at once
# ---------------------------------------------------------------------------


def compute_all_metrics(
    demand: ArrayLike,
    fulfilled: ArrayLike,
    on_hand: ArrayLike,
    procurement_costs: ArrayLike,
    holding_costs: ArrayLike,
    penalty_costs: ArrayLike,
    spoilage_costs: ArrayLike,
    spoilage_units: ArrayLike,
    baseline_total_cost: float,
    procurement_volumes: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Compute all five paper metrics and return as a dict.

    IMPORTANT: Spoilage is NOT double-counted.  The ``spoilage_costs`` array
    is used only for the ``total_cost`` calculation (monetary).  The
    ``spoilage_units`` array is used only for the ``spoilage_rate`` calculation
    (units as a fraction of demand).  These are separate, independent metrics.

    Args:
        demand: Daily demand array.
        fulfilled: Daily fulfilled units array.
        on_hand: Daily on-hand stock array.
        procurement_costs: Daily procurement cost array.
        holding_costs: Daily holding cost array.
        penalty_costs: Daily penalty cost array.
        spoilage_costs: Daily spoilage cost array (for total_cost only).
        spoilage_units: Daily spoiled units array (for spoilage_rate only).
        baseline_total_cost: Reference cost for normalisation.
        procurement_volumes: ``{category: {supplier_id: volume}}``.

    Returns:
        Dict with keys:
        ``{stockout_rate, fill_rate, avg_inventory, total_cost, div_index, spoilage_rate}``.
        
        Each key represents a separate, independent metric with no redundancy.
    """
    return {
        "stockout_rate": stockout_rate(demand, fulfilled),
        "fill_rate": fill_rate(demand, fulfilled),
        "avg_inventory": average_inventory_ratio(on_hand, demand),
        "total_cost": total_cost_normalised(
            procurement_costs,
            holding_costs,
            penalty_costs,
            spoilage_costs,
            baseline_total_cost,
        ),
        "div_index": supplier_diversification_index(procurement_volumes),
        "spoilage_rate": spoilage_rate(demand, spoilage_units),
    }
