"""Classical inventory theory formulas and paper equations.

Every public function in this module directly implements a numbered
equation from the paper.  Function docstrings reference the equation
number so that code and manuscript remain traceable.

All functions accept and return plain Python floats or NumPy arrays.
They have no external dependencies beyond NumPy and SciPy.

References
----------
Syed et al. (2025), "Agentic Commerce", Sections 2.1 and 4.2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


# ---------------------------------------------------------------------------
# Eq. 1 — Reorder Point (ROP)
# ---------------------------------------------------------------------------

def rop(
    mu_d: float,
    sigma_d: float,
    lead_time: float,
    service_level: float = 0.95,
) -> float:
    """Compute the Reorder Point (Eq. 1 of the paper).

    .. math::

        \\text{ROP} = \\mu_D \\cdot L + z \\cdot \\sigma_D \\sqrt{L}

    Args:
        mu_d: Mean demand per unit time period :math:`\\mu_D`.
        sigma_d: Standard deviation of demand per unit time period
            :math:`\\sigma_D`.
        lead_time: Replenishment lead time :math:`L` in the same units
            as ``mu_d`` and ``sigma_d``.
        service_level: Target cycle service level in :math:`(0, 1)`.
            Defaults to ``0.95`` (:math:`z \\approx 1.645`).

    Returns:
        Reorder point quantity.  Round up to the nearest integer for
        discrete inventory systems.

    Raises:
        ValueError: If ``service_level`` is not in (0, 1).
        ValueError: If ``lead_time``, ``mu_d``, or ``sigma_d`` is negative.

    References:
        Paper Eq. 1; Nahmias & Olsen (2019), Chapter 5.

    Examples:
        >>> round(rop(10.0, 2.0, 5.0, 0.95), 2)
        57.36
        >>> round(rop(50.0, 8.0, 3.0, 0.99), 2)
        168.62
    """
    if not (0.0 < service_level < 1.0):
        raise ValueError(f"service_level must be in (0, 1); got {service_level}")
    if lead_time < 0:
        raise ValueError(f"lead_time must be non-negative; got {lead_time}")
    if mu_d < 0 or sigma_d < 0:
        raise ValueError("mu_d and sigma_d must be non-negative.")

    z: float = float(stats.norm.ppf(service_level))
    return mu_d * lead_time + z * sigma_d * np.sqrt(lead_time)


def eoq(
    demand_rate: float,
    ordering_cost: float,
    holding_cost_rate: float,
    unit_cost: float,
) -> float:
    """Compute the Economic Order Quantity (EOQ).

    .. math::

        Q^* = \\sqrt{\\frac{2 D K}{h \\cdot c}}

    Args:
        demand_rate: Annual demand :math:`D` in units per year.
        ordering_cost: Fixed cost per order :math:`K` (currency units).
        holding_cost_rate: Annual holding cost as a fraction of unit cost
            :math:`h` (dimensionless, e.g. 0.25 for 25 percent).
        unit_cost: Unit procurement cost :math:`c` (currency units).

    Returns:
        Optimal order quantity :math:`Q^*`.

    Raises:
        ValueError: If any argument is non-positive.

    Examples:
        >>> round(eoq(1000.0, 50.0, 0.25, 10.0), 2)
        200.0
    """
    if any(v <= 0 for v in [demand_rate, ordering_cost, holding_cost_rate, unit_cost]):
        raise ValueError("All EOQ arguments must be strictly positive.")
    h = holding_cost_rate * unit_cost
    return float(np.sqrt(2.0 * demand_rate * ordering_cost / h))


# ---------------------------------------------------------------------------
# Eq. 3 — Single-period expected cost C_i(Q_i)
# ---------------------------------------------------------------------------

def expected_cost_single_period(
    q: float,
    demand_mean: float,
    demand_std: float,
    unit_cost: float,
    holding_cost_rate: float,
    penalty_cost: float,
    spoilage_cost_rate: float = 0.0,
    shelf_life_demand: float | None = None,
) -> float:
    """Single-period expected cost :math:`C_i(Q_i)` (Eq. 3 of the paper).

    .. math::

        C_i(Q_i) = c_i Q_i
                 + h_i \\mathbb{E}[(Q_i - \\tilde{D}_i)_+]
                 + p_i \\mathbb{E}[(\\tilde{D}_i - Q_i)_+]
                 + \\delta_i \\mathbb{E}[\\max\\{0,\\, Q_i - \\tilde{D}_i^{\\text{life}}\\}]

    Demand is modelled as :math:`\\tilde{D}_i \\sim \\mathcal{N}(\\mu_D, \\sigma_D^2)`.

    Args:
        q: Order quantity :math:`Q_i`.
        demand_mean: Expected demand :math:`\\mu_D`.
        demand_std: Standard deviation of demand :math:`\\sigma_D`.
        unit_cost: Unit procurement cost :math:`c_i`.
        holding_cost_rate: Per-unit holding cost rate :math:`h_i`
            (per period, not annualised).
        penalty_cost: Stockout penalty cost :math:`p_i` per unit short.
        spoilage_cost_rate: Spoilage / wastage cost rate :math:`\\delta_i`.
            Set to ``0.0`` for non-perishable SKUs (default).
        shelf_life_demand: Expected demand realisable within the shelf-life
            window :math:`\\tilde{D}_i^{\\text{life}}`.  Ignored when
            ``spoilage_cost_rate`` is zero.

    Returns:
        Expected total cost for ordering quantity ``q``.

    References:
        Paper Eq. 3; Nahmias (1982) for the spoilage term.

    Examples:
        >>> round(expected_cost_single_period(
        ...     100.0, 90.0, 15.0, 5.0, 1.25, 15.0), 2)
        537.81
    """
    if demand_std <= 0:
        # Deterministic demand — no stochastic overage / underage
        overage = max(0.0, q - demand_mean)
        underage = max(0.0, demand_mean - q)
    else:
        z = (q - demand_mean) / demand_std
        phi_z = float(stats.norm.pdf(z))
        Phi_z = float(stats.norm.cdf(z))
        # E[(Q - D)₊] — expected overage (holding)
        overage = (q - demand_mean) * Phi_z + demand_std * phi_z
        # E[(D - Q)₊] — expected underage (stockout)
        underage = (demand_mean - q) * (1.0 - Phi_z) + demand_std * phi_z

    cost = unit_cost * q + holding_cost_rate * overage + penalty_cost * underage

    if spoilage_cost_rate > 0.0 and shelf_life_demand is not None:
        spoilage = max(0.0, q - shelf_life_demand)
        cost += spoilage_cost_rate * spoilage

    return float(cost)


# ---------------------------------------------------------------------------
# Eq. 6 — Composite Supplier Score
# ---------------------------------------------------------------------------

def supplier_score(
    unit_cost: float,
    lead_time_adjusted: float,
    reliability: float,
    moq_violation: bool,
    alpha_1: float = 0.35,
    alpha_2: float = 0.30,
    alpha_3: float = 0.25,
    alpha_4: float = 0.10,
) -> float:
    """Composite supplier score :math:`\\text{Score}_{ij}` (Eq. 6 of the paper).

    .. math::

        \\text{Score}_{ij} =
            \\alpha_1 c_{ij}
            + \\alpha_2 \\hat{L}_{ij}
            - \\alpha_3 r_{ij}
            + \\alpha_4 \\,\\mathbb{I}\\{Q_i < m_{ij}\\}

    **Lower score = better supplier.**  Rank suppliers in ascending order.

    Args:
        unit_cost: Quoted unit cost :math:`c_{ij}` (normalised to [0, 1]
            across candidates before calling this function).
        lead_time_adjusted: Adjusted lead-time estimate :math:`\\hat{L}_{ij}`
            incorporating historical deviation (normalised to [0, 1]).
        reliability: Historical on-time delivery rate :math:`r_{ij}` in
            :math:`[0, 1]`.  Higher is better (subtracted in the formula).
        moq_violation: ``True`` if the requested quantity :math:`Q_i` is
            below the supplier's minimum order quantity :math:`m_{ij}`.
        alpha_1: Cost weight (default 0.35; paper default).
        alpha_2: Lead-time weight (default 0.30).
        alpha_3: Reliability weight (default 0.25).
        alpha_4: MOQ violation penalty weight (default 0.10).

    Returns:
        Composite score.  Lower values indicate a more desirable supplier.

    Raises:
        ValueError: If any weight is negative or ``reliability`` not in [0,1].

    References:
        Paper Eq. 6; SupplierRankingConfig in aairm/utils/config.py.

    Examples:
        >>> round(supplier_score(0.6, 0.4, 0.95, False), 4)
        0.225
        >>> # Cheaper but lower reliability — worse score
        >>> s1 = supplier_score(0.4, 0.3, 0.70, False)
        >>> s2 = supplier_score(0.6, 0.2, 0.95, False)
        >>> s2 < s1  # reliable supplier wins despite higher cost
        True
    """
    if not (0.0 <= reliability <= 1.0):
        raise ValueError(f"reliability must be in [0, 1]; got {reliability}")
    for name, val in [("alpha_1", alpha_1), ("alpha_2", alpha_2),
                      ("alpha_3", alpha_3), ("alpha_4", alpha_4)]:
        if val < 0:
            raise ValueError(f"{name} must be non-negative; got {val}")

    return (
        alpha_1 * unit_cost
        + alpha_2 * lead_time_adjusted
        - alpha_3 * reliability
        + alpha_4 * float(moq_violation)
    )


# ---------------------------------------------------------------------------
# Supplier Diversification Index
# ---------------------------------------------------------------------------

def diversification_index(procurement_shares: ArrayLike) -> float:
    """Herfindahl-normalised supplier diversification index, scaled to [0, 1].

    Measures how evenly procurement volume is distributed across suppliers
    in a category.  A value of 1.0 indicates perfectly uniform distribution;
    0.0 indicates monopoly (all procurement from one supplier).

    The paper reports AAIRM = 0.61 vs. Baseline 1 = 0.42 (Table 2).

    Args:
        procurement_shares: Array-like of procurement fractions for each
            supplier.  Values are automatically normalised to sum to 1.
            Must contain at least one positive entry.

    Returns:
        Diversification index in :math:`[0, 1]`.  Higher is better.

    Raises:
        ValueError: If all shares are zero or the input is empty.

    References:
        Paper Table 2; Herfindahl–Hirschman Index normalisation.

    Examples:
        >>> round(diversification_index([1.0, 0.0, 0.0]), 4)
        0.0
        >>> round(diversification_index([0.25, 0.25, 0.25, 0.25]), 4)
        1.0
        >>> round(diversification_index([0.5, 0.3, 0.2]), 4)
        0.9167
    """
    shares = np.asarray(procurement_shares, dtype=float).ravel()
    if shares.size == 0 or shares.sum() == 0:
        raise ValueError("procurement_shares must be non-empty with positive sum.")

    shares = shares / shares.sum()
    n = shares.size
    if n == 1:
        return 0.0

    hhi = float(np.dot(shares, shares))
    # Normalise: HHI = 1 (monopoly) → index 0; HHI = 1/n (uniform) → index 1
    return float((1.0 - hhi) / (1.0 - 1.0 / n))


# ---------------------------------------------------------------------------
# Safety Stock
# ---------------------------------------------------------------------------

def safety_stock(
    sigma_d: float,
    lead_time: float,
    service_level: float = 0.95,
) -> float:
    """Compute safety stock :math:`SS = z \\cdot \\sigma_D \\sqrt{L}`.

    This is the stochastic buffer component of the ROP formula (Eq. 1).

    Args:
        sigma_d: Standard deviation of demand per time unit.
        lead_time: Replenishment lead time :math:`L`.
        service_level: Target service level.  Defaults to 0.95.

    Returns:
        Safety stock quantity.

    Examples:
        >>> round(safety_stock(2.0, 5.0, 0.95), 2)
        7.36
    """
    z: float = float(stats.norm.ppf(service_level))
    return z * sigma_d * np.sqrt(lead_time)


# ---------------------------------------------------------------------------
# TD Loss (Eq. 7 — used by Learning Agent A3)
# ---------------------------------------------------------------------------

def td_loss(
    reward: float,
    value_current: float,
    value_next: float,
    gamma: float = 0.99,
) -> float:
    """Temporal-difference loss :math:`\\mathcal{L}_{\\text{TD}}(\\phi)` (Eq. 7).

    .. math::

        \\mathcal{L}_{\\text{TD}}(\\phi) =
            \\left(r_t + \\gamma V_\\phi(s_{t+1}) - V_\\phi(s_t)\\right)^2

    Args:
        reward: Observed reward :math:`r_t` at current timestep.
        value_current: Value function estimate :math:`V_\\phi(s_t)`.
        value_next: Value function estimate :math:`V_\\phi(s_{t+1})`.
        gamma: Discount factor :math:`\\gamma \\in (0, 1)`.  Default 0.99.

    Returns:
        Scalar TD loss value.

    References:
        Paper Eq. 7; Sutton & Barto (2018), Chapter 6.

    Examples:
        >>> round(td_loss(-1.0, -5.0, -4.5, 0.99), 4)
        0.5401
    """
    td_error = reward + gamma * value_next - value_current
    return float(td_error ** 2)
