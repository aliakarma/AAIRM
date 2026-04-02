"""ROP–EOQ Baseline Policy (Baseline 1).

A traditional reorder-point and economic-order-quantity policy as
described in paper Section 5.2:

  - Demand means and variances estimated from historical sales.
  - Fixed safety factor corresponding to the 95% service level.
  - Orders triggered whenever available stock falls below the ROP.
  - Order quantity equals the EOQ.

This is the primary comparison baseline reported in Tables 2 and 3.

Expected paper performance:
    stockout_rate   = 8.7%
    fill_rate       = 93.1%
    avg_inventory   = 1.45  (normalised)
    total_cost      = 1.00  (normalised reference)
    div_index       = 0.42

References
----------
Paper Section 5.2; Nahmias & Olsen (2019), Chapter 5.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aairm.utils.math_utils import rop, eoq


class ROPEOQPolicy:
    """Classical ROP–EOQ inventory policy.

    Args:
        service_level: Target cycle service level (default 0.95).
        ordering_cost: Fixed cost per order K (default 50.0 currency units).
        holding_cost_rate: Annual holding cost fraction h (default 0.25).
    """

    def __init__(
        self,
        service_level: float = 0.95,
        ordering_cost: float = 50.0,
        holding_cost_rate: float = 0.25,
    ) -> None:
        self._sl = service_level
        self._k = ordering_cost
        self._h = holding_cost_rate

        # Fitted per-SKU parameters
        self._mu: dict[str, float] = {}
        self._sigma: dict[str, float] = {}
        self._lead_times: dict[str, float] = {}
        self._unit_costs: dict[str, float] = {}
        self._fitted = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        historical_demand: dict[str, np.ndarray],
        lead_times: dict[str, float] | None = None,
        unit_costs: dict[str, float] | None = None,
    ) -> "ROPEOQPolicy":
        """Estimate μ_D and σ_D per SKU from historical demand.

        Args:
            historical_demand: ``{sku_id: np.ndarray of daily demand}``.
            lead_times: ``{sku_id: float}`` — average lead time in days.
                Defaults to 5 days for all SKUs.
            unit_costs: ``{sku_id: float}`` — unit procurement cost.
                Defaults to 10.0 for all SKUs.

        Returns:
            Self.
        """
        for sku_id, series in historical_demand.items():
            nonzero = series[series > 0]
            if len(nonzero) == 0:
                nonzero = np.array([1.0])
            self._mu[sku_id] = float(np.mean(nonzero))
            self._sigma[sku_id] = float(np.std(nonzero) + 1e-8)
            self._lead_times[sku_id] = (
                lead_times[sku_id] if lead_times and sku_id in lead_times else 5.0
            )
            self._unit_costs[sku_id] = (
                unit_costs[sku_id] if unit_costs and sku_id in unit_costs else 10.0
            )
        self._fitted = True
        return self

    def get_orders(
        self, inventory_snapshot: dict[str, dict[str, Any]]
    ) -> dict[str, float]:
        """Return order quantities for all triggered SKUs.

        A SKU is triggered when its effective available stock ≤ its ROP.
        The order quantity is the EOQ.

        Args:
            inventory_snapshot: Per-SKU inventory dict from ERPStub.

        Returns:
            ``{sku_id: Q*}`` for all triggered SKUs.  Empty dict if
            no SKUs are below their reorder points.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before get_orders().")

        orders: dict[str, float] = {}
        for sku_id, rec in inventory_snapshot.items():
            effective = float(rec.get("effective_available", 0.0))
            lead_time = self._lead_times.get(sku_id, 5.0)
            mu = self._mu.get(sku_id, 10.0)
            sigma = self._sigma.get(sku_id, 2.0)
            unit_cost = self._unit_costs.get(sku_id, 10.0)

            reorder_point = rop(mu, sigma, lead_time, self._sl)

            if effective <= reorder_point:
                # Annual demand estimate for EOQ
                annual_demand = mu * 365.0
                q = eoq(annual_demand, self._k, self._h, unit_cost)
                orders[sku_id] = round(q, 2)

        return orders

    def get_top_supplier(
        self, sku_id: str, supplier_offers: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Return the cheapest supplier without negotiation (Baseline 1 logic).

        Args:
            sku_id: SKU identifier.
            supplier_offers: List of supplier offer dicts.

        Returns:
            The offer with the lowest unit cost, or None if empty.
        """
        if not supplier_offers:
            return None
        return min(supplier_offers, key=lambda o: float(o.get("unit_cost", 9999.0)))
