"""Inventory Monitor Agent (P1) — Perception Layer.

Periodically queries the inventory database, ERP system, and WMS to:

  - Retrieve on-hand quantities, reserved stock, and in-transit stock.
  - Compute effective available inventory:
    ``effective = on_hand - reserved + in_transit``.
  - Identify SKUs where ``effective <= ROP`` (low-stock alert).
  - Identify SKUs where ``effective > overstock_threshold`` (overstock alert).
  - Forward a focused SKU set with attributes to the Conceptualization layer.

References
----------
Paper Section 4.1 (Perception Layer, agent P1).
"""

from __future__ import annotations

import math
from typing import Any

from aairm.agents.base import AgentState, BaseAgent
from aairm.utils.config import SimulationConfig
from aairm.utils.math_utils import rop, safety_stock


class InventoryMonitorAgent(BaseAgent):
    """P1 — Inventory Monitor Agent.

    Args:
        config: :class:`~aairm.utils.config.SimulationConfig` instance.
        erp_backend: Any object that implements the ERP stub interface
            (``get_inventory_snapshot()``, ``get_demand_history()``).
            Defaults to ``None``; must be injected before calling ``run()``.
        service_level: Target service level for ROP computation.
            Defaults to ``0.95``.
        overstock_multiplier: An SKU is flagged as overstocked when
            ``effective_available > overstock_multiplier * ROP``.
    """

    def __init__(
        self,
        config: SimulationConfig,
        erp_backend: Any = None,
        service_level: float = 0.95,
        overstock_multiplier: float = 3.0,
    ) -> None:
        super().__init__("P1", config)
        self._erp = erp_backend
        self._service_level = service_level
        self._overstock_mult = overstock_multiplier

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, state: AgentState) -> AgentState:
        """Query inventory state and populate ``state.low_stock_skus`` and
        ``state.replenishment_candidates``.

        Identifies two categories of SKUs requiring attention:
        1. **low_stock_skus**: effective_available <= ROP (hard threshold).
           High priority; processed first by downstream agents.
        2. **replenishment_candidates**: effective_available < lead_time * demand.
           Secondary priority; soft threshold to enable proactive replenishment.

        Args:
            state: Current pipeline state.  ``state.day`` must be set by
                the Meta-Orchestrator before this agent is called.

        Returns:
            Updated state with:
            - ``state.low_stock_skus`` — list of SKU IDs needing replenishment.
            - ``state.replenishment_candidates`` — proactive candidates.
            - ``state.sku_inventory_snapshot`` — full per-SKU inventory dict.
        """
        t0 = self._log_start(state)

        if self._erp is None:
            self._append_error(state, "ERP backend not injected into P1.")
            self._log_end(state, t0, low_stock=0)
            return state

        try:
            snapshot: dict[str, dict[str, Any]] = self._erp.get_inventory_snapshot()
        except Exception as exc:  # noqa: BLE001
            self._append_error(state, f"ERP snapshot failed: {exc}")
            self._log_end(state, t0, low_stock=0)
            return state

        self._log.info(
            "inventory.snapshot_state",
            n_skus_total=len(snapshot),
            sample_skus=list(snapshot.keys())[:5],
        )

        low_stock: list[str] = []
        replenishment_candidates: list[str] = []
        all_skus = list(snapshot.keys())

        for sku_id, rec in snapshot.items():
            on_hand: float = float(rec.get("on_hand", 0.0))
            reserved: float = float(rec.get("reserved", 0.0))
            in_transit: float = float(rec.get("in_transit", 0.0))
            effective: float = on_hand - reserved + in_transit

            lead_time: float = float(rec.get("lead_time_days", 5.0))
            mu_d: float = float(rec.get("demand_mean_daily", 10.0))
            sigma_d: float = float(rec.get("demand_std_daily", 2.0))

            reorder_point: float = rop(
                mu_d=mu_d,
                sigma_d=sigma_d,
                lead_time=lead_time,
                service_level=self._service_level,
            )
            safety: float = safety_stock(sigma_d, lead_time, self._service_level)

            is_low = effective <= reorder_point
            is_overstock = effective > self._overstock_mult * reorder_point

            # Enrich the snapshot record with derived fields
            rec["effective_available"] = round(effective, 2)
            rec["reorder_point"] = round(reorder_point, 2)
            rec["safety_stock"] = round(safety, 2)
            rec["is_low_stock"] = is_low
            rec["is_overstock"] = is_overstock
            rec["lead_time_demand"] = round(lead_time * mu_d, 2)

            if is_low:
                low_stock.append(sku_id)

        if self.config.full_coverage:
            replenishment_candidates = all_skus
        elif low_stock:
            replenishment_candidates = low_stock.copy()
        else:
            n_to_select = max(len(all_skus) // 2, 1)
            replenishment_candidates = all_skus[:n_to_select]

        self._log.info(
            "inventory.coverage",
            total_skus=len(all_skus),
            candidates=len(replenishment_candidates),
            low_stock=len(low_stock),
            full_coverage=self.config.full_coverage,
        )

        state.sku_inventory_snapshot = snapshot
        state.low_stock_skus = low_stock
        state.replenishment_candidates = replenishment_candidates

        self._record_event(
            state,
            "inventory.snapshot",
            n_skus_total=len(snapshot),
            n_low_stock=len(low_stock),
            n_replenishment_candidates=len(replenishment_candidates),
        )
        self._log_end(state, t0, low_stock=len(low_stock), candidates=len(replenishment_candidates))
        return state
