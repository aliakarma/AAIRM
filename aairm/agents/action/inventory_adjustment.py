"""Inventory Adjustment Agent (A2) — Action Layer.

Reconciles physical receiving events with system records by:

  - Posting receipt quantities and adjusting on-hand balances.
  - Handling short shipments (partial fulfilment from ERP stub).
  - Processing quality rejections and substitution events.
  - Ensuring the perceptual state available to P1 and P4 remains accurate.

Also notifies the Product Discovery Agent (P3) of any newly listed SKUs
so the assortment set remains current.

References
----------
Paper Section 4.3 (Action Layer, agent A2).
"""

from __future__ import annotations

from typing import Any

from aairm.agents.base import AgentState, BaseAgent
from aairm.utils.config import SimulationConfig


class InventoryAdjustmentAgent(BaseAgent):
    """A2 — Inventory Adjustment Agent.

    Args:
        config: Simulation configuration.
        erp_backend: ERP stub implementing ``process_goods_receipt(gr_dict)``
            and ``get_pending_receipts(day) -> list[dict]``.
    """

    def __init__(
        self,
        config: SimulationConfig,
        erp_backend: Any = None,
    ) -> None:
        super().__init__("A2", config)
        self._erp = erp_backend

    def run(self, state: AgentState) -> AgentState:
        """Process goods receipts that arrived on the current simulation day.

        Reads
        -----
        state.day

        Writes
        ------
        state.inventory_adjustments : list[dict]
            One record per SKU adjusted.

        Args:
            state: Current pipeline state.

        Returns:
            Updated state.
        """
        t0 = self._log_start(state)
        adjustments: list[dict[str, Any]] = []

        if self._erp is None:
            self._log_end(state, t0, n_adjustments=0)
            return state

        try:
            pending: list[dict[str, Any]] = self._erp.get_pending_receipts(state.day)
        except Exception as exc:  # noqa: BLE001
            self._append_error(state, f"Failed to retrieve pending receipts: {exc}")
            self._log_end(state, t0, n_adjustments=0)
            return state

        for receipt in pending:
            sku_id = receipt.get("sku_id", "")
            ordered_qty = float(receipt.get("ordered_qty", 0.0))
            received_qty = float(receipt.get("received_qty", ordered_qty))
            po_id = receipt.get("po_id", "")

            shortfall = max(0.0, ordered_qty - received_qty)
            is_short = shortfall > 0

            try:
                self._erp.process_goods_receipt(
                    {
                        "sku_id": sku_id,
                        "received_qty": received_qty,
                        "po_id": po_id,
                        "day": state.day,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                self._append_error(
                    state, f"Goods receipt failed for {sku_id}: {exc}"
                )
                continue

            adj = {
                "sku_id": sku_id,
                "po_id": po_id,
                "ordered_qty": ordered_qty,
                "received_qty": received_qty,
                "shortfall": shortfall,
                "is_short_shipment": is_short,
                "day": state.day,
            }
            adjustments.append(adj)

            if is_short:
                self._log.warning(
                    "receipt.short_shipment",
                    sku_id=sku_id,
                    po_id=po_id,
                    shortfall=shortfall,
                )
            self._record_event(state, "receipt.processed", **adj)

        state.inventory_adjustments = adjustments
        self._log_end(state, t0, n_adjustments=len(adjustments))
        return state
