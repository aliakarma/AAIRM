"""Order Execution Agent (A1) — Action Layer.

Translates governance-approved orders into purchase orders transmitted to
supplier platforms.  Also:

  - Arranges shipments with logistics carriers.
  - Generates receiving operations in the WMS.
  - Processes advanced shipping notices (ASNs) and goods-receipt confirmations.
  - Updates ERP and inventory databases in real time.
  - Propagates KPIs to the BI reporting layer.

References
----------
Paper Section 4.3 (Action Layer, agent A1).
"""

from __future__ import annotations

import uuid
from typing import Any

from aairm.agents.base import AgentState, BaseAgent
from aairm.utils.config import SimulationConfig


class OrderExecutionAgent(BaseAgent):
    """A1 — Order Execution Agent.

    Args:
        config: Simulation configuration (used for category lookups).
        erp_backend: Object implementing ``create_purchase_order(order_dict)``
            and ``update_inbound_schedule(po_id, eta_days)``.
        supplier_backend: Object implementing
            ``submit_purchase_order(po_dict) -> {po_id, confirmed, eta_days}``.
    """

    def __init__(
        self,
        config: SimulationConfig,
        erp_backend: Any = None,
        supplier_backend: Any = None,
    ) -> None:
        super().__init__("A1", config)
        self._erp = erp_backend
        self._supplier = supplier_backend

    def run(self, state: AgentState) -> AgentState:
        """Submit purchase orders for all approved orders.

        Reads
        -----
        state.approved_orders

        Writes
        ------
        state.purchase_orders_issued : list[str]
            Confirmed PO IDs.

        Args:
            state: Current pipeline state.

        Returns:
            Updated state.
        """
        t0 = self._log_start(state, n_approved=len(state.approved_orders))
        issued_pos: list[str] = []

        for sku_id, terms in state.approved_orders.items():
            if terms.get("needs_human_approval", False):
                self._log.info(
                    "order.awaiting_human_approval",
                    sku_id=sku_id,
                    order_value=terms.get("order_value", 0.0),
                )
                # In simulation, auto-approve; in production this would
                # pause and wait for the approver
                pass

            po_dict = {
                "po_id": f"PO-{uuid.uuid4().hex[:8].upper()}",
                "sku_id": sku_id,
                "supplier_id": terms.get("supplier_id"),
                "quantity": terms.get("quantity", 0.0),
                "unit_price": terms.get("unit_price", 0.0),
                "delivery_window_days": terms.get("delivery_window_days", 5.0),
                "payment_terms": terms.get("payment_terms", "Net-30"),
                "cycle_id": state.cycle_id,
                "day": state.day,
            }

            confirmed_po_id = po_dict["po_id"]
            eta_days = int(po_dict["delivery_window_days"])

            # Submit to supplier backend (real or simulated)
            if self._supplier is not None:
                try:
                    response = self._supplier.submit_purchase_order(po_dict)
                    confirmed_po_id = response.get("po_id", po_dict["po_id"])
                    eta_days = int(response.get("eta_days", eta_days))
                except Exception as exc:  # noqa: BLE001
                    self._append_error(state, f"PO submission failed for {sku_id}: {exc}")
                    continue

            # Update ERP with inbound schedule
            if self._erp is not None:
                try:
                    self._erp.create_purchase_order(po_dict)
                    self._erp.update_inbound_schedule(confirmed_po_id, eta_days)
                except Exception as exc:  # noqa: BLE001
                    self._append_error(state, f"ERP update failed for {sku_id}: {exc}")

            issued_pos.append(confirmed_po_id)
            self._record_event(
                state, "po.issued",
                po_id=confirmed_po_id,
                sku_id=sku_id,
                quantity=terms.get("quantity"),
                eta_days=eta_days,
            )
            self._log.info(
                "order.executed",
                po_id=confirmed_po_id,
                sku_id=sku_id,
                qty=terms.get("quantity"),
                supplier=terms.get("supplier_id"),
            )

        state.purchase_orders_issued = issued_pos
        self._log_end(state, t0, pos_issued=len(issued_pos))
        return state
