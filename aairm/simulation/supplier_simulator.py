"""Supplier Simulator.

Generates per-SKU supplier catalogues and simulates lead-time distributions,
reliability scores, and partial-fulfilment events.

Deliberate imperfections matching the paper's simulation setup:
  - 15% of shipments delayed by 1–3 days.
  - 5% of POs receive partial fulfilment (50–90% of ordered quantity).
  - 3% of confirmations arrive with a simulated 4–12 hour delay flag.

References
----------
Paper Section 5.1; Repo Guide Section 6.3 (ERPStub imperfections).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aairm.utils.config import AAIRMConfig

_DELAY_PROBABILITY = 0.15
_PARTIAL_PROBABILITY = 0.05
_NOISY_ACK_PROBABILITY = 0.03
_PARTIAL_RANGE = (0.50, 0.90)


@dataclass
class SupplierOffer:
    """Single supplier offer for one SKU."""

    supplier_id: str
    sku_id: str
    unit_cost: float        # base quoted price
    lead_time_mean: float   # mean lead time in days
    lead_time_std: float    # standard deviation of lead time
    reliability: float      # historical on-time delivery rate in [0,1]
    moq: int                # minimum order quantity
    country: str


class SupplierSimulator:
    """Generates and manages the supplier catalogue for all SKUs.

    For each SKU, 3–5 suppliers are generated with heterogeneous pricing,
    lead times, reliability scores, and MOQ constraints.

    Args:
        catalog: :class:`~aairm.simulation.sku_catalog.SKUCatalog`.
        n_suppliers_min: Minimum suppliers per SKU (default 3).
        n_suppliers_max: Maximum suppliers per SKU (default 5).
        seed: Random seed.
    """

    COUNTRIES = ["SA", "CN", "IN", "TR", "AE", "DE", "US", "PK"]

    def __init__(
        self,
        catalog: SKUCatalog,
        n_suppliers_min: int = 3,
        n_suppliers_max: int = 5,
        seed: int = 42,
        config: AAIRMConfig | None = None,
    ) -> None:
        self._catalog = catalog
        self._rng = np.random.default_rng(seed)
        self._n_min = n_suppliers_min
        self._n_max = n_suppliers_max
        self._config = config
        # {sku_id: [SupplierOffer, ...]}
        self._catalogue: dict[str, list[SupplierOffer]] = {}
        self._generate_all()

        # Pending purchase orders: {po_id: {sku_id, quantity, eta_day, ...}}
        self._pending_pos: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public interface (used by ERPStub and tool wrappers)
    # ------------------------------------------------------------------

    def query_catalogue(self, sku_id: str) -> list[dict[str, Any]]:
        """Return all supplier offers for a SKU as plain dicts.

        Args:
            sku_id: SKU identifier.

        Returns:
            List of offer dicts (one per supplier).
        """
        offers = self._catalogue.get(sku_id, [])
        return [self._offer_to_dict(o) for o in offers]

    def submit_purchase_order(self, po_dict: dict[str, Any]) -> dict[str, Any]:
        """Accept a purchase order and return a confirmation with ETA.

        Applies deliberate imperfections:
          - 15% chance of +1 to +3 day delay.
          - 5% chance of partial fulfilment.
          - 3% chance of noisy (delayed) acknowledgement flag.

        Args:
            po_dict: PO dict from A1 with ``{po_id, sku_id, supplier_id,
                quantity, delivery_window_days}``.

        Returns:
            Confirmation dict with ``{po_id, confirmed, eta_days,
            fulfilled_quantity, noisy_ack}``.
        """
        po_id = po_dict.get("po_id", str(uuid.uuid4())[:8])
        qty = float(po_dict.get("quantity", 0.0))
        sku_id = po_dict.get("sku_id")
        base_lead = float(po_dict.get("delivery_window_days", 5.0))

        # Sample stochastic lead time
        if self._config and hasattr(self._config, 'supplier') and sku_id:
            rec = self._catalog[sku_id]
            lt_params = self._config.supplier.lead_time_days.get(rec.category, {"mean": base_lead, "std": 1, "min": 1, "max": 10})
            sampled_lead = self._rng.normal(lt_params["mean"], lt_params["std"])
            sampled_lead = np.clip(sampled_lead, lt_params["min"], lt_params["max"])
            eta_days = int(round(sampled_lead))
        else:
            eta_days = int(base_lead)

        # Delay
        delayed = self._rng.random() < _DELAY_PROBABILITY
        delay_days = int(self._rng.integers(1, 4)) if delayed else 0
        eta_days += delay_days

        # Partial fulfilment
        partial = self._rng.random() < _PARTIAL_PROBABILITY
        fulfilled_qty = (
            qty * float(self._rng.uniform(*_PARTIAL_RANGE))
            if partial
            else qty
        )

        # Noisy ACK
        noisy_ack = self._rng.random() < _NOISY_ACK_PROBABILITY

        result = {
            "po_id": po_id,
            "confirmed": True,
            "eta_days": eta_days,
            "fulfilled_quantity": round(fulfilled_qty, 2),
            "partial_fulfilment": partial,
            "noisy_ack": noisy_ack,
            "delay_days": delay_days,
            "lead_time_realized": eta_days - delay_days,  # realized before delay
        }
        self._pending_pos[po_id] = {
            "sku_id": po_dict.get("sku_id"),
            "ordered_qty": qty,
            "received_qty": round(fulfilled_qty, 2),
            "eta_day": po_dict.get("day", 0) + eta_days,
        }
        return result

    def get_pending_receipts(self, day: int) -> list[dict[str, Any]]:
        """Return POs scheduled to arrive on ``day``.

        Args:
            day: Current simulation day.

        Returns:
            List of receipt dicts ``{po_id, sku_id, ordered_qty, received_qty}``.
        """
        due = [
            {"po_id": po_id, **info}
            for po_id, info in self._pending_pos.items()
            if info["eta_day"] == day
        ]
        # Remove processed POs
        for r in due:
            self._pending_pos.pop(r["po_id"], None)
        return due

    def request_discount(
        self, supplier_id: str, sku_id: str, quantity: float
    ) -> str:
        """Simulation-mode discount response (used by C4).

        Args:
            supplier_id: Supplier identifier.
            sku_id: SKU identifier.
            quantity: Requested quantity.

        Returns:
            Human-readable discount response string.
        """
        offers = self._catalogue.get(sku_id, [])
        for offer in offers:
            if offer.supplier_id == supplier_id:
                moq = offer.moq
                if quantity > 2.0 * moq:
                    return f"5% discount approved for qty {quantity:.0f} (MOQ {moq})."
                return "No discount applicable (quantity below 2×MOQ threshold)."
        return "Supplier not found in catalogue."

    def propose_schedule(
        self, supplier_id: str, sku_id: str, delta_days: int
    ) -> str:
        """Simulation-mode schedule proposal response (used by C4)."""
        delta_days = int(np.clip(delta_days, -2, 2))
        if delta_days == 0:
            return "Current schedule maintained."
        direction = "earlier" if delta_days < 0 else "later"
        return (
            f"Schedule adjusted by {abs(delta_days)} day(s) {direction}. "
            "Confirmed by supplier."
        )

    # ------------------------------------------------------------------
    # Private generation
    # ------------------------------------------------------------------

    def _generate_all(self) -> None:
        """Generate suppliers for every SKU in the catalog."""
        sup_counter = 1
        for sku_id in self._catalog.sku_ids:
            rec = self._catalog[sku_id]
            n_sup = int(self._rng.integers(self._n_min, self._n_max + 1))
            offers = []
            for _ in range(n_sup):
                markup = float(self._rng.uniform(0.90, 1.35))
                reliability = round(float(self._rng.uniform(0.65, 0.99)), 3)
                
                # Use config lead times if available
                if self._config and hasattr(self._config, 'supplier'):
                    lt_params = self._config.supplier.lead_time_days.get(rec.category, {"mean": 5, "std": 1, "min": 3, "max": 10})
                    lt_mean = lt_params["mean"]
                    lt_std = lt_params["std"]
                else:
                    lt_mean = round(float(self._rng.uniform(2.0, 10.0)), 1)
                    lt_std = round(float(self._rng.uniform(0.3, 2.0)), 2)
                
                moq = int(self._rng.choice([5, 10, 20, 50, 100]))
                country = str(self._rng.choice(self.COUNTRIES))
                offers.append(
                    SupplierOffer(
                        supplier_id=f"SUP-{sup_counter:05d}",
                        sku_id=sku_id,
                        unit_cost=round(rec.unit_cost * markup, 2),
                        lead_time_mean=lt_mean,
                        lead_time_std=lt_std,
                        reliability=reliability,
                        moq=moq,
                        country=country,
                    )
                )
                sup_counter += 1
            self._catalogue[sku_id] = offers

    @staticmethod
    def _offer_to_dict(offer: SupplierOffer) -> dict[str, Any]:
        return {
            "supplier_id": offer.supplier_id,
            "sku_id": offer.sku_id,
            "unit_cost": offer.unit_cost,
            "lead_time_mean": offer.lead_time_mean,
            "lead_time_std": offer.lead_time_std,
            "reliability": offer.reliability,
            "moq": offer.moq,
            "country": offer.country,
        }
