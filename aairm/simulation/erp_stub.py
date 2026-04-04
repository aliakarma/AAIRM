"""Mock ERP / WMS Backend (ERPStub).

Implements the same interface as a real ERP system, delegating to the
DemandGenerator and SupplierSimulator for data.  Used by:

  - P1 (InventoryMonitorAgent)  — get_inventory_snapshot()
  - P4 (ContextEngine)           — get_demand_history()
  - A1 (OrderExecutionAgent)     — create_purchase_order(), update_inbound_schedule()
  - A2 (InventoryAdjustmentAgent)— get_pending_receipts(), process_goods_receipt()

The stub maintains an in-memory inventory state that evolves across
simulation days.  It is NOT thread-safe; use a single instance per
simulation run.

References
----------
Paper Section 5.1; Repo Guide Section 6.3.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aairm.simulation.demand_generator import DemandGenerator
from aairm.simulation.sku_catalog import SKUCatalog
from aairm.simulation.supplier_simulator import SupplierSimulator


class ERPStub:
    """Mock ERP and WMS backend for simulation mode.

    Args:
        catalog: SKU catalog with all SKU metadata.
        demand_gen: Demand generator for history and realised demand.
        supplier_sim: Supplier simulator for PO fulfilment.
        initial_inventory_days: Initial stock-on-hand expressed as days
            of average demand (default 14 days = 2 weeks of stock).
    """

    def __init__(
        self,
        catalog: SKUCatalog,
        demand_gen: DemandGenerator,
        supplier_sim: SupplierSimulator,
        initial_inventory_days: float = 14.0,
        seed: int = 42,
    ) -> None:
        self._catalog = catalog
        self._gen = demand_gen
        self._sup = supplier_sim
        self._day: int = 0
        self._rng = np.random.default_rng(seed)
        self._expiry_rate_multiplier: float = 1.0

        # In-memory inventory: {sku_id: {on_hand, reserved, in_transit, ...}}
        self._inventory: dict[str, dict[str, Any]] = {}
        self._init_inventory(initial_inventory_days)

        # Purchase order register: {po_id: dict}
        self._pos: dict[str, dict[str, Any]] = {}
        self._ordering_cost_by_day: dict[int, float] = {}
        self._last_day_metrics: dict[str, float] = {
            "total_demand": 0.0,
            "fulfilled_units": 0.0,
            "stockout_units": 0.0,
            "expired_units": 0.0,
            "ordering_cost": 0.0,
        }

    # ------------------------------------------------------------------
    # Inventory interface (used by P1)
    # ------------------------------------------------------------------

    def get_inventory_snapshot(self) -> dict[str, dict[str, Any]]:
        """Return the full inventory snapshot for the current day.

        Returns:
            Dict mapping sku_id to inventory record with fields:
            ``{on_hand, reserved, in_transit, effective_available,
               unit_cost, unit_volume, category, is_perishable,
               days_to_expiry, demand_mean_daily, demand_std_daily,
               lead_time_days}``.
        """
        snapshot: dict[str, dict[str, Any]] = {}
        for sku_id, state in self._inventory.items():
            rec = self._catalog[sku_id]
            stats = self._gen.get_demand_stats(sku_id, self._day)

            # Update days to expiry for perishables
            days_to_expiry = 9999.0
            if rec.is_perishable and rec.shelf_life_days is not None:
                # Decrement from initial shelf life based on elapsed days
                days_to_expiry = max(
                    1.0,
                    float(rec.shelf_life_days) - (self._day % rec.shelf_life_days),
                )

            snapshot[sku_id] = {
                "on_hand": round(state["on_hand"], 2),
                "reserved": round(state["reserved"], 2),
                "in_transit": round(state["in_transit"], 2),
                "effective_available": round(
                    state["on_hand"] - state["reserved"] + state["in_transit"], 2
                ),
                "unit_cost": rec.unit_cost,
                "unit_volume": rec.unit_volume,
                "category": rec.category,
                "is_perishable": rec.is_perishable,
                "shelf_life_days": rec.shelf_life_days,
                "days_to_expiry": days_to_expiry,
                "demand_mean_daily": stats["mu_d"],
                "demand_std_daily": stats["sigma_d"],
                "lead_time_days": state.get("avg_lead_time", 5.0),
                "last_demand": float(state.get("last_demand", 0.0)),
                "last_fulfilled": float(state.get("last_fulfilled", 0.0)),
                "last_stockout": float(state.get("last_stockout", 0.0)),
                "last_expired": float(state.get("last_expired", 0.0)),
            }
        return snapshot

    # ------------------------------------------------------------------
    # Demand history interface (used by P4)
    # ------------------------------------------------------------------

    def get_demand_history(self, sku_id: str, n_days: int) -> np.ndarray:
        """Return recent demand history for a SKU.

        Args:
            sku_id: SKU identifier.
            n_days: Number of history days to return.

        Returns:
            NumPy array of shape ``(n_days,)`` with daily demand values.
        """
        return self._gen.get_history(sku_id, n_days, self._day)

    def get_trend_signals(self, day: int) -> list[dict[str, Any]]:
        """Return trend signals for P2.

        Args:
            day: Simulation day.

        Returns:
            List of trend signal dicts.
        """
        return self._gen.get_trend_signals(day)

    # ------------------------------------------------------------------
    # PO interface (used by A1)
    # ------------------------------------------------------------------

    def create_purchase_order(self, po_dict: dict[str, Any]) -> None:
        """Register a new purchase order in the ERP.

        Args:
            po_dict: PO dict with ``{po_id, sku_id, quantity, ...}``.
        """
        po_id = po_dict.get("po_id", "")
        sku_id = po_dict.get("sku_id", "")
        quantity = float(po_dict.get("quantity", 0.0))
        self._pos[po_id] = dict(po_dict)

        # Mark as in-transit
        if sku_id in self._inventory:
            self._inventory[sku_id]["in_transit"] = (
                self._inventory[sku_id].get("in_transit", 0.0) + quantity
            )

        # Track ordering cost for reward shaping and diagnostics.
        unit_price = float(po_dict.get("unit_price", 0.0))
        if unit_price <= 0.0:
            rec = self._catalog.get(sku_id)
            if rec is not None:
                unit_price = float(rec.unit_cost)
        day = int(po_dict.get("day", self._day))
        self._ordering_cost_by_day[day] = (
            self._ordering_cost_by_day.get(day, 0.0) + quantity * unit_price
        )

    def update_inbound_schedule(self, po_id: str, eta_days: int) -> None:
        """Record the expected arrival day for a PO.

        Args:
            po_id: Purchase order identifier.
            eta_days: Days until expected arrival from today.
        """
        if po_id in self._pos:
            self._pos[po_id]["eta_day"] = self._day + eta_days

    def submit_purchase_order(self, po_dict: dict[str, Any]) -> dict[str, Any]:
        """Delegate PO submission to SupplierSimulator.

        Args:
            po_dict: PO dict.

        Returns:
            Confirmation dict from SupplierSimulator.
        """
        return self._sup.submit_purchase_order(po_dict)

    # ------------------------------------------------------------------
    # Goods receipt interface (used by A2)
    # ------------------------------------------------------------------

    def get_pending_receipts(self, day: int) -> list[dict[str, Any]]:
        """Return receipts due today from SupplierSimulator.

        Args:
            day: Current simulation day.

        Returns:
            List of receipt dicts.
        """
        return self._sup.get_pending_receipts(day)

    def process_goods_receipt(self, receipt: dict[str, Any]) -> None:
        """Update on-hand inventory and clear in-transit stock.

        Args:
            receipt: Receipt dict with ``{sku_id, received_qty, po_id}``.
        """
        sku_id = receipt.get("sku_id", "")
        received = float(receipt.get("received_qty", 0.0))

        if sku_id in self._inventory:
            self._inventory[sku_id]["on_hand"] = self._inventory[sku_id]["on_hand"] + received
            # Clear in-transit (may be partial)
            po_id = receipt.get("po_id", "")
            if po_id in self._pos:
                ordered = float(self._pos[po_id].get("quantity", received))
                self._inventory[sku_id]["in_transit"] = max(
                    0.0,
                    self._inventory[sku_id]["in_transit"] - ordered,
                )
                self._pos.pop(po_id, None)

    # ------------------------------------------------------------------
    # Simulation step (called by RetailEnv.step)
    # ------------------------------------------------------------------

    def advance_day(self, day: int) -> dict[str, float]:
        """Consume daily demand and return per-SKU realised demand.

        Called once per simulation day by RetailEnv.  Updates on-hand
        inventory by subtracting realised demand and handles stockouts.

        Args:
            day: The simulation day to advance to.

        Returns:
            ``{sku_id: realised_demand}`` for metric computation.
        """
        self._day = day

        # Process inbound receipts scheduled for today before demand arrives.
        for receipt in self._sup.get_pending_receipts(day):
            self.process_goods_receipt(receipt)

        realised: dict[str, float] = {}
        total_demand = 0.0
        total_fulfilled = 0.0
        total_stockout = 0.0
        total_expired = 0.0

        for sku_id in self._catalog.sku_ids:
            demand = self._gen.get_demand(sku_id, day)
            state = self._inventory[sku_id]
            available = max(0.0, state["on_hand"] - state["reserved"])
            fulfilled = min(available, demand)
            stockout = max(0.0, demand - fulfilled)
            state["on_hand"] = max(0.0, state["on_hand"] - fulfilled)

            # Perishable inventory naturally decays; this creates non-zero spoilage.
            rec = self._catalog[sku_id]
            expired = 0.0
            if rec.is_perishable and rec.shelf_life_days is not None and state["on_hand"] > 0:
                # Category-specific shelf life adjustments
                category_shelf_life_multipliers = {
                    "frozen_food": 2.0,  # Frozen food lasts longer
                    "apparel": float("inf"),  # Apparel doesn't expire
                    "dry_fruits": 1.0,  # Standard expiry
                    "cosmetics": 1.0,  # Standard expiry
                    "grocery": 1.0,  # Standard expiry
                }
                shelf_life_mult = category_shelf_life_multipliers.get(rec.category, 1.0)

                # Category-specific expiry rate multipliers
                category_expiry_multipliers = {
                    "frozen_food": 0.5,  # Lowest spoilage
                    "apparel": 0.0,  # No spoilage
                    "dry_fruits": 1.5,  # Highest spoilage
                    "cosmetics": 1.0,  # Standard
                    "grocery": 1.0,  # Standard
                }
                expiry_mult = category_expiry_multipliers.get(rec.category, 1.0)

                if shelf_life_mult != float("inf"):
                    base_expiry_frac = 1.0 / max(float(rec.shelf_life_days) * shelf_life_mult, 1.0)
                    stochastic = float(self._rng.uniform(0.7, 1.3))
                    expired = min(
                        state["on_hand"],
                        state["on_hand"]
                        * base_expiry_frac
                        * stochastic
                        * self._expiry_rate_multiplier
                        * expiry_mult,
                    )
                    state["on_hand"] = max(0.0, state["on_hand"] - expired)

            state["last_demand"] = float(demand)
            state["last_fulfilled"] = float(fulfilled)
            state["last_stockout"] = float(stockout)
            state["last_expired"] = float(expired)

            realised[sku_id] = demand  # store total demand (not fulfilled)
            total_demand += demand
            total_fulfilled += fulfilled
            total_stockout += stockout
            total_expired += expired

        self._last_day_metrics = {
            "total_demand": float(total_demand),
            "fulfilled_units": float(total_fulfilled),
            "stockout_units": float(total_stockout),
            "expired_units": float(total_expired),
            "ordering_cost": float(self._ordering_cost_by_day.pop(day, 0.0)),
        }

        return realised

    def get_last_day_metrics(self) -> dict[str, float]:
        """Return aggregated metrics from the most recent `advance_day` call."""
        return dict(self._last_day_metrics)

    def set_expiry_rate_multiplier(self, multiplier: float) -> None:
        """Set multiplicative factor for perishable expiry dynamics."""
        self._expiry_rate_multiplier = float(max(0.1, multiplier))

    # ------------------------------------------------------------------
    # Supplier catalogue access
    # ------------------------------------------------------------------

    def query_catalogue(self, sku_id: str) -> list[dict[str, Any]]:
        """Query supplier offers for a SKU.

        Args:
            sku_id: SKU identifier.

        Returns:
            List of supplier offer dicts.
        """
        return self._sup.query_catalogue(sku_id)

    def request_discount(self, supplier_id: str, sku_id: str, qty: float) -> str:
        """Proxy to SupplierSimulator discount negotiation."""
        return self._sup.request_discount(supplier_id, sku_id, qty)

    def propose_schedule(self, supplier_id: str, sku_id: str, delta: int) -> str:
        """Proxy to SupplierSimulator schedule negotiation."""
        return self._sup.propose_schedule(supplier_id, sku_id, delta)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_inventory(self, initial_days: float) -> None:
        """Seed initial on-hand stock at ``initial_days`` of demand."""
        for sku_id in self._catalog.sku_ids:
            rec = self._catalog[sku_id]
            on_hand = round(rec.base_demand_daily * initial_days, 2)
            self._inventory[sku_id] = {
                "on_hand": on_hand,
                "reserved": 0.0,
                "in_transit": 0.0,
                "avg_lead_time": 5.0,
            }
