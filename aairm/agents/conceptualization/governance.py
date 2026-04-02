"""Governance and Policy Agent (C5) — Conceptualization Layer.

Acts as policy enforcer and cross-category coordinator.  Validates every
proposed order against four constraint categories before forwarding to
the Action layer:

    1. Budget cap     — category-level spending limits.
    2. Diversification — flag if one supplier captures > 60% of a category.
    3. Storage capacity — frozen vs. ambient zone volumetric limits.
    4. Shelf-life alignment — perishable Q* ≤ D_life × safety_margin.

Also reviews anomaly alerts from P5 before approving any affected SKU.

References
----------
Paper Section 4.2.5 (Governance and Policy Agent, C5).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from aairm.agents.base import AgentState, BaseAgent
from aairm.utils.config import GovernanceConfig


class GovernanceAgent(BaseAgent):
    """C5 — Governance and Policy Agent.

    Args:
        config: :class:`~aairm.utils.config.GovernanceConfig`.
        category_budgets: Optional per-category budget caps.  If not
            provided, the total budget is split equally.
        total_budget: Total purchasing budget (fallback when
            ``category_budgets`` is None).
    """

    def __init__(
        self,
        config: GovernanceConfig,
        category_budgets: dict[str, float] | None = None,
        total_budget: float = 1_000_000.0,
    ) -> None:
        super().__init__("C5", config)
        self._cfg = config
        self._category_budgets = category_budgets or {}
        self._total_budget = total_budget

        # Track per-category procurement volume per cycle for diversification
        self._category_supplier_volume: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

    def run(self, state: AgentState) -> AgentState:
        """Validate all negotiated terms and populate approved_orders.

        Reads
        -----
        state.negotiated_terms, state.order_proposals,
        state.sku_inventory_snapshot, state.anomaly_alerts

        Writes
        ------
        state.approved_orders : dict[str, dict]
            Orders that pass all four constraint checks, annotated with
            constraint flags.

        Args:
            state: Current pipeline state.

        Returns:
            Updated state.
        """
        t0 = self._log_start(state, n_proposals=len(state.negotiated_terms))

        # Pre-compute anomalous SKUs for fast lookup
        anomalous_skus: set[str] = {
            a["sku_id"] for a in state.anomaly_alerts
            if a.get("type") in ("inventory_discrepancy", "data_quality")
        }

        # Reset cycle-level volume tracking
        self._category_supplier_volume = defaultdict(lambda: defaultdict(float))

        approved: dict[str, dict[str, Any]] = {}
        frozen_used = 0.0
        ambient_used = 0.0

        for sku_id, terms in state.negotiated_terms.items():
            rec = state.sku_inventory_snapshot.get(sku_id, {})
            category = str(rec.get("category", "grocery"))
            unit_volume = float(rec.get("unit_volume", 0.05))
            unit_price = float(terms.get("unit_price", 0.0))
            quantity = float(terms.get("quantity", 0.0))
            supplier_id = str(terms.get("supplier_id", "UNKNOWN"))
            days_to_expiry = float(rec.get("days_to_expiry", 9999.0))

            order_value = quantity * unit_price
            order_volume = quantity * unit_volume
            is_frozen = category == "frozen_food"
            is_perishable = days_to_expiry < 9000

            flags: list[str] = []

            # 1. Anomaly block
            if sku_id in anomalous_skus:
                flags.append("anomaly_hold")
                self._log.warning(
                    "governance.anomaly_hold", sku_id=sku_id, cycle_id=state.cycle_id
                )

            # 2. Budget cap check
            cat_budget = self._category_budgets.get(
                category, self._total_budget / 5.0
            )
            cat_spent = sum(
                float(approved[s].get("order_value", 0.0))
                for s in approved
                if state.sku_inventory_snapshot.get(s, {}).get("category") == category
            )
            if cat_spent + order_value > cat_budget:
                flags.append("budget_cap_exceeded")
                quantity = max(0.0, (cat_budget - cat_spent) / max(unit_price, 1e-6))
                order_value = quantity * unit_price
                order_volume = quantity * unit_volume

            # 3. Storage capacity check
            if is_frozen:
                if frozen_used + order_volume > self._cfg.frozen_zone_capacity:
                    flags.append("frozen_capacity_exceeded")
                    max_q = max(0.0,
                        (self._cfg.frozen_zone_capacity - frozen_used) / max(unit_volume, 1e-6)
                    )
                    quantity = min(quantity, max_q)
                    order_value = quantity * unit_price
                    order_volume = quantity * unit_volume
                frozen_used += order_volume
            else:
                if ambient_used + order_volume > self._cfg.ambient_zone_capacity:
                    flags.append("ambient_capacity_exceeded")
                    max_q = max(0.0,
                        (self._cfg.ambient_zone_capacity - ambient_used) / max(unit_volume, 1e-6)
                    )
                    quantity = min(quantity, max_q)
                    order_value = quantity * unit_price
                    order_volume = quantity * unit_volume
                ambient_used += order_volume

            # 4. Shelf-life alignment for perishables
            if is_perishable:
                demand_mean = float(rec.get("demand_mean_daily", 10.0))
                shelf_life_demand = demand_mean * days_to_expiry
                max_q_shelf = shelf_life_demand * self._cfg.shelf_life_safety_margin
                if quantity > max_q_shelf:
                    flags.append("shelf_life_alignment")
                    quantity = max(0.0, max_q_shelf)
                    order_value = quantity * unit_price

            # Human-approval threshold
            needs_human = order_value > self._cfg.human_approval_threshold
            if needs_human:
                flags.append("human_approval_required")

            # Track diversification
            if quantity > 0:
                self._category_supplier_volume[category][supplier_id] += order_value

            # Check diversification within category
            cat_vol = self._category_supplier_volume[category]
            total_cat_vol = sum(cat_vol.values())
            if total_cat_vol > 0:
                max_frac = max(cat_vol.values()) / total_cat_vol
                if max_frac > self._cfg.max_single_supplier_fraction:
                    flags.append("diversification_flag")

            if quantity > 0 and "anomaly_hold" not in flags:
                approved_terms = dict(terms)
                approved_terms.update(
                    {
                        "quantity": round(quantity, 2),
                        "order_value": round(order_value, 2),
                        "governance_flags": flags,
                        "needs_human_approval": needs_human,
                    }
                )
                approved[sku_id] = approved_terms

        state.approved_orders = approved
        n_rejected = len(state.negotiated_terms) - len(approved)

        self._record_event(
            state, "governance.completed",
            n_approved=len(approved),
            n_rejected=n_rejected,
        )
        self._log_end(state, t0, approved=len(approved), rejected=n_rejected)
        return state
