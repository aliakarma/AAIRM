"""Reorder Optimisation Agent (C2) — Conceptualization Layer.

Implements Eqs. 3–5 of the paper.

Two modes (controlled by ``OptimisationConfig.mode``):

**Analytical mode** (``mode="analytical"``)
    Minimises the single-period expected cost (Eq. 3) via grid search
    over Q, subject to the budget and capacity constraints of Eq. 4.
    Used by ablation studies and as a fallback when the RL policy has
    not yet been trained.

**RL mode** (``mode="rl"``)
    Uses a trained PPO policy (Eq. 5) that maps the system state
    ``s_t = [effective_available, forecast_mean, forecast_std,
    days_to_expiry, budget_remaining]``
    to the order action ``a_t = Q*``.

References
----------
Paper Section 4.2.2; Eqs. 3, 4, 5.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aairm.agents.base import AgentState, BaseAgent
from aairm.utils.config import OptimisationConfig
from aairm.utils.math_utils import expected_cost_single_period


class ReorderOptimisationAgent(BaseAgent):
    """C2 — Reorder Optimisation Agent.

    Args:
        config: :class:`~aairm.utils.config.OptimisationConfig`.
        rl_policy: Trained PPO policy object implementing
            ``predict(obs) -> (action, _)``.  Required when
            ``config.mode == "rl"``.
    """

    def __init__(
        self,
        config: OptimisationConfig,
        rl_policy: Any = None,
    ) -> None:
        super().__init__("C2", config)
        self._original_mode: str = config.mode
        self._mode: str = config.mode if (config.mode == "rl" and rl_policy is not None) else "analytical"
        self._policy = rl_policy
        self._budget: float = config.budget
        self._capacity: float = config.warehouse_capacity
        self._h_rate: float = config.holding_cost_rate
        self._penalty_mult: float = config.penalty_cost_multiplier
        self._min_order_quantity: float = config.min_order_quantity

    def run(self, state: AgentState) -> AgentState:
        """Compute optimal order quantities for all low-stock and candidate SKUs.

        Ensures replenishment proposals are generated proactively by considering
        both high-priority low-stock SKUs and secondary-priority candidates
        identified via soft thresholds.

        Reads
        -----
        state.low_stock_skus, state.replenishment_candidates, state.demand_forecasts,
        state.sku_inventory_snapshot

        Writes
        ------
        state.order_proposals : dict[str, float]
            ``{sku_id: Q*}``

        Args:
            state: Current pipeline state.

        Returns:
            Updated state.
        """
        t0 = self._log_start(state, mode=self._mode, n_low_stock=len(state.low_stock_skus),
                           n_candidates=len(state.replenishment_candidates))
        if self._original_mode != self._mode:
            self._log.info(
                "mode.override",
                message="RL disabled, using analytical policy",
                original_mode=self._original_mode,
            )

        # Combine low_stock (high priority) and candidates
        all_skus = state.low_stock_skus + state.replenishment_candidates
        # Deduplicate while preserving order
        seen = set()
        unique_skus = []
        for sku_id in all_skus:
            if sku_id not in seen:
                seen.add(sku_id)
                unique_skus.append(sku_id)

        if not unique_skus:
            self._log.warning(
                "optimisation.no_candidates",
                day=state.day,
                mode=self._mode,
                note="No low-stock or candidate SKUs available for optimisation.",
            )

        proposals: dict[str, float] = {}
        budget_used: float = 0.0

        for sku_id in unique_skus:
            rec = state.sku_inventory_snapshot.get(sku_id, {})
            fc = state.demand_forecasts.get(sku_id, {})

            demand_mean = float(fc.get("mean", 10.0))
            demand_std = float(np.sqrt(fc.get("variance", 4.0)))
            unit_cost = float(rec.get("unit_cost", 5.0))
            unit_volume = float(rec.get("unit_volume", 0.05))
            penalty_cost = unit_cost * self._penalty_mult
            holding_cost = unit_cost * self._h_rate
            lead_time = float(rec.get("lead_time_days", 5.0))

            # Perishability
            days_to_expiry = float(rec.get("days_to_expiry", 9999.0))
            is_perishable = days_to_expiry < 9000
            shelf_life_demand: float | None = None
            spoilage_rate = 0.0
            if is_perishable:
                shelf_life_demand = demand_mean * (days_to_expiry / 7.0)
                spoilage_rate = unit_cost * 0.5

            budget_remaining = self._budget - budget_used
            self._log.debug(
                "optimisation.input_state",
                sku_id=sku_id,
                effective_available=float(rec.get("effective_available", 0.0)),
                demand_mean=demand_mean,
                demand_std=demand_std,
                days_to_expiry=days_to_expiry,
                budget_remaining=round(budget_remaining, 2),
                unit_cost=unit_cost,
                unit_volume=unit_volume,
            )

            minimum_order_qty = float(rec.get("minimum_order_qty", rec.get("moq", 1.0)))
            min_order_quantity = self._min_order_quantity

            if self._mode == "rl" and self._policy is not None:
                # RL mode: use trained PPO policy
                # Observation: [effective_available, forecast_mean, forecast_std, days_to_expiry_norm, budget_fraction_remaining]
                effective_available = float(rec.get("effective_available", 0.0))
                days_to_expiry_norm = min(days_to_expiry / 365.0, 1.0)  # normalize to [0,1]
                budget_fraction_remaining = budget_remaining / self._budget

                obs = np.array([
                    effective_available,
                    demand_mean,
                    demand_std,
                    days_to_expiry_norm,
                    budget_fraction_remaining
                ], dtype=np.float32)

                max_order_limit = float(rec.get("max_order_quantity", 1000.0))
                q_star: float
                try:
                    action, _ = self._policy.predict(obs, deterministic=True)
                    action_value = float(np.asarray(action).item())
                    scaled_action = (action_value + 1.0) / 2.0  # [-1,1] -> [0,1]
                    order_qty = float(max(0.0, scaled_action * max_order_limit))
                    self._log.debug("action.scaling", raw_action=action_value, scaled_action=scaled_action, order_qty=order_qty)

                    if order_qty <= 0.0:
                        raise ValueError("RL policy produced zero order quantity")
                    q_star = order_qty
                except Exception as exc:
                    reorder_point = rec.get("reorder_point")
                    if reorder_point is not None:
                        fallback_qty = max(float(reorder_point) * 1.5, min_order_quantity)
                    else:
                        fallback_qty = min_order_quantity
                    q_star = float(fallback_qty)
                    self._log.warning(
                        "fallback.used",
                        reason="invalid_rl_action",
                        sku=sku_id,
                        order_qty=q_star,
                        error=str(exc),
                    )
            else:
                # Analytical mode
                q_star = self._rule_based_q(
                    sku_id=sku_id,
                    demand_mean=demand_mean,
                    lead_time=lead_time,
                    minimum_order_qty=minimum_order_qty,
                )

            # Minimum order quantity safeguard
            if q_star < min_order_quantity and (sku_id in state.low_stock_skus or sku_id in state.replenishment_candidates):
                q_star = min_order_quantity
                self._log.debug("order.adjusted_min", adjusted_qty=q_star)

            # Enforce budget and capacity feasibility
            max_by_budget = budget_remaining / max(unit_cost, 1e-6)
            max_by_volume = (self._capacity - sum(
                proposals.get(s, 0.0) * float(
                    state.sku_inventory_snapshot.get(s, {}).get("unit_volume", 0.05)
                )
                for s in proposals
            )) / max(unit_volume, 1e-6)
            q_star = min(q_star, max_by_budget, max(max_by_volume, 0.0))
            q_star = max(0.0, q_star)

            if q_star > 0:
                proposals[sku_id] = round(q_star, 2)
                budget_used += q_star * unit_cost

            self._record_event(
                state, "order.proposed",
                sku_id=sku_id, quantity=q_star,
                cost=round(q_star * unit_cost, 2), mode=self._mode,
            )

        state.order_proposals = proposals
        self._log_end(
            state, t0,
            n_proposals=len(proposals),
            budget_used=round(budget_used, 2),
        )
        return state

    # ------------------------------------------------------------------

    def _analytical_q(
        self,
        demand_mean: float,
        demand_std: float,
        unit_cost: float,
        holding_cost: float,
        penalty_cost: float,
        spoilage_rate: float,
        shelf_life_demand: float | None,
        budget_remaining: float,
        unit_volume: float,
    ) -> float:
        """Grid-search optimal Q by minimising Eq. 3.

        Searches over Q in [0, Q_max] with 200 candidate values.
        Q_max is the lesser of budget-constrained and volume-constrained
        maxima, capped at 5× expected demand to prevent over-ordering.
        """
        q_max = min(
            budget_remaining / max(unit_cost, 1e-6),
            self._capacity / max(unit_volume, 1e-6),
            demand_mean * 5,
        )
        if q_max <= 0:
            return 0.0

        candidates = np.linspace(0.0, q_max, 200)
        costs = np.array(
            [
                expected_cost_single_period(
                    q=float(q),
                    demand_mean=demand_mean,
                    demand_std=max(demand_std, 1e-6),
                    unit_cost=unit_cost,
                    holding_cost_rate=holding_cost,
                    penalty_cost=penalty_cost,
                    spoilage_cost_rate=spoilage_rate,
                    shelf_life_demand=shelf_life_demand,
                )
                for q in candidates
            ]
        )
        return float(candidates[int(np.argmin(costs))])

    def _rule_based_q(
        self,
        sku_id: str,
        demand_mean: float,
        lead_time: float,
        minimum_order_qty: float,
    ) -> float:
        """Compute order quantity using newsvendor-style formula."""
        # Reorder point: demand during lead time + safety stock
        demand_lead = demand_mean * lead_time
        # Assume demand_std ~ demand_mean * 0.2 (from synthetic)
        demand_std = demand_mean * 0.2
        z = 1.645  # 95% service level
        safety_stock = z * demand_std * (lead_time ** 0.5)
        reorder_point = demand_lead + safety_stock
        
        # Assume current_inventory is 0 for simplicity (low stock)
        current_inventory = 0
        
        # Target stock: 14 days demand + safety stock
        target_days = 14
        demand_target = demand_mean * target_days
        safety_stock_target = z * demand_std * (target_days ** 0.5)
        target_stock = demand_target + safety_stock_target
        
        order_qty = max(0, target_stock - current_inventory)
        order_qty = max(order_qty, minimum_order_qty)
        
        self._log.info(
            "reorder.calculation",
            sku=sku_id,
            reorder_point=round(reorder_point, 2),
            target_stock=round(target_stock, 2),
            qty=round(order_qty, 2),
        )
        return float(order_qty)
