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
        self._mode: str = config.mode
        self._policy = rl_policy
        self._budget: float = config.budget
        self._capacity: float = config.warehouse_capacity
        self._h_rate: float = config.holding_cost_rate
        self._penalty_mult: float = config.penalty_cost_multiplier

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

            if self._mode == "rl" and self._policy is not None and hasattr(self._policy, '_model') and self._policy._model is not None:
                obs = np.array(
                    [
                        float(rec.get("effective_available", 0.0)),
                        demand_mean,
                        demand_std,
                        days_to_expiry if days_to_expiry < 9000 else 365.0,
                        budget_remaining / self._budget,  # normalised
                    ],
                    dtype=np.float32,
                )
                try:
                    action, _ = self._policy.predict(obs, deterministic=True)
                    q_star = float(np.clip(action[0], 0.0, 1e6))
                except Exception as exc:  # noqa: BLE001
                    self._append_error(state, f"RL policy failed for {sku_id}: {exc}")
                    q_star = self._analytical_q(
                        demand_mean, demand_std, unit_cost,
                        holding_cost, penalty_cost, spoilage_rate,
                        shelf_life_demand, budget_remaining, unit_volume,
                    )
            else:
                # Fallback to analytical if RL not available or not built
                q_star = self._analytical_q(
                    demand_mean, demand_std, unit_cost,
                    holding_cost, penalty_cost, spoilage_rate,
                    shelf_life_demand, budget_remaining, unit_volume,
                )

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
