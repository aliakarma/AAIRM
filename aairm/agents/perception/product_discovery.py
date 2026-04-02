"""Product Discovery Agent (P3) — Perception Layer.

Cross-references external trend signals (from P2) with the mart's existing
assortment to surface new or underrepresented SKUs that merit stocking
evaluation.  Works jointly with P2 to extend the framework's procurement
scope beyond the current catalogue.

References
----------
Paper Section 4.1 (agent P3).
"""

from __future__ import annotations

from typing import Any

from aairm.agents.base import AgentState, BaseAgent
from aairm.utils.config import SimulationConfig


class ProductDiscoveryAgent(BaseAgent):
    """P3 — Product Discovery Agent.

    Args:
        config: Simulation configuration.
        existing_sku_ids: Set of SKU IDs currently in the mart's assortment.
            Updated by the Inventory Adjustment Agent (A2) after each cycle.
        min_trend_score: Minimum trend score for a new SKU to be surfaced.
    """

    def __init__(
        self,
        config: SimulationConfig,
        existing_sku_ids: set[str] | None = None,
        min_trend_score: float = 0.60,
    ) -> None:
        super().__init__("P3", config)
        self._existing_skus: set[str] = existing_sku_ids or set()
        self._min_trend_score = min_trend_score

    def run(self, state: AgentState) -> AgentState:
        """Surface new SKU candidates from trend signals not in the assortment.

        Args:
            state: Pipeline state.  Reads ``state.trend_signals`` (from P2).

        Returns:
            Updated state with ``state.new_sku_candidates`` populated.
        """
        t0 = self._log_start(state)

        candidates: list[dict[str, Any]] = []
        for signal in state.trend_signals:
            product_id = signal.get("product_id", "")
            trend_score = float(signal.get("trend_score", 0.0))

            if (
                product_id
                and product_id not in self._existing_skus
                and trend_score >= self._min_trend_score
            ):
                candidates.append(
                    {
                        "product_id": product_id,
                        "product_name": signal.get("product_name", ""),
                        "category": signal.get("category", ""),
                        "trend_score": trend_score,
                        "source": signal.get("source", "external"),
                        "recommendation": "evaluate_for_stocking",
                    }
                )

        state.new_sku_candidates = candidates
        self._record_event(
            state, "discovery.candidates", n_candidates=len(candidates)
        )
        self._log_end(state, t0, n_candidates=len(candidates))
        return state

    def update_assortment(self, sku_ids: set[str]) -> None:
        """Update the known assortment after new SKUs are added.

        Called by A2 after successful goods receipt of newly listed products.

        Args:
            sku_ids: Complete current set of SKU IDs in the mart.
        """
        self._existing_skus = sku_ids
