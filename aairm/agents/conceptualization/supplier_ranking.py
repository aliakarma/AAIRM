"""Supplier Ranking Agent (C3) — Conceptualization Layer.

For each ``(sku_id, Q*)`` pair approved by C2, queries supplier catalogues
and ranks candidates using the composite score in Eq. 6 of the paper:

    Score_ij = α₁·c_ij + α₂·L̂_ij − α₃·r_ij + α₄·𝟙{Q < m_ij}

Lower score = more desirable supplier.  Forwards the top-3 shortlist
per SKU to the Negotiation Agent (C4) and Governance Agent (C5).

References
----------
Paper Section 4.2.3; Eq. 6.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aairm.agents.base import AgentState, BaseAgent
from aairm.utils.config import SupplierRankingConfig
from aairm.utils.math_utils import supplier_score


class SupplierRankingAgent(BaseAgent):
    """C3 — Supplier Ranking Agent.

    Args:
        config: :class:`~aairm.utils.config.SupplierRankingConfig`.
        supplier_backend: Object implementing
            ``query_catalogue(sku_id) -> list[dict]`` where each dict
            contains ``{supplier_id, unit_cost, lead_time_mean,
            lead_time_std, reliability, moq}``.
        top_k: Number of top suppliers to shortlist per SKU (default 3).
    """

    def __init__(
        self,
        config: SupplierRankingConfig,
        supplier_backend: Any = None,
        top_k: int = 3,
    ) -> None:
        super().__init__("C3", config)
        self._backend = supplier_backend
        self._top_k = top_k
        self._a1 = config.alpha_1
        self._a2 = config.alpha_2
        self._a3 = config.alpha_3
        self._a4 = config.alpha_4

    def run(self, state: AgentState) -> AgentState:
        """Rank suppliers for each SKU in ``state.order_proposals``.

        Reads
        -----
        state.order_proposals

        Writes
        ------
        state.supplier_rankings : dict[str, list[dict]]
            Top-``top_k`` ranked suppliers per SKU, ascending by score.

        Args:
            state: Current pipeline state.

        Returns:
            Updated state.
        """
        t0 = self._log_start(state, n_skus=len(state.order_proposals))
        rankings: dict[str, list[dict[str, Any]]] = {}

        for sku_id, q_star in state.order_proposals.items():
            if self._backend is None:
                self._append_error(state, "Supplier backend not injected into C3.")
                rankings[sku_id] = []
                continue

            try:
                offers: list[dict[str, Any]] = self._backend.query_catalogue(sku_id)
            except Exception as exc:  # noqa: BLE001
                self._append_error(state, f"Supplier query failed for {sku_id}: {exc}")
                rankings[sku_id] = []
                continue

            if not offers:
                self._append_error(state, f"No supplier offers found for {sku_id}.")
                rankings[sku_id] = []
                continue

            # Normalise unit_cost and lead_time to [0,1] across candidates
            costs = np.array([float(o["unit_cost"]) for o in offers])
            lead_times = np.array(
                [
                    float(o["lead_time_mean"])
                    + float(o.get("lead_time_std", 0.0))
                    for o in offers
                ]
            )
            c_range = costs.max() - costs.min() + 1e-9
            l_range = lead_times.max() - lead_times.min() + 1e-9
            c_norm = (costs - costs.min()) / c_range
            l_norm = (lead_times - lead_times.min()) / l_range

            scored: list[tuple[float, dict[str, Any]]] = []
            for idx, offer in enumerate(offers):
                moq = float(offer.get("moq", 1.0))
                score = supplier_score(
                    unit_cost=float(c_norm[idx]),
                    lead_time_adjusted=float(l_norm[idx]),
                    reliability=float(offer.get("reliability", 0.85)),
                    moq_violation=q_star < moq,
                    alpha_1=self._a1,
                    alpha_2=self._a2,
                    alpha_3=self._a3,
                    alpha_4=self._a4,
                )
                offer_copy = dict(offer)
                offer_copy["composite_score"] = round(score, 6)
                offer_copy["moq_violation"] = q_star < moq
                scored.append((score, offer_copy))

            # Sort ascending (lower = better)
            scored.sort(key=lambda x: x[0])
            top_suppliers = [o for _, o in scored[: self._top_k]]
            rankings[sku_id] = top_suppliers

            self._record_event(
                state, "supplier.ranked",
                sku_id=sku_id,
                n_offers=len(offers),
                top_supplier=top_suppliers[0].get("supplier_id", "unknown"),
                top_score=top_suppliers[0].get("composite_score", 0.0),
            )

        state.supplier_rankings = rankings
        self._log_end(state, t0, n_ranked=len(rankings))
        return state
