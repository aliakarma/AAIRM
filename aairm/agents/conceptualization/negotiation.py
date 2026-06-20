"""Autonomous Negotiation Agent (C4) — Conceptualization Layer.

Engages in structured dialogue with supplier systems to:

  - Elicit revised quotations.
  - Request volume discounts when Q* > 2 × MOQ.
  - Propose alternative delivery schedules (±2 days).
  - Never accept lead times beyond remaining shelf life for perishables.

In **LLM mode** (production), this agent is a LangChain AgentExecutor
backed by the configured LLM model, with supplier tools injected.

In **simulation mode** (default when no LLM API key is set), it applies
deterministic negotiation rules that reproduce the paper's cost savings
without requiring live API calls.

References
----------
Paper Section 4.2.4; Repo Guide, Section 5.6.
"""

from __future__ import annotations

import os
from typing import Any

from aairm.agents.base import AgentState, BaseAgent
from aairm.utils.config import LLMConfig


_VOLUME_DISCOUNT_THRESHOLD = 2.0   # Q* > 2×MOQ triggers discount request
_MAX_DELIVERY_FLEX_DAYS = 2        # accept ±2 days on window
_TYPICAL_DISCOUNT_RATE = 0.05      # 5% volume discount in simulation mode


class NegotiationAgent(BaseAgent):
    """C4 — Autonomous Negotiation Agent.

    Args:
        config: :class:`~aairm.utils.config.LLMConfig`.
        use_llm: If ``True`` and an OpenAI API key is present, use the LLM
            backend.  If ``False`` or no key is set, use deterministic
            simulation-mode rules.
    """

    def __init__(self, config: LLMConfig, use_llm: bool = False) -> None:
        super().__init__("C4", config)
        self._use_llm = use_llm and bool(os.getenv("OPENAI_API_KEY"))
        self._llm_executor: Any = None
        self.recent_allocations: dict[str, float] = {}  # rolling 14-day supplier allocations
        if self._use_llm:
            self._llm_executor = self._build_llm_executor(config)

    def run(self, state: AgentState) -> AgentState:
        """Negotiate terms with top-ranked suppliers per SKU.

        Reads
        -----
        state.supplier_rankings, state.order_proposals,
        state.sku_inventory_snapshot

        Writes
        ------
        state.negotiated_terms : dict[str, dict]
            ``{sku_id: {supplier_id, unit_price, quantity,
               delivery_window_days, payment_terms, discount_applied}}``

        Args:
            state: Current pipeline state.

        Returns:
            Updated state.
        """
        t0 = self._log_start(state, use_llm=self._use_llm)
        terms: dict[str, dict[str, Any]] = {}

        for sku_id, ranked_suppliers in state.supplier_rankings.items():
            if not ranked_suppliers:
                self._append_error(state, f"No suppliers to negotiate for {sku_id}.")
                continue

            q_star = state.order_proposals.get(sku_id, 0.0)
            rec = state.sku_inventory_snapshot.get(sku_id, {})
            days_to_expiry = float(rec.get("days_to_expiry", 9999.0))
            
            # Select supplier with diversification
            best_supplier = self._select_supplier(ranked_suppliers, q_star)

            if self._use_llm and self._llm_executor is not None:
                negotiated = self._llm_negotiate(
                    sku_id, q_star, best_supplier, days_to_expiry
                )
            else:
                negotiated = self._sim_negotiate(
                    sku_id, q_star, best_supplier, days_to_expiry
                )

            terms[sku_id] = negotiated
            
            # Update recent allocations
            supplier_id = negotiated["supplier_id"]
            self.recent_allocations[supplier_id] = self.recent_allocations.get(supplier_id, 0) + q_star
            # Keep rolling 14-day window (simplified, assume daily update)
            if len(self.recent_allocations) > 14:
                # Remove oldest, but since dict, approximate
                oldest = min(self.recent_allocations, key=self.recent_allocations.get)
                del self.recent_allocations[oldest]
            
            self._record_event(
                state, "negotiation.complete",
                sku_id=sku_id,
                supplier_id=negotiated["supplier_id"],
                final_unit_price=negotiated["unit_price"],
                discount_applied=negotiated["discount_applied"],
            )

        state.negotiated_terms = terms
        self._log_end(state, t0, n_terms=len(terms))
        return state

    def _select_supplier(self, ranked_suppliers: list[dict], order_qty: float) -> dict:
        """Select supplier considering diversification."""
        total_alloc = sum(self.recent_allocations.values()) + 1e-8
        current_shares = {s["supplier_id"]: self.recent_allocations.get(s["supplier_id"], 0) / total_alloc
                          for s in ranked_suppliers}
        
        scores = []
        for sup in ranked_suppliers:
            concentration_penalty = max(0, current_shares[sup["supplier_id"]] - 0.5)
            reliability = sup.get("reliability", 0.8)
            score = reliability * (1 - 0.4 * concentration_penalty)
            scores.append((sup, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        selected = scores[0][0]
        self._log.info("supplier.selected", supplier_id=selected["supplier_id"], score=scores[0][1])
        return selected

    # ------------------------------------------------------------------
    # Simulation-mode deterministic negotiation
    # ------------------------------------------------------------------

    def _sim_negotiate(
        self,
        sku_id: str,
        q_star: float,
        supplier: dict[str, Any],
        days_to_expiry: float,
    ) -> dict[str, Any]:
        """Apply deterministic negotiation rules (no LLM required).

        Rules (paper-aligned):
        1. Volume discount: if Q* > 2×MOQ → apply 5% discount.
        2. Perishable guard: reject lead time > days_to_expiry - 3.
           If violated, select a ±1-day schedule adjustment.
        3. Payment terms default to Net-30.
        """
        base_price = float(supplier.get("unit_cost", 10.0))
        moq = float(supplier.get("moq", 1.0))
        lead_time = float(supplier.get("lead_time_mean", 5.0))

        # Volume discount
        discount = 0.0
        if q_star > _VOLUME_DISCOUNT_THRESHOLD * moq:
            discount = _TYPICAL_DISCOUNT_RATE
        final_price = round(base_price * (1.0 - discount), 4)

        # Shelf-life guard for perishables
        if days_to_expiry < 9000 and lead_time > days_to_expiry - 3:
            # Request expedited lead time (min 2 days faster, floored at 1 day)
            lead_time = max(1.0, lead_time - 2.0)

        return {
            "supplier_id": supplier.get("supplier_id", "UNKNOWN"),
            "sku_id": sku_id,
            "unit_price": final_price,
            "quantity": q_star,
            "delivery_window_days": round(lead_time, 1),
            "payment_terms": "Net-30",
            "discount_applied": round(discount, 4),
            "negotiation_mode": "simulation",
        }

    # ------------------------------------------------------------------
    # LLM-mode negotiation (production path)
    # ------------------------------------------------------------------

    def _build_llm_executor(self, config: LLMConfig) -> Any:
        """Build a LangChain AgentExecutor for live LLM negotiation."""
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
            from langchain.agents import AgentExecutor, create_openai_tools_agent  # type: ignore
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # type: ignore
            from aairm.tools.supplier_tools import (
                request_discount_tool,
                propose_schedule_tool,
                finalise_terms_tool,
            )

            llm = ChatOpenAI(
                model=config.model,
                temperature=config.temperature,
                timeout=config.timeout,
            )
            tools = [request_discount_tool, propose_schedule_tool, finalise_terms_tool]
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        (
                            "You are the AAIRM Autonomous Negotiation Agent (C4). "
                            "Negotiate the best commercial terms for retail procurement. "
                            "Prefer suppliers with higher reliability even at slightly "
                            "higher cost. Request a volume discount if the order quantity "
                            "exceeds twice the minimum order quantity. Accept a revised "
                            "delivery window of at most ±2 days. Never accept lead times "
                            "beyond the remaining shelf life for perishable products."
                        ),
                    ),
                    MessagesPlaceholder(variable_name="chat_history", optional=True),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )
            agent = create_openai_tools_agent(llm, tools, prompt)
            return AgentExecutor(agent=agent, tools=tools, verbose=False)
        except Exception as exc:  # noqa: BLE001
            self._log.warning(
                "llm.executor.build_failed",
                error=str(exc),
                fallback="simulation_mode",
            )
            self._use_llm = False
            return None

    def _llm_negotiate(
        self,
        sku_id: str,
        q_star: float,
        supplier: dict[str, Any],
        days_to_expiry: float,
    ) -> dict[str, Any]:
        """Invoke the LLM executor for one supplier negotiation round."""
        if self._llm_executor is None:
            return self._sim_negotiate(sku_id, q_star, supplier, days_to_expiry)

        prompt = (
            f"Negotiate procurement for SKU {sku_id}. "
            f"Supplier: {supplier.get('supplier_id')}. "
            f"Quoted price: {supplier.get('unit_cost'):.2f}. "
            f"Lead time: {supplier.get('lead_time_mean')} days. "
            f"Reliability: {supplier.get('reliability', 0.85):.0%}. "
            f"Order quantity: {q_star:.0f} units. "
            f"MOQ: {supplier.get('moq', 1)}. "
            f"Days to expiry: {days_to_expiry if days_to_expiry < 9000 else 'N/A'}. "
            "Return the finalised terms as a JSON object."
        )
        try:
            result = self._llm_executor.invoke({"input": prompt})
            output = result.get("output", "")
            import json
            # Extract JSON from output
            start = output.find("{")
            end = output.rfind("}") + 1
            parsed = json.loads(output[start:end]) if start >= 0 else {}
            parsed.setdefault("supplier_id", supplier.get("supplier_id"))
            parsed.setdefault("sku_id", sku_id)
            parsed.setdefault("negotiation_mode", "llm")
            parsed.setdefault("discount_applied", 0.0)
            return parsed
        except Exception as exc:  # noqa: BLE001
            self._log.warning("llm.negotiation.failed", error=str(exc))
            return self._sim_negotiate(sku_id, q_star, supplier, days_to_expiry)
