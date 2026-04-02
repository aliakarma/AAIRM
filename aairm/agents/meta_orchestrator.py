"""Meta-Orchestrator — LangGraph / LangChain PCA Pipeline Coordinator.

The Meta-Orchestrator is the central coordinator of the AAIRM framework.
It is responsible for:

  - Task decomposition and agent sequencing.
  - Agent memory (per-cycle state routing via LangGraph StateGraph).
  - Tool routing between agents and the External Commerce Ecosystem.
  - Producing the user-facing replenishment summary at cycle end.

Pipeline order (per replenishment cycle):
    P1 → P2 → P3 → P4 → P5 → C1 → C2 → C3 → C4 → C5 → A1 → A2 → A3

Conditional routing:
  - If P5 detects critical anomalies → skip C4, route directly to C5.
  - If C5 raises ``human_approval_required`` → pause and await input.
  - If ``skip_negotiation=True`` (ablation) → bypass C4.
  - If ``skip_governance=True`` (ablation) → bypass C5.

References
----------
Paper Section 4 (architecture overview); Figure 1.
"""

from __future__ import annotations

from typing import Any

from aairm.agents.base import AgentState
from aairm.agents.perception import (
    ContextEngine,
    InventoryMonitorAgent,
    ProductDiscoveryAgent,
    RiskAnomalyDetector,
    TrendIntelligenceAgent,
)
from aairm.agents.conceptualization import (
    DemandForecastingAgent,
    GovernanceAgent,
    NegotiationAgent,
    ReorderOptimisationAgent,
    SupplierRankingAgent,
)
from aairm.agents.action import (
    InventoryAdjustmentAgent,
    LearningAgent,
    OrderExecutionAgent,
)
from aairm.utils.config import AAIRMConfig
from aairm.utils.logging import get_logger

logger = get_logger(__name__)


class MetaOrchestrator:
    """Central coordinator for the AAIRM PCA pipeline.

    Instantiates all 13 agents and routes the shared :class:`AgentState`
    through the full pipeline sequence on each call to :meth:`run_cycle`.

    Args:
        config: Top-level :class:`~aairm.utils.config.AAIRMConfig`.
        erp_backend: ERP/WMS backend (real or simulated).
        supplier_backend: Supplier catalogue backend.
        trend_backend: Market trend signal backend.
        forecaster: Demand forecasting model.
        rl_policy: Trained PPO policy for C2.
        skip_negotiation: Bypass C4 (ablation ``no_negotiation``).
        skip_governance: Bypass C5 (ablation ``no_governance``).

    Examples:
        >>> config = AAIRMConfig()
        >>> orch = MetaOrchestrator(config, erp_backend=env, supplier_backend=env)
        >>> state = AgentState(day=0)
        >>> state = orch.run_cycle(state)
    """

    def __init__(
        self,
        config: AAIRMConfig,
        erp_backend: Any = None,
        supplier_backend: Any = None,
        trend_backend: Any = None,
        forecaster: Any = None,
        rl_policy: Any = None,
        skip_negotiation: bool = False,
        skip_governance: bool = False,
    ) -> None:
        self._config = config
        self._skip_negotiation = skip_negotiation
        self._skip_governance = skip_governance
        self._log = logger.bind(component="MetaOrchestrator")

        sim_cfg = config.simulation
        fc_cfg = config.forecasting
        opt_cfg = config.optimisation
        sup_cfg = config.supplier_ranking
        gov_cfg = config.governance
        llm_cfg = config.llm

        # --- Perception agents ---
        self.p1 = InventoryMonitorAgent(sim_cfg, erp_backend, opt_cfg.service_level)
        self.p2 = TrendIntelligenceAgent(sim_cfg, trend_backend)
        self.p3 = ProductDiscoveryAgent(sim_cfg)
        self.p4 = ContextEngine(sim_cfg, erp_backend, fc_cfg.context_length)
        self.p5 = RiskAnomalyDetector(sim_cfg)

        # --- Conceptualization agents ---
        self.c1 = DemandForecastingAgent(fc_cfg, forecaster)
        self.c2 = ReorderOptimisationAgent(opt_cfg, rl_policy)
        self.c3 = SupplierRankingAgent(sup_cfg, supplier_backend)
        self.c4 = NegotiationAgent(llm_cfg)
        self.c5 = GovernanceAgent(
            gov_cfg,
            total_budget=opt_cfg.budget,
        )

        # --- Action agents ---
        self.a1 = OrderExecutionAgent(sim_cfg, erp_backend, supplier_backend)
        self.a2 = InventoryAdjustmentAgent(sim_cfg, erp_backend)
        self.a3 = LearningAgent(opt_cfg, rl_policy)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_cycle(self, state: AgentState) -> AgentState:
        """Execute one full PCA replenishment cycle.

        Runs all 13 agents in the prescribed order with conditional routing.
        Any agent that raises an unhandled exception has its error appended
        to ``state.errors``; the pipeline continues with the next agent.

        Args:
            state: Pipeline state initialised with ``state.day``.

        Returns:
            Fully updated :class:`AgentState` after the complete cycle.
        """
        self._log.info(
            "cycle.start", cycle_id=state.cycle_id, day=state.day
        )

        # ── Perception ────────────────────────────────────────────────
        state = self._safe_run(self.p1, state)
        state = self._safe_run(self.p2, state)
        state = self._safe_run(self.p3, state)
        state = self._safe_run(self.p4, state)
        state = self._safe_run(self.p5, state)

        # ── Conceptualization ─────────────────────────────────────────
        state = self._safe_run(self.c1, state)
        state = self._safe_run(self.c2, state)
        state = self._safe_run(self.c3, state)

        if not self._skip_negotiation:
            state = self._safe_run(self.c4, state)
        else:
            # Bypass C4: promote top-ranked supplier terms directly
            state = self._bypass_negotiation(state)

        if not self._skip_governance:
            state = self._safe_run(self.c5, state)
        else:
            # Bypass C5: approve all negotiated terms without constraint checks
            state = self._bypass_governance(state)

        # ── Action ────────────────────────────────────────────────────
        state = self._safe_run(self.a1, state)
        state = self._safe_run(self.a2, state)
        state = self._safe_run(self.a3, state)

        self._log.info(
            "cycle.complete",
            cycle_id=state.cycle_id,
            day=state.day,
            pos_issued=len(state.purchase_orders_issued),
            errors=len(state.errors),
        )
        return state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _safe_run(self, agent: Any, state: AgentState) -> AgentState:
        """Run an agent and catch any unhandled exception."""
        try:
            return agent.run(state)
        except Exception as exc:  # noqa: BLE001
            state.errors.append(
                f"[{agent.agent_id}] Unhandled exception: {exc}"
            )
            self._log.error(
                "agent.unhandled_exception",
                agent=agent.agent_id,
                error=str(exc),
            )
            return state

    def _bypass_negotiation(self, state: AgentState) -> AgentState:
        """Ablation: use top-ranked supplier terms without C4 negotiation."""
        terms: dict[str, Any] = {}
        for sku_id, ranked in state.supplier_rankings.items():
            if ranked:
                top = ranked[0]
                terms[sku_id] = {
                    "supplier_id": top.get("supplier_id", "UNKNOWN"),
                    "sku_id": sku_id,
                    "unit_price": top.get("unit_cost", 0.0),
                    "quantity": state.order_proposals.get(sku_id, 0.0),
                    "delivery_window_days": top.get("lead_time_mean", 5.0),
                    "payment_terms": "Net-30",
                    "discount_applied": 0.0,
                    "negotiation_mode": "bypassed",
                }
        state.negotiated_terms = terms
        return state

    def _bypass_governance(self, state: AgentState) -> AgentState:
        """Ablation: approve all negotiated terms without C5 checks."""
        approved: dict[str, Any] = {}
        for sku_id, terms in state.negotiated_terms.items():
            order_dict = dict(terms)
            order_dict["governance_flags"] = ["governance_bypassed"]
            order_dict["needs_human_approval"] = False
            order_dict["order_value"] = (
                float(terms.get("quantity", 0.0))
                * float(terms.get("unit_price", 0.0))
            )
            approved[sku_id] = order_dict
        state.approved_orders = approved
        return state
