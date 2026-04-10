"""Abstract base class and shared pipeline state for all AAIRM agents.

Every agent in the Perception, Conceptualization, and Action layers must
inherit from :class:`BaseAgent` and implement :meth:`BaseAgent.run`.

The :class:`AgentState` dataclass is the single mutable object passed
through the entire PCA pipeline.  Agents read from it, write to it, and
return it.  The :class:`MetaOrchestrator` (LangGraph) routes it between
agents based on the pipeline configuration.

References
----------
AAIRM paper, Section 4; Repo Guide, Section 5.1.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from aairm.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared Pipeline State
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """Mutable state object passed between all agents in the PCA pipeline.

    The Meta-Orchestrator initialises one ``AgentState`` per replenishment
    cycle and routes it through every agent in sequence.  Each agent reads
    from fields populated by upstream agents and writes to its own output
    fields before returning the state.

    Agents must never replace this object; they mutate its fields and
    return the same instance.

    Attributes
    ----------
    cycle_id : str
        Unique identifier for this replenishment cycle, used in audit logs.
    day : int
        Current simulation day (0-indexed from start of test horizon).
    errors : list[str]
        Human-readable error strings appended by any agent that catches a
        recoverable failure.  Non-empty errors are surfaced to the user
        summary.

    Perception fields (written by P1–P5)
    -------------------------------------
    low_stock_skus : list[str]
        SKU IDs below their reorder-point threshold (populated by P1).
        Priority candidates; processed first by downstream agents.
    replenishment_candidates : list[str]
        SKU IDs where effective_available < lead_time_days * forecast_demand.
        Secondary candidates identified via soft thresholds (populated by P1).
    sku_inventory_snapshot : dict[str, dict]
        Per-SKU inventory record:
        ``{on_hand, reserved, in_transit, effective_available,
           lead_time_estimate, days_to_expiry}``.
    trend_signals : list[dict]
        Ranked external trend signals from P2.  Each entry:
        ``{product_name, trend_score, source, category}``.
    new_sku_candidates : list[dict]
        Underrepresented SKUs surfaced by P3 for stocking evaluation.
    context_features : dict[str, Any]
        Feature-rich contextual state assembled by P4 for forecasting.
    anomaly_alerts : list[dict]
        Irregular signals detected by P5, routed to C5 for review.

    Conceptualization fields (written by C1–C5)
    --------------------------------------------
    demand_forecasts : dict[str, dict]
        Per-SKU forecast output from C1:
        ``{mean, variance, p10, p50, p90, horizon_days}``.
    order_proposals : dict[str, float]
        Per-SKU optimal order quantities ``Q*`` from C2.
    supplier_rankings : dict[str, list[dict]]
        Per-SKU ranked supplier shortlists from C3 (top 3 per SKU).
    negotiated_terms : dict[str, dict]
        Finalised commercial terms from C4:
        ``{supplier_id, unit_price, quantity, delivery_window, payment_terms}``.
    approved_orders : dict[str, dict]
        Governance-approved orders from C5, annotated with constraint flags.

    Action fields (written by A1–A3)
    ---------------------------------
    purchase_orders_issued : list[str]
        Purchase order IDs confirmed by supplier systems (written by A1).
    inventory_adjustments : list[dict]
        Reconciliation records written by A2 after goods receipt.
    learning_events : list[dict]
        Structured event stream consumed by A3 for policy/model updates.

    Metrics (written by the Benchmarker, not agents)
    ------------------------------------------------
    metrics : dict[str, float]
        Cycle-level metric snapshot for real-time monitoring.
    """

    # --- metadata ---
    cycle_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    day: int = 0
    errors: list[str] = field(default_factory=list)

    # --- perception ---
    low_stock_skus: list[str] = field(default_factory=list)
    replenishment_candidates: list[str] = field(default_factory=list)
    sku_inventory_snapshot: dict[str, dict[str, Any]] = field(default_factory=dict)
    trend_signals: list[dict[str, Any]] = field(default_factory=list)
    new_sku_candidates: list[dict[str, Any]] = field(default_factory=list)
    context_features: dict[str, Any] = field(default_factory=dict)
    anomaly_alerts: list[dict[str, Any]] = field(default_factory=list)

    # --- conceptualization ---
    demand_forecasts: dict[str, dict[str, Any]] = field(default_factory=dict)
    order_proposals: dict[str, float] = field(default_factory=dict)
    supplier_rankings: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    negotiated_terms: dict[str, dict[str, Any]] = field(default_factory=dict)
    approved_orders: dict[str, dict[str, Any]] = field(default_factory=dict)

    # --- action ---
    purchase_orders_issued: list[str] = field(default_factory=list)
    inventory_adjustments: list[dict[str, Any]] = field(default_factory=list)
    learning_events: list[dict[str, Any]] = field(default_factory=list)

    # --- metrics ---
    metrics: dict[str, float] = field(default_factory=dict)

    def has_errors(self) -> bool:
        """Return True if any agent recorded a recoverable error this cycle."""
        return len(self.errors) > 0

    def summary(self) -> dict[str, Any]:
        """Return a compact summary dict suitable for user-facing logging."""
        return {
            "cycle_id": self.cycle_id,
            "day": self.day,
            "low_stock_count": len(self.low_stock_skus),
            "orders_proposed": len(self.order_proposals),
            "orders_approved": len(self.approved_orders),
            "pos_issued": len(self.purchase_orders_issued),
            "errors": len(self.errors),
            "metrics": self.metrics,
        }


# ---------------------------------------------------------------------------
# Abstract Base Agent
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """Abstract base for all Perception, Conceptualization, and Action agents.

    Subclasses must implement :meth:`run`.  They should use
    :meth:`_log_start` / :meth:`_log_end` and :meth:`_record_event` for
    consistent structured logging and audit-trail compliance.

    Attributes
    ----------
    agent_id : str
        Canonical identifier matching the paper notation (e.g. ``"P1"``,
        ``"C2"``, ``"A3"``).
    config : Any
        Agent-specific configuration object (Pydantic model).
    """

    def __init__(self, agent_id: str, config: Any) -> None:
        """Initialise the agent.

        Args:
            agent_id: Canonical identifier from the paper (e.g. ``"P1"``).
            config: Pydantic configuration object for this agent's layer.
        """
        self.agent_id = agent_id
        self.config = config
        self._log = get_logger(__name__).bind(agent=agent_id)

    @abstractmethod
    def run(self, state: AgentState) -> AgentState:
        """Execute this agent's primary function and return the updated state.

        Args:
            state: Current shared pipeline state.  Read upstream fields;
                write to this agent's designated output fields; return the
                same object.

        Returns:
            The updated :class:`AgentState` object.

        Notes:
            - Must not raise unhandled exceptions.  Catch tool-call failures,
              append to ``state.errors``, fall back to a safe default, and
              return the state.
            - Must call :meth:`_log_start` at entry and :meth:`_log_end`
              at exit.
        """

    # ------------------------------------------------------------------
    # Protected helpers
    # ------------------------------------------------------------------

    def _log_start(self, state: AgentState, **extra: Any) -> float:
        """Log agent entry and return the wall-clock start time."""
        self._log.info(
            "agent.start",
            cycle_id=state.cycle_id,
            day=state.day,
            **extra,
        )
        return time.perf_counter()

    def _log_end(self, state: AgentState, start_time: float, **extra: Any) -> None:
        """Log agent exit with elapsed time."""
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
        self._log.info(
            "agent.complete",
            cycle_id=state.cycle_id,
            day=state.day,
            elapsed_ms=elapsed_ms,
            errors=len(state.errors),
            **extra,
        )

    def _record_event(
        self,
        state: AgentState,
        event_type: str,
        **payload: Any,
    ) -> None:
        """Append a structured event to the learning stream for A3.

        Args:
            state: Current pipeline state.
            event_type: Short dot-separated event identifier
                (e.g. ``"order.proposed"``).
            **payload: Arbitrary key-value pairs logged with the event.
        """
        state.learning_events.append(
            {
                "agent": self.agent_id,
                "event": event_type,
                "cycle_id": state.cycle_id,
                "day": state.day,
                **payload,
            }
        )

    def _append_error(self, state: AgentState, message: str) -> None:
        """Record a recoverable error and log it at WARNING level."""
        state.errors.append(f"[{self.agent_id}] {message}")
        self._log.warning("agent.error", cycle_id=state.cycle_id, message=message)
