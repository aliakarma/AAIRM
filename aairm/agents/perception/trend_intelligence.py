"""Trend and Market Intelligence Agent (P2) — Perception Layer.

Monitors external signals to identify products or attributes exhibiting
strong positive momentum.  In simulation mode, trend scores are derived
from the synthetic demand generator's promotional and seasonal signals.
In production, this agent would query social media APIs, search-trend
APIs, and third-party marketplace feeds.

References
----------
Paper Section 4.1 (agent P2).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aairm.agents.base import AgentState, BaseAgent
from aairm.utils.config import SimulationConfig


class TrendIntelligenceAgent(BaseAgent):
    """P2 — Trend and Market Intelligence Agent.

    Args:
        config: Simulation configuration.
        trend_backend: Object implementing ``get_trend_signals(day)``
            that returns a list of trend-signal dicts.  In simulation mode
            this is provided by the DemandGenerator.
        top_k: Maximum number of trend signals to forward downstream.
    """

    def __init__(
        self,
        config: SimulationConfig,
        trend_backend: Any = None,
        top_k: int = 20,
    ) -> None:
        super().__init__("P2", config)
        self._backend = trend_backend
        self._top_k = top_k

    def run(self, state: AgentState) -> AgentState:
        """Retrieve and rank external trend signals.

        Args:
            state: Current pipeline state.

        Returns:
            Updated state with ``state.trend_signals`` populated.
        """
        t0 = self._log_start(state)

        if self._backend is None:
            # Graceful degradation: return empty trend list
            state.trend_signals = []
            self._log_end(state, t0, n_trends=0)
            return state

        try:
            raw_signals: list[dict[str, Any]] = self._backend.get_trend_signals(
                state.day
            )
        except Exception as exc:  # noqa: BLE001
            self._append_error(state, f"Trend backend failed: {exc}")
            state.trend_signals = []
            self._log_end(state, t0, n_trends=0)
            return state

        # Rank by trend_score descending; take top-k
        ranked = sorted(raw_signals, key=lambda x: x.get("trend_score", 0.0), reverse=True)
        state.trend_signals = ranked[: self._top_k]

        self._record_event(
            state,
            "trends.retrieved",
            n_signals=len(state.trend_signals),
            top_score=state.trend_signals[0].get("trend_score", 0.0)
            if state.trend_signals
            else 0.0,
        )
        self._log_end(state, t0, n_trends=len(state.trend_signals))
        return state
