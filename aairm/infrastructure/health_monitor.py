"""Agent Health Monitor — Trusted Agent Infrastructure.

Detects and isolates failing or anomalous agents by tracking per-agent
error rates across cycles.  An agent is flagged as degraded when its
rolling error rate exceeds the configured threshold.

In production this component would trigger alerts and optionally swap in
a fallback agent implementation.  In simulation mode it logs warnings and
continues pipeline execution with degraded-mode agents.

References
----------
Paper Section 4.3 (Trusted Agent Infrastructure).
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from aairm.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_WINDOW = 20      # rolling window size (cycles)
_ERROR_THRESHOLD = 0.40   # flag agent if >40% of recent cycles had errors


class AgentHealthMonitor:
    """Monitor error rates for all registered agents.

    Args:
        window_size: Number of recent cycles to include in the rolling
            error rate.  Defaults to 20.
        error_threshold: Fraction of errored cycles above which an agent
            is considered degraded.  Defaults to 0.40.
    """

    def __init__(
        self,
        window_size: int = _DEFAULT_WINDOW,
        error_threshold: float = _ERROR_THRESHOLD,
    ) -> None:
        self._window = window_size
        self._threshold = error_threshold
        # {agent_id: deque of 0/1 (0=ok, 1=error)}
        self._history: dict[str, deque[int]] = defaultdict(
            lambda: deque(maxlen=self._window)
        )

    def record_cycle(self, agent_id: str, had_error: bool) -> None:
        """Record the outcome of one agent execution.

        Args:
            agent_id: Agent identifier (e.g. ``"P1"``).
            had_error: ``True`` if the agent produced an error this cycle.
        """
        self._history[agent_id].append(int(had_error))

    def error_rate(self, agent_id: str) -> float:
        """Return the rolling error rate for an agent.

        Args:
            agent_id: Agent identifier.

        Returns:
            Error rate in [0, 1].  Returns 0.0 if no history.
        """
        hist = self._history.get(agent_id)
        if not hist:
            return 0.0
        return float(sum(hist) / len(hist))

    def is_degraded(self, agent_id: str) -> bool:
        """Return True if the agent's error rate exceeds the threshold.

        Args:
            agent_id: Agent identifier.

        Returns:
            ``True`` if the agent is considered degraded.
        """
        rate = self.error_rate(agent_id)
        degraded = rate >= self._threshold
        if degraded:
            logger.warning(
                "health.agent_degraded",
                agent=agent_id,
                error_rate=round(rate, 3),
                threshold=self._threshold,
            )
        return degraded

    def status_report(self) -> dict[str, Any]:
        """Return a health status report for all monitored agents.

        Returns:
            Dict mapping agent_id to
            ``{error_rate, degraded, cycles_monitored}``.
        """
        return {
            agent_id: {
                "error_rate": round(self.error_rate(agent_id), 4),
                "degraded": self.is_degraded(agent_id),
                "cycles_monitored": len(hist),
            }
            for agent_id, hist in self._history.items()
        }
