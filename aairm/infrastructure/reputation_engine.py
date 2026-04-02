"""Reputation and Scoring Engine — Trusted Agent Infrastructure.

Maintains longitudinal performance records for each agent and supplier.
Scores are updated after every cycle and are used by:

  - C3 (Supplier Ranking): to incorporate updated reliability scores.
  - A3 (Learning Agent):   to weight recent vs. historical performance.
  - The health monitor:    to flag degraded components.

References
----------
Paper Section 4.3 (Trusted Agent Infrastructure).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from aairm.utils.logging import get_logger

logger = get_logger(__name__)

_DECAY_FACTOR = 0.95   # exponential decay for historical data


class ReputationEngine:
    """Longitudinal performance tracker for agents and suppliers.

    Uses exponential moving averages (EMA) to weight recent performance
    more heavily than historical data.

    Args:
        decay: EMA decay factor in (0, 1).  Higher = slower decay
            (longer memory).  Defaults to 0.95.
    """

    def __init__(self, decay: float = _DECAY_FACTOR) -> None:
        self._decay = decay
        # {entity_id: {metric: ema_value}}
        self._scores: dict[str, dict[str, float]] = defaultdict(dict)
        # {entity_id: int}
        self._update_count: dict[str, int] = defaultdict(int)

    def update(self, entity_id: str, metric: str, value: float) -> None:
        """Update the EMA score for an entity on a given metric.

        Args:
            entity_id: Agent ID (e.g. ``"C3"``) or supplier ID.
            metric: Metric name (e.g. ``"reliability"``, ``"fill_rate"``).
            value: New observed value.
        """
        current = self._scores[entity_id].get(metric, value)
        updated = self._decay * current + (1.0 - self._decay) * value
        self._scores[entity_id][metric] = updated
        self._update_count[entity_id] += 1

    def get_score(self, entity_id: str, metric: str, default: float = 0.85) -> float:
        """Return the current EMA score for an entity on a given metric.

        Args:
            entity_id: Entity identifier.
            metric: Metric name.
            default: Default value if no history exists.

        Returns:
            Current EMA score.
        """
        return float(self._scores[entity_id].get(metric, default))

    def get_supplier_reliability(self, supplier_id: str) -> float:
        """Convenience method for C3 supplier reliability lookups.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            EMA on-time delivery rate in [0, 1].
        """
        return self.get_score(supplier_id, "reliability", default=0.85)

    def report(self) -> dict[str, Any]:
        """Return the full reputation report for all tracked entities.

        Returns:
            Nested dict: ``{entity_id: {metric: score, ..., n_updates: int}}``.
        """
        return {
            entity_id: {
                **metrics,
                "n_updates": self._update_count[entity_id],
            }
            for entity_id, metrics in self._scores.items()
        }
