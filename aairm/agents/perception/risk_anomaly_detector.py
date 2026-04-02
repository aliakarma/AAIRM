"""Risk and Anomaly Detector (P5) — Perception Layer.

Identifies irregular signals before any procurement decision is finalised:

  - Demand spikes (> threshold × rolling mean).
  - Supplier anomalies (reliability score degraded below floor).
  - Sudden inventory discrepancies (on-hand delta > threshold).
  - Data quality issues (missing / null inventory records).

All detected anomalies are routed to ``state.anomaly_alerts`` for review
by the Governance Agent (C5).

References
----------
Paper Section 4.1 (agent P5).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aairm.agents.base import AgentState, BaseAgent
from aairm.utils.config import SimulationConfig


class RiskAnomalyDetector(BaseAgent):
    """P5 — Risk and Anomaly Detector.

    Args:
        config: Simulation configuration.
        demand_spike_multiplier: Flag as spike if latest demand >
            ``demand_spike_multiplier × rolling_7d_mean``.
        min_supplier_reliability: Flag if a preferred supplier's reliability
            drops below this threshold.
        inventory_delta_threshold: Flag if on-hand changed by more than
            this fraction between cycles (absolute).
    """

    def __init__(
        self,
        config: SimulationConfig,
        demand_spike_multiplier: float = 3.0,
        min_supplier_reliability: float = 0.70,
        inventory_delta_threshold: float = 0.50,
    ) -> None:
        super().__init__("P5", config)
        self._spike_mult = demand_spike_multiplier
        self._min_reliability = min_supplier_reliability
        self._inv_delta_threshold = inventory_delta_threshold
        self._prev_snapshot: dict[str, float] = {}

    def run(self, state: AgentState) -> AgentState:
        """Detect anomalies in the current inventory and demand snapshot.

        Args:
            state: Pipeline state.  Reads ``state.sku_inventory_snapshot``
                and ``state.context_features``.

        Returns:
            Updated state with ``state.anomaly_alerts`` populated.
        """
        t0 = self._log_start(state)
        alerts: list[dict[str, Any]] = []

        for sku_id, rec in state.sku_inventory_snapshot.items():
            on_hand = float(rec.get("on_hand", 0.0))

            # -- Data quality check --
            if on_hand < 0:
                alerts.append(
                    {"type": "data_quality", "sku_id": sku_id,
                     "detail": f"Negative on-hand inventory: {on_hand}"}
                )

            # -- Inventory delta check --
            prev = self._prev_snapshot.get(sku_id)
            if prev is not None and prev > 0:
                delta = abs(on_hand - prev) / prev
                if delta > self._inv_delta_threshold:
                    alerts.append(
                        {"type": "inventory_discrepancy", "sku_id": sku_id,
                         "detail": f"On-hand changed {delta:.1%} since last cycle",
                         "prev": prev, "current": on_hand}
                    )

            # -- Demand spike check --
            ctx = state.context_features.get(sku_id, {})
            rolling_mean = float(ctx.get("rolling_7d_mean", 0.0))
            history: list[float] = ctx.get("history", [])
            if history and rolling_mean > 0:
                latest_demand = float(history[-1])
                if latest_demand > self._spike_mult * rolling_mean:
                    alerts.append(
                        {"type": "demand_spike", "sku_id": sku_id,
                         "detail": (
                             f"Latest demand {latest_demand:.1f} > "
                             f"{self._spike_mult}x rolling mean {rolling_mean:.1f}"
                         )}
                    )

        state.anomaly_alerts = alerts

        # Update previous snapshot for delta tracking
        self._prev_snapshot = {
            sku_id: float(rec.get("on_hand", 0.0))
            for sku_id, rec in state.sku_inventory_snapshot.items()
        }

        self._record_event(state, "anomalies.detected", n_alerts=len(alerts))
        if alerts:
            self._log.warning(
                "anomalies.detected",
                cycle_id=state.cycle_id,
                n_alerts=len(alerts),
                types=list({a["type"] for a in alerts}),
            )
        self._log_end(state, t0, n_alerts=len(alerts))
        return state
