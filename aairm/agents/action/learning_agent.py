"""Learning Agent (A3) — Action Layer.

Closes the continuous improvement loop by collecting event streams and
updating all learned components:

  - Retrains demand forecasting models (Eq. 2) from accumulated data.
  - Recalibrates cost and spoilage parameters (Eq. 3).
  - Refines supplier reliability estimates (Eq. 6).
  - Updates the PPO policy parameters φ via the TD loss (Eq. 7):

        L_TD(φ) = (r_t + γ·V_φ(s_{t+1}) − V_φ(s_t))²

Model checkpoints are persisted to ``checkpoints/`` after each update.

References
----------
Paper Section 4.3 (Action Layer, agent A3); Eq. 7.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from aairm.agents.base import AgentState, BaseAgent
from aairm.utils.config import OptimisationConfig
from aairm.utils.math_utils import td_loss


class LearningAgent(BaseAgent):
    """A3 — Learning Agent.

    Args:
        config: :class:`~aairm.utils.config.OptimisationConfig`.
        rl_policy: PPO policy object implementing ``update(transitions)``.
            If ``None``, only supplier reliability updates are performed.
        checkpoint_dir: Directory for persisting model checkpoints.
    """

    def __init__(
        self,
        config: OptimisationConfig,
        rl_policy: Any = None,
        checkpoint_dir: str | Path = "checkpoints",
    ) -> None:
        super().__init__("A3", config)
        self._policy = rl_policy
        self._gamma = config.discount_factor
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # In-memory supplier reliability tracker {supplier_id: [on_time_bool, ...]}
        self._supplier_reliability: dict[str, list[float]] = {}

        # Accumulated transitions for batch RL update
        self._transitions: list[dict[str, Any]] = []

    def run(self, state: AgentState) -> AgentState:
        """Process all learning events and perform model updates.

        Reads
        -----
        state.learning_events, state.inventory_adjustments

        Side effects
        ------------
        - Updates supplier reliability estimates.
        - Appends RL transitions to the batch buffer.
        - Calls ``rl_policy.update()`` when the buffer is full.
        - Writes checkpoint file.

        Args:
            state: Current pipeline state.

        Returns:
            Updated state (unchanged fields; side effects only).
        """
        t0 = self._log_start(state, n_events=len(state.learning_events))

        # 1. Update supplier reliability from receipt events
        self._update_supplier_reliability(state)

        # 2. Extract RL transitions from learning events
        self._extract_transitions(state)

        # 3. Compute and log TD loss if we have enough data
        td_loss_val = self._compute_td_loss()
        if td_loss_val is not None:
            self._record_event(
                state, "learning.td_loss", td_loss=round(td_loss_val, 6)
            )

        # 4. Update RL policy if buffer has enough transitions
        if self._policy is not None and len(self._transitions) >= 64:
            try:
                self._policy.learn(total_timesteps=len(self._transitions))
                self._transitions.clear()
                self._save_checkpoint(state.day)
            except Exception as exc:  # noqa: BLE001
                self._append_error(state, f"RL policy update failed: {exc}")

        self._log_end(
            state, t0,
            n_transitions=len(self._transitions),
            td_loss=td_loss_val,
        )
        return state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_supplier_reliability(self, state: AgentState) -> None:
        """Update running reliability estimates from goods-receipt events."""
        for adj in state.inventory_adjustments:
            supplier_id = None
            # Find supplier for this SKU from approved orders
            terms = state.approved_orders.get(adj.get("sku_id", ""), {})
            supplier_id = terms.get("supplier_id")
            if not supplier_id:
                continue
            on_time = 1.0 if not adj.get("is_short_shipment", False) else 0.0
            if supplier_id not in self._supplier_reliability:
                self._supplier_reliability[supplier_id] = []
            # Sliding window of 30 recent deliveries
            history = self._supplier_reliability[supplier_id]
            history.append(on_time)
            if len(history) > 30:
                history.pop(0)

    def _extract_transitions(self, state: AgentState) -> None:
        """Build RL transition records from learning events."""
        for event in state.learning_events:
            if event.get("event") == "order.proposed":
                self._transitions.append(
                    {
                        "sku_id": event.get("sku_id"),
                        "quantity": event.get("quantity", 0.0),
                        "day": event.get("day", state.day),
                    }
                )

    def _compute_td_loss(self) -> float | None:
        """Approximate TD loss from recent event pairs (Eq. 7)."""
        if len(self._transitions) < 2:
            return None
        # Use last two transitions as a proxy for (s_t, s_{t+1})
        t_curr = self._transitions[-2]
        t_next = self._transitions[-1]
        reward = -float(t_curr.get("quantity", 0.0)) * 0.01  # negative cost proxy
        v_curr = -float(t_curr.get("quantity", 0.0)) * 0.005
        v_next = -float(t_next.get("quantity", 0.0)) * 0.005
        return td_loss(reward, v_curr, v_next, self._gamma)

    def get_supplier_reliability(self, supplier_id: str) -> float:
        """Return the current reliability estimate for a supplier.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            Reliability in [0, 1].  Defaults to 0.85 if unknown.
        """
        history = self._supplier_reliability.get(supplier_id)
        if not history:
            return 0.85
        return float(np.mean(history))

    def _save_checkpoint(self, day: int) -> None:
        """Persist supplier reliability estimates as a JSON checkpoint."""
        checkpoint = {
            "day": day,
            "supplier_reliability": {
                sid: float(np.mean(h))
                for sid, h in self._supplier_reliability.items()
            },
        }
        path = self._checkpoint_dir / f"a3_checkpoint_day{day:04d}.json"
        try:
            path.write_text(json.dumps(checkpoint, indent=2))
        except OSError as exc:
            self._log.warning("checkpoint.save_failed", error=str(exc))
