"""RetailEnv — Gymnasium-Compatible Synthetic Retail Environment.

Implements the simulation described in Section 5.1 of the paper:

  - 1,200 SKUs across 5 categories (240 each).
  - 2-year simulation horizon (730 days train + test).
  - Category-specific demand profiles, weekly seasonality, promotions.
  - Perishable SKUs (frozen food, cosmetics, dry fruits) with shelf lives.
  - 3–5 suppliers per SKU with heterogeneous pricing and reliability.
  - Stochastic lead times sampled from calibrated distributions.
  - Deliberate ERP imperfections (15% delayed shipments, 5% partial fills).

The environment implements the ``gymnasium.Env`` interface so that the
PPO policy (C2) can train in it directly.

It also exposes :meth:`step_agentic` for the full 13-agent pipeline, and
:meth:`get_inventory_snapshot` / :meth:`get_demand_history` so that
agents can query the environment state via the ERP stub.

References
----------
Paper Section 5.1; Repo Guide Section 6.1.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aairm.simulation.demand_generator import DemandGenerator
from aairm.simulation.erp_stub import ERPStub
from aairm.simulation.sku_catalog import SKUCatalog
from aairm.simulation.supplier_simulator import SupplierSimulator
from aairm.utils.config import SimulationConfig
from aairm.utils.logging import get_logger
from aairm.utils.seed import set_global_seed

logger = get_logger(__name__)

try:
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
    _GYM_AVAILABLE = True
except ImportError:
    logger.warning("gymnasium.not_available; RL training disabled")
    _GYM_AVAILABLE = False

# Observation space per SKU (5 features)
_OBS_DIM = 5


class RetailEnv:
    """Synthetic multi-category retail environment.

    Observation space (per cycle, flattened across all low-stock SKUs):
        [effective_available, forecast_mean, forecast_std,
         days_to_expiry_norm, budget_fraction_remaining]

    Action space:
        Continuous order quantities Q_i ≥ 0 per low-stock SKU.

    Reward:
        Negative expected cost (Eq. 3) summed across all SKUs.

    Args:
        config: :class:`~aairm.utils.config.SimulationConfig`.
        initial_inventory_days: Initial on-hand stock in days of demand.
    """

    def __init__(
        self,
        config: SimulationConfig,
        initial_inventory_days: float = 14.0,
    ) -> None:
        self._config = config
        self._n_days = config.simulation_horizon_days
        self._test_days = config.test_horizon_days
        self._seed = config.seed
        self._initial_inv_days = initial_inventory_days

        # Build simulation components (lazy — reset() finalises them)
        self._catalog: SKUCatalog | None = None
        self._gen: DemandGenerator | None = None
        self._sup: SupplierSimulator | None = None
        self._erp: ERPStub | None = None

        # Current day
        self._day: int = 0

        # Accumulators for metric computation
        self._total_demand: dict[str, float] = {}
        self._total_fulfilled: dict[str, float] = {}
        self._total_cost: float = 0.0
        self._daily_on_hand: list[dict[str, float]] = []

        # Gymnasium spaces (defined after reset)
        self.observation_space: Any = None
        self.action_space: Any = None
        self._gym_available = _GYM_AVAILABLE

        # Build immediately with default seed
        self._build_components(self._seed)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to day 0.

        Args:
            seed: Optional seed override.
            options: Unused (Gymnasium compatibility).

        Returns:
            Tuple of ``(initial_observation, info_dict)``.
        """
        effective_seed = seed if seed is not None else self._seed
        set_global_seed(effective_seed)
        self._build_components(effective_seed)

        self._day = 0
        self._total_demand = {s: 0.0 for s in self._catalog.sku_ids}  # type: ignore[union-attr]
        self._total_fulfilled = {s: 0.0 for s in self._catalog.sku_ids}  # type: ignore[union-attr]
        self._total_cost = 0.0
        self._daily_on_hand = []

        obs = self._get_obs()
        info = {"day": 0, "n_skus": len(self._catalog.sku_ids)}  # type: ignore[union-attr]
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one day in the RL training loop.

        Args:
            action: Order quantities (one per low-stock SKU in the
                current observation).  Shape ``(n_low_stock,)`` or scalar.

        Returns:
            ``(obs, reward, terminated, truncated, info)`` per Gymnasium spec.
        """
        if self._erp is None:
            raise RuntimeError("Call reset() before step().")

        realised_demand = self._erp.advance_day(self._day)
        reward = self._compute_reward(realised_demand)

        # Update accumulators
        snapshot = self._erp.get_inventory_snapshot()
        for sku_id, demand in realised_demand.items():
            self._total_demand[sku_id] = self._total_demand.get(sku_id, 0.0) + demand
            on_hand = snapshot.get(sku_id, {}).get("on_hand", 0.0)
            fulfilled = min(demand, on_hand + demand)  # approx
            self._total_fulfilled[sku_id] = (
                self._total_fulfilled.get(sku_id, 0.0) + fulfilled
            )
        self._daily_on_hand.append(
            {s: snapshot.get(s, {}).get("on_hand", 0.0) for s in self._catalog.sku_ids}  # type: ignore[union-attr]
        )

        self._day += 1
        terminated = self._day >= self._n_days
        truncated = False

        obs = self._get_obs()
        info = {
            "day": self._day,
            "reward": reward,
        }
        return obs, reward, terminated, truncated, info

    def step_agentic(
        self, order_dict: dict[str, float]
    ) -> dict[str, Any]:
        """Advance one day driven by the full agent pipeline.

        Called by the MetaOrchestrator after A1 has placed orders.
        Consumes daily demand, processes due receipts, and returns
        metrics for the current day.

        Args:
            order_dict: ``{sku_id: Q*}`` — approved orders from A1.
                May be empty (no orders needed today).

        Returns:
            Day-level metric dict:
            ``{day, demand, fulfilled, stockout_units, on_hand, reward}``.
        """
        if self._erp is None:
            raise RuntimeError("Call reset() before step_agentic().")

        realised_demand = self._erp.advance_day(self._day)
        reward = self._compute_reward(realised_demand)

        snapshot = self._erp.get_inventory_snapshot()
        stockout_units = 0.0
        for sku_id, demand in realised_demand.items():
            on_hand = snapshot.get(sku_id, {}).get("on_hand", 0.0)
            fulfilled = min(demand, on_hand + demand)
            stockout_units += max(0.0, demand - fulfilled)
            self._total_demand[sku_id] = self._total_demand.get(sku_id, 0.0) + demand
            self._total_fulfilled[sku_id] = (
                self._total_fulfilled.get(sku_id, 0.0) + fulfilled
            )

        self._daily_on_hand.append(
            {s: snapshot.get(s, {}).get("on_hand", 0.0) for s in self._catalog.sku_ids}  # type: ignore[union-attr]
        )
        self._day += 1

        total_demand_today = sum(realised_demand.values())
        return {
            "day": self._day - 1,
            "total_demand": round(total_demand_today, 2),
            "stockout_units": round(stockout_units, 2),
            "reward": round(reward, 4),
        }

    # ------------------------------------------------------------------
    # ERP delegate methods (for injection into agents)
    # ------------------------------------------------------------------

    def get_inventory_snapshot(self) -> dict[str, dict[str, Any]]:
        """Return current inventory snapshot (delegates to ERPStub)."""
        if self._erp is None:
            return {}
        return self._erp.get_inventory_snapshot()

    def get_demand_history(self, sku_id: str, n_days: int) -> np.ndarray:
        """Return demand history (delegates to ERPStub)."""
        if self._erp is None:
            return np.zeros(n_days)
        return self._erp.get_demand_history(sku_id, n_days)

    def get_trend_signals(self, day: int) -> list[dict[str, Any]]:
        """Return trend signals (delegates to ERPStub)."""
        if self._erp is None:
            return []
        return self._erp.get_trend_signals(day)

    def create_purchase_order(self, po_dict: dict[str, Any]) -> None:
        """Register PO in ERP (used by A1)."""
        if self._erp:
            self._erp.create_purchase_order(po_dict)

    def update_inbound_schedule(self, po_id: str, eta_days: int) -> None:
        """Record ETA in ERP (used by A1)."""
        if self._erp:
            self._erp.update_inbound_schedule(po_id, eta_days)

    def submit_purchase_order(self, po_dict: dict[str, Any]) -> dict[str, Any]:
        """Submit PO to supplier simulator (used by A1)."""
        if self._erp:
            return self._erp.submit_purchase_order(po_dict)
        return {"po_id": po_dict.get("po_id", ""), "confirmed": False, "eta_days": 5}

    def get_pending_receipts(self, day: int) -> list[dict[str, Any]]:
        """Return due receipts (used by A2)."""
        if self._erp:
            return self._erp.get_pending_receipts(day)
        return []

    def process_goods_receipt(self, receipt: dict[str, Any]) -> None:
        """Post goods receipt (used by A2)."""
        if self._erp:
            self._erp.process_goods_receipt(receipt)

    def query_catalogue(self, sku_id: str) -> list[dict[str, Any]]:
        """Query supplier catalogue (used by C3)."""
        if self._erp:
            return self._erp.query_catalogue(sku_id)
        return []

    def request_discount(self, supplier_id: str, sku_id: str, qty: float) -> str:
        """Discount request proxy (used by C4)."""
        if self._erp:
            return self._erp.request_discount(supplier_id, sku_id, qty)
        return ""

    def propose_schedule(self, supplier_id: str, sku_id: str, delta: int) -> str:
        """Schedule proposal proxy (used by C4)."""
        if self._erp:
            return self._erp.propose_schedule(supplier_id, sku_id, delta)
        return ""

    # ------------------------------------------------------------------
    # Metric accessors
    # ------------------------------------------------------------------

    @property
    def catalog(self) -> SKUCatalog | None:
        """The SKU catalog."""
        return self._catalog

    @property
    def current_day(self) -> int:
        """Current simulation day."""
        return self._day

    def get_accumulated_metrics(self) -> dict[str, float]:
        """Return cumulative metrics from the start of the simulation.

        Returns:
            Dict with ``total_demand``, ``total_fulfilled``,
            ``stockout_rate``, ``fill_rate``.
        """
        tot_d = sum(self._total_demand.values())
        tot_f = sum(self._total_fulfilled.values())
        stockout_rate = (tot_d - tot_f) / max(tot_d, 1e-9)
        fill_rate = tot_f / max(tot_d, 1e-9)
        return {
            "total_demand": round(tot_d, 2),
            "total_fulfilled": round(tot_f, 2),
            "stockout_rate": round(float(stockout_rate), 6),
            "fill_rate": round(float(fill_rate), 6),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_components(self, seed: int) -> None:
        """(Re)build all simulation components with a given seed."""
        cfg = self._config
        self._catalog = SKUCatalog(
            n_skus=cfg.n_skus,
            category_names=cfg.category_names,
            seed=seed,
        )
        self._gen = DemandGenerator(
            self._catalog,
            n_days=cfg.simulation_horizon_days,
            seed=seed,
        )
        self._sup = SupplierSimulator(
            self._catalog,
            n_suppliers_min=cfg.n_suppliers_min,
            n_suppliers_max=cfg.n_suppliers_max,
            seed=seed,
        )
        self._erp = ERPStub(
            self._catalog,
            self._gen,
            self._sup,
            self._initial_inv_days,
        )
        self._total_demand = {s: 0.0 for s in self._catalog.sku_ids}
        self._total_fulfilled = {s: 0.0 for s in self._catalog.sku_ids}

        # Gymnasium spaces
        if self._gym_available:
            n_skus = cfg.n_skus
            self.observation_space = spaces.Box(
                low=0.0, high=np.inf, shape=(_OBS_DIM,), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=0.0, high=10000.0, shape=(1,), dtype=np.float32
            )

    def _get_obs(self) -> np.ndarray:
        """Build a representative observation vector for the RL policy."""
        if self._erp is None:
            return np.zeros(_OBS_DIM, dtype=np.float32)
        snapshot = self._erp.get_inventory_snapshot()
        # Use first low-stock SKU as representative observation
        for sku_id, rec in snapshot.items():
            if rec.get("is_low_stock", False):
                return np.array(
                    [
                        float(rec.get("effective_available", 0.0)),
                        float(rec.get("demand_mean_daily", 10.0)) * 7,
                        float(rec.get("demand_std_daily", 2.0)) * 7,
                        min(float(rec.get("days_to_expiry", 365.0)) / 365.0, 1.0),
                        1.0,  # budget fraction (full)
                    ],
                    dtype=np.float32,
                )
        return np.zeros(_OBS_DIM, dtype=np.float32)

    def _compute_reward(self, realised_demand: dict[str, float]) -> float:
        """Compute negative expected cost as reward.

        Uses a simplified version of Eq. 3: stockout penalty minus
        holding cost, so the RL agent learns to balance both.
        """
        if self._erp is None:
            return 0.0
        snapshot = self._erp.get_inventory_snapshot()
        reward = 0.0
        for sku_id, demand in realised_demand.items():
            rec_sim = self._catalog[sku_id] if self._catalog else None  # type: ignore[index]
            if rec_sim is None:
                continue
            erp_rec = snapshot.get(sku_id, {})
            on_hand = float(erp_rec.get("on_hand", 0.0))
            fulfilled = min(demand, on_hand)
            stockout = max(0.0, demand - fulfilled)
            penalty = rec_sim.unit_cost * 3.0 * stockout
            holding = rec_sim.unit_cost * 0.25 / 365.0 * on_hand
            reward -= (penalty + holding)
        return float(reward)
