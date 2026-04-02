"""Benchmarker — Runs All Three Policies and Collects Results.

Runs AAIRM, Baseline 1 (ROP–EOQ), and Baseline 2 (ML + Static) over
the one-year test horizon and returns a structured BenchmarkResult
matching Tables 2 and 3 of the paper.

References
----------
Paper Section 5.2–5.3; Repo Guide Section 8.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aairm.evaluation.metrics import compute_all_metrics
from aairm.utils.config import AAIRMConfig
from aairm.utils.logging import get_logger

logger = get_logger(__name__)

# Paper expected results (Table 2) — used for assertion checks
PAPER_RESULTS: dict[str, dict[str, float]] = {
    "aairm": {
        "stockout_rate": 0.039, "fill_rate": 0.978,
        "avg_inventory": 1.19,  "total_cost": 0.84, "div_index": 0.61,
    },
    "baseline1": {
        "stockout_rate": 0.087, "fill_rate": 0.931,
        "avg_inventory": 1.45,  "total_cost": 1.00, "div_index": 0.42,
    },
    "baseline2": {
        "stockout_rate": 0.062, "fill_rate": 0.954,
        "avg_inventory": 1.32,  "total_cost": 0.93, "div_index": 0.47,
    },
}
TOLERANCE = 0.005   # ±0.5 percentage points


@dataclass
class BenchmarkResult:
    """Container for one policy's benchmark output.

    Attributes
    ----------
    policy_name : str
        Human-readable policy label.
    overall : dict[str, float]
        Aggregate metrics over the full test horizon.
    per_category : dict[str, dict[str, float]]
        Per-category breakdown (keys match category names).
    timeseries : dict[str, np.ndarray]
        Daily metric arrays for plotting.
    rl_curve : list[tuple[int, float]] | None
        RL training curve ``[(episode, cost), ...]`` (AAIRM only).
    """

    policy_name: str
    overall: dict[str, float] = field(default_factory=dict)
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)
    timeseries: dict[str, np.ndarray] = field(default_factory=dict)
    rl_curve: list[tuple[int, float]] | None = None


class Benchmarker:
    """Run all three policies and collect BenchmarkResult objects.

    Args:
        config: Top-level :class:`~aairm.utils.config.AAIRMConfig`.
        env: :class:`~aairm.simulation.environment.RetailEnv` instance.
        orchestrator: Initialised MetaOrchestrator (AAIRM policy).
        baseline1: Fitted :class:`~aairm.baselines.rop_eoq.ROPEOQPolicy`.
        baseline2: Fitted :class:`~aairm.baselines.ml_static.MLStaticPolicy`.
    """

    def __init__(
        self,
        config: AAIRMConfig,
        env: Any,
        orchestrator: Any = None,
        baseline1: Any = None,
        baseline2: Any = None,
    ) -> None:
        self._config = config
        self._env = env
        self._orch = orchestrator
        self._bl1 = baseline1
        self._bl2 = baseline2
        self._test_days = config.simulation.test_horizon_days
        self._categories = config.simulation.category_names

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_all(
        self, assert_paper_results: bool = False
    ) -> dict[str, BenchmarkResult]:
        """Run all three policies and return results.

        Args:
            assert_paper_results: If ``True``, assert that AAIRM results
                are within TOLERANCE of paper Table 2 values.

        Returns:
            ``{policy_name: BenchmarkResult}``
        """
        results: dict[str, BenchmarkResult] = {}

        if self._bl1 is not None:
            logger.info("benchmarker.running", policy="baseline1")
            results["baseline1"] = self._run_baseline1()

        if self._bl2 is not None:
            logger.info("benchmarker.running", policy="baseline2")
            results["baseline2"] = self._run_baseline2()

        if self._orch is not None:
            logger.info("benchmarker.running", policy="aairm")
            results["aairm"] = self._run_aairm()

        if assert_paper_results and "aairm" in results:
            self._assert_results(results)

        return results

    def run_single(self, policy_name: str) -> BenchmarkResult:
        """Run one policy and return its BenchmarkResult.

        Args:
            policy_name: One of ``"aairm"``, ``"baseline1"``, ``"baseline2"``.

        Returns:
            BenchmarkResult for the specified policy.
        """
        if policy_name == "aairm":
            return self._run_aairm()
        elif policy_name == "baseline1":
            return self._run_baseline1()
        elif policy_name == "baseline2":
            return self._run_baseline2()
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

    # ------------------------------------------------------------------
    # Private runners
    # ------------------------------------------------------------------

    def _run_aairm(self) -> BenchmarkResult:
        """Run the full AAIRM pipeline for test_horizon_days."""
        from aairm.agents.base import AgentState

        env = self._env
        obs, info = env.reset()

        daily_demand, daily_fulfilled, daily_on_hand = [], [], []
        daily_proc_cost, daily_hold_cost = [], []
        daily_penalty_cost, daily_spoilage_cost = [], []
        procurement_volumes: dict[str, dict[str, float]] = {
            cat: {} for cat in self._categories
        }

        for day in range(self._test_days):
            state = AgentState(day=day)
            state = self._orch.run_cycle(state)

            # Step environment with approved orders
            order_dict = {
                sku: info.get("quantity", 0.0)
                for sku, info in state.approved_orders.items()
            }
            metrics = env.step_agentic(order_dict)

            snap = env.get_inventory_snapshot()
            day_demand = metrics.get("total_demand", 0.0)
            day_stockout = metrics.get("stockout_units", 0.0)
            day_fulfilled = max(0.0, day_demand - day_stockout)

            daily_demand.append(day_demand)
            daily_fulfilled.append(day_fulfilled)
            daily_on_hand.append(
                sum(rec.get("on_hand", 0.0) for rec in snap.values())
            )

            # Cost accumulation (approximate from approved orders)
            proc = sum(
                float(t.get("order_value", 0.0))
                for t in state.approved_orders.values()
            )
            hold = sum(
                float(rec.get("on_hand", 0.0)) * float(rec.get("unit_cost", 5.0))
                * (0.25 / 365.0)
                for rec in snap.values()
            )
            pen = day_stockout * 5.0 * 3.0    # avg unit_cost * penalty_mult
            daily_proc_cost.append(proc)
            daily_hold_cost.append(hold)
            daily_penalty_cost.append(pen)
            daily_spoilage_cost.append(0.0)

            # Track procurement volumes for diversification
            for sku, terms in state.approved_orders.items():
                cat = snap.get(sku, {}).get("category", "grocery")
                sup = str(terms.get("supplier_id", "UNKNOWN"))
                vol = float(terms.get("order_value", 0.0))
                if sup not in procurement_volumes[cat]:
                    procurement_volumes[cat][sup] = 0.0
                procurement_volumes[cat][sup] += vol

        baseline_cost = float(sum(daily_proc_cost) + sum(daily_hold_cost)
                               + sum(daily_penalty_cost))

        overall = compute_all_metrics(
            daily_demand, daily_fulfilled, daily_on_hand,
            daily_proc_cost, daily_hold_cost,
            daily_penalty_cost, daily_spoilage_cost,
            baseline_cost,   # self-normalised; will be adjusted by runner
            procurement_volumes,
        )

        return BenchmarkResult(
            policy_name="AAIRM (proposed)",
            overall=overall,
            per_category=self._per_category_stub(),
            timeseries={
                "demand": np.array(daily_demand),
                "fulfilled": np.array(daily_fulfilled),
                "on_hand": np.array(daily_on_hand),
            },
        )

    def _run_baseline1(self) -> BenchmarkResult:
        """Run ROP–EOQ baseline for test_horizon_days."""
        env = self._env
        obs, info = env.reset()
        snap = env.get_inventory_snapshot()

        daily_demand, daily_fulfilled, daily_on_hand = [], [], []
        daily_proc_cost, daily_hold_cost, daily_penalty_cost = [], [], []
        procurement_volumes: dict[str, dict[str, float]] = {
            c: {} for c in self._categories
        }

        for day in range(self._test_days):
            snap = env.get_inventory_snapshot()
            orders = self._bl1.get_orders(snap)

            # Submit orders via cheapest supplier (no negotiation)
            proc = 0.0
            for sku, qty in orders.items():
                cat = snap.get(sku, {}).get("category", "grocery")
                offers = env.query_catalogue(sku)
                supplier = self._bl1.get_top_supplier(sku, offers)
                if supplier:
                    sup_id = str(supplier.get("supplier_id", "UNKNOWN"))
                    cost = qty * float(supplier.get("unit_cost", 5.0))
                    proc += cost
                    if sup_id not in procurement_volumes[cat]:
                        procurement_volumes[cat][sup_id] = 0.0
                    procurement_volumes[cat][sup_id] += cost

            metrics = env.step_agentic(orders)
            day_demand = metrics.get("total_demand", 0.0)
            day_stockout = metrics.get("stockout_units", 0.0)
            day_fulfilled = max(0.0, day_demand - day_stockout)

            hold = sum(
                float(rec.get("on_hand", 0.0)) * float(rec.get("unit_cost", 5.0))
                * (0.25 / 365.0)
                for rec in snap.values()
            )
            pen = day_stockout * 5.0 * 3.0

            daily_demand.append(day_demand)
            daily_fulfilled.append(day_fulfilled)
            daily_on_hand.append(sum(r.get("on_hand", 0.0) for r in snap.values()))
            daily_proc_cost.append(proc)
            daily_hold_cost.append(hold)
            daily_penalty_cost.append(pen)

        baseline_cost = float(
            sum(daily_proc_cost) + sum(daily_hold_cost) + sum(daily_penalty_cost)
        )

        overall = compute_all_metrics(
            daily_demand, daily_fulfilled, daily_on_hand,
            daily_proc_cost, daily_hold_cost, daily_penalty_cost, [0.0],
            baseline_cost, procurement_volumes,
        )
        overall["total_cost"] = 1.00   # baseline is the reference

        return BenchmarkResult(
            policy_name="Baseline 1 (ROP–EOQ)",
            overall=overall,
            per_category=self._per_category_stub(),
            timeseries={
                "demand": np.array(daily_demand),
                "fulfilled": np.array(daily_fulfilled),
                "on_hand": np.array(daily_on_hand),
            },
        )

    def _run_baseline2(self) -> BenchmarkResult:
        """Run ML + Static baseline for test_horizon_days."""
        from aairm.baselines.ml_static import MLStaticPolicy

        env = self._env
        obs, info = env.reset()

        daily_demand, daily_fulfilled, daily_on_hand = [], [], []
        daily_proc_cost, daily_hold_cost, daily_penalty_cost = [], [], []
        procurement_volumes: dict[str, dict[str, float]] = {
            c: {} for c in self._categories
        }

        for day in range(self._test_days):
            snap = env.get_inventory_snapshot()

            # Build feature matrix for today
            demand_hist = {
                sku: env.get_demand_history(sku, 30) for sku in snap.keys()
            }
            X_today = MLStaticPolicy.build_feature_matrix(demand_hist, day)
            forecasts = self._bl2.predict_demand(X_today)
            orders = self._bl2.get_orders(snap, forecasts)

            proc = 0.0
            for sku, qty in orders.items():
                cat = snap.get(sku, {}).get("category", "grocery")
                offers = env.query_catalogue(sku)
                supplier = self._bl2.get_top_supplier(sku, offers)
                if supplier:
                    sup_id = str(supplier.get("supplier_id", "UNKNOWN"))
                    cost = qty * float(supplier.get("unit_cost", 5.0))
                    proc += cost
                    if sup_id not in procurement_volumes[cat]:
                        procurement_volumes[cat][sup_id] = 0.0
                    procurement_volumes[cat][sup_id] += cost

            metrics = env.step_agentic(orders)
            day_demand = metrics.get("total_demand", 0.0)
            day_stockout = metrics.get("stockout_units", 0.0)
            day_fulfilled = max(0.0, day_demand - day_stockout)

            hold = sum(
                float(r.get("on_hand", 0.0)) * float(r.get("unit_cost", 5.0))
                * (0.25 / 365.0) for r in snap.values()
            )
            pen = day_stockout * 5.0 * 3.0

            daily_demand.append(day_demand)
            daily_fulfilled.append(day_fulfilled)
            daily_on_hand.append(sum(r.get("on_hand", 0.0) for r in snap.values()))
            daily_proc_cost.append(proc)
            daily_hold_cost.append(hold)
            daily_penalty_cost.append(pen)

        baseline_cost = float(
            sum(daily_proc_cost) + sum(daily_hold_cost) + sum(daily_penalty_cost)
        )
        overall = compute_all_metrics(
            daily_demand, daily_fulfilled, daily_on_hand,
            daily_proc_cost, daily_hold_cost, daily_penalty_cost, [0.0],
            baseline_cost, procurement_volumes,
        )

        return BenchmarkResult(
            policy_name="Baseline 2 (ML + Static)",
            overall=overall,
            per_category=self._per_category_stub(),
            timeseries={
                "demand": np.array(daily_demand),
                "fulfilled": np.array(daily_fulfilled),
                "on_hand": np.array(daily_on_hand),
            },
        )

    def _per_category_stub(self) -> dict[str, dict[str, float]]:
        """Return per-category metric stubs (populated by post-processing)."""
        return {cat: {} for cat in self._categories}

    def _assert_results(self, results: dict[str, BenchmarkResult]) -> None:
        """Assert AAIRM results are within TOLERANCE of paper values."""
        aairm = results["aairm"].overall
        expected = PAPER_RESULTS["aairm"]
        for metric, exp_val in expected.items():
            got = aairm.get(metric, 0.0)
            assert abs(got - exp_val) <= TOLERANCE, (
                f"Metric '{metric}' out of tolerance: "
                f"expected {exp_val:.4f} ± {TOLERANCE}, got {got:.4f}"
            )
        logger.info("benchmarker.paper_results_verified")
