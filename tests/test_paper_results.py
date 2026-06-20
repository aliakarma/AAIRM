"""Guard tests: the project's canonical results equal the paper numbers.

These tests are the contract behind the alignment goal — they assert that the
canonical fixtures hold the exact values reported in ``product.tex`` and remain
internally consistent. If anyone edits a fixture and drifts from the paper,
these fail. They also exercise the new subsystems (BL3/4/5, FDL, BTL, WRMSSE)
and the table-export path.
"""

from __future__ import annotations

import numpy as np
import pytest

from aairm.evaluation.paper_results import CANONICAL_FILES, load_all, load_canonical


# ---------------------------------------------------------------------------
# Canonical fixtures load and equal the paper numbers
# ---------------------------------------------------------------------------

def test_all_canonical_fixtures_load():
    data = load_all()
    assert set(data) == set(CANONICAL_FILES)


def test_table3_primary_exact():
    p = load_canonical("table3_primary_100sku")["policies"]
    assert p["aairm"]["total_cost"] == {"mean": 0.868, "std": 0.014}
    assert p["aairm"]["fill_rate"]["mean"] == 92.29
    assert p["baseline1"]["total_cost"]["mean"] == 1.000
    assert p["baseline3"]["total_cost"]["mean"] == 0.962


def test_table4_ablation_decomposition_adds_up():
    t = load_canonical("table4_ablation")
    dec = t["decomposition"]
    assert dec["rl_contribution_pp"] == 8.8
    assert dec["governance_contribution_pp"] == 3.8
    assert dec["llm_contribution_pp"] == 0.6
    # Component decomposition is ~additive to the headline 13.2 pp.
    total = (dec["rl_contribution_pp"] + dec["governance_contribution_pp"]
             + dec["llm_contribution_pp"])
    assert abs(total - dec["total_pp"]) <= 0.2
    # RL contribution matches A vs BL1; governance matches A vs C; LLM C vs D.
    v = t["variants"]
    assert round(1.000 - v["A"]["total_cost"]["mean"], 3) == 0.088
    assert round(v["A"]["total_cost"]["mean"] - v["C"]["total_cost"]["mean"], 3) == 0.038
    assert round(v["C"]["total_cost"]["mean"] - v["D"]["total_cost"]["mean"], 3) == 0.006


def test_table5_rl_baselines_exact():
    p = load_canonical("table5_rl_baselines")["policies"]
    assert p["bl5_mappo"]["total_cost"]["mean"] == 0.885
    assert p["aairm"]["constraint_violation"]["mean"] == 0.0
    assert p["bl4_dqn"]["constraint_violation"]["mean"] == 14.1


def test_table7_scalability_exact():
    p = load_canonical("table7_scalability_500sku")["policies"]
    assert p["aairm_all"]["total_cost"]["mean"] == 0.829
    assert p["aairm_all"]["stockout_rate"]["mean"] == 14.34
    assert p["aairm_excl_dryfruits"]["total_cost"]["mean"] == 0.871


def test_table8_dryfruits_recal_monotonic():
    sweep = load_canonical("table8_dryfruits_recal")["sweep"]
    # Higher w_p -> higher fill rate and higher spoilage (the documented tension).
    fills = [s["fill_rate"]["mean"] for s in sweep]
    spoil = [s["spoilage_rate"]["mean"] for s in sweep]
    assert fills == sorted(fills)
    assert spoil == sorted(spoil)
    assert sweep[0]["stockout_rate"]["mean"] == 31.54  # the failure
    assert sweep[-1]["fill_rate"]["mean"] == 95.3       # the fix


def test_table9_fdl_ordering():
    r = load_canonical("table9_fdl")["regimes"]
    # Centralized best, FedProx <= FedAvg, local-only worst (WAPE).
    assert r["centralized"]["wape"]["mean"] == 18.4
    assert r["fedprox"]["wape"]["mean"] <= r["fedavg"]["wape"]["mean"]
    assert r["local_only"]["wape"]["mean"] == 24.7
    assert r["centralized"]["raw_data_leaves_store"] is True
    assert r["fedavg"]["raw_data_leaves_store"] is False


def test_table10_btl_exact():
    m = load_canonical("table10_btl")["metrics"]
    assert m["mean_commit_latency_ms"]["btl"] == 142
    assert m["decision_cycle_overhead_pct"]["btl"] == 6.6
    assert m["injected_mutation_detection_500"]["btl"] == "500/500"


def test_table11_m5_exact():
    rows = load_canonical("table11_m5")["rows"]
    assert rows["c1_wrmsse_centralized"]["m5"] == 0.66
    assert rows["cost_reduction_vs_rop_eoq_pct"]["m5"] == 10.2


def test_pareto_frontier_monotonic():
    f = load_canonical("figure_pareto")["frontier"]
    costs = [pt["total_cost"] for pt in f]
    stockouts = [pt["stockout_rate"] for pt in f]
    assert costs == sorted(costs)                 # cost rises as service improves
    assert stockouts == sorted(stockouts, reverse=True)
    assert f[0] == {"w_p": 1.2, "stockout_rate": 7.71, "total_cost": 0.868}


# ---------------------------------------------------------------------------
# Export path reproduces all tables
# ---------------------------------------------------------------------------

def test_export_paper_tables(tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import export_paper_tables as ept

    out = tmp_path / "tables"
    out.mkdir()
    for fname, builder in ept.TABLE_BUILDERS.items():
        (out / fname).write_text(builder(), encoding="utf-8")
    # Table 3 must contain the headline AAIRM cost exactly.
    assert "0.868" in (out / "table3_results-overview.tex").read_text(encoding="utf-8")
    # Figure coordinate data is well-formed.
    coords = ept._figure_coords()
    assert "(7.71,0.868)" in coords["pareto"]["frontier"]


# ---------------------------------------------------------------------------
# New subsystems run end-to-end
# ---------------------------------------------------------------------------

def test_new_baselines_run():
    from aairm.baselines import MAPPOPolicy, MLAdaptivePolicy, PerSKUDQNPolicy

    md = {f"s{i}": 10.0 for i in range(5)}
    snap = {s: {"on_hand": 2.0, "in_transit": 0.0, "shelf_life_days": 30.0} for s in md}
    fc = {s: 10.0 for s in md}

    dqn = PerSKUDQNPolicy(seed=1).fit(md)
    assert isinstance(dqn.get_orders(snap, fc), dict)

    mappo = MAPPOPolicy(seed=1).fit(md)
    mappo.get_orders(snap, fc, budget=10.0, unit_costs={s: 5.0 for s in md})
    assert 0.0 <= mappo.constraint_violation_rate <= 100.0

    adaptive = MLAdaptivePolicy()
    import pandas as pd
    adaptive.fit(pd.DataFrame(), pd.Series(dtype=float), {"s0": 5.0}, {"s0": 5.0})
    adaptive.observe_actuals({"s0": 9.0}, {"s0": 11.0})
    assert isinstance(adaptive.get_orders({"s0": {"effective_available": 0.0}}, {"s0": 10.0}), dict)


def test_fdl_federation_beats_local_only():
    from aairm.federated import FDLCoordinator

    rng = np.random.default_rng(0)
    n, f = 3000, 6
    X = rng.normal(size=(n, f))
    w = rng.normal(size=f)
    cats = rng.integers(0, 5, size=n)
    y = X @ w + 10 + cats + rng.normal(scale=0.3, size=n)
    Xe, ye = X[:400], y[:400]
    co = FDLCoordinator(n_features=f, rounds=15, seed=1)
    clients = co.partition_dirichlet(X, y, cats, n_clients=8, beta=0.5)
    fed = co.fit_federated(clients, Xe, ye, "fedavg").final_wape
    clients = co.partition_dirichlet(X, y, cats, n_clients=8, beta=0.5)
    local = co.fit_local_only(clients, Xe, ye).final_wape
    assert fed <= local + 1e-6  # federation regains cross-store statistical strength


def test_btl_mutation_detection_is_perfect():
    from aairm.infrastructure.btl_evaluator import BTLEvaluator

    detected, total, fp = BTLEvaluator().mutation_replay(n_events=2000, n_mutations=200)
    assert detected == total          # SHA-256 collision resistance
    assert fp == 0                    # no false positives on unmutated events


def test_wrmsse_rewards_better_forecasts():
    from aairm.evaluation.wrmsse import rmsse, wrmsse

    rng = np.random.default_rng(0)
    train = {s: rng.poisson(5, 100).astype(float) for s in ["a", "b"]}
    true = {s: rng.poisson(5, 28).astype(float) for s in ["a", "b"]}
    good = {s: true[s] + rng.normal(0, 0.3, 28) for s in ["a", "b"]}
    bad = {s: true[s] + rng.normal(0, 4.0, 28) for s in ["a", "b"]}
    assert wrmsse(train, true, good) < wrmsse(train, true, bad)
    assert rmsse(np.array([1.0]), np.array([1.0]), np.array([1.0])) == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
