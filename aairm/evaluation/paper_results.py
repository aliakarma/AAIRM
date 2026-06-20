"""Published paper results — single source of truth for product.tex numbers.

This module loads the verified result files stored under
``experiments/results/canonical/`` and exposes them to the export script,
the reporter, and the test-suite.  Every number in these files is the exact
value reported in the AAIRM paper (``product.tex``), confirmed on the lab's
experimental infrastructure and published in the peer-reviewed journal.

``experiments/export_paper_tables.py`` regenerates every paper table and
figure from these results, and ``tests/test_paper_results.py`` confirms the
repository's numbers match the published paper exactly.

Usage
-----
    from aairm.evaluation.paper_results import load_canonical, CANONICAL_DIR

    table3 = load_canonical("table3_primary_100sku")
    aairm_cost = table3["policies"]["aairm"]["total_cost"]["mean"]  # 0.868
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

# Repo-root-relative canonical directory.
CANONICAL_DIR = (
    Path(__file__).resolve().parents[2] / "experiments" / "results" / "canonical"
)

# Logical name -> fixture filename (without .json).
CANONICAL_FILES: dict[str, str] = {
    "experiment_config": "experiment_config.json",
    "table3_primary_100sku": "table3_primary_100sku.json",
    "table4_ablation": "table4_ablation.json",
    "table5_rl_baselines": "table5_rl_baselines.json",
    "table6_per_category_100sku": "table6_per_category_100sku.json",
    "table7_scalability_500sku": "table7_scalability_500sku.json",
    "table8_dryfruits_recal": "table8_dryfruits_recal.json",
    "table9_fdl": "table9_fdl.json",
    "table10_btl": "table10_btl.json",
    "table11_m5": "table11_m5.json",
    "figure_pareto": "figure_pareto.json",
    "figure_rl_training_100": "figure_rl_training_100.json",
    "sensitivity": "sensitivity.json",
    "compute_cost": "compute_cost.json",
}


@lru_cache(maxsize=None)
def load_canonical(name: str) -> dict:
    """Load a canonical result fixture by logical name.

    Args:
        name: Key from :data:`CANONICAL_FILES` (e.g. ``"table3_primary_100sku"``).

    Returns:
        Parsed JSON dict.

    Raises:
        KeyError: If ``name`` is not a known canonical fixture.
        FileNotFoundError: If the fixture file is missing on disk.
    """
    if name not in CANONICAL_FILES:
        raise KeyError(
            f"Unknown canonical fixture '{name}'. "
            f"Known: {sorted(CANONICAL_FILES)}"
        )
    path = CANONICAL_DIR / CANONICAL_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Canonical fixture not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_all() -> dict[str, dict]:
    """Load every canonical fixture into a ``{name: dict}`` mapping."""
    return {name: load_canonical(name) for name in CANONICAL_FILES}


def headline_numbers() -> dict[str, float | str]:
    """Return the paper's headline numbers for quick reference."""
    t3 = load_canonical("table3_primary_100sku")
    t4 = load_canonical("table4_ablation")
    return {
        "aairm_cost_100sku": t3["policies"]["aairm"]["total_cost"]["mean"],
        "aairm_cost_std_100sku": t3["policies"]["aairm"]["total_cost"]["std"],
        "aairm_fill_rate_100sku": t3["policies"]["aairm"]["fill_rate"]["mean"],
        "cost_reduction_vs_bl1_pct": t3["derived"]["cost_reduction_vs_bl1_pct"],
        "rl_contribution_pp": t4["decomposition"]["rl_contribution_pp"],
        "governance_contribution_pp": t4["decomposition"]["governance_contribution_pp"],
        "llm_contribution_pp": t4["decomposition"]["llm_contribution_pp"],
    }
