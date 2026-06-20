#!/usr/bin/env python3
"""Run every AAIRM paper experiment and regenerate all tables/figures.

Executes all runners (primary, scalability, ablation, RL baselines, Pareto,
dry-fruits recalibration, FDL, BTL, sensitivity, M5) and then the canonical
table/figure export. This is the one-command reproduction of every numeric
result in ``product.tex``.

Usage
-----
    python experiments/run_all_paper.py
    python experiments/run_all_paper.py --quick   # skip live FDL/BTL/baseline runs
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _paper_runner import banner  # noqa: E402

RUNNERS = [
    ("run_primary.py", []),
    ("run_scalability.py", []),
    ("run_ablation_variants.py", []),
    ("run_rl_baselines.py", ["--no-smoke"]),
    ("run_pareto.py", []),
    ("run_dryfruits_recal.py", []),
    ("run_fdl.py", ["--no-live"]),
    ("run_btl.py", ["--no-live"]),
    ("run_sensitivity.py", []),
    ("run_m5_validation.py", ["--no-smoke"]),
]


def _run(script: str, argv: list[str]) -> None:
    old = sys.argv
    try:
        sys.argv = [script, *argv]
        runpy.run_path(str(HERE / script), run_name="__main__")
    finally:
        sys.argv = old


def main() -> None:
    ap = argparse.ArgumentParser(description="Run all AAIRM paper experiments.")
    ap.add_argument("--quick", action="store_true",
                    help="Skip live FDL/BTL/baseline runs (canonical export only).")
    args = ap.parse_args()

    for script, argv in RUNNERS:
        if not args.quick:
            argv = [a for a in argv if a not in ("--no-live", "--no-smoke")]
        _run(script, argv)

    banner("Regenerating canonical LaTeX tables and figure data")
    _run("export_paper_tables.py", [])

    banner("All paper experiments complete")
    print("Every table and figure in product.tex has been reproduced from the "
          "canonical fixtures. See experiments/results/.")


if __name__ == "__main__":
    main()
