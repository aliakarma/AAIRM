"""Evaluation pipeline for AAIRM experiments.

metrics    — five paper metrics (stockout rate, fill rate, avg inventory,
             total cost, diversification index)
benchmarker — runs all three policies and collects BenchmarkResult
reporter   — generates LaTeX tables and matplotlib figures from results
"""

from aairm.evaluation.metrics import (
    stockout_rate,
    fill_rate,
    average_inventory_ratio,
    total_cost_normalised,
    supplier_diversification_index,
)
from aairm.evaluation.benchmarker import Benchmarker, BenchmarkResult
from aairm.evaluation.reporter import Reporter

__all__ = [
    "stockout_rate",
    "fill_rate",
    "average_inventory_ratio",
    "total_cost_normalised",
    "supplier_diversification_index",
    "Benchmarker",
    "BenchmarkResult",
    "Reporter",
]
