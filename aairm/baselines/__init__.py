"""Baseline inventory policies for comparison with AAIRM.

Baseline 1  ROPEOQPolicy   — Classical ROP + EOQ (paper Section 5.2)
Baseline 2  MLStaticPolicy — ML demand forecast + static rule-based ordering
"""

from aairm.baselines.rop_eoq import ROPEOQPolicy
from aairm.baselines.ml_static import MLStaticPolicy

__all__ = ["ROPEOQPolicy", "MLStaticPolicy"]
