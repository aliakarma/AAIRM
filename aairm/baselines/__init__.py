"""Baseline inventory policies for comparison with AAIRM.

Baseline 1  ROPEOQPolicy      — Classical ROP + EOQ (product.tex BL1)
Baseline 2  MLStaticPolicy    — XGBoost forecast + static rule-based ordering (BL2)
Baseline 3  MLAdaptivePolicy  — XGBoost forecast + adaptive safety stock (BL3)
Baseline 4  PerSKUDQNPolicy   — Independent per-SKU DQN, no coordination (BL4)
Baseline 5  MAPPOPolicy       — Shared-parameter MAPPO, centralized critic (BL5)
"""

from aairm.baselines.ml_adaptive import MLAdaptivePolicy
from aairm.baselines.ml_static import MLStaticPolicy
from aairm.baselines.mappo import MAPPOPolicy
from aairm.baselines.per_sku_dqn import PerSKUDQNPolicy
from aairm.baselines.rop_eoq import ROPEOQPolicy

__all__ = [
    "ROPEOQPolicy",
    "MLStaticPolicy",
    "MLAdaptivePolicy",
    "PerSKUDQNPolicy",
    "MAPPOPolicy",
]
