"""Synthetic retail simulation environment.

Implements the data-generating process described in Section 5.1 of the
paper: 1,200 SKUs across five product categories with category-specific
demand seasonality, perishability constraints, and stochastic lead times.

Modules
-------
environment        — RetailEnv gymnasium-compatible environment
demand_generator   — stochastic demand process (DGP)
supplier_simulator — lead-time and availability simulation
sku_catalog        — 1,200-SKU catalog generator
erp_stub           — mock ERP/WMS backend
"""

from aairm.simulation.environment import RetailEnv
from aairm.simulation.sku_catalog import SKUCatalog
from aairm.simulation.demand_generator import DemandGenerator
from aairm.simulation.supplier_simulator import SupplierSimulator
from aairm.simulation.erp_stub import ERPStub

__all__ = [
    "RetailEnv",
    "SKUCatalog",
    "DemandGenerator",
    "SupplierSimulator",
    "ERPStub",
]
