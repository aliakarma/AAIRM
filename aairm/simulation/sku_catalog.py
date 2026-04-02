"""SKU Catalog Generator.

Generates the 1,200-SKU product catalog used across all paper experiments.
Each SKU record contains all fields required by the simulation, baselines,
and evaluation pipeline.

Catalog schema (per SKU):
    sku_id              : str   — "GRO-0001", "FRZ-0001", etc.
    category            : str   — one of the five category names
    unit_cost           : float — unit procurement cost (currency)
    base_demand_daily   : float — mean daily demand μ_D
    demand_sigma_frac   : float — demand noise level σ_i / μ_D
    unit_volume         : float — volumetric footprint (cubic units)
    is_perishable       : bool
    shelf_life_days     : int | None
    seasonality_amp     : float — category-level amplitude (from paper DGP)
    margin_rate         : float — gross margin fraction

References
----------
Paper Section 5.1; demand_params_seed.json; Repo Guide Section 6.1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Category-level constants (from paper / demand_params_seed.json)
# ---------------------------------------------------------------------------

CATEGORY_PARAMS: dict[str, dict[str, Any]] = {
    "grocery": {
        "code": "GRO",
        "seasonality_amp": 0.15,
        "is_perishable": False,
        "shelf_life_range": None,
        "volume_range": (0.02, 0.15),
        "cost_range": (1.0, 25.0),
        "demand_range": (30.0, 180.0),
        "margin_range": (0.18, 0.35),
    },
    "frozen_food": {
        "code": "FRZ",
        "seasonality_amp": 0.25,
        "is_perishable": True,
        "shelf_life_range": (30, 180),
        "volume_range": (0.05, 0.20),
        "cost_range": (3.0, 40.0),
        "demand_range": (20.0, 120.0),
        "margin_range": (0.22, 0.40),
    },
    "apparel": {
        "code": "APP",
        "seasonality_amp": 0.40,
        "is_perishable": False,
        "shelf_life_range": None,
        "volume_range": (0.10, 0.50),
        "cost_range": (5.0, 80.0),
        "demand_range": (10.0, 80.0),
        "margin_range": (0.35, 0.65),
    },
    "cosmetics": {
        "code": "COS",
        "seasonality_amp": 0.20,
        "is_perishable": True,
        "shelf_life_range": (180, 720),
        "volume_range": (0.01, 0.05),
        "cost_range": (2.0, 60.0),
        "demand_range": (15.0, 100.0),
        "margin_range": (0.40, 0.70),
    },
    "dry_fruits": {
        "code": "DRY",
        "seasonality_amp": 0.30,
        "is_perishable": True,
        "shelf_life_range": (60, 365),
        "volume_range": (0.02, 0.10),
        "cost_range": (2.0, 35.0),
        "demand_range": (20.0, 150.0),
        "margin_range": (0.25, 0.50),
    },
}


@dataclass
class SKURecord:
    """Single SKU metadata record."""

    sku_id: str
    category: str
    unit_cost: float
    base_demand_daily: float
    demand_sigma_frac: float
    unit_volume: float
    is_perishable: bool
    shelf_life_days: int | None
    seasonality_amp: float
    margin_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain dict for DataFrame construction."""
        return {
            "sku_id": self.sku_id,
            "category": self.category,
            "unit_cost": self.unit_cost,
            "base_demand_daily": self.base_demand_daily,
            "demand_sigma_frac": self.demand_sigma_frac,
            "demand_std_daily": self.base_demand_daily * self.demand_sigma_frac,
            "unit_volume": self.unit_volume,
            "is_perishable": self.is_perishable,
            "shelf_life_days": self.shelf_life_days,
            "seasonality_amp": self.seasonality_amp,
            "margin_rate": self.margin_rate,
        }


class SKUCatalog:
    """Generator and container for the AAIRM SKU catalog.

    Args:
        n_skus: Total number of SKUs (must be divisible by n_categories).
        category_names: Ordered list of category names.
        seed: Random seed for reproducible catalog generation.

    Examples:
        >>> catalog = SKUCatalog(n_skus=1200, seed=42)
        >>> df = catalog.to_dataframe()
        >>> len(df)
        1200
        >>> df["category"].value_counts().to_dict()
        {'grocery': 240, 'frozen_food': 240, ...}
    """

    def __init__(
        self,
        n_skus: int = 1200,
        category_names: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        self._n_skus = n_skus
        self._categories = category_names or list(CATEGORY_PARAMS.keys())
        self._n_cats = len(self._categories)
        self._skus_per_cat = n_skus // self._n_cats
        self._rng = np.random.default_rng(seed)
        self._records: list[SKURecord] = []
        self._index: dict[str, SKURecord] = {}
        self._generate()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, sku_id: str) -> SKURecord:
        return self._index[sku_id]

    def get(self, sku_id: str, default: Any = None) -> SKURecord | None:
        """Return SKU record or default if not found."""
        return self._index.get(sku_id, default)

    @property
    def sku_ids(self) -> list[str]:
        """Ordered list of all SKU identifiers."""
        return [r.sku_id for r in self._records]

    @property
    def categories(self) -> list[str]:
        """All category names in the catalog."""
        return self._categories

    def skus_by_category(self, category: str) -> list[str]:
        """Return SKU IDs for a specific category.

        Args:
            category: Category name.

        Returns:
            List of SKU IDs in that category.
        """
        return [r.sku_id for r in self._records if r.category == category]

    def to_dataframe(self) -> pd.DataFrame:
        """Export catalog as a pandas DataFrame.

        Returns:
            DataFrame with one row per SKU and columns matching SKURecord fields.
        """
        return pd.DataFrame([r.to_dict() for r in self._records])

    def to_csv(self, path: str | Path) -> None:
        """Write catalog to CSV.

        Args:
            path: Output file path.
        """
        self.to_dataframe().to_csv(path, index=False)

    # ------------------------------------------------------------------
    # Private generation
    # ------------------------------------------------------------------

    def _generate(self) -> None:
        """Generate all SKU records deterministically from the seed."""
        for cat_name in self._categories:
            params = CATEGORY_PARAMS.get(cat_name, CATEGORY_PARAMS["grocery"])
            code = params["code"]

            for k in range(self._skus_per_cat):
                sku_id = f"{code}-{k + 1:04d}"

                unit_cost = float(
                    self._rng.uniform(*params["cost_range"])
                )
                base_demand = float(
                    self._rng.uniform(*params["demand_range"])
                )
                sigma_frac = float(self._rng.uniform(0.05, 0.25))
                unit_volume = float(
                    self._rng.uniform(*params["volume_range"])
                )
                margin_rate = float(
                    self._rng.uniform(*params["margin_range"])
                )

                shelf_life = None
                if params["is_perishable"] and params["shelf_life_range"]:
                    lo, hi = params["shelf_life_range"]
                    shelf_life = int(self._rng.integers(lo, hi + 1))

                rec = SKURecord(
                    sku_id=sku_id,
                    category=cat_name,
                    unit_cost=round(unit_cost, 2),
                    base_demand_daily=round(base_demand, 2),
                    demand_sigma_frac=round(sigma_frac, 4),
                    unit_volume=round(unit_volume, 5),
                    is_perishable=params["is_perishable"],
                    shelf_life_days=shelf_life,
                    seasonality_amp=params["seasonality_amp"],
                    margin_rate=round(margin_rate, 4),
                )
                self._records.append(rec)
                self._index[sku_id] = rec
