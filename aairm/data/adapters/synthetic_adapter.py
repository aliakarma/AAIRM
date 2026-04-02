"""Synthetic Data Adapter.

Converts AAIRM-generated simulation data into the unified schema used by
the DataLoader, making it interchangeable with real-world dataset adapters.

Unified schema columns:
    sku_id          : str
    date            : date
    demand          : float
    unit_price      : float
    category        : str
    is_perishable   : bool
    promotion_flag  : bool
    day_of_week     : int
    is_holiday      : bool
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from aairm.simulation.demand_generator import DemandGenerator
from aairm.simulation.sku_catalog import SKUCatalog
from aairm.simulation.supplier_simulator import SupplierSimulator


class SyntheticAdapter:
    """Wraps the AAIRM simulation into the unified DataLoader schema.

    Args:
        catalog: Populated :class:`~aairm.simulation.sku_catalog.SKUCatalog`.
        demand_gen: Initialised :class:`~aairm.simulation.demand_generator.DemandGenerator`.
        supplier_sim: Initialised :class:`~aairm.simulation.supplier_simulator.SupplierSimulator`.
        start_date: Simulation start date string (ISO format, default "2022-01-01").
    """

    def __init__(
        self,
        catalog: SKUCatalog,
        demand_gen: DemandGenerator,
        supplier_sim: SupplierSimulator,
        start_date: str = "2022-01-01",
    ) -> None:
        self._catalog = catalog
        self._gen = demand_gen
        self._sup = supplier_sim
        self._start = pd.Timestamp(start_date)

    def to_unified_schema(self, n_days: int | None = None) -> dict[str, Any]:
        """Export the simulation data in the unified AAIRM schema.

        Args:
            n_days: Number of days to export.  Defaults to the full
                simulation horizon.

        Returns:
            Dict with keys:
            ``{demand_history, sku_catalog, supplier_catalog, calendar}``.
        """
        all_rows = []
        sku_ids = self._catalog.sku_ids
        horizon = n_days or self._gen._n_days

        dates = pd.date_range(self._start, periods=horizon, freq="D")

        for sku_id in sku_ids:
            rec = self._catalog[sku_id]
            for day_idx, date in enumerate(dates):
                demand = self._gen.get_demand(sku_id, day_idx)
                doy = date.day_of_year
                is_hol = doy in {100, 101, 102, 175, 176, 177, 272, 273}
                all_rows.append(
                    {
                        "sku_id": sku_id,
                        "date": date.date(),
                        "demand": demand,
                        "unit_price": rec.unit_cost,
                        "category": rec.category,
                        "is_perishable": rec.is_perishable,
                        "promotion_flag": False,
                        "day_of_week": date.day_of_week,
                        "is_holiday": is_hol,
                    }
                )

        demand_df = pd.DataFrame(all_rows)

        # demand_history dict for baselines
        demand_history: dict[str, np.ndarray] = {}
        for sku_id in sku_ids:
            demand_history[sku_id] = self._gen.get_history(
                sku_id, horizon, horizon
            )

        # SKU catalog DataFrame
        sku_catalog = self._catalog.to_dataframe()

        # Supplier catalog
        sup_rows = []
        for sku_id in sku_ids:
            for offer in self._sup.query_catalogue(sku_id):
                sup_rows.append(offer)
        supplier_catalog = pd.DataFrame(sup_rows)

        # Calendar
        calendar = pd.DataFrame(
            {
                "date": dates.date,
                "day_of_week": dates.day_of_week,
                "day_of_year": dates.day_of_year,
                "is_holiday": [d in {100, 101, 102, 175, 176, 177} for d in dates.day_of_year],
            }
        )

        return {
            "demand_history": demand_history,
            "sku_catalog": sku_catalog,
            "supplier_catalog": supplier_catalog,
            "calendar": calendar,
            "demand_df": demand_df,
        }
