"""Corporación Favorita Grocery Sales Dataset Adapter.

Source: Kaggle — competitions/favorita-grocery-sales-forecasting
License: Competition-specific (see data/README.md)

Files required in data/raw/favorita/:
    train.csv, items.csv, stores.csv, transactions.csv,
    oil.csv, holidays_events.csv

Category mapping (Favorita family → AAIRM category):
    See FAVORITA_TO_AAIRM dict below.

References
----------
Paper Section 9.2 (Repo Guide).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aairm.utils.logging import get_logger

logger = get_logger(__name__)

FAVORITA_TO_AAIRM: dict[str, str] = {
    "GROCERY I": "grocery",
    "GROCERY II": "grocery",
    "BEVERAGES": "grocery",
    "BREAD/BAKERY": "frozen_food",
    "DAIRY": "frozen_food",
    "FROZEN FOODS": "frozen_food",
    "DELI": "frozen_food",
    "CLEANING": "cosmetics",
    "PERSONAL CARE": "cosmetics",
    "BEAUTY": "cosmetics",
    "CLOTHING": "apparel",
    "LADIESWEAR": "apparel",
    "MENSWEAR": "apparel",
    "HARDWARE": "dry_fruits",
    "HOME AND KITCHEN I": "dry_fruits",
    "HOME AND KITCHEN II": "dry_fruits",
}

PERISHABLE_CATEGORIES = {"frozen_food", "cosmetics", "dry_fruits"}
TOP_N_SKUS = 1200


class FavoritaAdapter:
    """Load and transform Corporación Favorita data to AAIRM unified schema.

    Args:
        data_dir: Path to directory containing raw Favorita CSV files.
    """

    def __init__(self, data_dir: str | Path = "data/raw/favorita") -> None:
        self._dir = Path(data_dir)

    def load(self) -> dict[str, Any]:
        """Load and transform Favorita data.

        Returns:
            Unified schema dict.

        Raises:
            FileNotFoundError: If required Favorita files are missing.
        """
        self._check_files()
        logger.info("favorita.loading", data_dir=str(self._dir))

        # --- Core files ---
        train = pd.read_csv(
            self._dir / "train.csv",
            parse_dates=["date"],
            dtype={"unit_sales": float},
            low_memory=False,
        )
        items = pd.read_csv(self._dir / "items.csv")
        holidays = pd.read_csv(self._dir / "holidays_events.csv", parse_dates=["date"])
        oil = pd.read_csv(self._dir / "oil.csv", parse_dates=["date"])

        logger.info("favorita.train_loaded", rows=len(train))

        # Clip negative sales (returns) to 0
        train["unit_sales"] = train["unit_sales"].clip(lower=0)
        # Some mirrors include nulls in onpromotion; normalize to bool.
        if "onpromotion" in train.columns:
            train["onpromotion"] = train["onpromotion"].fillna(False).astype(bool)

        # Merge item metadata
        train = train.merge(items[["item_nbr", "family", "perishable"]], on="item_nbr")
        train["sku_id"] = train["item_nbr"].astype(str) + "_" + train["store_nbr"].astype(str)
        train["category"] = train["family"].map(FAVORITA_TO_AAIRM).fillna("grocery")
        train["is_perishable"] = train["category"].isin(PERISHABLE_CATEGORIES) | (
            train["perishable"] == 1
        )

        # Select top SKUs by total sales
        sku_totals = train.groupby("sku_id")["unit_sales"].sum().nlargest(TOP_N_SKUS)
        top_skus = set(sku_totals.index)
        train = train[train["sku_id"].isin(top_skus)]
        logger.info("favorita.top_skus_selected", n=len(top_skus))

        # Build demand history
        train_pivot = train.groupby(["sku_id", "date"])["unit_sales"].sum().reset_index()
        demand_history: dict[str, np.ndarray] = {}
        for sku_id in top_skus:
            sku_data = train_pivot[train_pivot["sku_id"] == sku_id].sort_values("date")
            demand_history[sku_id] = sku_data["unit_sales"].values.astype(float)

        # Holiday calendar
        holiday_dates = set(holidays[holidays["type"].isin(["Holiday", "Bridge"])]["date"].dt.date)
        all_dates = sorted(train["date"].dt.date.unique())
        oil_indexed = (
            oil.set_index("date")["dcoilwtico"].reindex(pd.to_datetime(all_dates)).ffill().bfill()
        )

        calendar_out = pd.DataFrame(
            {
                "date": all_dates,
                "day_of_week": [pd.Timestamp(d).day_of_week for d in all_dates],
                "day_of_year": [pd.Timestamp(d).day_of_year for d in all_dates],
                "is_holiday": [d in holiday_dates for d in all_dates],
                "oil_price": oil_indexed.values,
            }
        )

        # SKU catalog
        sku_meta = (
            train.groupby("sku_id")
            .agg(
                category=("category", "first"),
                is_perishable=("is_perishable", "first"),
                mean_price=("unit_sales", "mean"),
            )
            .reset_index()
        )
        sku_meta["unit_cost"] = (sku_meta["mean_price"] * 0.65).round(2)
        sku_meta["unit_volume"] = 0.05
        sku_meta["margin_rate"] = 0.35
        sku_meta["shelf_life_days"] = None
        sku_meta = sku_meta.rename(columns={"mean_price": "base_demand_daily"})

        # Synthetic suppliers
        from aairm.data.adapters.m5_adapter import M5Adapter

        supplier_catalog = M5Adapter._build_synthetic_suppliers(sku_meta)

        logger.info("favorita.load_complete", n_skus=len(sku_meta))
        return {
            "demand_history": demand_history,
            "sku_catalog": sku_meta,
            "supplier_catalog": supplier_catalog,
            "calendar": calendar_out,
            "source": "favorita",
        }

    def _check_files(self) -> None:
        required = ["train.csv", "items.csv", "holidays_events.csv", "oil.csv"]
        for fname in required:
            path = self._dir / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"Favorita file not found: {path}\n" "Run: make download-data"
                )
