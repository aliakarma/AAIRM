"""M5 Forecasting Competition Dataset Adapter.

Converts Walmart M5 sales data into the AAIRM unified schema.

Source: Kaggle — competitions/m5-forecasting-accuracy
License: Competition-specific (see data/README.md)

Files required in data/raw/m5/:
    sales_train_validation.csv  — daily unit sales (30,490 series × 1,913 days)
    calendar.csv                — date → weekday, event flags, SNAP flags
    sell_prices.csv             — item-store-week → unit sell price

SKU selection strategy:
    Top 1,200 series by total sales volume, matching the paper's SKU count.

Category mapping (M5 dept → AAIRM category):
    FOODS_1, FOODS_2  → grocery
    FOODS_3           → frozen_food (perishable)
    HOBBIES_1         → apparel
    HOBBIES_2         → dry_fruits
    HOUSEHOLD_1       → cosmetics (perishable)
    HOUSEHOLD_2       → dry_fruits

References
----------
Paper Section 9.1 (Repo Guide); M5 Competition description.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aairm.utils.logging import get_logger

logger = get_logger(__name__)

M5_TO_AAIRM: dict[str, str] = {
    "FOODS_1":     "grocery",
    "FOODS_2":     "grocery",
    "FOODS_3":     "frozen_food",
    "HOBBIES_1":   "apparel",
    "HOBBIES_2":   "dry_fruits",
    "HOUSEHOLD_1": "cosmetics",
    "HOUSEHOLD_2": "dry_fruits",
}

PERISHABLE_CATEGORIES = {"frozen_food", "cosmetics", "dry_fruits"}

TOP_N_SKUS = 1200


class M5Adapter:
    """Load and transform M5 Walmart data to the AAIRM unified schema.

    Args:
        data_dir: Path to directory containing the raw M5 CSV files.
    """

    def __init__(self, data_dir: str | Path = "data/raw/m5") -> None:
        self._dir = Path(data_dir)

    def load(self) -> dict[str, Any]:
        """Load and transform M5 data.

        Returns:
            Unified schema dict with keys:
            ``{demand_history, sku_catalog, supplier_catalog, calendar}``.

        Raises:
            FileNotFoundError: If required M5 files are missing.
        """
        self._check_files()

        logger.info("m5.loading", data_dir=str(self._dir))

        # --- Sales data ---
        sales = pd.read_csv(self._dir / "sales_train_validation.csv")
        calendar = pd.read_csv(self._dir / "calendar.csv")
        prices = pd.read_csv(self._dir / "sell_prices.csv")

        # Identify day columns
        day_cols = [c for c in sales.columns if c.startswith("d_")]
        logger.info("m5.sales_loaded", n_series=len(sales), n_days=len(day_cols))

        # Select top-1200 SKUs by total sales
        sales["total_sales"] = sales[day_cols].sum(axis=1)
        top_sales = sales.nlargest(TOP_N_SKUS, "total_sales").reset_index(drop=True)
        logger.info("m5.top_skus_selected", n=len(top_sales))

        # Build sku_id = item_id + "_" + store_id
        top_sales["sku_id"] = top_sales["item_id"] + "_" + top_sales["store_id"]

        # Map departments to AAIRM categories
        top_sales["category"] = top_sales["dept_id"].map(M5_TO_AAIRM).fillna("grocery")
        top_sales["is_perishable"] = top_sales["category"].isin(PERISHABLE_CATEGORIES)

        # Build demand_history dict
        demand_history: dict[str, np.ndarray] = {}
        for _, row in top_sales.iterrows():
            sku_id = row["sku_id"]
            demand_history[sku_id] = row[day_cols].values.astype(float)

        # Build calendar DataFrame
        calendar["date"] = pd.to_datetime(calendar["date"])
        calendar_out = pd.DataFrame(
            {
                "date": calendar["date"].dt.date,
                "day_of_week": calendar["date"].dt.day_of_week,
                "day_of_year": calendar["date"].dt.day_of_year,
                "is_holiday": (
                    calendar["event_name_1"].notna()
                    | calendar["event_name_2"].notna()
                ).astype(bool),
                "snap": (
                    calendar.get("snap_CA", pd.Series(0))
                    | calendar.get("snap_TX", pd.Series(0))
                    | calendar.get("snap_WI", pd.Series(0))
                ).astype(bool),
            }
        )

        # Build SKU catalog
        avg_prices = (
            prices.groupby("item_id")["sell_price"].mean().reset_index()
        )
        sku_catalog_rows = []
        for _, row in top_sales.iterrows():
            item_id = row["item_id"]
            avg_price_row = avg_prices[avg_prices["item_id"] == item_id]
            unit_price = (
                float(avg_price_row["sell_price"].iloc[0])
                if not avg_price_row.empty
                else 5.0
            )
            mu_d = float(np.mean(demand_history[row["sku_id"]]))
            sku_catalog_rows.append(
                {
                    "sku_id": row["sku_id"],
                    "category": row["category"],
                    "unit_cost": round(unit_price * 0.65, 2),   # approximate wholesale
                    "base_demand_daily": round(mu_d, 2),
                    "demand_sigma_frac": 0.15,
                    "is_perishable": row["is_perishable"],
                    "shelf_life_days": None,
                    "unit_volume": 0.05,
                    "margin_rate": 0.35,
                }
            )
        sku_catalog = pd.DataFrame(sku_catalog_rows)

        # Synthetic supplier catalog (M5 has no supplier data)
        supplier_catalog = self._build_synthetic_suppliers(sku_catalog)

        logger.info("m5.load_complete", n_skus=len(sku_catalog))
        return {
            "demand_history": demand_history,
            "sku_catalog": sku_catalog,
            "supplier_catalog": supplier_catalog,
            "calendar": calendar_out,
            "source": "m5",
        }

    def _check_files(self) -> None:
        """Verify required files exist."""
        required = [
            "sales_train_validation.csv",
            "calendar.csv",
            "sell_prices.csv",
        ]
        for fname in required:
            path = self._dir / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"M5 file not found: {path}\n"
                    "Run: make download-data  (requires Kaggle API credentials)"
                )

    @staticmethod
    def _build_synthetic_suppliers(sku_catalog: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic supplier catalog for M5 (no real supplier data)."""
        rng = np.random.default_rng(42)
        rows = []
        for i, (_, row) in enumerate(sku_catalog.iterrows()):
            n_sup = rng.integers(3, 6)
            for s in range(n_sup):
                markup = rng.uniform(0.90, 1.30)
                rows.append(
                    {
                        "supplier_id": f"M5-SUP-{i * 5 + s + 1:06d}",
                        "sku_id": row["sku_id"],
                        "unit_cost": round(float(row["unit_cost"]) * markup, 2),
                        "lead_time_mean": round(rng.uniform(2.0, 10.0), 1),
                        "lead_time_std": round(rng.uniform(0.3, 2.0), 2),
                        "reliability": round(rng.uniform(0.70, 0.99), 3),
                        "moq": int(rng.choice([10, 20, 50, 100])),
                        "country": str(rng.choice(["US", "CN", "MX", "CA"])),
                    }
                )
        return pd.DataFrame(rows)
