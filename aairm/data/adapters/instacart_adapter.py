"""Instacart Market Basket Analysis Dataset Adapter.

Unlike M5 and Favorita, Instacart does NOT produce a demand time-series.
Instead it produces two outputs consumed by AAIRM agents:

    1. Product trend features (→ P2 TrendIntelligenceAgent)
       {reorder_rate, avg_add_to_cart_order, department_popularity}

    2. Category co-purchase matrix (→ P4 ContextEngine)
       A (n_categories × n_categories) co-occurrence probability matrix.

Source: Kaggle — competitions/instacart-market-basket-analysis
License: CC BY-SA 4.0 (see data/README.md)

Files required in data/raw/instacart/:
    orders.csv
    order_products__prior.csv
    products.csv
    departments.csv

References
----------
Paper Section 9.3 (Repo Guide).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aairm.utils.logging import get_logger

logger = get_logger(__name__)

INSTACART_DEPT_TO_AAIRM: dict[str, str] = {
    "produce":       "grocery",
    "dairy eggs":    "frozen_food",
    "meat seafood":  "frozen_food",
    "frozen":        "frozen_food",
    "bakery":        "grocery",
    "deli":          "grocery",
    "canned goods":  "dry_fruits",
    "dry goods pasta":"dry_fruits",
    "snacks":        "dry_fruits",
    "beverages":     "grocery",
    "personal care": "cosmetics",
    "beauty":        "cosmetics",
    "household":     "cosmetics",
    "clothes":       "apparel",
    "babies":        "apparel",
}

AAIRM_CATEGORIES = ["grocery", "frozen_food", "apparel", "cosmetics", "dry_fruits"]


class InstacartAdapter:
    """Load Instacart data and produce trend features + co-purchase matrix.

    Args:
        data_dir: Path to directory containing raw Instacart CSV files.
    """

    def __init__(self, data_dir: str | Path = "data/raw/instacart") -> None:
        self._dir = Path(data_dir)

    def load(self) -> dict[str, Any]:
        """Load Instacart data and extract trend features.

        Returns:
            Dict with keys:
            ``{product_trend_features, category_copurchase_matrix}``.

        Raises:
            FileNotFoundError: If required Instacart files are missing.
        """
        self._check_files()
        logger.info("instacart.loading", data_dir=str(self._dir))

        orders = pd.read_csv(self._dir / "orders.csv")
        prior = pd.read_csv(self._dir / "order_products__prior.csv")
        products = pd.read_csv(self._dir / "products.csv")
        departments = pd.read_csv(self._dir / "departments.csv")

        # Merge metadata
        products = products.merge(departments, on="department_id")
        products["aairm_category"] = (
            products["department"].str.lower().map(INSTACART_DEPT_TO_AAIRM).fillna("grocery")
        )
        prior = prior.merge(products[["product_id", "product_name", "department", "aairm_category"]], on="product_id")

        logger.info("instacart.prior_orders", rows=len(prior))

        # --- Trend features per product ---
        product_stats = (
            prior.groupby("product_id")
            .agg(
                reorder_rate=("reordered", "mean"),
                avg_cart_position=("add_to_cart_order", "mean"),
                order_count=("order_id", "count"),
                product_name=("product_name", "first"),
                aairm_category=("aairm_category", "first"),
            )
            .reset_index()
        )

        # Normalise to [0,1] trend score
        max_orders = product_stats["order_count"].max()
        product_stats["trend_score"] = (
            0.5 * product_stats["reorder_rate"]
            + 0.3 * (1.0 - product_stats["avg_cart_position"] / product_stats["avg_cart_position"].max())
            + 0.2 * (product_stats["order_count"] / max(max_orders, 1))
        ).clip(0, 1)

        trend_features = product_stats[
            ["product_id", "product_name", "aairm_category",
             "reorder_rate", "avg_cart_position", "trend_score"]
        ].to_dict(orient="records")

        # --- Category co-purchase matrix ---
        # Join products to get categories for each order
        order_categories = prior.merge(
            orders[["order_id", "user_id"]], on="order_id"
        )[["order_id", "aairm_category"]].drop_duplicates()

        n_cat = len(AAIRM_CATEGORIES)
        cat_idx = {c: i for i, c in enumerate(AAIRM_CATEGORIES)}
        copurchase = np.zeros((n_cat, n_cat), dtype=float)

        for _, group in order_categories.groupby("order_id"):
            cats = group["aairm_category"].unique()
            idxs = [cat_idx[c] for c in cats if c in cat_idx]
            for i in idxs:
                for j in idxs:
                    copurchase[i, j] += 1.0

        # Row-normalise
        row_sums = copurchase.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        copurchase_prob = copurchase / row_sums

        copurchase_df = pd.DataFrame(
            copurchase_prob,
            index=AAIRM_CATEGORIES,
            columns=AAIRM_CATEGORIES,
        )

        logger.info("instacart.load_complete", n_products=len(trend_features))
        return {
            "product_trend_features": trend_features,
            "category_copurchase_matrix": copurchase_df,
            "source": "instacart",
        }

    def _check_files(self) -> None:
        required = [
            "orders.csv",
            "order_products__prior.csv",
            "products.csv",
            "departments.csv",
        ]
        for fname in required:
            path = self._dir / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"Instacart file not found: {path}\n"
                    "Run: make download-data"
                )
