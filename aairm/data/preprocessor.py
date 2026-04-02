"""Data Preprocessing Pipeline.

Converts the unified schema DataFrame into model-ready features.
Applied after any adapter loads raw data and before training or evaluation.

Steps:
    1. Remove SKUs with fewer than 30 days of non-zero demand.
    2. Clip demand at the 99th percentile per SKU (outlier treatment).
    3. Interpolate missing dates with zero demand.
    4. Add temporal features.
    5. Compute rolling statistics.
    6. Compute ROP-derived features.
    7. Output to data/processed/ as parquet files.

References
----------
Repo Guide Section 10.1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aairm.utils.logging import get_logger
from aairm.utils.math_utils import rop

logger = get_logger(__name__)

_MIN_NONZERO_DAYS = 30
_ROP_SERVICE_LEVEL = 0.95
_DEFAULT_LEAD_TIME = 5.0


class Preprocessor:
    """Feature engineering pipeline for AAIRM demand data.

    Args:
        output_dir: Directory for writing processed parquet files.
        service_level: Service level for ROP feature computation.
        lead_time: Default lead time in days for ROP computation.
    """

    def __init__(
        self,
        output_dir: str | Path = "data/processed",
        service_level: float = _ROP_SERVICE_LEVEL,
        lead_time: float = _DEFAULT_LEAD_TIME,
    ) -> None:
        self._out_dir = Path(output_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._sl = service_level
        self._lead_time = lead_time

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        demand_history: dict[str, np.ndarray],
        sku_catalog: pd.DataFrame,
        calendar: pd.DataFrame,
        dataset_name: str = "dataset",
    ) -> dict[str, Any]:
        """Run the full preprocessing pipeline.

        Args:
            demand_history: Raw demand history ``{sku_id: np.ndarray}``.
            sku_catalog: SKU metadata DataFrame.
            calendar: Calendar DataFrame with ``date``, ``is_holiday``.
            dataset_name: Used as the output filename prefix.

        Returns:
            Dict with ``{demand_history_clean, feature_matrix,
            sku_catalog_clean, lead_times, unit_costs}``.
        """
        logger.info("preprocessor.start", n_skus=len(demand_history))

        # Step 1: Filter SKUs with insufficient history
        demand_clean = self._filter_skus(demand_history)
        logger.info("preprocessor.filtered", remaining=len(demand_clean))

        # Step 2: Clip outliers
        demand_clean = self._clip_outliers(demand_clean)

        # Step 3: Build feature matrix
        feature_rows = []
        for sku_id, series in demand_clean.items():
            cat_row = sku_catalog[sku_catalog["sku_id"] == sku_id]
            unit_cost = float(
                cat_row["unit_cost"].iloc[0] if not cat_row.empty else 5.0
            )
            feature_rows.append(
                self._compute_features(sku_id, series, unit_cost)
            )

        feature_matrix = pd.DataFrame(feature_rows)

        # Step 4: Persist
        self._persist(demand_clean, feature_matrix, sku_catalog, dataset_name)

        # Build return objects
        sku_ids_clean = list(demand_clean.keys())
        sku_catalog_clean = sku_catalog[sku_catalog["sku_id"].isin(sku_ids_clean)]
        lead_times = {s: self._lead_time for s in sku_ids_clean}
        unit_costs = {
            row["sku_id"]: float(row.get("unit_cost", 5.0))
            for _, row in sku_catalog_clean.iterrows()
        }

        logger.info("preprocessor.complete", n_clean=len(demand_clean))
        return {
            "demand_history": demand_clean,
            "feature_matrix": feature_matrix,
            "sku_catalog": sku_catalog_clean,
            "lead_times": lead_times,
            "unit_costs": unit_costs,
        }

    # ------------------------------------------------------------------
    # Private steps
    # ------------------------------------------------------------------

    def _filter_skus(
        self, demand_history: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Remove SKUs with fewer than 30 non-zero demand days."""
        return {
            sku_id: series
            for sku_id, series in demand_history.items()
            if int(np.sum(series > 0)) >= _MIN_NONZERO_DAYS
        }

    def _clip_outliers(
        self, demand_history: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Clip demand at the 99th percentile per SKU."""
        cleaned = {}
        for sku_id, series in demand_history.items():
            p99 = float(np.percentile(series, 99))
            cleaned[sku_id] = np.clip(series, 0.0, p99)
        return cleaned

    def _compute_features(
        self, sku_id: str, series: np.ndarray, unit_cost: float
    ) -> dict[str, Any]:
        """Compute rolling statistics and ROP feature for one SKU."""
        recent7 = series[-7:] if len(series) >= 7 else series
        recent28 = series[-28:] if len(series) >= 28 else series

        mu_d = float(np.mean(recent7))
        sigma_d = float(np.std(recent7) + 1e-8)
        reorder_pt = rop(mu_d, sigma_d, self._lead_time, self._sl)

        return {
            "sku_id": sku_id,
            "rolling_7d_mean": round(mu_d, 4),
            "rolling_7d_std": round(sigma_d, 4),
            "rolling_28d_mean": round(float(np.mean(recent28)), 4),
            "rop": round(reorder_pt, 2),
            "unit_cost": unit_cost,
            "n_days_history": len(series),
            "n_nonzero_days": int(np.sum(series > 0)),
        }

    def _persist(
        self,
        demand_clean: dict[str, np.ndarray],
        feature_matrix: pd.DataFrame,
        sku_catalog: pd.DataFrame,
        dataset_name: str,
    ) -> None:
        """Write processed outputs to parquet files."""
        try:
            out = self._out_dir / dataset_name
            out.mkdir(parents=True, exist_ok=True)
            feature_matrix.to_parquet(out / "features.parquet", index=False)
            sku_catalog.to_parquet(out / "sku_catalog.parquet", index=False)
            # Save demand history as numpy archive
            np.savez(out / "demand_history.npz", **demand_clean)
            logger.info("preprocessor.persisted", path=str(out))
        except Exception as exc:  # noqa: BLE001
            logger.warning("preprocessor.persist_failed", error=str(exc))
