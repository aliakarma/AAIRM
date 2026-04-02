#!/usr/bin/env python3
"""Preprocess All Downloaded Datasets.

Runs the data adapter and Preprocessor pipeline for every available
real-world dataset, writing processed parquet files to data/processed/.

Must be run after scripts/download_datasets.py.

Usage
-----
    python scripts/preprocess_all.py
    python scripts/preprocess_all.py --dataset m5
    make preprocess-data
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aairm.utils.seed import set_global_seed
from aairm.utils.logging import configure_logging, get_logger
from aairm.data.preprocessor import Preprocessor

logger = get_logger(__name__)

AVAILABLE = ["m5", "favorita"]


def preprocess_m5() -> None:
    """Preprocess M5 dataset."""
    from aairm.data.adapters.m5_adapter import M5Adapter
    logger.info("preprocess.m5.start")
    t0 = time.perf_counter()
    adapter = M5Adapter("data/raw/m5")
    raw = adapter.load()
    preprocessor = Preprocessor("data/processed")
    preprocessor.fit_transform(
        raw["demand_history"], raw["sku_catalog"], raw["calendar"],
        dataset_name="m5",
    )
    logger.info("preprocess.m5.done", elapsed_s=round(time.perf_counter() - t0, 1))


def preprocess_favorita() -> None:
    """Preprocess Favorita dataset."""
    from aairm.data.adapters.favorita_adapter import FavoritaAdapter
    logger.info("preprocess.favorita.start")
    t0 = time.perf_counter()
    adapter = FavoritaAdapter("data/raw/favorita")
    raw = adapter.load()
    preprocessor = Preprocessor("data/processed")
    preprocessor.fit_transform(
        raw["demand_history"], raw["sku_catalog"], raw["calendar"],
        dataset_name="favorita",
    )
    logger.info("preprocess.favorita.done", elapsed_s=round(time.perf_counter() - t0, 1))


def main() -> None:
    p = argparse.ArgumentParser(description="Preprocess real-world datasets.")
    p.add_argument(
        "--dataset", choices=AVAILABLE + ["all"], default="all",
    )
    args = p.parse_args()

    configure_logging(level="INFO", fmt="console")
    set_global_seed(42)

    datasets = AVAILABLE if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        try:
            if ds == "m5":
                preprocess_m5()
            elif ds == "favorita":
                preprocess_favorita()
            print(f"✓ {ds} preprocessing complete → data/processed/{ds}/")
        except FileNotFoundError as exc:
            print(f"✗ {ds}: {exc}\n  Run: make download-data first.")
        except Exception as exc:  # noqa: BLE001
            logger.error("preprocess.failed", dataset=ds, error=str(exc))


if __name__ == "__main__":
    main()
