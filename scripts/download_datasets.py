#!/usr/bin/env python3
"""Download All Real-World Datasets from Kaggle.

Downloads M5 Forecasting Competition, Corporación Favorita Grocery Sales,
and Instacart Market Basket Analysis datasets to data/raw/.

Requires:
    - Kaggle API credentials in .env: KAGGLE_USERNAME and KAGGLE_KEY
    - pip install kaggle

Usage
-----
    python scripts/download_datasets.py
    python scripts/download_datasets.py --dataset m5
    make download-data

References
----------
Repo Guide Section 9.4.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ImportError:
    pass

DATASETS = {
    "m5": {
        "source_type": "competition",
        "competition": "m5-forecasting-accuracy",
        "fallback_dataset": "aryayadav0513/m5-forecasting-accuracy",
        "target_dir": "data/raw/m5",
        "required_files": [
            "sales_train_validation.csv",
            "calendar.csv",
            "sell_prices.csv",
        ],
        "description": "M5 Forecasting Competition (Walmart daily sales)",
        "approx_size_mb": 70,
    },
    "favorita": {
        "source_type": "competition",
        "competition": "favorita-grocery-sales-forecasting",
        "fallback_dataset": "siliconx/favoritagrocerysalesforecastingextracted",
        "target_dir": "data/raw/favorita",
        "required_files": [
            "train.csv",
            "items.csv",
            "stores.csv",
            "transactions.csv",
            "oil.csv",
            "holidays_events.csv",
        ],
        "description": "Corporación Favorita Grocery Sales",
        "approx_size_mb": 370,
    },
    "instacart": {
        "source_type": "dataset",
        "dataset": "psparks/instacart-market-basket-analysis",
        "target_dir": "data/raw/instacart",
        "required_files": [
            "orders.csv",
            "order_products__prior.csv",
            "products.csv",
            "departments.csv",
        ],
        "description": "Instacart Market Basket Analysis",
        "approx_size_mb": 200,
    },
}


def check_credentials() -> bool:
    """Verify Kaggle credentials are available."""
    username = os.getenv("KAGGLE_USERNAME", "")
    key = os.getenv("KAGGLE_KEY", "")
    if not username or not key:
        print(
            "ERROR: Kaggle credentials not found.\n"
            "Set KAGGLE_USERNAME and KAGGLE_KEY in your .env file.\n"
            "See .env.example for the required format.\n"
            "Get your API key from: https://www.kaggle.com/settings"
        )
        return False
    return True


def is_already_downloaded(dataset_cfg: dict) -> bool:
    """Check if all required files are already present."""
    target = Path(dataset_cfg["target_dir"])
    if not target.exists():
        return False
    return all((target / f).exists() for f in dataset_cfg["required_files"])


def download_dataset(name: str, dataset_cfg: dict) -> bool:
    """Download and extract one dataset.

    Args:
        name: Dataset name key.
        dataset_cfg: Dataset configuration dict.

    Returns:
        True on success, False on failure.
    """
    target = Path(dataset_cfg["target_dir"])

    if is_already_downloaded(dataset_cfg):
        print(
            f"  ✓ {name}: all required files already present " f"in {target} — skipping download."
        )
        return True

    target.mkdir(parents=True, exist_ok=True)
    print(
        f"\n  Downloading {name} ({dataset_cfg['description']}) "
        f"~{dataset_cfg['approx_size_mb']} MB ..."
    )

    try:
        import kaggle  # type: ignore

        kaggle.api.authenticate()
        t0 = time.time()
        source_type = dataset_cfg.get("source_type", "competition")
        if source_type == "competition":
            try:
                kaggle.api.competition_download_files(
                    competition=dataset_cfg["competition"],
                    path=str(target),
                    quiet=False,
                )
            except requests.exceptions.HTTPError as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                fallback = dataset_cfg.get("fallback_dataset")
                if status == 401 and fallback:
                    print(
                        "  Competition access unauthorized (401). "
                        f"Trying fallback dataset mirror: {fallback}"
                    )
                    kaggle.api.dataset_download_files(
                        dataset=fallback,
                        path=str(target),
                        quiet=False,
                    )
                else:
                    raise
        elif source_type == "dataset":
            kaggle.api.dataset_download_files(
                dataset=dataset_cfg["dataset"],
                path=str(target),
                quiet=False,
            )
        else:
            raise ValueError(f"Unsupported source_type for {name}: {source_type}")
        elapsed = round(time.time() - t0, 1)
        print(f"  Download complete ({elapsed}s). Extracting ...")

        # Extract all zip files
        for zip_path in target.glob("*.zip"):
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(target)
            zip_path.unlink()  # remove zip after extraction

        # Verify required files
        missing = [f for f in dataset_cfg["required_files"] if not (target / f).exists()]
        if missing:
            # Try one level deeper (some competitions nest files)
            for sub in target.iterdir():
                if sub.is_dir():
                    for f in missing[:]:
                        if (sub / f).exists():
                            import shutil

                            shutil.move(str(sub / f), str(target / f))
                            missing.remove(f)

        if missing:
            print(f"  WARNING: missing files after extraction: {missing}")
            return False

        # Report directory contents
        files = list(target.glob("*.csv"))
        total_mb = sum(f.stat().st_size for f in files) / 1024**2
        print(f"  ✓ {name}: {len(files)} CSV files, " f"{total_mb:.1f} MB total in {target}")
        return True

    except ImportError:
        print("  ERROR: 'kaggle' package not installed.\n" "  Run: pip install kaggle")
        return False
    except Exception as exc:
        print(f"  ERROR downloading {name}: {exc}")
        return False


def main() -> None:
    p = argparse.ArgumentParser(description="Download AAIRM real-world datasets.")
    p.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Dataset to download (default: all).",
    )
    args = p.parse_args()

    print("\n" + "=" * 60)
    print("AAIRM Dataset Downloader")
    print("=" * 60)

    if not check_credentials():
        sys.exit(1)

    to_download = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    success_all = True
    for name in to_download:
        ok = download_dataset(name, DATASETS[name])
        if not ok:
            success_all = False

    print("\n" + "=" * 60)
    if success_all:
        print("✓ All requested datasets downloaded successfully.")
        print("\nNext step:  python scripts/preprocess_all.py")
    else:
        print("✗ Some downloads failed. Check error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
