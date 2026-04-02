#!/usr/bin/env python3
"""Generate the AAIRM Synthetic Retail Simulation Dataset.

Produces the 1,200-SKU, 730-day synthetic dataset described in
Section 5.1 of the paper and saves it to data/synthetic/.

Output files:
    data/synthetic/sku_catalog.csv          — 1,200 SKU records
    data/synthetic/supplier_catalog.csv     — 3–5 suppliers per SKU
    data/synthetic/demand_matrix.npz        — (n_skus × n_days) demand array
    data/synthetic/demand_params_seed.json  — DGP parameters (reference)

Usage
-----
    python scripts/generate_synthetic.py
    python scripts/generate_synthetic.py --n-skus 100 --n-days 60   # fast
    make generate-synthetic

References
----------
Paper Section 5.1; Repo Guide Section 9.4.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from aairm.utils.seed import set_global_seed
from aairm.utils.logging import configure_logging, get_logger
from aairm.simulation.sku_catalog import SKUCatalog
from aairm.simulation.demand_generator import DemandGenerator
from aairm.simulation.supplier_simulator import SupplierSimulator

logger = get_logger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate AAIRM synthetic simulation dataset."
    )
    p.add_argument("--n-skus",   type=int, default=1200)
    p.add_argument("--n-days",   type=int, default=730)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--output",   default="data/synthetic")
    args = p.parse_args()

    configure_logging(level="INFO", fmt="console")
    set_global_seed(args.seed)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # ── SKU Catalog ────────────────────────────────────────────────────────
    print(f"\nGenerating SKU catalog ({args.n_skus} SKUs, seed={args.seed}) ...")
    t0 = time.perf_counter()

    n_cats = 5 if args.n_skus >= 5 else 1
    if args.n_skus % n_cats != 0:
        args.n_skus = (args.n_skus // n_cats) * n_cats
        print(f"  Adjusted n_skus to {args.n_skus} for even category split.")

    catalog = SKUCatalog(n_skus=args.n_skus, seed=args.seed)
    sku_df = catalog.to_dataframe()
    sku_path = out / "sku_catalog.csv"
    sku_df.to_csv(sku_path, index=False)
    print(f"  ✓ {len(sku_df)} SKUs written to {sku_path}")
    print(f"  Category breakdown: {sku_df['category'].value_counts().to_dict()}")

    # ── Supplier Catalog ───────────────────────────────────────────────────
    print(f"\nGenerating supplier catalog ...")
    sup = SupplierSimulator(catalog, seed=args.seed)
    sup_rows = []
    for sku_id in catalog.sku_ids:
        sup_rows.extend(sup.query_catalogue(sku_id))
    import pandas as pd
    sup_df = pd.DataFrame(sup_rows)
    sup_path = out / "supplier_catalog.csv"
    sup_df.to_csv(sup_path, index=False)
    print(f"  ✓ {len(sup_df)} supplier records written to {sup_path}")

    # ── Demand Matrix ──────────────────────────────────────────────────────
    print(f"\nGenerating demand matrix ({args.n_skus} × {args.n_days} days) ...")
    gen = DemandGenerator(catalog, n_days=args.n_days, seed=args.seed)

    # gen._demand_matrix is shape (n_skus, n_days)
    demand_matrix = gen._demand_matrix
    dem_path = out / "demand_matrix.npz"
    sku_id_arr = np.array(catalog.sku_ids)
    np.savez(dem_path, demand=demand_matrix, sku_ids=sku_id_arr)
    print(
        f"  ✓ Demand matrix {demand_matrix.shape} written to {dem_path}"
        f"\n  Mean daily demand: {demand_matrix.mean():.2f} units/SKU/day"
        f"\n  Total demand over horizon: {demand_matrix.sum():,.0f} units"
    )

    # ── DGP Parameters ─────────────────────────────────────────────────────
    dgp_params = {
        "paper_reference": "Syed et al. (2025), Section 5.1",
        "generated_at": str(pd.Timestamp.now()),
        "n_skus": args.n_skus,
        "n_days": args.n_days,
        "seed": args.seed,
        "category_seasonality_amplitude": {
            "grocery": 0.15, "frozen_food": 0.25, "apparel": 0.40,
            "cosmetics": 0.20, "dry_fruits": 0.30,
        },
        "base_demand_range": {"min": 20.0, "max": 200.0, "dist": "Uniform"},
        "demand_noise": {"dist": "LogNormal", "mu": 0.0, "sigma_range": [0.05, 0.25]},
        "promotional_uplift": {"fraction": 0.10, "multiplier_range": [1.5, 2.5]},
        "holiday_spikes": [
            {"name": "Eid_Al_Fitr",  "doy": 100, "dur": 3, "uplift": 2.0},
            {"name": "Eid_Al_Adha",  "doy": 175, "dur": 3, "uplift": 1.8},
            {"name": "Ramadan",      "doy": 60,  "dur": 30,"uplift": 1.3},
            {"name": "National_Day", "doy": 272, "dur": 2, "uplift": 1.4},
        ],
    }
    dgp_path = out / "demand_params.json"
    dgp_path.write_text(json.dumps(dgp_params, indent=2))
    print(f"  ✓ DGP parameters written to {dgp_path}")

    elapsed = round(time.perf_counter() - t0, 1)
    print(f"\n✓ Synthetic dataset generation complete in {elapsed}s")
    print(f"  Output directory: {out.resolve()}")


if __name__ == "__main__":
    main()
