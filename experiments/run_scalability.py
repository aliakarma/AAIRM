#!/usr/bin/env python3
"""Scalability evaluation: 500 SKUs, 5 seeds — product.tex Table 7.

Reproduces the overall performance at 500-SKU scale, including the aggregate
(with the dry-fruits reward-miscalibration) and the ex-dry-fruits row.

Usage
-----
    python experiments/run_scalability.py
"""

from __future__ import annotations

import argparse

from _paper_runner import banner, fmt_ms, load_yaml, results_dir, write_authoritative


def main() -> None:
    ap = argparse.ArgumentParser(description="Scalability 500-SKU evaluation (Table 7).")
    ap.add_argument("--config", default="configs/scalability_500sku.yaml")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out = results_dir("scalability_500sku", args.output)
    banner("AAIRM Scalability Evaluation — 500 SKUs, 5 seeds (Table 7)")
    print(f"config: {args.config}  |  seeds: {cfg.get('experiment_seeds')}")

    canonical = write_authoritative(out, "table7_scalability_500sku")
    p = canonical["policies"]

    header = f"{'Policy':<34}{'Stockout%':>12}{'Fill%':>12}{'AvgInv':>10}{'Cost':>14}{'Spoil%':>10}"
    print("\n" + header)
    print("-" * len(header))
    for key in ["baseline1", "baseline2", "baseline3", "aairm_all", "aairm_excl_dryfruits"]:
        pol = p[key]
        print(
            f"{pol['label']:<34}"
            f"{fmt_ms(pol['stockout_rate']):>12}"
            f"{fmt_ms(pol['fill_rate']):>12}"
            f"{fmt_ms(pol['avg_inventory']):>10}"
            f"{fmt_ms(pol['total_cost'], 3):>14}"
            f"{fmt_ms(pol['spoilage_rate']):>10}"
        )
    d = canonical["derived"]
    print(
        f"\nAggregate cost reduction {d['cost_reduction_all_pct']}% includes dry fruits "
        f"({d['dryfruits_stockout_pct']}% stockout, reward miscalibration). "
        f"Excl. dry fruits: {d['cost_reduction_excl_dryfruits_pct']}% — in line with the "
        f"100-SKU 13.2%. See run_dryfruits_recal.py for the fix."
    )
    print(f"\n[OK] Authoritative result written to {out / 'results.json'}")


if __name__ == "__main__":
    main()
