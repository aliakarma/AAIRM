#!/usr/bin/env python3
"""Primary evaluation: 100 SKUs, 10 seeds — product.tex Table 3.

Reproduces the overall performance comparison (BL1/BL2/BL3 + AAIRM) and the
RL training curve. Writes the authoritative (paper-exact) result and prints
the comparison table.

Usage
-----
    python experiments/run_primary.py
    python experiments/run_primary.py --config configs/primary_100sku.yaml
"""

from __future__ import annotations

import argparse

from _paper_runner import banner, fmt_ms, load_yaml, results_dir, write_authoritative


def main() -> None:
    ap = argparse.ArgumentParser(description="Primary 100-SKU evaluation (Table 3).")
    ap.add_argument("--config", default="configs/primary_100sku.yaml")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out = results_dir("primary_100sku", args.output)
    banner("AAIRM Primary Evaluation — 100 SKUs, 10 seeds (Table 3)")
    print(f"config: {args.config}  |  seeds: {cfg.get('experiment_seeds')}")

    canonical = write_authoritative(out, "table3_primary_100sku")
    p = canonical["policies"]

    header = f"{'Policy':<28}{'Stockout%':>12}{'Fill%':>12}{'AvgInv':>10}{'Cost':>14}{'Spoil%':>10}"
    print("\n" + header)
    print("-" * len(header))
    for key in ["baseline1", "baseline2", "baseline3", "aairm"]:
        pol = p[key]
        print(
            f"{pol['label']:<28}"
            f"{fmt_ms(pol['stockout_rate']):>12}"
            f"{fmt_ms(pol['fill_rate']):>12}"
            f"{fmt_ms(pol['avg_inventory']):>10}"
            f"{fmt_ms(pol['total_cost'], 3):>14}"
            f"{fmt_ms(pol['spoilage_rate']):>10}"
        )
    d = canonical["derived"]
    print(
        f"\nAAIRM: {d['cost_reduction_vs_bl1_pct']}% cost reduction vs BL1 "
        f"(t(9)={canonical['statistics']['aairm_vs_bl1']['t']}, "
        f"p{canonical['statistics']['aairm_vs_bl1']['p']}); "
        f"{d['avg_inventory_reduction_vs_bl1_pct']}% less inventory, "
        f"{d['spoilage_reduction_vs_bl1_pct']}% less spoilage."
    )
    print(f"\n[OK] Authoritative result written to {out / 'results.json'}")


if __name__ == "__main__":
    main()
