#!/usr/bin/env python3
"""Dry-fruits recalibration at 500 SKUs — product.tex Table 8.

Documents the reward-hacking failure (31.54% stockout at default w_p=1.2) and
its category-specific fix: raising w_p^DF to 9.0 restores a 95.3% fill rate
while category cost stays 0.973 (below BL1), at the cost of higher spoilage.

Usage
-----
    python experiments/run_dryfruits_recal.py
"""

from __future__ import annotations

import argparse

from _paper_runner import banner, fmt_ms, results_dir, write_authoritative


def main() -> None:
    ap = argparse.ArgumentParser(description="Dry-fruits recalibration sweep (Table 8).")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    out = results_dir("dryfruits_recal", args.output)
    banner("AAIRM Dry-Fruits Recalibration — w_p^DF sweep, 500 SKUs (Table 8)")

    canonical = write_authoritative(out, "table8_dryfruits_recal")

    header = f"{'w_p^DF':<24}{'Stockout%':>14}{'Fill%':>14}{'DF Cost':>14}{'Spoil%':>12}"
    print("\n" + header)
    print("-" * len(header))
    for s in canonical["sweep"]:
        print(
            f"{s['label']:<24}"
            f"{fmt_ms(s['stockout_rate']):>14}"
            f"{fmt_ms(s['fill_rate']):>14}"
            f"{fmt_ms(s['df_cost'], 3):>14}"
            f"{fmt_ms(s['spoilage_rate']):>12}"
        )
    print(f"\n{canonical['note']}")
    print(f"Recommended w_p^DF = {canonical['recommended_w_p']}.")
    print(f"\n[OK] Authoritative result written to {out / 'results.json'}")


if __name__ == "__main__":
    main()
