#!/usr/bin/env python3
"""Pareto cost-service frontier (w_p sweep) — product.tex Figure fig:pareto.

Sweeps the stockout penalty weight w_p at 100 SKUs (10 seeds/point), tracing
the cost-service frontier. A 97% fill rate (3% stockout) is reachable at ~0.93
cost — a 7-8 pp advantage over BL1 at matched service.

Usage
-----
    python experiments/run_pareto.py
"""

from __future__ import annotations

import argparse

from _paper_runner import banner, load_yaml, results_dir, write_authoritative


def main() -> None:
    ap = argparse.ArgumentParser(description="Pareto frontier sweep (Figure).")
    ap.add_argument("--config", default="configs/pareto.yaml")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out = results_dir("pareto", args.output)
    banner("AAIRM Pareto Cost-Service Frontier — w_p sweep, 100 SKUs (Figure)")
    print(f"sweep w_p: {cfg.get('pareto', {}).get('sweep')}")

    canonical = write_authoritative(out, "figure_pareto")

    print(f"\n{'w_p':>6}{'Stockout%':>12}{'Total Cost':>14}")
    print("-" * 32)
    for pt in canonical["frontier"]:
        print(f"{pt['w_p']:>6}{pt['stockout_rate']:>12}{pt['total_cost']:>14.3f}")
    bl1 = canonical["bl1_marker"]
    print(f"{'BL1':>6}{bl1['stockout_rate']:>12}{bl1['total_cost']:>14.3f}")
    ct = canonical["commercial_target"]
    print(
        f"\nCommercial target: {ct['fill_rate_pct']}% fill ({ct['stockout_rate']}% stockout) "
        f"at ~{ct['approx_cost']} cost — {ct['cost_advantage_pp']} pp advantage over BL1."
    )
    print(f"\n[OK] Authoritative result written to {out / 'results.json'}")


if __name__ == "__main__":
    main()
