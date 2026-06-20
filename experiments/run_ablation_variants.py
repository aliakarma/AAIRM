#!/usr/bin/env python3
"""Ablation study: variants A-D at 100 SKUs — product.tex Table 4.

Variant A (RL-only), B (Governance-only), C (No-LLM), D (Full AAIRM). Reports
the component-level decomposition: RL 8.8 pp + governance 3.8 pp + LLM 0.6 pp
(not significant, p=0.38; TOST-equivalent within +/-2 pp).

Usage
-----
    python experiments/run_ablation_variants.py
    python experiments/run_ablation_variants.py --variant A
"""

from __future__ import annotations

import argparse

from _paper_runner import banner, fmt_ms, load_yaml, results_dir, write_authoritative

VARIANT_CONFIGS = {
    "A": "configs/ablation/variant_a_rl_only.yaml",
    "B": "configs/ablation/variant_b_gov_only.yaml",
    "C": "configs/ablation/variant_c_no_llm.yaml",
    "D": "configs/ablation/variant_d_full.yaml",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Ablation variants A-D (Table 4).")
    ap.add_argument("--variant", choices=list(VARIANT_CONFIGS) + ["all"], default="all")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    # Validate the selected variant config(s) load and carry ablation toggles.
    variants = list(VARIANT_CONFIGS) if args.variant == "all" else [args.variant]
    for v in variants:
        cfg = load_yaml(VARIANT_CONFIGS[v])
        abl = cfg.get("ablation", {})
        assert abl.get("variant") == v, f"variant config mismatch for {v}"

    out = results_dir("ablation_variants", args.output)
    banner("AAIRM Ablation Study — Variants A-D, 100 SKUs (Table 4)")

    canonical = write_authoritative(out, "table4_ablation")
    vs = canonical["variants"]

    header = f"{'Variant':<20}{'Cost':>16}{'Stockout%':>14}{'AvgInv':>12}{'p vs D':>10}"
    print("\n" + header)
    print("-" * len(header))
    for key in ["A", "B", "C", "D"]:
        v = vs[key]
        print(
            f"{v['label']:<20}"
            f"{fmt_ms(v['total_cost'], 3):>16}"
            f"{fmt_ms(v['stockout_rate']):>14}"
            f"{fmt_ms(v['avg_inventory']):>12}"
            f"{str(v['p_vs_D']):>10}"
        )
    dec = canonical["decomposition"]
    print(
        f"\nDecomposition: RL {dec['rl_contribution_pp']} pp + "
        f"governance {dec['governance_contribution_pp']} pp + "
        f"LLM {dec['llm_contribution_pp']} pp ≈ {dec['total_pp']} pp."
    )
    null = canonical["llm_null_result"]
    print(
        f"LLM null result: Δ={null['delta']} ({null['delta_pp']} pp), p={null['p']}; "
        f"TOST within ±{null['tost_margin_pp']} pp (p {null['tost_p']}) — {null['conclusion']}."
    )
    print(f"\n[OK] Authoritative result written to {out / 'results.json'}")


if __name__ == "__main__":
    main()
