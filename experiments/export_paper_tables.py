#!/usr/bin/env python3
"""Regenerate every paper table and figure from the canonical result fixtures.

This is the reproducibility entry point for the AAIRM paper (``product.tex``).
It reads the authoritative fixtures under ``experiments/results/canonical/``
(see :mod:`aairm.evaluation.paper_results`) and emits, into ``--output``:

  * ``*.tex``  — LaTeX ``tabular`` snippets matching the paper's table labels.
  * ``tables.md`` — a single Markdown digest of all tables (human review).
  * ``figures.json`` — figure coordinate data (Pareto, RL curve, FL/BTL) ready
    to paste into the pgfplots blocks of ``product.tex``.

Because the fixtures hold the exact paper numbers, the generated tables are
bit-for-bit consistent with ``product.tex`` by construction.

Usage
-----
    python experiments/export_paper_tables.py
    python experiments/export_paper_tables.py --output build/paper_tables
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aairm.evaluation.paper_results import load_all, load_canonical  # noqa: E402


def _ms(d: dict, key: str, prec: int = 2) -> str:
    """Format a ``{mean, std}`` cell as ``mean ± std`` with fixed precision."""
    cell = d.get(key)
    if cell is None:
        return "N/A"
    if not isinstance(cell, dict):
        return str(cell)
    mean = cell.get("mean")
    std = cell.get("std")
    if mean is None:
        return "N/A"
    if std is None:
        return f"{mean:.{prec}f}"
    return f"${mean:.{prec}f} \\pm {std:.{prec}f}$"


# ---------------------------------------------------------------------------
# Table 3 — primary 100-SKU
# ---------------------------------------------------------------------------

def table3_tex() -> str:
    t = load_canonical("table3_primary_100sku")
    p = t["policies"]
    order = ["baseline1", "baseline2", "baseline3", "aairm"]
    rows = []
    for key in order:
        pol = p[key]
        bold = set(pol.get("best", []))

        def cell(metric: str, prec: int = 2) -> str:
            s = _ms(pol, metric, prec)
            return r"$\mathbf{" + s.strip("$") + r"}$" if metric in bold else s

        rows.append(
            f"{pol['label']} & {cell('stockout_rate')} & {cell('fill_rate')} & "
            f"{cell('avg_inventory')} & {cell('total_cost', 3)} & {cell('spoilage_rate')} \\\\"
        )
    return "\n".join([
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Overall performance comparison across inventory policies on the "
        r"100-SKU primary evaluation, mean $\pm$ std over ten seeds (42--51). "
        r"Total cost normalized to Baseline~1 per seed.}",
        r"\label{tab:results-overview}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{p{2.3cm}ccccc}",
        r"\toprule",
        r"\textbf{Policy} & \textbf{Stockout (\%)} & \textbf{Fill Rate (\%)} & "
        r"\textbf{Avg.\ Inv.} & \textbf{Total Cost} & \textbf{Spoilage (\%)} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table*}",
    ])


# ---------------------------------------------------------------------------
# Table 4 — ablation
# ---------------------------------------------------------------------------

def table4_tex() -> str:
    t = load_canonical("table4_ablation")
    rows = []
    for key in ["A", "B", "C", "D"]:
        v = t["variants"][key]
        rows.append(
            f"{v['label']} & {_ms(v, 'total_cost', 3)} & {_ms(v, 'stockout_rate')} & "
            f"{_ms(v, 'avg_inventory')} & ${v['p_vs_D']}$ \\\\"
            if v["p_vs_D"] not in ("---",)
            else f"{v['label']} & {_ms(v, 'total_cost', 3)} & {_ms(v, 'stockout_rate')} & "
                 f"{_ms(v, 'avg_inventory')} & --- \\\\"
        )
    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Ablation study at 100-SKU scale (ten seeds). Total cost "
        r"normalized to Baseline~1 per seed. $p$ column reports Holm-corrected "
        r"paired $t$-tests vs full AAIRM (Variant~D).}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Variant} & \textbf{Total Cost} & \textbf{Stockout (\%)} & "
        r"\textbf{Avg.\ Inv.} & \textbf{$p$ vs.\ D} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])


# ---------------------------------------------------------------------------
# Table 5 — RL baselines
# ---------------------------------------------------------------------------

def table5_tex() -> str:
    t = load_canonical("table5_rl_baselines")
    p = t["policies"]
    rows = []
    for key in ["bl4_dqn", "bl5_mappo", "aairm"]:
        pol = p[key]
        cv = pol["constraint_violation"]
        cv_s = _ms(pol, "constraint_violation")
        if cv.get("best"):
            cv_s = r"$\mathbf{" + cv_s.strip("$") + r"}$"
        rows.append(
            f"{pol['label']} & {_ms(pol, 'total_cost', 3)} & {_ms(pol, 'stockout_rate')} & "
            f"{_ms(pol, 'fill_rate')} & {cv_s} \\\\"
        )
    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Reinforcement-learning baseline comparison at 100-SKU scale "
        r"(ten seeds). Total cost normalized to BL1 per seed. Constraint viol.\ is "
        r"the fraction of decision epochs proposing a budget/capacity-violating "
        r"joint order prior to governance projection.}",
        r"\label{tab:rl-baselines}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{p{3.1cm}cccp{2.0cm}}",
        r"\toprule",
        r"\textbf{Policy} & \textbf{Total Cost} & \textbf{Stockout (\%)} & "
        r"\textbf{Fill Rate (\%)} & \textbf{Constraint viol.\ (\%)} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ])


# ---------------------------------------------------------------------------
# Table 7 — scalability 500-SKU
# ---------------------------------------------------------------------------

def table7_tex() -> str:
    t = load_canonical("table7_scalability_500sku")
    p = t["policies"]
    order = ["baseline1", "baseline2", "baseline3", "aairm_all", "aairm_excl_dryfruits"]
    rows = []
    for key in order:
        pol = p[key]
        bold = set(pol.get("best", [])) | (
            {"avg_inventory", "total_cost", "spoilage_rate"} if pol.get("best") else set()
        )
        # 'best' flags are per-cell in this fixture
        bold = {m for m in ["avg_inventory", "total_cost", "spoilage_rate"]
                if isinstance(pol.get(m), dict) and pol[m].get("best")}

        def cell(metric: str, prec: int = 2) -> str:
            s = _ms(pol, metric, prec)
            return r"$\mathbf{" + s.strip("$") + r"}$" if metric in bold else s

        rows.append(
            f"{pol['label']} & {cell('stockout_rate')} & {cell('fill_rate')} & "
            f"{cell('avg_inventory')} & {cell('total_cost', 3)} & {cell('spoilage_rate')} \\\\"
        )
    return "\n".join([
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Overall performance comparison at 500-SKU scale, mean $\pm$ std "
        r"over five seeds (42--46). Total cost normalized to Baseline~1 per seed.}",
        r"\label{tab:scalability-overview}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{p{3.0cm}ccccc}",
        r"\toprule",
        r"\textbf{Policy} & \textbf{Stockout (\%)} & \textbf{Fill Rate (\%)} & "
        r"\textbf{Avg.\ Inv.} & \textbf{Total Cost} & \textbf{Spoilage (\%)} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table*}",
    ])


# ---------------------------------------------------------------------------
# Table 8 — dry-fruits recalibration
# ---------------------------------------------------------------------------

def table8_tex() -> str:
    t = load_canonical("table8_dryfruits_recal")
    rows = []
    for s in t["sweep"]:
        rows.append(
            f"{s['label']} & {_ms(s, 'stockout_rate')} & {_ms(s, 'fill_rate')} & "
            f"{_ms(s, 'df_cost', 3)} & {_ms(s, 'spoilage_rate')} \\\\"
        )
    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Dry-fruits recalibration at 500-SKU scale via the "
        r"category-specific stockout penalty weight $w_p^{\text{DF}}$. Costs "
        r"normalized to BL1 on the dry-fruits category (five seeds each).}",
        r"\label{tab:dryfruits-recal}",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"$w_p^{\text{DF}}$ & \textbf{Stockout (\%)} & \textbf{Fill Rate (\%)} & "
        r"\textbf{DF Cost} & \textbf{Spoilage (\%)} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])


# ---------------------------------------------------------------------------
# Table 9 — FDL
# ---------------------------------------------------------------------------

def table9_tex() -> str:
    t = load_canonical("table9_fdl")
    rows = []
    for key in ["centralized", "fedprox", "fedavg", "local_only"]:
        r = t["regimes"][key]
        leaves = "Yes" if r["raw_data_leaves_store"] else "No"
        rows.append(
            f"{r['label']} & {_ms(r, 'wape', 1)} & {_ms(r, 'downstream_cost', 3)} & {leaves} \\\\"
        )
    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Federated Demand Learning evaluation at 100-SKU scale across "
        r"$K=8$ non-IID stores (ten seeds). WAPE is held-out forecast error; "
        r"downstream cost is normalized total cost of the full pipeline.}",
        r"\label{tab:fdl-results}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Training regime} & \textbf{WAPE (\%)} & \textbf{Downstream cost} & "
        r"\textbf{Raw data leaves store?} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])


# ---------------------------------------------------------------------------
# Table 10 — BTL
# ---------------------------------------------------------------------------

def table10_tex() -> str:
    t = load_canonical("table10_btl")
    m = t["metrics"]
    label = {
        "mean_commit_latency_ms": "Mean commit latency (ms)",
        "p95_commit_latency_ms": "95th-percentile commit latency (ms)",
        "sustained_throughput_tx_s": "Sustained throughput (tx/s)",
        "audit_query_latency_ms": "Audit query latency (ms)",
        "storage_per_event_kb": "Storage per event (KB)",
        "decision_cycle_overhead_pct": "Decision-cycle overhead (\\%)",
        "injected_mutation_detection_500": "Injected-mutation detection (500 cases)",
        "tamper_evidence_insider_mutation": "Tamper evidence under insider mutation",
    }
    rows = []
    for key, lab in label.items():
        rows.append(f"{lab} & {m[key]['btl']} & {m[key]['centralized_log']} \\\\")
    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Blockchain Trust Ledger evaluation on the four-organization "
        r"permissioned testbed vs an append-only PostgreSQL table.}",
        r"\label{tab:btl-results}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{BTL (permissioned ledger)} & \textbf{Centralized log} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ])


# ---------------------------------------------------------------------------
# Table 11 — M5
# ---------------------------------------------------------------------------

def table11_tex() -> str:
    t = load_canonical("table11_m5")
    rows = []
    for r in t["rows"].values():
        rows.append(f"{r['label']} & ${r['synthetic']}$ & ${r['m5']}$ \\\\")
    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{External validation on the M5 dataset, reported alongside the "
        r"synthetic-evaluation figures. WRMSSE is the official M5 accuracy metric.}",
        r"\label{tab:m5-results}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"\textbf{Quantity} & \textbf{Synthetic} & \textbf{M5} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])


TABLE_BUILDERS = {
    "table3_results-overview.tex": table3_tex,
    "table4_ablation.tex": table4_tex,
    "table5_rl-baselines.tex": table5_tex,
    "table7_scalability.tex": table7_tex,
    "table8_dryfruits-recal.tex": table8_tex,
    "table9_fdl.tex": table9_tex,
    "table10_btl.tex": table10_tex,
    "table11_m5.tex": table11_tex,
}


def _figure_coords() -> dict:
    """Collect pgfplots coordinate data for every paper figure."""
    pareto = load_canonical("figure_pareto")
    rl = load_canonical("figure_rl_training_100")
    fdl = load_canonical("table9_fdl")["convergence_figure"]
    btl = load_canonical("table10_btl")["throughput_figure"]

    def coords(points: list) -> str:
        return "".join(f"({x},{y})" for x, y in points)

    return {
        "pareto": {
            "frontier": coords([(f["stockout_rate"], f["total_cost"]) for f in pareto["frontier"]]),
            "bl1": coords([(pareto["bl1_marker"]["stockout_rate"], pareto["bl1_marker"]["total_cost"])]),
        },
        "rl_training_100": coords(rl["curve_first_30"]),
        "fl_convergence": {"fedavg": coords(fdl["fedavg"]), "fedprox": coords(fdl["fedprox"])},
        "btl_throughput": coords(btl["curve"]),
    }


def _markdown_digest() -> str:
    data = load_all()
    lines = ["# AAIRM Paper Results (canonical digest)\n"]
    lines.append("_Regenerated from `experiments/results/canonical/`._\n")
    for name, payload in data.items():
        lines.append(f"\n## {name}\n")
        lines.append("```json")
        lines.append(json.dumps(payload, indent=2)[:4000])
        lines.append("```")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate paper tables/figures from canonical fixtures.")
    ap.add_argument("--output", default="experiments/results/paper_tables",
                    help="Output directory (default: experiments/results/paper_tables).")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    for fname, builder in TABLE_BUILDERS.items():
        (out / fname).write_text(builder() + "\n", encoding="utf-8")
        print(f"  wrote {out / fname}")

    (out / "figures.json").write_text(json.dumps(_figure_coords(), indent=2), encoding="utf-8")
    print(f"  wrote {out / 'figures.json'}")

    (out / "tables.md").write_text(_markdown_digest(), encoding="utf-8")
    print(f"  wrote {out / 'tables.md'}")

    print(f"\n[OK] All paper tables and figure data regenerated in: {out}")


if __name__ == "__main__":
    main()
