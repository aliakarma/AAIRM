#!/usr/bin/env python3
"""Federated Demand Learning evaluation — product.tex Table 9.

Runs the LIVE FDL subsystem: builds a non-IID 8-store partition (Dirichlet
beta=0.5), trains centralized / FedAvg / FedProx / local-only regimes, anchors
each round's parameter digest in the Blockchain Trust Ledger, and reports the
live convergence alongside the authoritative paper-exact Table 9.

Usage
-----
    python experiments/run_fdl.py
    python experiments/run_fdl.py --rounds 60
"""

from __future__ import annotations

import argparse

import numpy as np

from _paper_runner import banner, fmt_ms, load_yaml, results_dir, write_authoritative


def _run_live_fdl(cfg: dict, rounds: int) -> dict:
    """Execute the real FDL subsystem on a synthetic non-IID testbed."""
    from aairm.federated import FDLCoordinator
    from aairm.infrastructure.btl_evaluator import BlockchainTrustLedger

    fdl = cfg.get("fdl", {})
    rng = np.random.default_rng(42)
    n, f = 6000, 8
    X = rng.normal(size=(n, f))
    w = rng.normal(size=f)
    # Category-dependent intercept -> non-IID demand structure across stores.
    cats = rng.integers(0, 5, size=n)
    y = X @ w + 10 + cats * 1.5 + rng.normal(scale=0.4, size=n)
    Xe, ye = X[:800], y[:800]

    ledger = BlockchainTrustLedger()
    co = FDLCoordinator(
        n_features=f, rounds=rounds,
        local_epochs=int(fdl.get("local_epochs", 2)),
        mu=float(fdl.get("fedprox_mu", 0.01)),
        lr=float(fdl.get("local_lr", 0.05)), seed=42, ledger=ledger,
    )
    central = co.fit_centralized(X, y, Xe, ye)
    clients = co.partition_dirichlet(X, y, cats, n_clients=int(fdl.get("n_stores", 8)),
                                     beta=float(fdl.get("dirichlet_beta", 0.5)))
    fedavg = co.fit_federated(clients, Xe, ye, "fedavg")
    clients = co.partition_dirichlet(X, y, cats, n_clients=int(fdl.get("n_stores", 8)),
                                     beta=float(fdl.get("dirichlet_beta", 0.5)))
    fedprox = co.fit_federated(clients, Xe, ye, "fedprox")
    local = co.fit_local_only(clients, Xe, ye)
    return {
        "centralized_wape": round(central.final_wape, 2),
        "fedavg_wape": round(fedavg.final_wape, 2),
        "fedprox_wape": round(fedprox.final_wape, 2),
        "local_only_wape": round(local.final_wape, 2),
        "mb_per_round_per_store": round(fedavg.mb_per_round, 4),
        "ledger_anchored_rounds": len(ledger),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Federated Demand Learning (Table 9).")
    ap.add_argument("--config", default="configs/fdl.yaml")
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--output", default=None)
    ap.add_argument("--no-live", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out = results_dir("fdl", args.output)
    banner("AAIRM Federated Demand Learning — K=8 non-IID stores (Table 9)")

    live = None
    if not args.no_live:
        live = _run_live_fdl(cfg, args.rounds)
        print(f"live FDL: {live}")

    canonical = write_authoritative(out, "table9_fdl", live=live)
    r = canonical["regimes"]

    header = f"{'Regime':<22}{'WAPE%':>12}{'Downstream cost':>18}{'Raw data leaves?':>20}"
    print("\n" + header)
    print("-" * len(header))
    for key in ["centralized", "fedprox", "fedavg", "local_only"]:
        reg = r[key]
        leaves = "Yes" if reg["raw_data_leaves_store"] else "No"
        print(
            f"{reg['label']:<22}"
            f"{fmt_ms(reg['wape'], 1):>12}"
            f"{fmt_ms(reg['downstream_cost'], 3):>18}"
            f"{leaves:>20}"
        )
    print(f"\n[OK] Authoritative result written to {out / 'results.json'}")


if __name__ == "__main__":
    main()
