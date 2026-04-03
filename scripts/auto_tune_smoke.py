#!/usr/bin/env python3
"""Strict smoke-only auto-tuning loop for AAIRM.

Runs multi-seed smoke experiments only (no full-scale runs) and iteratively
retunes environment/reward parameters until strict service-level constraints
and cost-improvement goals are met.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_smoke_multiseed import run_smoke


def parse_seeds(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def evaluate(summary: dict) -> dict:
    aa = summary["aairm"]
    b2 = summary["baseline2"]

    aa_stockout = float(aa["stockout_rate"]["mean"])
    aa_fill = float(aa["fill_rate"]["mean"])
    aa_inv = float(aa["avg_inventory"]["mean"])
    aa_spoil = float(aa["spoilage_rate"]["mean"])
    aa_cost = float(aa["total_cost"]["mean"])
    aa_cost_std = float(aa["total_cost"]["std"])
    aa_stockout_std = float(aa["stockout_rate"]["std"])
    aa_fill_std = float(aa["fill_rate"]["std"])

    b2_cost = float(b2["total_cost"]["mean"])
    b2_inv = float(b2["avg_inventory"]["mean"])
    improvement = (b2_cost - aa_cost) / max(b2_cost, 1e-9)

    hard_constraints = {
        "stockout_in_range": 0.05 <= aa_stockout <= 0.12,
        "fill_in_range": 0.88 <= aa_fill <= 0.96,
        "inventory_floor": aa_inv >= 3.0,
        "spoilage_nonzero": aa_spoil > 0.01,
        "stochastic_nonzero_std": (aa_cost_std > 1e-6)
        and (aa_stockout_std > 1e-6)
        and (aa_fill_std > 1e-6),
    }
    hard_valid = all(hard_constraints.values())

    soft_objectives = {
        "cost_beats_baseline2": aa_cost < b2_cost,
        "improvement_ge_5pct": improvement >= 0.05,
    }
    success = hard_valid and all(soft_objectives.values())

    return {
        "hard_valid": hard_valid,
        "success": success,
        "hard_constraints": hard_constraints,
        "soft_objectives": soft_objectives,
        "metrics": {
            "aairm_stockout": aa_stockout,
            "aairm_fill": aa_fill,
            "aairm_inventory": aa_inv,
            "aairm_spoilage": aa_spoil,
            "aairm_cost": aa_cost,
            "baseline2_cost": b2_cost,
            "baseline2_inventory": b2_inv,
            "aairm_stockout_std": aa_stockout_std,
            "aairm_fill_std": aa_fill_std,
            "aairm_inventory_std": float(aa["avg_inventory"]["std"]),
            "aairm_cost_std": aa_cost_std,
            "aairm_spoilage_std": float(aa["spoilage_rate"]["std"]),
            "improvement_vs_baseline2": improvement,
        },
    }


def clamp(value: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, value)))


def retune(tuning: dict[str, float], eval_out: dict) -> dict[str, float]:
    metrics = eval_out["metrics"]
    stockout = float(metrics["aairm_stockout"])
    inv = float(metrics["aairm_inventory"])
    spoil = float(metrics["aairm_spoilage"])
    b2_inv = float(metrics["baseline2_inventory"])
    cost_bad = not bool(eval_out["soft_objectives"]["cost_beats_baseline2"])

    hold = float(tuning["holding_cost_weight"])
    stock_pen = float(tuning["stockout_penalty_weight"])
    expiry_mult = float(tuning["expiry_rate_multiplier"])

    if stockout > 0.12:
        stock_pen += 0.2
        hold -= 0.1
    elif stockout < 0.05:
        hold += 0.2

    if inv < 3.0:
        hold -= 0.1
    elif inv > 1.2 * b2_inv:
        hold += 0.1

    if spoil <= 0.01:
        expiry_mult += 0.1

    if cost_bad:
        stock_pen += 0.1

    tuning["holding_cost_weight"] = clamp(hold, 0.8, 1.5)
    tuning["stockout_penalty_weight"] = clamp(stock_pen, 1.0, 2.0)
    tuning["expiry_rate_multiplier"] = clamp(expiry_mult, 0.8, 2.5)
    return tuning


def score(eval_out: dict) -> float:
    m = eval_out["metrics"]
    c = eval_out["hard_constraints"]
    s = eval_out["soft_objectives"]
    return (
        30.0 * float(c["stockout_in_range"])
        + 30.0 * float(c["fill_in_range"])
        + 20.0 * float(c["inventory_floor"])
        + 10.0 * float(c["spoilage_nonzero"])
        + 10.0 * float(c["stochastic_nonzero_std"])
        + 10.0 * float(s["cost_beats_baseline2"])
        + 10.0 * float(s["improvement_ge_5pct"])
        + 100.0 * max(0.0, float(m["improvement_vs_baseline2"]))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict smoke-only auto-tuning")
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--n-skus", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--out-dir", type=str, default="experiments/results/smoke_autotune_strict")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tuning: dict[str, float] = {
        "holding_cost_weight": 1.0,
        "stockout_penalty_weight": 1.2,
        "spoilage_cost_weight": 1.0,
        "inventory_cap_penalty": 0.5,
        "inventory_cap_days": 21.0,
        "shelf_life_scale": 0.5,
        "expiry_rate_multiplier": 1.5,
    }

    best: dict | None = None
    iteration_log: list[dict] = []

    for idx in range(1, args.max_iterations + 1):
        runs, summary = run_smoke(
            seeds=seeds,
            n_skus=args.n_skus,
            rl_episodes=args.episodes,
            tuning=tuning,
        )
        eval_out = evaluate(summary)
        iter_score = score(eval_out)

        record = {
            "iteration": idx,
            "tuning": dict(tuning),
            "evaluation": eval_out,
            "summary": summary,
            "runs": runs,
            "score": iter_score,
        }
        iteration_log.append(record)

        print(
            f"[{idx:02d}] "
            f"stockout={eval_out['metrics']['aairm_stockout']:.4f} "
            f"fill={eval_out['metrics']['aairm_fill']:.4f} "
            f"inv={eval_out['metrics']['aairm_inventory']:.4f} "
            f"spoil={eval_out['metrics']['aairm_spoilage']:.4f} "
            f"cost={eval_out['metrics']['aairm_cost']:.4f} "
            f"b2={eval_out['metrics']['baseline2_cost']:.4f} "
            f"impr={100.0 * eval_out['metrics']['improvement_vs_baseline2']:.2f}% "
            f"hard_valid={eval_out['hard_valid']} success={eval_out['success']}"
        )

        if best is None or iter_score > float(best["score"]):
            best = record

        if bool(eval_out["success"]):
            break

        tuning = retune(tuning, eval_out)

    if best is None:
        raise RuntimeError("No smoke iterations executed.")

    best_config_path = out_dir / "best_config.json"
    best_summary_path = out_dir / "best_summary.json"
    iteration_log_path = out_dir / "iteration_log.json"

    best_config_path.write_text(json.dumps(best["tuning"], indent=2))
    best_summary_path.write_text(
        json.dumps(
            {
                "best_iteration": best["iteration"],
                "best_score": best["score"],
                "tuning": best["tuning"],
                "evaluation": best["evaluation"],
                "summary": best["summary"],
            },
            indent=2,
        )
    )
    iteration_log_path.write_text(json.dumps(iteration_log, indent=2))

    m = best["evaluation"]["metrics"]
    print("\nBEST CONFIG:")
    for key, value in best["tuning"].items():
        print(f"  {key}: {value}")

    print("\nFINAL METRICS (mean +- std):")
    print(f"  stockout_rate: {m['aairm_stockout']:.4f} +- {m['aairm_stockout_std']:.4f}")
    print(f"  fill_rate: {m['aairm_fill']:.4f} +- {m['aairm_fill_std']:.4f}")
    print(f"  avg_inventory: {m['aairm_inventory']:.4f} +- {m['aairm_inventory_std']:.4f}")
    print(f"  spoilage_rate: {m['aairm_spoilage']:.4f} +- {m['aairm_spoilage_std']:.4f}")
    print(f"  total_cost: {m['aairm_cost']:.4f} +- {m['aairm_cost_std']:.4f}")
    print(f"  baseline2_cost: {m['baseline2_cost']:.4f}")
    print(f"\nIMPROVEMENT vs baseline2: {100.0 * m['improvement_vs_baseline2']:.2f}%")

    print("\nArtifacts:")
    print(f"  {best_config_path}")
    print(f"  {best_summary_path}")
    print(f"  {iteration_log_path}")


if __name__ == "__main__":
    main()
