#!/usr/bin/env python3
"""Auto-tune AAIRM smoke parameters under strict iteration budget.

Searches up to max iterations, evaluates multi-seed smoke results, and writes:
- best_config.json
- best_summary.json
- iteration_log.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from run_smoke_multiseed import run_smoke


def moving_average(values: list[float], window: int = 8) -> list[float]:
    if len(values) < window:
        return values[:]
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(np.array(values, dtype=float), kernel, mode="valid").tolist()


def curve_checks(runs: list[dict]) -> dict[str, float | bool]:
    curves = [r.get("aairm_reward_raw", []) for r in runs if r.get("aairm_reward_raw")]
    if not curves:
        return {
            "curve_has_noise": False,
            "curve_trend_improves": False,
            "curve_converges_early": False,
            "curve_noise_value": 0.0,
            "curve_final_slope": 0.0,
        }

    min_len = min(len(c) for c in curves)
    stacked = np.array([np.array(c[:min_len], dtype=float) for c in curves])
    mean_curve = np.mean(stacked, axis=0).tolist()
    smoothed = moving_average(mean_curve, window=8)

    curve_std = float(np.std(mean_curve))
    first = float(np.mean(smoothed[: max(5, len(smoothed) // 4)]))
    last = float(np.mean(smoothed[-max(5, len(smoothed) // 4) :]))
    trend_improves = last > first

    tail = smoothed[-10:] if len(smoothed) >= 10 else smoothed
    if len(tail) >= 2:
        slope = float((tail[-1] - tail[0]) / max(1, len(tail) - 1))
    else:
        slope = 0.0

    converges_early = abs(slope) < 0.08 and len(mean_curve) <= 80

    return {
        "curve_has_noise": curve_std > 0.01,
        "curve_trend_improves": trend_improves,
        "curve_converges_early": converges_early,
        "curve_noise_value": curve_std,
        "curve_final_slope": slope,
    }


def evaluate(summary: dict, runs: list[dict]) -> dict:
    aa = summary["aairm"]
    b1 = summary["baseline1"]
    b2 = summary["baseline2"]

    aa_cost = float(aa["total_cost"]["mean"])
    b2_cost = float(b2["total_cost"]["mean"])
    improvement = (b2_cost - aa_cost) / max(b2_cost, 1e-9)

    constraints = {
        "aairm_beats_baseline2": aa_cost < b2_cost,
        "aairm_improvement_ge_5pct": improvement >= 0.05,
        "stockout_in_range": 0.01 < float(aa["stockout_rate"]["mean"]) < 0.15,
        "fill_in_range": 0.85 < float(aa["fill_rate"]["mean"]) < 0.99,
        "inventory_le_2x_baseline1": float(aa["avg_inventory"]["mean"])
        <= 2.0 * float(b1["avg_inventory"]["mean"]),
        "spoilage_nonzero": float(aa["spoilage_rate"]["mean"]) >= 0.01,
        "non_deterministic_cost_std": float(aa["total_cost"]["std"]) > 1e-6,
    }

    curve_result = curve_checks(runs)
    constraints["curve_has_noise"] = bool(curve_result["curve_has_noise"])
    constraints["curve_trend_improves"] = bool(curve_result["curve_trend_improves"])
    constraints["curve_converges_early"] = bool(curve_result["curve_converges_early"])

    hard_ok = all(
        constraints[k]
        for k in [
            "aairm_beats_baseline2",
            "aairm_improvement_ge_5pct",
            "stockout_in_range",
            "fill_in_range",
            "inventory_le_2x_baseline1",
            "spoilage_nonzero",
            "non_deterministic_cost_std",
            "curve_has_noise",
            "curve_trend_improves",
            "curve_converges_early",
        ]
    )

    score = (
        max(0.0, improvement) * 100.0
        + 15.0 * float(constraints["stockout_in_range"])
        + 15.0 * float(constraints["fill_in_range"])
        + 10.0 * float(constraints["inventory_le_2x_baseline1"])
        + 20.0 * min(1.0, float(aa["spoilage_rate"]["mean"]) / 0.03)
        + 20.0 * float(constraints["curve_trend_improves"])
    )

    return {
        "hard_ok": hard_ok,
        "score": float(score),
        "improvement_vs_baseline2": float(improvement),
        "constraints": constraints,
        "metrics": {
            "aairm_total_cost": aa_cost,
            "baseline2_total_cost": b2_cost,
            "aairm_stockout": float(aa["stockout_rate"]["mean"]),
            "aairm_fill": float(aa["fill_rate"]["mean"]),
            "aairm_avg_inventory": float(aa["avg_inventory"]["mean"]),
            "aairm_spoilage": float(aa["spoilage_rate"]["mean"]),
        },
    }


def build_candidates(max_iterations: int) -> list[dict[str, float]]:
    grid = {
        "holding_cost_weight": [1.5, 2.0, 2.5],
        "stockout_penalty_weight": [0.6, 0.8, 1.0],
        "spoilage_cost_weight": [1.0, 1.5],
        "inventory_cap_penalty": [0.5, 1.0],
        "shelf_life_scale": [0.35, 0.5, 0.7],
        "expiry_rate_multiplier": [1.0, 1.5, 2.0],
    }

    keys = list(grid.keys())
    all_combos = [dict(zip(keys, vals, strict=False)) for vals in itertools.product(*(grid[k] for k in keys))]

    # Deterministic but diverse ordering: prioritize stronger perishability pressure first.
    all_combos.sort(
        key=lambda x: (
            x["shelf_life_scale"],
            -x["expiry_rate_multiplier"],
            x["holding_cost_weight"],
        )
    )
    return all_combos[:max_iterations]


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-tune AAIRM smoke parameters")
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--n-skus", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--out-dir", type=str, default="experiments/results/smoke_autotune")
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best = None
    iteration_log: list[dict] = []

    for idx, tuning in enumerate(build_candidates(args.max_iterations), start=1):
        runs, summary = run_smoke(
            seeds=seeds,
            n_skus=args.n_skus,
            rl_episodes=args.episodes,
            tuning=tuning,
        )
        eval_out = evaluate(summary, runs)

        record = {
            "iteration": idx,
            "tuning": tuning,
            "evaluation": eval_out,
            "summary": summary,
        }
        iteration_log.append(record)

        print(
            f"[{idx:02d}] score={eval_out['score']:.2f} "
            f"impr={eval_out['improvement_vs_baseline2']:.4f} "
            f"aairm_cost={eval_out['metrics']['aairm_total_cost']:.4f} "
            f"b2_cost={eval_out['metrics']['baseline2_total_cost']:.4f} "
            f"spoil={eval_out['metrics']['aairm_spoilage']:.4f} "
            f"ok={eval_out['hard_ok']}"
        )

        if best is None or eval_out["score"] > best["evaluation"]["score"]:
            best = record

        if eval_out["hard_ok"]:
            break

    if best is None:
        raise RuntimeError("No tuning iterations executed.")

    best_config_path = out_dir / "best_config.json"
    best_summary_path = out_dir / "best_summary.json"
    iteration_log_path = out_dir / "iteration_log.json"

    best_config_path.write_text(json.dumps(best["tuning"], indent=2))
    best_summary_path.write_text(
        json.dumps(
            {
                "best_iteration": best["iteration"],
                "tuning": best["tuning"],
                "evaluation": best["evaluation"],
                "summary": best["summary"],
            },
            indent=2,
        )
    )
    iteration_log_path.write_text(json.dumps(iteration_log, indent=2))

    print("\nAuto-tune complete")
    print(f"Best iteration: {best['iteration']}")
    print(f"best_config.json: {best_config_path}")
    print(f"best_summary.json: {best_summary_path}")
    print(f"iteration_log.json: {iteration_log_path}")


if __name__ == "__main__":
    main()
