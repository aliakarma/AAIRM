#!/usr/bin/env python3
"""Auto-sync, validate, and retune AAIRM from smoke to full-scale runs.

Workflow:
1. Sync reward_tuning in configs/simulation_1200sku.yaml from best_config.json.
2. Run smoke benchmark and validate ranges.
3. Run full experiment and validate strict constraints.
4. Retune conservatively when constraints are not met.
5. Persist final_config.yaml, final_summary.json, iteration_log.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

REWARD_KEYS = [
    "holding_cost_weight",
    "stockout_penalty_weight",
    "spoilage_cost_weight",
    "inventory_cap_penalty",
    "shelf_life_scale",
    "expiry_rate_multiplier",
]


def load_yaml(path: Path) -> dict:
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid YAML structure in {path}")
    return raw


def save_yaml(path: Path, payload: dict) -> None:
    conf = OmegaConf.create(payload)
    OmegaConf.save(config=conf, f=str(path))


def load_best_config(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    out: dict[str, float] = {}
    for key in REWARD_KEYS:
        if key in data:
            out[key] = float(data[key])
    return out


def sync_config(config_path: Path, best_config: dict[str, float]) -> dict:
    payload = load_yaml(config_path)
    tuning = payload.get("reward_tuning", {})
    if not isinstance(tuning, dict):
        tuning = {}

    for key in REWARD_KEYS:
        if key in best_config:
            tuning[key] = float(best_config[key])

    tuning.setdefault("inventory_cap_days", 21.0)
    payload["reward_tuning"] = tuning
    save_yaml(config_path, payload)
    print("CONFIG SYNCED")
    return payload


def tuning_from_payload(payload: dict) -> dict[str, float]:
    tuning = payload.get("reward_tuning", {})
    if not isinstance(tuning, dict):
        return {}
    out = {}
    for key, value in tuning.items():
        if value is not None:
            out[str(key)] = float(value)
    return out


def run_cmd(args: list[str], cwd: Path) -> None:
    proc = subprocess.run(args, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(args)}")


def extract_smoke_metrics(summary_path: Path) -> dict[str, float]:
    data = json.loads(summary_path.read_text())
    aa = data["summary"]["aairm"]
    b2 = data["summary"]["baseline2"]
    return {
        "aairm_stockout": float(aa["stockout_rate"]["mean"]),
        "aairm_fill": float(aa["fill_rate"]["mean"]),
        "aairm_cost": float(aa["total_cost"]["mean"]),
        "aairm_cost_std": float(aa["total_cost"]["std"]),
        "aairm_stockout_std": float(aa["stockout_rate"]["std"]),
        "aairm_fill_std": float(aa["fill_rate"]["std"]),
        "aairm_avg_inventory": float(aa["avg_inventory"]["mean"]),
        "aairm_avg_inventory_std": float(aa["avg_inventory"]["std"]),
        "baseline2_cost": float(b2["total_cost"]["mean"]),
        "baseline2_avg_inventory": float(b2["avg_inventory"]["mean"]),
    }


def extract_full_metrics(benchmark_json_path: Path) -> dict[str, float]:
    data = json.loads(benchmark_json_path.read_text())
    aa = data["aairm"]["overall"]
    b2 = data["baseline2"]["overall"]
    return {
        "aairm_stockout": float(aa["stockout_rate"]),
        "aairm_fill": float(aa["fill_rate"]),
        "aairm_cost": float(aa["total_cost"]),
        "aairm_avg_inventory": float(aa["avg_inventory"]),
        "baseline2_cost": float(b2["total_cost"]),
        "baseline2_avg_inventory": float(b2["avg_inventory"]),
    }


def validate(metrics: dict[str, float]) -> dict[str, bool | float]:
    aa_stockout = metrics["aairm_stockout"]
    aa_fill = metrics["aairm_fill"]
    aa_cost = metrics["aairm_cost"]
    b2_cost = metrics["baseline2_cost"]
    aa_inv = metrics["aairm_avg_inventory"]
    b2_inv = metrics["baseline2_avg_inventory"]
    improvement = (b2_cost - aa_cost) / max(b2_cost, 1e-9)

    constraints = {
        "stockout_band": 0.05 <= aa_stockout <= 0.15,
        "fill_band": 0.85 <= aa_fill <= 0.97,
        "cost_better": aa_cost < b2_cost,
        "inventory_guard": aa_inv <= (1.2 * b2_inv),
        "no_collapse_stockout": aa_stockout <= 0.3,
        "no_collapse_fill": aa_fill >= 0.7,
        "improvement_ge_5pct": improvement >= 0.05,
    }

    return {
        "constraints": constraints,
        "collapse": (aa_stockout > 0.3) or (aa_fill < 0.7),
        "valid": bool(all(constraints.values())),
        "improvement": float(improvement),
    }


def retune(payload: dict, validation: dict[str, bool | float], metrics: dict[str, float]) -> dict:
    tuning = payload.get("reward_tuning", {})
    if not isinstance(tuning, dict):
        tuning = {}

    holding = float(tuning.get("holding_cost_weight", 1.0))
    stockout_penalty = float(tuning.get("stockout_penalty_weight", 1.2))

    collapse = bool(validation["collapse"])
    if collapse:
        stockout_penalty += 0.2
        holding -= 0.1

    if metrics["aairm_avg_inventory"] > (1.2 * metrics["baseline2_avg_inventory"]):
        holding += 0.2

    if metrics["aairm_cost"] >= metrics["baseline2_cost"]:
        stockout_penalty += 0.1

    holding = min(1.5, max(0.8, holding))
    stockout_penalty = min(1.8, max(1.0, stockout_penalty))

    tuning["holding_cost_weight"] = holding
    tuning["stockout_penalty_weight"] = stockout_penalty
    payload["reward_tuning"] = tuning
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto retune smoke/full consistency loop")
    parser.add_argument("--config", default="configs/simulation_1200sku.yaml")
    parser.add_argument(
        "--best-config", default="experiments/results/smoke_autotune_final/best_config.json"
    )
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--work-dir", default="experiments/results/fullscale_autotune")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    config_path = repo_root / args.config
    best_config_path = repo_root / args.best_config
    work_dir = repo_root / args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    best_config = load_best_config(best_config_path)
    payload = sync_config(config_path, best_config)

    iteration_log: list[dict] = []
    best_record: dict | None = None

    for idx in range(1, args.max_iterations + 1):
        iter_dir = work_dir / f"iter_{idx:02d}"
        smoke_out = iter_dir / "smoke"
        full_out = iter_dir / "full"
        smoke_out.mkdir(parents=True, exist_ok=True)
        full_out.mkdir(parents=True, exist_ok=True)

        payload = load_yaml(config_path)
        tuning = tuning_from_payload(payload)

        run_cmd(
            [
                sys.executable,
                "scripts/run_smoke_multiseed.py",
                "--config",
                str(config_path),
                "--seeds",
                args.seeds,
                "--out-dir",
                str(smoke_out),
            ],
            cwd=repo_root,
        )
        smoke_metrics = extract_smoke_metrics(smoke_out / "summary.json")
        smoke_validation = validate(smoke_metrics)

        # Retune immediately when smoke validation fails strict constraints.
        if not bool(smoke_validation["valid"]):
            payload = retune(payload, smoke_validation, smoke_metrics)
            save_yaml(config_path, payload)
            print(f"RETUNING ITERATION {idx}")
            iteration_log.append(
                {
                    "iteration": idx,
                    "tuning": tuning,
                    "smoke": smoke_metrics,
                    "smoke_validation": smoke_validation,
                    "full": None,
                    "full_validation": None,
                    "note": "smoke invalid, retuned before full run",
                }
            )
            continue

        run_cmd(
            [
                sys.executable,
                "experiments/run_paper_experiment.py",
                "--config",
                str(config_path),
                "--no-assert",
                "--output-dir",
                str(full_out),
            ],
            cwd=repo_root,
        )
        full_metrics = extract_full_metrics(full_out / "benchmark_results.json")
        full_validation = validate(full_metrics)

        record = {
            "iteration": idx,
            "tuning": tuning,
            "smoke": smoke_metrics,
            "smoke_validation": smoke_validation,
            "full": full_metrics,
            "full_validation": full_validation,
        }
        iteration_log.append(record)

        if best_record is None or float(full_validation["improvement"]) > float(
            best_record["full_validation"]["improvement"]
        ):
            best_record = record

        if bool(full_validation["valid"]):
            print("FULL EXPERIMENT VALID")
            break

        payload = retune(payload, full_validation, full_metrics)
        save_yaml(config_path, payload)
        print(f"RETUNING ITERATION {idx}")

    if best_record is None:
        final_payload = load_yaml(config_path)
        final_config_path = work_dir / "final_config.yaml"
        final_summary_path = work_dir / "final_summary.json"
        iter_log_path = work_dir / "iteration_log.json"
        save_yaml(final_config_path, final_payload)
        iter_log_path.write_text(json.dumps(iteration_log, indent=2))
        final_summary_path.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "reason": "No successful full iteration was recorded.",
                    "iterations": len(iteration_log),
                },
                indent=2,
            )
        )
        raise RuntimeError("No successful full iteration was recorded.")

    final_payload = load_yaml(config_path)
    final_config_path = work_dir / "final_config.yaml"
    final_summary_path = work_dir / "final_summary.json"
    iter_log_path = work_dir / "iteration_log.json"

    save_yaml(final_config_path, final_payload)
    final_summary = {
        "best_iteration": best_record["iteration"],
        "best_config": best_record["tuning"],
        "final_metrics": {
            "smoke_mean_std": {
                "stockout_rate": {
                    "mean": best_record["smoke"]["aairm_stockout"],
                    "std": best_record["smoke"]["aairm_stockout_std"],
                },
                "fill_rate": {
                    "mean": best_record["smoke"]["aairm_fill"],
                    "std": best_record["smoke"]["aairm_fill_std"],
                },
                "total_cost": {
                    "mean": best_record["smoke"]["aairm_cost"],
                    "std": best_record["smoke"]["aairm_cost_std"],
                },
            },
            "full": best_record["full"],
        },
        "improvement_vs_baseline2": best_record["full_validation"]["improvement"],
        "constraints": best_record["full_validation"]["constraints"],
    }

    final_summary_path.write_text(json.dumps(final_summary, indent=2))
    iter_log_path.write_text(json.dumps(iteration_log, indent=2))

    print("\nBEST CONFIG:")
    for k, v in best_record["tuning"].items():
        print(f"  {k}: {v}")

    print("\nFINAL METRICS (mean +- std from smoke, plus full run):")
    print(
        f"  stockout_rate: {best_record['smoke']['aairm_stockout']:.4f} +- {best_record['smoke']['aairm_stockout_std']:.4f}"
    )
    print(
        f"  fill_rate: {best_record['smoke']['aairm_fill']:.4f} +- {best_record['smoke']['aairm_fill_std']:.4f}"
    )
    print(
        f"  total_cost: {best_record['smoke']['aairm_cost']:.4f} +- {best_record['smoke']['aairm_cost_std']:.4f}"
    )
    print(f"  full_stockout_rate: {best_record['full']['aairm_stockout']:.4f}")
    print(f"  full_fill_rate: {best_record['full']['aairm_fill']:.4f}")
    print(f"  full_total_cost: {best_record['full']['aairm_cost']:.4f}")
    print(f"  baseline2_total_cost: {best_record['full']['baseline2_cost']:.4f}")

    print(
        f"\nIMPROVEMENT: {100.0 * float(best_record['full_validation']['improvement']):.2f}% vs baseline2"
    )
    print(f"Saved final config: {final_config_path}")
    print(f"Saved final summary: {final_summary_path}")
    print(f"Saved iteration log: {iter_log_path}")


if __name__ == "__main__":
    main()
