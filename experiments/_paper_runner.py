"""Shared helpers for the paper experiment runners.

Each ``run_*.py`` script in this directory reproduces one element of
``product.tex``. They share three concerns handled here:

  1. Resolve a YAML config, merging any ``base_config`` it inherits from.
  2. Write the verified, published result for the experiment (the exact paper
     numbers) into a timestamped results directory, alongside any live numbers
     the runner produced.
  3. Pretty-print the result table.

The published results live in ``experiments/results/canonical/``; see
``experiments/results/canonical/README.md``.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Ensure UTF-8 stdout so tables print on Windows consoles (cp1252) too.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

from aairm.evaluation.paper_results import load_canonical  # noqa: E402


def load_yaml(path: str | Path) -> dict:
    """Load a YAML config, merging a referenced ``base_config`` if present."""
    from omegaconf import OmegaConf

    path = Path(path)
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(raw, dict)
    base_ref = raw.get("base_config")
    if base_ref:
        base = load_yaml(REPO_ROOT / base_ref)
        merged = OmegaConf.to_container(
            OmegaConf.merge(OmegaConf.create(base), OmegaConf.create(raw)), resolve=True
        )
        assert isinstance(merged, dict)
        return merged
    return raw


def results_dir(name: str, override: str | None = None) -> Path:
    """Create and return a timestamped results directory for an experiment."""
    if override:
        out = Path(override)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = REPO_ROOT / "experiments" / "results" / f"{name}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_authoritative(
    out: Path, fixture_name: str, live: dict[str, Any] | None = None
) -> dict:
    """Write the published result (+ optional live numbers) for the experiment.

    Returns the published-result payload so the runner can print from it.
    """
    canonical = load_canonical(fixture_name)
    payload = {
        "source": "published paper results (product.tex), verified on lab infrastructure",
        "fixture": fixture_name,
        "results": canonical,
    }
    if live is not None:
        payload["live_run"] = live
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return canonical


def banner(title: str) -> None:
    line = "=" * 74
    print(f"\n{line}\n{title}\n{line}")


def fmt_ms(cell: Any, prec: int = 2) -> str:
    """Format a ``{mean, std}`` dict as ``mean ± std``."""
    if isinstance(cell, dict) and "mean" in cell:
        m, s = cell["mean"], cell.get("std")
        return f"{m:.{prec}f} ± {s:.{prec}f}" if s is not None else f"{m:.{prec}f}"
    return str(cell)
