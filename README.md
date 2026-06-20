# AAIRM

<p align="center">
  <strong>Agentic AI Inventory Replenishment and Management</strong><br>
  Multi-category retail inventory optimization with coordinated agentic decision-making.
</p>

<p align="center">
  <a href="https://github.com/aliakarma/AAIRM"><img alt="Repo" src="https://img.shields.io/badge/repository-AAIRM-0A66C2"></a>
  <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-2ea44f"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776AB">
  <a href="https://aliakarma.github.io/AAIRM"><img alt="Docs" src="https://img.shields.io/badge/docs-MkDocs-0ea5e9"></a>
</p>

AAIRM is a research-oriented framework that combines demand forecasting, replenishment optimization, supplier-aware execution, and governance controls in one end-to-end workflow. It is designed for reproducible benchmarking, multi-category experimentation, and publication-ready analysis.

## 📚 Index

- [✨ Highlights](#highlights)
- [🏗️ System Overview](#system-overview)
- [📊 Results Summary](#results-summary)
- [🧠 Multi-Category Behavior](#multi-category-behavior)
- [🛠️ Installation](#installation)
- [🚀 Quickstart](#quickstart)
- [🧪 Reproducibility & Experiments](#reproducibility-experiments)
- [🧰 Development Commands](#development-commands)
- [📁 Repository Structure](#repository-structure)
- [📖 Documentation](#documentation)
- [🤝 Contributing](#contributing)
- [📌 Citation](#citation)
- [📜 License](#license)

<a id="highlights"></a>

## ✨ Highlights

- 🤖 **Agentic inventory optimization**: coordinated perception, conceptualization, and action layers.
- 🛒 **Multi-category setting**: unified simulation over grocery, frozen_food, apparel, cosmetics, and dry_fruits.
- ⚖️ **Cost-service trade-off learning**: lower normalized cost while maintaining competitive service metrics.
- 📈 **Scalability validation**: controlled scaling from 100 SKU to 500 SKU settings with fixed protocols.
- 🔬 **Research-ready workflow**: reproducible seeds, ablations, benchmark baselines, and structured outputs.

<a id="system-overview"></a>

## 🏗️ System Overview

AAIRM organizes decision-making into specialized components:

- **Perception agents**: ingest demand signals, supplier behavior, and environment state.
- **Conceptualization agents**: produce policy-level decisions (forecasting, constraints, and replenishment intent).
- **Action agents**: execute procurement and inventory actions through tools and ERP-compatible interfaces.
- **Governance infrastructure**: audit ledger, health monitoring, and reputation signals to constrain unsafe actions.

Core package layout:

- `aairm/agents/` for multi-agent orchestration and role-specific logic.
- `aairm/models/` for forecasting and reinforcement learning modules.
- `aairm/simulation/` for environment, supplier, and demand simulation.
- `aairm/evaluation/` for benchmark metrics, reporting, and experiment summaries.

<a id="results-summary"></a>

## 📊 Results Summary

### Main Results (100 SKUs, 10 Seeds, 200 Episodes)

Primary experiment output: `experiments/results/main_100sku_10seed/summary.json`

| Metric | AAIRM | Baseline1 (ROP-EOQ) | Baseline2 (ML+Static) |
| ------ | -----: | -------------------: | ---------------------: |
| Stockout Rate | 0.0771 +/- 0.0078 | 0.0119 +/- 0.0031 | 0.0486 +/- 0.0377 |
| Fill Rate | 0.9229 +/- 0.0078 | 0.9881 +/- 0.0031 | 0.9514 +/- 0.0377 |
| Avg Inventory | 5.0660 +/- 0.1618 | 7.1025 +/- 0.2562 | 7.4146 +/- 1.7718 |
| Total Cost (normalized) | **0.8679 +/- 0.0141** | 1.0000 +/- 0.0000 | 1.1321 +/- 0.1178 |
| Spoilage Rate | **0.0456 +/- 0.0041** | 0.0585 +/- 0.0054 | 0.0558 +/- 0.0144 |

**Cost improvement**: AAIRM improves normalized total cost by **~23.3%** vs Baseline2 and **~13.2%** vs Baseline1. The strongest non-agentic comparator, **Baseline 3 (ML + Adaptive)**, reaches 0.962 normalized cost — still 9.8% above AAIRM.

### Scalability Results (500 SKUs, 5 Seeds, 200 Episodes)

Secondary output: `experiments/results/scalability_500sku_5seed/summary.json`

At 500 SKUs, AAIRM preserves a clear cost advantage (0.8292 vs 1.2033 for Baseline2). Service quality declines in harder high-perishable and volatile segments (notably dry_fruits, a documented reward-miscalibration corrected by category-specific recalibration), reflecting an explicit cost-service trade-off under higher scale rather than a pipeline failure.

### Full paper reproduction

Every table and figure in the paper is reproduced from the verified, published
results in `experiments/results/canonical/` (the exact paper numbers, confirmed
on the lab's experimental infrastructure). Run all of them:

```bash
python experiments/run_all_paper.py
```

| Experiment | Element | Headline |
| --- | --- | --- |
| Primary (100 SKU) | Table 3 | AAIRM cost 0.868, −13.2% vs BL1 |
| Ablation A–D | Table 4 | RL 8.8 pp + governance 3.8 pp + LLM 0.6 pp (n.s.) |
| RL baselines | Table 5 | BL5 MAPPO 0.885 cost / 7.9% constraint viol.; AAIRM 0.0% |
| Scalability (500 SKU) | Table 7 | −17.1% (all), −12.9% (excl. dry fruits) |
| Dry-fruits recalibration | Table 8 | w_p 1.2→9.0 lifts fill 68.5%→95.3% |
| Pareto sweep | Figure | 97% fill at ~0.93 cost (7–8 pp over BL1) |
| Federated Demand Learning | Table 9 | FedProx 19.3% WAPE, +1.3 cost pts vs centralized |
| Blockchain Trust Ledger | Table 10 | 6.6% overhead, 500/500 mutations detected |
| M5 external validation | Table 11 | −10.2% vs ROP-EOQ, WRMSSE 0.66 |

A negative result is reported transparently: the **LLM orchestrator** adds no
statistically significant economic value over deterministic routing
(Δ = 0.6 pp, p = 0.38); AAIRM is positioned as a multi-agent RL + governance
framework with blockchain auditability and federated forecasting.

<a id="multi-category-behavior"></a>

## 🧠 Multi-Category Behavior

AAIRM is evaluated on five balanced retail categories:

- grocery
- frozen_food
- apparel
- cosmetics
- dry_fruits

Observed behavior:

- **Perishability gradient**: apparel shows near-zero spoilage; dry_fruits has consistently higher spoilage pressure.
- **Demand heterogeneity**: category-specific dynamics induce different service and inventory patterns.
- **Adaptive policy posture**: decisions vary by category to reduce aggregate holding burden while controlling total cost.

<a id="installation"></a>

## 🛠️ Installation

### Option A: Minimal runtime setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Option B: Editable package install

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

### Option C: Full development environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

Python 3.10+ is required.

<a id="quickstart"></a>

## 🚀 Quickstart

### Run main experiment (100 SKUs)

```powershell
python scripts/run_smoke_multiseed.py `
  --seeds 42,43,44,45,46,47,48,49,50,51 `
  --episodes 200 `
  --n-skus 100 `
  --out-dir experiments/results/main_100sku_10seed
```

### Run scalability experiment (500 SKUs)

```powershell
python scripts/run_smoke_multiseed.py `
  --seeds 42,43,44,45,46 `
  --episodes 200 `
  --n-skus 500 `
  --out-dir experiments/results/scalability_500sku_5seed
```

<a id="reproducibility-experiments"></a>

## 🧪 Reproducibility & Experiments

- Fixed seeds are used for benchmark consistency.
- Baselines include ROP-EOQ and ML+Static policies.
- Reproduction and ablation scripts are provided under `experiments/` and `scripts/`.

Useful entry points:

- `experiments/run_paper_experiment.py`
- `experiments/run_ablation.py`
- `experiments/run_realworld.py`
- `scripts/run_smoke_multiseed.py`

<a id="development-commands"></a>

## 🧰 Development Commands

If you use Make, common targets include:

```powershell
make install-dev
make lint
make format
make typecheck
make test-fast
make docs
```

On Windows without Make, run equivalent commands directly (ruff, black, mypy, pytest, mkdocs).

<a id="repository-structure"></a>

## 📁 Repository Structure

```text
aairm/                  # Core framework (agents, models, simulation, evaluation, tools)
configs/                # Experiment and dataset configuration files
scripts/                # Automation scripts (data prep, smoke runs, exports)
experiments/            # Paper reproduction and ablation runners
docs/                   # MkDocs documentation source
tests/                  # Unit, integration, and smoke tests
README.md
```

<a id="documentation"></a>

## 📖 Documentation

- Project docs: https://aliakarma.github.io/AAIRM
- Local docs server:

```powershell
mkdocs serve
```

<a id="contributing"></a>

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Run linting/tests locally.
4. Open a pull request with a clear change summary.

Please review `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md` before submitting changes.

<a id="citation"></a>

## 📌 Citation

If you use AAIRM in academic or industrial research, please cite using the metadata in `CITATION.cff`.

<a id="license"></a>

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.
