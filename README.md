# AAIRM: Agentic AI Inventory Replenishment and Management

[![Paper](https://img.shields.io/badge/Paper-Frontiers-blue)](https://doi.org/[DOI-PLACEHOLDER])
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/[author-handle]/aairm/actions/workflows/ci.yml/badge.svg)](https://github.com/[author-handle]/aairm/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/[author-handle]/aairm/badge.svg)](https://codecov.io/gh/[author-handle]/aairm)
[![Docs](https://img.shields.io/badge/docs-gh--pages-brightgreen)](https://[author-handle].github.io/aairm)

Companion code for:

> **Syed, T.A., El-Boghdadi, H.M., Naqash, M.T., Alghamdi, T., Alshahrani, A.,
> Lee, I.E., Akarma, A. (2025).** *Agentic Commerce: Economic Implications of
> AI-Driven Forecasting, Inventory Management, and Product Personalization in Retail.*
> Frontiers in [Journal]. https://doi.org/[DOI-PLACEHOLDER]

---

## Abstract

Retail marts with broad product assortments face persistent challenges in maintaining
optimal stock levels, responding to volatile demand, and identifying high-potential
new products across heterogeneous categories. This paper proposes the **AAIRM
framework**: a multi-agent, LangChain-orchestrated system that implements autonomous
inventory replenishment and product discovery through a structured
Perception–Conceptualization–Action (PCA) workflow. Evaluated on a structurally
realistic synthetic simulation comprising 1,200 SKUs across five product categories,
AAIRM achieves a stockout rate of 3.9% (vs. 8.7% for classical ROP–EOQ), reduces
total operational cost by 16%, and raises fill rate to 97.8%.

---

## Framework Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Meta-Orchestrator                         │
│              (LangGraph / LangChain)                        │
└──────┬──────────────────┬──────────────────┬───────────────┘
       │                  │                  │
  PERCEPTION         CONCEPTUAL.          ACTION
  P1 Inventory       C1 Forecasting       A1 Order Execution
  P2 Trend Intel     C2 Reorder Optim     A2 Inventory Adj.
  P3 Discovery       C3 Supplier Rank     A3 Learning Agent
  P4 Context Eng     C4 Negotiation
  P5 Risk Detect     C5 Governance
       │                  │                  │
       └──────────────────┴──────────────────┘
                          │
              External Commerce Ecosystem
         (ERP / WMS / Supplier APIs / Logistics)
```

---

## Key Results (Paper Table 2)

| Policy | Stockout (%) | Fill Rate (%) | Avg. Inv. | Total Cost | Div. Index |
|---|---|---|---|---|---|
| Baseline 1 (ROP–EOQ) | 8.7 | 93.1 | 1.45 | 1.00 | 0.42 |
| Baseline 2 (ML + Static) | 6.2 | 95.4 | 1.32 | 0.93 | 0.47 |
| **AAIRM (proposed)** | **3.9** | **97.8** | **1.19** | **0.84** | **0.61** |

All values reproduced with `seed=42`. Results verified within ±0.5% via `make run-paper-experiment`.

---

## Installation

```bash
git clone https://github.com/[author-handle]/aairm.git
cd aairm

# Minimal install (simulation + baselines, no GPU required)
pip install -e .

# Full dev install
pip install -e ".[dev]"
pre-commit install

# Verify installation (< 60 seconds)
make smoke
```

**Optional extras:**
```bash
pip install -e ".[llm]"         # C4 LLM negotiation (requires OpenAI key)
pip install -e ".[rl]"          # C2 PPO policy (requires PyTorch)
pip install -e ".[forecasting]" # C1 TFT/LSTM (requires pytorch-forecasting)
pip install -e ".[datasets]"    # Kaggle dataset downloads
```

---

## Quick Start

```python
from aairm.utils.config import AAIRMConfig
from aairm.utils.seed import set_global_seed
from aairm.simulation.environment import RetailEnv
from aairm.agents.meta_orchestrator import MetaOrchestrator
from aairm.agents.base import AgentState
from aairm.models.forecasting.naive_forecaster import NaiveForecaster

set_global_seed(42)
config = AAIRMConfig()
env = RetailEnv(config.simulation)
env.reset()

orchestrator = MetaOrchestrator(
    config=config,
    erp_backend=env,
    supplier_backend=env,
    trend_backend=env,
    forecaster=NaiveForecaster(),
)

for day in range(7):
    state = AgentState(day=day)
    state = orchestrator.run_cycle(state)
    metrics = env.step_agentic(
        {sku: t.get("quantity", 0.0) for sku, t in state.approved_orders.items()}
    )
    print(f"Day {day+1}: demand={metrics['total_demand']:.0f}  "
          f"orders={len(state.purchase_orders_issued)}")
```

---

## Reproducing Paper Results

```bash
# 1. Generate synthetic dataset (seed=42, 1,200 SKUs)
make generate-synthetic

# 2. Run full paper experiment (Tables 2 & 3, Figures 3 & 4)
make run-paper-experiment
```

Expected terminal output:
```
──────────────────────────────────────────────────────────────
Table 2 — Overall Performance (paper Section 5.3)
──────────────────────────────────────────────────────────────
Policy                         Stockout%  FillRate%  AvgInv  TotalCost  DivIdx
Baseline 1 (ROP-EOQ)               8.7%     93.1%    1.45      1.00     0.42
Baseline 2 (ML + Static)           6.2%     95.4%    1.32      0.93     0.47
AAIRM (proposed)                   3.9%     97.8%    1.19      0.84     0.61
──────────────────────────────────────────────────────────────
✓ Results saved to: experiments/results/paper_experiment_YYYYMMDD_HHMMSS/
```

---

## Real-World Datasets

AAIRM supports three real-world datasets. Download requires Kaggle API credentials.

```bash
# Add credentials to .env (see .env.example)
make download-data        # M5, Favorita, Instacart (~640 MB total)
make preprocess-data      # feature engineering → data/processed/

python experiments/run_realworld.py --dataset m5
python experiments/run_realworld.py --dataset favorita
```

| Dataset | Source | SKUs used | Description |
|---|---|---|---|
| M5 | Kaggle (Walmart) | 1,200 | Daily retail sales, 5 years |
| Favorita | Kaggle (Ecuador) | 1,200 | Grocery sales + oil shocks |
| Instacart | Kaggle | all products | Trend features + co-purchase matrix |

---

## Ablation Studies

```bash
make run-ablation
```

| Ablation config | What is disabled | Expected change |
|---|---|---|
| `no_rl` | PPO → analytical C2 | stockout ↑ ~1–2pp |
| `no_negotiation` | C4 bypassed | total cost ↑ ~3–5% |
| `no_governance` | C5 bypassed | div\_index ↓ ~0.05 |
| `single_category` | Grocery only | interpretive baseline |

---

## Project Structure

```
aairm/
├── aairm/                  # Main Python package (67 modules)
│   ├── agents/             # 13 PCA agents + MetaOrchestrator
│   ├── models/             # TFT, LSTM, Naive forecasters; PPO policy
│   ├── simulation/         # RetailEnv, DemandGenerator, ERPStub
│   ├── baselines/          # ROPEOQPolicy, MLStaticPolicy
│   ├── evaluation/         # metrics, benchmarker, reporter
│   ├── data/               # adapters (M5, Favorita, Instacart, Synthetic)
│   ├── tools/              # LangChain tool wrappers
│   ├── infrastructure/     # HealthMonitor, ReputationEngine, AuditLedger
│   └── utils/              # config, math_utils, seed, logging
├── configs/                # YAML experiment configurations
├── data/synthetic/         # Committed seed files (no download needed)
├── experiments/            # Reproducible experiment scripts
├── notebooks/              # 6 Jupyter notebooks
├── scripts/                # Download, preprocess, generate, export
├── tests/                  # Unit (100% math/metrics), integration, smoke
├── docs/                   # MkDocs documentation source
└── paper/                  # Companion LaTeX manuscript + bibliography
```

---

## Running Tests

```bash
make test-fast        # unit tests only (~30s)
make smoke            # 10-SKU end-to-end smoke test (<60s)
make test             # full suite with coverage report
make test-integration # integration tests (full environment)
```

Coverage targets: `math_utils.py` and `metrics.py` at 100%; overall ≥ 80%.

---

## Documentation

Full documentation: [https://[author-handle].github.io/aairm](https://[author-handle].github.io/aairm)

Build locally:
```bash
make docs        # strict build
make serve-docs  # live preview at http://localhost:8000
```

---

## Citation

If you use AAIRM in your research, please cite:

```bibtex
@article{syed2025agentic,
  author  = {Syed, Toqeer Ali and El-Boghdadi, Hatem M. and
             Naqash, Muhammad Tayyab and Alghamdi, Turki and
             Alshahrani, Abdulaziz and Lee, It Ee and Akarma, Ali},
  title   = {Agentic Commerce: Economic Implications of {AI}-Driven
             Forecasting, Inventory Management, and Product
             Personalization in Retail},
  journal = {Frontiers in [Journal]},
  year    = {2025},
  doi     = {[DOI-PLACEHOLDER]}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

This work was supported by the Islamic University of Madinah (Saudi Arabia)
and Multimedia University (Malaysia). The authors thank the M5 competition
organizers, Corporación Favorita, and Instacart for making their datasets
publicly available.
