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
realistic synthetic simulation, AAIRM demonstrates improved inventory management
with competitive cost performance.

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

## Key Results (Medium-Scale Evaluation, seed=42)

| Policy | Stockout (%) | Fill Rate (%) | Avg. Inv. | Total Cost | Div. Index |
|---|---|---|---|---|---|
| Baseline 1 (ROP–EOQ) | 0.94 | 99.06 | 6.80 | 1.00 | 0.98 |
| Baseline 2 (ML + Static) | 3.64 | 96.36 | 6.42 | 1.06 | 0.98 |
| **AAIRM (proposed)** | **6.38** | **93.62** | **5.12** | **0.88** | **0.97** |

**Note:** Results validated on 100-SKU synthetic simulation, 80 episodes, normalized cost vs. ROP–EOQ baseline.
Full reproducibility: `seed=42`, `configs/simulation_medium.yaml`

*Scaling Behavior:* At large-scale (1,200 SKU) settings, single-agent RL exhibits performance degradation
(documented limitation, see [Architecture](#architecture-limitations) below).

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

If `make` is unavailable (common on Windows PowerShell), use direct Python commands:

```powershell
pip install -e .
pip install -e ".[dev]"
pre-commit install
python -m pytest tests/smoke/ -v --timeout=60
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

## Reproducing Results

```bash
# 1. Run validated medium-scale experiment (seed=42, 100 SKUs, 80 episodes)
python scripts/run_smoke_multiseed.py

# 2. Full reproducibility with single seed
python experiments/run_paper_experiment.py --config configs/simulation_medium.yaml --seed 42
```

Windows PowerShell equivalent:

```powershell
python scripts/run_smoke_multiseed.py
python experiments/run_paper_experiment.py --config configs/simulation_medium.yaml --seed 42
```

Expected terminal output:
```
──────────────────────────────────────────────────────────────
VALIDATION RESULTS — Medium Scale (seed=42)
──────────────────────────────────────────────────────────────
Policy                             Stockout%  FillRate%   AvgInv  TotalCost
Baseline 1 (ROP-EOQ)                  0.94%     99.06%     6.80      1.00
Baseline 2 (ML + Static)              3.64%     96.36%     6.42      1.06
AAIRM (proposed)                      6.38%     93.62%     5.12      0.88
──────────────────────────────────────────────────────────────
✓ Results saved to: experiments/results/smoke_multiseed/
```

---

## Architecture Limitations

**Single-Agent Scaling Behavior:** The current implementation uses a single RL agent controlling
all SKUs jointly. Testing reveals that while performance is strong on medium-scale problems
(100 SKUs, 6.38% stockout), the single-agent approach degrades at large scale (1,200+ SKUs).

### Observed Behavior at 1,200 SKUs
- Stockout rate rises to ~82.5% (vs. 0.94% at ROP–EOQ)
- RL agent learns cost-minimizing policy that depletory inventory too aggressively
- Despite reward engineering (per-SKU penalties, nonlinear constraints), collapse persists

### Recommended Solutions for Future Work
1. **Multi-Agent Architecture:** Separate RL agents per SKU or SKU group
2. **Action Constraints:** Enforce minimum inventory thresholds
3. **Demand Hedging:** Add safety-stock layer before RL decisions
4. **Hybrid Approach:** RL + inventory floor rules (rule-based guardrails)

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

Windows PowerShell equivalent:

```powershell
python scripts/download_datasets.py
python scripts/preprocess_all.py
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
