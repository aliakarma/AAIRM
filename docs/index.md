# AAIRM Documentation

**Agentic AI Inventory Replenishment and Management**

Companion code for:

> Syed, T.A., El-Boghdadi, H.M., Naqash, M.T., Alghamdi, T., Alshahrani, A.,
> Lee, I.E., Akarma, A. (2025). *Agentic Commerce: Economic Implications of
> AI-Driven Forecasting, Inventory Management, and Product Personalization in
> Retail*. Frontiers in [Journal]. https://doi.org/[DOI-PLACEHOLDER]

## What is AAIRM?

AAIRM is a multi-agent, LangChain-orchestrated framework for **autonomous
inventory replenishment and product discovery** in multi-category retail
environments. It organises 13 specialised agents across three functional layers:

| Layer | Agents | Function |
|---|---|---|
| **Perception** | P1–P5 | Inventory monitoring, trend signals, context assembly |
| **Conceptualization** | C1–C5 | Forecasting, reorder optimisation, supplier ranking, negotiation, governance |
| **Action** | A1–A3 | Order execution, inventory adjustment, learning |

## Key Results (Table 2, paper)

| Policy | Stockout | Fill Rate | Total Cost | Div. Index |
|---|---|---|---|---|
| Baseline 1 (ROP–EOQ) | 8.7% | 93.1% | 1.00 | 0.42 |
| Baseline 2 (ML + Static) | 6.2% | 95.4% | 0.93 | 0.47 |
| **AAIRM (proposed)** | **3.9%** | **97.8%** | **0.84** | **0.61** |

## Quick Navigation

- [Installation](installation.md) — get running in 5 minutes
- [Quick Start](quickstart.md) — run your first cycle
- [Architecture](architecture.md) — understand the PCA design
- [Datasets](datasets.md) — M5, Favorita, Instacart setup
- [Experiments](experiments.md) — reproduce all paper results
