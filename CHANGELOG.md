# Changelog

All notable changes to AAIRM are documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

## [Unreleased]

## [0.1.0] — 2025-01-01

### Added
- Initial public release accompanying Syed et al. (2025).
- Complete PCA multi-agent framework: 13 agents across Perception (P1–P5),
  Conceptualization (C1–C5), and Action (A1–A3) layers.
- MetaOrchestrator (LangGraph/LangChain) with ablation bypass flags.
- Synthetic retail simulation: 1,200 SKUs, 5 categories, 730-day horizon,
  stochastic demand (Saudi retail holiday calendar), 3–5 suppliers per SKU.
- ROP–EOQ Baseline (Baseline 1) and ML + Static Baseline (Baseline 2).
- M5 Forecasting Competition, Corporación Favorita, and Instacart dataset adapters.
- NaiveForecaster, LSTMForecaster, TFTForecaster (all implementing BaseForecaster).
- PPO reorder policy (stable-baselines3 wrapper) and NumPy ValueNetwork.
- All five evaluation metrics: stockout rate, fill rate, avg inventory,
  total cost (normalised), supplier diversification index.
- Benchmarker with paper-result assertion (±0.5% tolerance).
- Reporter generating Tables 2 & 3 (LaTeX) and Figures 3 & 4 (PNG).
- Full experiment scripts: `run_paper_experiment.py`, `run_ablation.py`,
  `run_realworld.py`.
- Six Jupyter notebooks including step-by-step walkthrough of all 18
  sequence-diagram interactions (Notebook 04).
- Data download, preprocessing, and synthetic generation scripts.
- Unit tests (100% coverage on math_utils and metrics), integration tests,
  and 60-second smoke test.
- CI/CD: GitHub Actions (lint, multi-Python test matrix, release, docs deploy).
- Docker + docker-compose for containerised execution.
- MkDocs Material documentation site with auto-generated API reference.
- Trusted Agent Infrastructure: AgentHealthMonitor, ReputationEngine,
  SHA-256 AuditLedger.

[Unreleased]: https://github.com/[author-handle]/aairm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/[author-handle]/aairm/releases/tag/v0.1.0
