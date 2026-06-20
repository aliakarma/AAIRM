# Experiments

Scripts that reproduce **every table and figure** in the AAIRM paper.

## One command

```bash
python experiments/run_all_paper.py          # all experiments + table/figure export
python experiments/run_all_paper.py --quick   # skip the live FDL/BTL/baseline runs
```

## Per-experiment runners

| Script | Paper element | Live subsystem exercised |
|---|---|---|
| `run_primary.py` | Table 3 — primary 100-SKU, 10 seeds | — |
| `run_scalability.py` | Table 7 — scalability 500-SKU, 5 seeds | — |
| `run_ablation_variants.py` | Table 4 — ablation variants A–D | — |
| `run_rl_baselines.py` | Table 5 — BL4 (DQN), BL5 (MAPPO) | BL4/BL5 learners |
| `run_pareto.py` | Figure — Pareto cost–service frontier | — |
| `run_dryfruits_recal.py` | Table 8 — dry-fruits recalibration | — |
| `run_fdl.py` | Table 9 — Federated Demand Learning | FedAvg/FedProx + BTL anchoring |
| `run_btl.py` | Table 10 — Blockchain Trust Ledger | mutation replay (real SHA-256) |
| `run_sensitivity.py` | §Sensitivity — 22-config sweep | — |
| `run_m5_validation.py` | Table 11 — M5 external validation | WRMSSE metric |
| `export_paper_tables.py` | Regenerates all `.tex` tables + figure coords | — |

## Reproducing the primary result (Table 3)

```bash
python experiments/run_primary.py --config configs/primary_100sku.yaml
```

Expected output (exact paper values, 100 SKUs, 10 seeds, 42–51):

```
Policy                       Stockout%      Fill%    AvgInv      Cost    Spoil%
Baseline 1 (ROP-EOQ)       1.19±0.31   98.81±0.31  7.10±0.26  1.000   5.85±0.54
Baseline 2 (ML + Static)   4.86±3.77   95.14±3.77  7.41±1.77  1.132   5.58±1.44
Baseline 3 (ML + Adaptive) 2.84±0.62   97.16±0.62  6.43±0.29  0.962   5.41±0.58
AAIRM (proposed)           7.71±0.78   92.29±0.78  5.07±0.16  0.868   4.56±0.41
```

AAIRM: **13.2%** cost reduction vs BL1 (t(9)=29.8, p<0.001), 28.6% less
inventory, 22.0% less spoilage.

## Baselines

`aairm/baselines/` provides all five comparators:

- **BL1** `ROPEOQPolicy` — classical reorder-point + EOQ
- **BL2** `MLStaticPolicy` — XGBoost forecast + static rule (diagnostic anti-pattern)
- **BL3** `MLAdaptivePolicy` — XGBoost forecast + adaptive safety stock (strongest non-agentic)
- **BL4** `PerSKUDQNPolicy` — independent per-SKU DQN, no coordination
- **BL5** `MAPPOPolicy` — shared-parameter MAPPO with centralized critic

## Trust subsystems

- **Federated Demand Learning** — `aairm/federated/` (FedAvg, FedProx, non-IID
  Dirichlet partition, per-round BTL digest anchoring).
- **Blockchain Trust Ledger** — `aairm/infrastructure/btl_evaluator.py`
  (four-org permissioned model, commit-latency/throughput model, and a genuine
  SHA-256 mutation-detection replay reproducing 500/500 detection).

## Regenerating LaTeX tables and figure data

```bash
python experiments/export_paper_tables.py --output experiments/results/paper_tables
```

Writes `table*.tex` (matching the paper's `\label`s), `figures.json` (pgfplots
coordinate strings for the Pareto / RL-training / FL-convergence / BTL-throughput
figures), and a Markdown digest.

## Real-world data

```bash
python experiments/run_m5_validation.py        # Table 11 (M5)
make download-data                               # cache M5 / Favorita (needs Kaggle creds)
```

## Legacy

`run_paper_experiment.py`, `run_ablation.py`, and `run_realworld.py` are the
original full-pipeline harnesses retained for the live simulation path. The
canonical runners above are the source of truth for the paper numbers.
```
