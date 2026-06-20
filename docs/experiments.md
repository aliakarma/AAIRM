# Experiments

Every table and figure in the paper (`product.tex`) is reproduced from
authoritative fixtures in `experiments/results/canonical/` (the exact paper
numbers). See `experiments/README.md` for the full runner reference.

## Reproduce all paper results

```bash
# All experiments + regenerate every LaTeX table and figure
python experiments/run_all_paper.py

# Skip the live FDL/BTL/baseline runs (canonical export only)
python experiments/run_all_paper.py --quick
```

## Per-experiment runners

| Script | Paper element |
|---|---|
| `run_primary.py` | Table 3 — primary 100-SKU, 10 seeds |
| `run_scalability.py` | Table 7 — scalability 500-SKU, 5 seeds |
| `run_ablation_variants.py` | Table 4 — ablation variants A–D |
| `run_rl_baselines.py` | Table 5 — BL4 (DQN), BL5 (MAPPO) |
| `run_pareto.py` | Figure — Pareto cost–service frontier |
| `run_dryfruits_recal.py` | Table 8 — dry-fruits recalibration |
| `run_fdl.py` | Table 9 — Federated Demand Learning |
| `run_btl.py` | Table 10 — Blockchain Trust Ledger |
| `run_sensitivity.py` | Hyperparameter sensitivity sweep |
| `run_m5_validation.py` | Table 11 — M5 external validation |

## Ablation study (Table 4)

Four variants isolate each component's contribution:

| Variant | Configuration | Total cost |
|---|---|---|
| A (RL-only) | PPO ordering, no governance, no LLM | 0.912 |
| B (Governance-only) | ROP-EOQ ordering + governance + LLM | 0.978 |
| C (No-LLM) | RL + governance, deterministic router | 0.874 |
| D (Full AAIRM) | RL + governance + LLM orchestrator | 0.868 |

```bash
python experiments/run_ablation_variants.py            # all variants
python experiments/run_ablation_variants.py --variant A
```

Decomposition: RL **8.8 pp** + governance **3.8 pp** + LLM **0.6 pp** (not
significant, p = 0.38; TOST-equivalent within ±2 pp) ≈ the 13.2 pp headline.

## RL baselines (Table 5)

```bash
python experiments/run_rl_baselines.py
```

BL5 (MAPPO) nearly matches AAIRM on cost (0.885 vs 0.868) but incurs 7.9%
constraint violations; AAIRM's governance layer guarantees feasibility (0.0%).

## Trust subsystems

```bash
python experiments/run_fdl.py      # Federated Demand Learning (Table 9)
python experiments/run_btl.py      # Blockchain Trust Ledger (Table 10)
```

The BTL mutation replay is a genuine SHA-256 test reproducing 500/500 detection
with zero false positives. FDL runs FedAvg/FedProx over a non-IID 8-store
partition and anchors each round's parameter digest in the ledger.

## Real-world evaluation

```bash
python experiments/run_m5_validation.py            # Table 11 (M5)
make download-data                                  # cache M5/Favorita (Kaggle creds)
python experiments/run_realworld.py --dataset m5    # legacy full-pipeline harness
```

## Configuration

Paper-aligned configs live in `configs/`:

- `primary_100sku.yaml`, `scalability_500sku.yaml`
- `ablation/variant_{a,b,c,d}_*.yaml`
- `fdl.yaml`, `btl.yaml`, `pareto.yaml`

```bash
python experiments/run_primary.py --config configs/primary_100sku.yaml
```

## Regenerate LaTeX tables and figure data

```bash
python experiments/export_paper_tables.py
```
