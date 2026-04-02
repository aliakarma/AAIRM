# Experiments

This directory contains scripts to reproduce all results from
Syed et al. (2025), "Agentic Commerce".

## Quick Reference

| Script | Purpose | Runtime |
|---|---|---|
| `run_paper_experiment.py` | Reproduce Tables 2 & 3 | ~15–30 min |
| `run_ablation.py` | Four ablation studies | ~60–90 min |
| `run_realworld.py` | M5 + Favorita evaluation | ~60–120 min |

## Reproducing Paper Results

```bash
# 1. Generate synthetic simulation data
make generate-synthetic

# 2. Run paper experiment (seed=42, 1,200 SKUs, 365-day test)
make run-paper-experiment

# 3. Check results in experiments/results/paper_experiment_*/
```

Expected output (Table 2):
```
Policy                          Stockout%   FillRate%   AvgInv  TotalCost   DivIdx
Baseline 1 (ROP-EOQ)               8.7%      93.1%     1.45      1.00      0.42
Baseline 2 (ML + Static)           6.2%      95.4%     1.32      0.93      0.47
AAIRM (proposed)                   3.9%      97.8%     1.19      0.84      0.61
```

All results are within ±0.5% of reported values when `seed=42`.

## Ablation Studies

```bash
make run-ablation
# Or individual ablations:
python experiments/run_ablation.py --ablation no_rl
python experiments/run_ablation.py --ablation no_negotiation
```

## Real-World Evaluation

```bash
# Requires Kaggle credentials
make download-data
python experiments/run_realworld.py --dataset m5
python experiments/run_realworld.py --dataset favorita
```

## Fast Smoke Test

```bash
python experiments/run_paper_experiment.py --fast
# 10 SKUs, 30-day horizon, completes in < 60 seconds
```
