# Experiments

## Reproduce Paper Results

```bash
# Full paper experiment (Tables 2 & 3, Figures 3 & 4)
make run-paper-experiment

# Fast smoke version (10 SKUs, 30 days, ~30 seconds)
make run-paper-experiment-fast
```

## Ablation Studies

```bash
make run-ablation
```

| Ablation | Change | Expected delta vs. AAIRM |
|---|---|---|
| `no_rl` | C2 uses analytical optimisation | stockout_rate ↑ ~1–2pp |
| `no_negotiation` | C4 bypassed | total_cost ↑ ~3–5% |
| `no_governance` | C5 bypassed | div_index ↓ ~0.05–0.10 |
| `single_category` | Grocery only | Context for cross-category value |

## Real-World Evaluation

```bash
make download-data      # requires Kaggle credentials
make preprocess-data
python experiments/run_realworld.py --dataset m5
python experiments/run_realworld.py --dataset favorita
```

## Configuration

All experiments accept a `--config` argument pointing to any YAML in `configs/`:

```bash
python experiments/run_paper_experiment.py --config configs/realworld_m5.yaml
```
