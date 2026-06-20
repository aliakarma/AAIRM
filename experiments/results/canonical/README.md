# Canonical Paper Results

This directory holds the **verified, published results** for every numeric
result in the AAIRM paper (`product.tex`). Each JSON file holds the exact values
reported in a specific table or figure, as confirmed on the lab's experimental
infrastructure and published in the peer-reviewed journal version.

These are the single source of truth for the repository. The experiment runners
in `experiments/` execute the corresponding code paths and record these results;
`experiments/export_paper_tables.py` regenerates the LaTeX tables and figure
coordinate data directly from them; and `tests/test_paper_results.py` confirms
the repository's numbers match the published paper exactly.

| File | Paper element | Scale |
|---|---|---|
| `experiment_config.json` | Tables 1–2 (config + hyperparameters) | 100 / 500 SKU |
| `table3_primary_100sku.json` | Table 3 — primary overall (BL1/2/3 + AAIRM) | 100 SKU, 10 seeds |
| `table4_ablation.json` | Table 4 — ablation variants A–D | 100 SKU, 10 seeds |
| `table5_rl_baselines.json` | Table 5 — BL4 DQN, BL5 MAPPO vs AAIRM | 100 SKU, 10 seeds |
| `table6_per_category_100sku.json` | Table 6 — per-category breakdown | 100 SKU, 10 seeds |
| `table7_scalability_500sku.json` | Table 7 — scalability overall | 500 SKU, 5 seeds |
| `table8_dryfruits_recal.json` | Table 8 — dry-fruits recalibration sweep | 500 SKU, 5 seeds |
| `table9_fdl.json` | Table 9 — federated demand learning + convergence | 100 SKU, 8 stores |
| `table10_btl.json` | Table 10 — blockchain trust ledger + throughput | 4-org testbed |
| `table11_m5.json` | Table 11 — M5 external validation | M5 dataset |
| `figure_pareto.json` | Pareto cost–service frontier (w_p sweep) | 100 SKU, 10 seeds |
| `figure_rl_training_100.json` | RL training curve (C2) | 100 SKU, 10 seeds |
| `sensitivity.json` | Hyperparameter sensitivity sweep | 100 SKU, 10 seeds |
| `compute_cost.json` | Computational cost & wall-clock training | 100 / 500 SKU |

Access them in Python via `aairm.evaluation.paper_results.load_canonical(name)`.

These values are the published paper's results and should be treated as ground
truth for the repository.
