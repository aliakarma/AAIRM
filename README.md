# AAIRM

> Agentic AI Inventory Replenishment and Management for multi-category retail inventory optimization.

AAIRM is a research-oriented inventory intelligence framework that combines forecasting, replenishment optimization, supplier-aware execution, and governance checks in one agentic workflow. The repository includes reproducible benchmark pipelines, validated multi-category experiments, and publication-ready result artifacts.

---

## 🚀 Key Contributions

- **Agentic inventory optimization:** Coordinated decision flow across perception, conceptualization, and action layers.
- **Multi-category modeling:** Unified simulation over grocery, frozen_food, apparel, cosmetics, and dry_fruits.
- **Cost vs service trade-off learning:** AAIRM improves normalized total cost while maintaining competitive fill performance under realistic constraints.
- **Scalability validation:** Controlled scaling from 100 SKU to 500 SKU settings with consistent evaluation protocol.

---

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

**Cost improvement:** AAIRM improves normalized total cost by **~23.3%** vs Baseline2 and **~13.2%** vs Baseline1.

### 📈 Scalability (500 SKUs, 5 Seeds, 200 Episodes)

Secondary scalability output: `experiments/results/scalability_500sku_5seed/summary.json`

At 500 SKUs, AAIRM still maintains clear cost advantage (0.8292 vs 1.2033 for Baseline2), while service quality degrades in harder high-perishable / volatile segments (especially dry_fruits). This behavior reflects an explicit cost-service trade-off under increased problem scale rather than a pipeline failure, and supports the paper discussion on scaling challenges.

---

## 🧠 Multi-Category Behavior

AAIRM is evaluated on five balanced retail categories:

- **grocery**
- **frozen_food**
- **apparel**
- **cosmetics**
- **dry_fruits**

Key observations:

- **Perishability differences:** apparel has near-zero spoilage, frozen_food remains below dry_fruits spoilage, and dry_fruits consistently shows highest spoilage pressure.
- **Demand heterogeneity:** demand and service behavior differ by category, reflecting category-level dynamics.
- **Policy adaptation:** AAIRM adapts inventory posture by category, reducing aggregate holding burden while controlling overall cost.

---

## ⚙️ How to Run

### 🔹 Environment Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 🔹 Run Main Experiment (100 SKUs)

```powershell
python scripts/run_smoke_multiseed.py `
  --seeds 42,43,44,45,46,47,48,49,50,51 `
  --episodes 200 `
  --n-skus 100 `
  --out-dir experiments/results/main_100sku_10seed
```

### 🔹 Run Scalability Experiment (500 SKUs)

```powershell
python scripts/run_smoke_multiseed.py `
  --seeds 42,43,44,45,46 `
  --episodes 200 `
  --n-skus 500 `
  --out-dir experiments/results/scalability_500sku_5seed
```

---

## 📁 Project Structure

```text
aairm/
configs/
scripts/
experiments/results/
README.md
```

---

## 📌 Notes

- Results are reproducible via fixed seeds.
- Simulations use a multi-category retail setup.
- Repository is prepared for research and publication workflows.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
