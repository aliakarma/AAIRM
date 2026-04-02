# Quick Start

## 5-Line Example

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
    print(f"Day {day}: demand={metrics['total_demand']:.0f}  "
          f"stockout={metrics['stockout_units']:.0f}")
```

## Reproduce Paper Results

```bash
make generate-synthetic
make run-paper-experiment
```

Expected output (Table 2):
```
Policy                         Stockout%  FillRate%  TotalCost  DivIdx
Baseline 1 (ROP-EOQ)              8.7%     93.1%      1.00      0.42
Baseline 2 (ML + Static)          6.2%     95.4%      0.93      0.47
AAIRM (proposed)                  3.9%     97.8%      0.84      0.61
```
