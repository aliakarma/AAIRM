# Architecture

AAIRM implements a **Perception–Conceptualization–Action (PCA)** workflow
coordinated by a central Meta-Orchestrator (LangGraph / LangChain).

## PCA Pipeline

```
User request
    │
    ▼
Meta-Orchestrator
    │
    ├── P1 Inventory Monitor
    ├── P2 Trend Intelligence
    ├── P3 Product Discovery
    ├── P4 Context Engine
    ├── P5 Risk & Anomaly Detector
    │
    ├── C1 Demand Forecasting   (Eq. 2)
    ├── C2 Reorder Optimisation (Eqs. 3–5, PPO policy)
    ├── C3 Supplier Ranking     (Eq. 6)
    ├── C4 Autonomous Negotiation
    ├── C5 Governance & Policy
    │
    ├── A1 Order Execution
    ├── A2 Inventory Adjustment
    └── A3 Learning Agent       (Eq. 7, TD update)
```

## Key Equations

| Equation | Description | Location |
|---|---|---|
| Eq. 1 | Reorder Point: $\text{ROP} = \mu_D L + z\sigma_D\sqrt{L}$ | `math_utils.rop` |
| Eq. 2 | Demand forecast: $\hat{y}_{i,t+h} = f_\theta(\mathbf{x}_{i,t}, h)$ | C1 agent |
| Eq. 3 | Single-period expected cost $C_i(Q_i)$ | `math_utils.expected_cost_single_period` |
| Eq. 4 | Budget + capacity constrained optimisation | C2 agent |
| Eq. 5 | RL objective: $\max_{\pi_\phi} \mathbb{E}[\sum \gamma^t r(s_t, a_t)]$ | `ppo_policy.py` |
| Eq. 6 | Supplier score: $\alpha_1 c_{ij} + \alpha_2\hat{L}_{ij} - \alpha_3 r_{ij} + \alpha_4\mathbb{I}$ | `math_utils.supplier_score` |
| Eq. 7 | TD loss: $\mathcal{L}_\text{TD}(\phi) = (r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t))^2$ | `math_utils.td_loss` |
