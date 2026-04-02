"""AAIRM agent modules.

Exposes all agent classes and the shared AgentState / BaseAgent contract.

Layers
------
Perception (P1–P5)
    inventory_monitor, trend_intelligence, product_discovery,
    context_engine, risk_anomaly_detector

Conceptualization (C1–C5)
    demand_forecasting, reorder_optimisation, supplier_ranking,
    negotiation, governance

Action (A1–A3)
    order_execution, inventory_adjustment, learning_agent

Orchestration
    meta_orchestrator  — LangGraph-based PCA pipeline coordinator
"""

from aairm.agents.base import AgentState, BaseAgent

__all__ = ["AgentState", "BaseAgent"]
