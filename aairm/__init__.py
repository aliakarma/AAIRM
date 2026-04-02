"""AAIRM: Agentic AI Inventory Replenishment and Management.

A multi-agent, LangChain-orchestrated framework for autonomous inventory
replenishment and product discovery in multi-category retail environments.

Architecture
------------
The framework organises specialised agents across three functional layers:

    Perception Layer (P1–P5)
        Ingests real-time signals from inventory databases, ERP/WMS systems,
        external trend APIs, and anomaly detectors.

    Conceptualization Layer (C1–C5)
        Performs demand forecasting (C1), reorder optimisation with PPO-based
        RL policy (C2), supplier ranking (C3), autonomous negotiation (C4),
        and governance enforcement (C5).

    Action Layer (A1–A3)
        Executes approved purchase orders (A1), reconciles inventory records
        (A2), and closes the learning loop via temporal-difference updates (A3).

A central Meta-Orchestrator (LangGraph / LangChain) handles task
decomposition, agent memory, and tool routing across all layers.

Paper Reference
---------------
Syed, T.A., El-Boghdadi, H.M., Naqash, M.T., Alghamdi, T., Alshahrani, A.,
Lee, I.E., Akarma, A. (2025). Agentic Commerce: Economic Implications of
AI-Driven Forecasting, Inventory Management, and Product Personalization
in Retail. Frontiers in [Journal]. https://doi.org/[DOI-PLACEHOLDER]

Usage
-----
    from aairm.utils.config import AAIRMConfig
    from aairm.simulation.environment import RetailEnv
    from aairm.agents.meta_orchestrator import MetaOrchestrator
    from aairm.utils.seed import set_global_seed

    set_global_seed(42)
    config = AAIRMConfig()
    env = RetailEnv(config.simulation)
    orchestrator = MetaOrchestrator(config)
    state = env.reset()
    for _ in range(365):
        state = orchestrator.run_cycle(state, env)
"""

from aairm.version import __version__, __version_info__

__all__ = ["__version__", "__version_info__"]
__author__ = (
    "Toqeer Ali Syed, Hatem M. El-Boghdadi, Muhammad Tayyab Naqash, "
    "Turki Alghamdi, Abdulaziz Alshahrani, It Ee Lee, Ali Akarma"
)
__email__ = "mnaqash@iu.edu.sa"
__license__ = "MIT"
