"""Perception Layer agents (P1–P5).

P1  InventoryMonitorAgent     — on-hand stock monitoring and low-stock detection
P2  TrendIntelligenceAgent    — external market trend signals
P3  ProductDiscoveryAgent     — new SKU candidate surfacing
P4  ContextEngine             — feature assembly for forecasting
P5  RiskAnomalyDetector       — irregular signal detection
"""

from aairm.agents.perception.inventory_monitor import InventoryMonitorAgent
from aairm.agents.perception.trend_intelligence import TrendIntelligenceAgent
from aairm.agents.perception.product_discovery import ProductDiscoveryAgent
from aairm.agents.perception.context_engine import ContextEngine
from aairm.agents.perception.risk_anomaly_detector import RiskAnomalyDetector

__all__ = [
    "InventoryMonitorAgent",
    "TrendIntelligenceAgent",
    "ProductDiscoveryAgent",
    "ContextEngine",
    "RiskAnomalyDetector",
]
