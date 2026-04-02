"""Action Layer agents (A1–A3).

A1  OrderExecutionAgent      — transmits POs to suppliers, updates ERP
A2  InventoryAdjustmentAgent — reconciles physical receipts with system records
A3  LearningAgent            — closes the feedback loop via TD-update (Eq. 7)
"""

from aairm.agents.action.order_execution import OrderExecutionAgent
from aairm.agents.action.inventory_adjustment import InventoryAdjustmentAgent
from aairm.agents.action.learning_agent import LearningAgent

__all__ = ["OrderExecutionAgent", "InventoryAdjustmentAgent", "LearningAgent"]
