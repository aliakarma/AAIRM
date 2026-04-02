"""Inventory and ERP tool wrappers for LangChain agents.

These tools wrap the ERP stub interface so that any LangChain-based agent
(including the Negotiation Agent C4) can query and update inventory state
via natural-language tool calls.

In simulation mode the tools delegate to the RetailEnv ERP stub.
In production they would call real ERP REST APIs.
"""

from __future__ import annotations

from typing import Any

try:
    from langchain.tools import StructuredTool  # type: ignore
    from pydantic import BaseModel, Field  # type: ignore[assignment]
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False

from aairm.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Global backend reference — injected at orchestrator startup
# ---------------------------------------------------------------------------
_ERP_BACKEND: Any = None


def set_erp_backend(backend: Any) -> None:
    """Inject the ERP backend for all inventory tools.

    Args:
        backend: Object implementing the ERP stub interface.
    """
    global _ERP_BACKEND  # noqa: PLW0603
    _ERP_BACKEND = backend


# ---------------------------------------------------------------------------
# Tool implementations (plain functions, wrapped below)
# ---------------------------------------------------------------------------

def get_inventory_snapshot_impl(sku_id: str) -> dict[str, Any]:
    """Query on-hand, reserved, and in-transit stock for a single SKU.

    Args:
        sku_id: The SKU identifier to query.

    Returns:
        Inventory record dict or an error dict.
    """
    if _ERP_BACKEND is None:
        return {"error": "ERP backend not configured."}
    try:
        snapshot = _ERP_BACKEND.get_inventory_snapshot()
        return snapshot.get(sku_id, {"error": f"SKU {sku_id} not found."})
    except Exception as exc:  # noqa: BLE001
        logger.error("tool.inventory.get_snapshot.failed", error=str(exc))
        return {"error": str(exc)}


def get_demand_history_impl(sku_id: str, n_days: int = 30) -> list[float]:
    """Retrieve the last ``n_days`` of daily demand for a SKU.

    Args:
        sku_id: The SKU identifier.
        n_days: Number of history days to retrieve (default 30).

    Returns:
        List of daily demand values (floats).
    """
    if _ERP_BACKEND is None:
        return []
    try:
        import numpy as np
        history = _ERP_BACKEND.get_demand_history(sku_id, n_days)
        return [float(x) for x in history]
    except Exception as exc:  # noqa: BLE001
        logger.error("tool.inventory.get_history.failed", error=str(exc))
        return []


def update_inbound_schedule_impl(po_id: str, eta_days: int) -> str:
    """Record the expected arrival date of a purchase order in the ERP.

    Args:
        po_id: Purchase order identifier.
        eta_days: Days until expected arrival.

    Returns:
        Confirmation string.
    """
    if _ERP_BACKEND is None:
        return "ERP backend not configured."
    try:
        _ERP_BACKEND.update_inbound_schedule(po_id, eta_days)
        return f"Inbound schedule updated: PO {po_id} ETA {eta_days} days."
    except Exception as exc:  # noqa: BLE001
        logger.error("tool.inventory.update_inbound.failed", error=str(exc))
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# LangChain tool wrappers (only created when LangChain is installed)
# ---------------------------------------------------------------------------

if _LANGCHAIN_AVAILABLE:
    class _InventoryQueryInput(BaseModel):
        sku_id: str = Field(description="SKU identifier to query.")

    class _DemandHistoryInput(BaseModel):
        sku_id: str = Field(description="SKU identifier.")
        n_days: int = Field(30, description="Number of history days.")

    inventory_read_tool = StructuredTool.from_function(
        func=get_inventory_snapshot_impl,
        name="inventory_read",
        description=(
            "Query real-time on-hand, reserved, and in-transit inventory "
            "for a given SKU.  Returns a dict with inventory details."
        ),
        args_schema=_InventoryQueryInput,
    )

    demand_history_tool = StructuredTool.from_function(
        func=get_demand_history_impl,
        name="demand_history",
        description="Retrieve recent daily demand history for a SKU.",
        args_schema=_DemandHistoryInput,
    )
else:
    inventory_read_tool = None  # type: ignore[assignment]
    demand_history_tool = None  # type: ignore[assignment]
