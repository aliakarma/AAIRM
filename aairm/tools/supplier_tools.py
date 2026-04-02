"""Supplier catalogue and negotiation tool wrappers for LangChain agents.

Used by the Supplier Ranking Agent (C3) and Negotiation Agent (C4).
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

_SUPPLIER_BACKEND: Any = None


def set_supplier_backend(backend: Any) -> None:
    """Inject the supplier backend for all supplier tools."""
    global _SUPPLIER_BACKEND  # noqa: PLW0603
    _SUPPLIER_BACKEND = backend


def query_catalogue_impl(sku_id: str) -> list[dict[str, Any]]:
    """Query all supplier offers for a given SKU.

    Args:
        sku_id: SKU identifier.

    Returns:
        List of supplier offer dicts:
        ``{supplier_id, unit_cost, lead_time_mean, lead_time_std,
           reliability, moq, country}``.
    """
    if _SUPPLIER_BACKEND is None:
        return []
    try:
        return _SUPPLIER_BACKEND.query_catalogue(sku_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("tool.supplier.query_catalogue.failed", error=str(exc))
        return []


def request_discount_impl(supplier_id: str, sku_id: str, quantity: float) -> str:
    """Request a volume discount from a supplier.

    Args:
        supplier_id: Supplier identifier.
        sku_id: SKU identifier.
        quantity: Requested order quantity.

    Returns:
        Discount offer string (e.g. "5% discount approved for qty >= 100").
    """
    if _SUPPLIER_BACKEND is None:
        return "Supplier backend not configured."
    try:
        result = _SUPPLIER_BACKEND.request_discount(supplier_id, sku_id, quantity)
        return str(result)
    except Exception as exc:  # noqa: BLE001
        return f"Discount request failed: {exc}"


def propose_schedule_impl(supplier_id: str, sku_id: str, delta_days: int) -> str:
    """Propose an alternative delivery schedule adjustment.

    Args:
        supplier_id: Supplier identifier.
        sku_id: SKU identifier.
        delta_days: Schedule adjustment in days (positive = later, negative = earlier).

    Returns:
        Supplier response string.
    """
    if _SUPPLIER_BACKEND is None:
        return "Supplier backend not configured."
    try:
        result = _SUPPLIER_BACKEND.propose_schedule(supplier_id, sku_id, delta_days)
        return str(result)
    except Exception as exc:  # noqa: BLE001
        return f"Schedule proposal failed: {exc}"


def finalise_terms_impl(
    supplier_id: str,
    sku_id: str,
    unit_price: float,
    quantity: float,
    delivery_window_days: float,
) -> dict[str, Any]:
    """Finalise commercial terms with a supplier.

    Args:
        supplier_id: Supplier identifier.
        sku_id: SKU identifier.
        unit_price: Agreed unit price.
        quantity: Agreed quantity.
        delivery_window_days: Agreed delivery window.

    Returns:
        Confirmed terms dict.
    """
    return {
        "supplier_id": supplier_id,
        "sku_id": sku_id,
        "unit_price": unit_price,
        "quantity": quantity,
        "delivery_window_days": delivery_window_days,
        "payment_terms": "Net-30",
        "status": "confirmed",
    }


if _LANGCHAIN_AVAILABLE:
    class _DiscountInput(BaseModel):
        supplier_id: str = Field(description="Supplier identifier.")
        sku_id: str = Field(description="SKU identifier.")
        quantity: float = Field(description="Requested order quantity.")

    class _ScheduleInput(BaseModel):
        supplier_id: str = Field(description="Supplier identifier.")
        sku_id: str = Field(description="SKU identifier.")
        delta_days: int = Field(description="Schedule adjustment in days (±2 max).")

    class _FinaliseInput(BaseModel):
        supplier_id: str = Field(description="Supplier identifier.")
        sku_id: str = Field(description="SKU identifier.")
        unit_price: float = Field(description="Agreed unit price.")
        quantity: float = Field(description="Agreed quantity.")
        delivery_window_days: float = Field(description="Agreed lead time in days.")

    request_discount_tool = StructuredTool.from_function(
        func=request_discount_impl,
        name="request_discount",
        description="Request a volume discount from a supplier for a given SKU and quantity.",
        args_schema=_DiscountInput,
    )

    propose_schedule_tool = StructuredTool.from_function(
        func=propose_schedule_impl,
        name="propose_schedule",
        description="Propose an alternative delivery schedule (±2 days max).",
        args_schema=_ScheduleInput,
    )

    finalise_terms_tool = StructuredTool.from_function(
        func=finalise_terms_impl,
        name="finalise_terms",
        description="Finalise and confirm commercial terms with a supplier.",
        args_schema=_FinaliseInput,
    )
else:
    request_discount_tool = None  # type: ignore[assignment]
    propose_schedule_tool = None  # type: ignore[assignment]
    finalise_terms_tool = None    # type: ignore[assignment]
