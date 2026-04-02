"""ERP / WMS integration helpers.

Thin wrappers around the ERP backend that normalise return formats and
add structured logging.  Used by perception and action agents.
"""

from __future__ import annotations

from typing import Any

from aairm.utils.logging import get_logger

logger = get_logger(__name__)

_ERP_BACKEND: Any = None


def set_erp_backend(backend: Any) -> None:
    """Inject the ERP backend (same reference as inventory_tools)."""
    global _ERP_BACKEND  # noqa: PLW0603
    _ERP_BACKEND = backend


def create_purchase_order(order: dict[str, Any]) -> str:
    """Write a new purchase order to the ERP system.

    Args:
        order: PO dict as produced by A1.

    Returns:
        ERP PO reference string.
    """
    if _ERP_BACKEND is None:
        return "ERP not configured."
    try:
        _ERP_BACKEND.create_purchase_order(order)
        ref = order.get("po_id", "UNKNOWN")
        logger.info("erp.po_created", po_id=ref, sku_id=order.get("sku_id"))
        return ref
    except Exception as exc:  # noqa: BLE001
        logger.error("erp.po_create.failed", error=str(exc))
        return f"Error: {exc}"


def process_goods_receipt(receipt: dict[str, Any]) -> str:
    """Post a goods receipt in the ERP and update on-hand balance.

    Args:
        receipt: Receipt dict with ``{sku_id, received_qty, po_id, day}``.

    Returns:
        Confirmation string.
    """
    if _ERP_BACKEND is None:
        return "ERP not configured."
    try:
        _ERP_BACKEND.process_goods_receipt(receipt)
        logger.info(
            "erp.goods_received",
            sku_id=receipt.get("sku_id"),
            qty=receipt.get("received_qty"),
        )
        return f"Receipt posted for SKU {receipt.get('sku_id')}."
    except Exception as exc:  # noqa: BLE001
        logger.error("erp.goods_receipt.failed", error=str(exc))
        return f"Error: {exc}"
