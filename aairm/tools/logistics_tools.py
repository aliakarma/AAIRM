"""Logistics carrier tool wrappers for the Order Execution Agent (A1)."""

from __future__ import annotations

from typing import Any

from aairm.utils.logging import get_logger

logger = get_logger(__name__)


def book_shipment(
    supplier_id: str,
    sku_id: str,
    quantity: float,
    destination: str = "WAREHOUSE_MAIN",
) -> dict[str, Any]:
    """Book a shipment with a logistics carrier.

    In simulation mode, always returns a confirmed booking with a
    deterministic tracking number.

    Args:
        supplier_id: Supplier dispatching the goods.
        sku_id: SKU being shipped.
        quantity: Shipment quantity in units.
        destination: Destination warehouse code.

    Returns:
        Booking confirmation dict:
        ``{tracking_id, carrier, estimated_transit_days, status}``.
    """
    tracking_id = f"TRK-{supplier_id[:4].upper()}-{sku_id[:4].upper()}"
    logger.info(
        "logistics.shipment_booked",
        tracking_id=tracking_id,
        sku_id=sku_id,
        quantity=quantity,
    )
    return {
        "tracking_id": tracking_id,
        "carrier": "SimCarrier",
        "estimated_transit_days": 2,
        "destination": destination,
        "status": "confirmed",
    }


def get_shipment_status(tracking_id: str) -> dict[str, Any]:
    """Query the status of an in-transit shipment.

    Args:
        tracking_id: Tracking identifier returned by :func:`book_shipment`.

    Returns:
        Status dict: ``{tracking_id, status, estimated_arrival_day}``.
    """
    return {
        "tracking_id": tracking_id,
        "status": "in_transit",
        "estimated_arrival_day": "N/A",
    }
