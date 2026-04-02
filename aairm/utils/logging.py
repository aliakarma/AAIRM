"""Structured logging configuration for AAIRM.

Uses ``structlog`` to produce machine-readable JSON logs in production
and human-readable coloured output in development / notebook contexts.

Usage
-----
    from aairm.utils.logging import get_logger

    logger = get_logger(__name__)
    logger.info("order.proposed", sku_id="F123", quantity=50, cost=1200.0)

All agent ``run()`` methods must use this logger, not ``print()``.

Log Levels
----------
    DEBUG   — per-step simulation internals (very verbose)
    INFO    — per-agent entry/exit, every economic decision
    WARNING — anomaly alerts, governance constraint violations
    ERROR   — tool call failures, ERP connectivity issues

References
----------
AAIRM Repo Guide, Section 21.3: Logging requirements.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(
    level: str = "INFO",
    fmt: str = "console",
) -> None:
    """Configure structlog for the process.

    Must be called once, before any ``get_logger`` calls, typically from
    the top of a script or from :class:`~aairm.utils.config.AAIRMConfig`
    initialisation.

    Args:
        level: Log level string — one of ``"DEBUG"``, ``"INFO"``,
            ``"WARNING"``, ``"ERROR"``.
        fmt: Output format — ``"console"`` for human-readable coloured
            output; ``"json"`` for machine-readable structured output
            (use in production / CI).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if fmt == "json":
        processors = shared_processors + [structlog.processors.JSONRenderer()]
    else:
        processors = shared_processors + [structlog.dev.ConsoleRenderer(colors=True)]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "aairm") -> structlog.BoundLogger:
    """Return a bound structlog logger.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A structlog ``BoundLogger`` instance.

    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.info("agent.start", agent="P1", cycle_id="cycle-001")
    """
    return structlog.get_logger(name)
