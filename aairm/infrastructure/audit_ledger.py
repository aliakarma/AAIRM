"""Blockchain-inspired Audit Ledger — Trusted Agent Infrastructure.

Provides an immutable, append-only audit trail of all procurement events
using SHA-256 chained hashing.  Each entry's hash incorporates the
previous entry's hash, making any retrospective modification detectable.

NOTE: This implementation uses local SHA-256 hashing for the simulation.
In production deployment, replace the ``_hash_entry`` method with calls
to a distributed ledger service (e.g. Hyperledger Fabric or a public
blockchain API).

References
----------
Paper Section 4.3 (Trusted Agent Infrastructure — Blockchain Trust Ledger).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from aairm.utils.logging import get_logger

logger = get_logger(__name__)

_GENESIS_HASH = "0" * 64   # conventional genesis block hash


class AuditLedger:
    """Immutable SHA-256 chained audit ledger.

    Args:
        persist_path: Optional path for writing the ledger to disk as
            newline-delimited JSON.  If ``None``, the ledger is in-memory
            only and is not persisted across restarts.
    """

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._entries: list[dict[str, Any]] = []
        self._prev_hash: str = _GENESIS_HASH
        self._persist_path = Path(persist_path) if persist_path else None

        if self._persist_path:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event_type: str, payload: dict[str, Any]) -> str:
        """Append a new event to the ledger.

        Args:
            event_type: Short identifier (e.g. ``"po.issued"``).
            payload: Arbitrary event payload dict.

        Returns:
            SHA-256 hash of the new entry (chain link).
        """
        entry = {
            "sequence": len(self._entries),
            "timestamp": time.time(),
            "event_type": event_type,
            "payload": payload,
            "prev_hash": self._prev_hash,
        }
        entry_hash = self._hash_entry(entry)
        entry["hash"] = entry_hash
        self._entries.append(entry)
        self._prev_hash = entry_hash

        if self._persist_path:
            self._write_entry(entry)

        logger.debug(
            "ledger.appended",
            seq=entry["sequence"],
            event_name=event_type,
            hash_prefix=entry_hash[:8],
        )
        return entry_hash

    def verify(self) -> bool:
        """Verify the integrity of the entire chain.

        Re-computes each entry's hash and confirms the chain links are
        unbroken.

        Returns:
            ``True`` if all hashes are consistent; ``False`` if any
            tampering is detected.
        """
        prev = _GENESIS_HASH
        for entry in self._entries:
            stored_hash = entry.get("hash", "")
            entry_without_hash = {k: v for k, v in entry.items() if k != "hash"}
            entry_without_hash["prev_hash"] = prev
            computed = self._hash_entry(entry_without_hash)
            if computed != stored_hash:
                logger.error(
                    "ledger.integrity_violation",
                    sequence=entry.get("sequence"),
                    expected=computed[:8],
                    found=stored_hash[:8],
                )
                return False
            prev = stored_hash
        return True

    def query(self, event_type: str | None = None) -> list[dict[str, Any]]:
        """Return ledger entries, optionally filtered by event type.

        Args:
            event_type: If provided, return only entries of this type.

        Returns:
            List of ledger entries (each includes ``hash`` and ``prev_hash``).
        """
        if event_type is None:
            return list(self._entries)
        return [e for e in self._entries if e["event_type"] == event_type]

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_entry(entry: dict[str, Any]) -> str:
        """Compute SHA-256 hash of a ledger entry."""
        serialised = json.dumps(entry, sort_keys=True, default=str).encode()
        return hashlib.sha256(serialised).hexdigest()

    def _write_entry(self, entry: dict[str, Any]) -> None:
        """Append one entry to the persistent ledger file."""
        try:
            with open(self._persist_path, "a", encoding="utf-8") as f:  # type: ignore[arg-type]
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError as exc:
            logger.error("ledger.persist_failed", error=str(exc))
