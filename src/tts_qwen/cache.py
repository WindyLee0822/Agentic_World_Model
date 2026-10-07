"""Read cache with optional selective invalidation.

Within one decision the environment never changes (simulators only read), so key reads are
perfectly cacheable and are repeated across the ``N`` rollouts that probe the same action's
keys.  After a commit, the cache is invalidated — selectively (only the committed action's
keys) or fully — per the ``selective_invalidation`` knob.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

from .types import Row, RowKey


@dataclass
class ReadCache:
    """A key -> row cache with hit/miss accounting."""

    enabled: bool = True
    store: Dict[RowKey, Row] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get_many(self, keys: Iterable[RowKey]):
        """Return ``(cached_rows, missing_keys)``; counts a hit/miss per key."""
        cached: Dict[RowKey, Row] = {}
        missing = []
        for k in keys:
            if self.enabled and k in self.store:
                cached[k] = dict(self.store[k])
                self.hits += 1
            else:
                missing.append(k)
                if self.enabled:
                    self.misses += 1
        return cached, missing

    def put_many(self, rows: Dict[RowKey, Row]) -> None:
        if not self.enabled:
            return
        for k, v in rows.items():
            self.store[k] = dict(v)

    def invalidate(self, keys: Iterable[RowKey], selective: bool = True) -> None:
        if not self.enabled:
            return
        if not selective:
            self.store.clear()
            return
        for k in list(keys):
            self.store.pop(k, None)

    def reset_counters(self) -> None:
        self.hits = 0
        self.misses = 0


__all__ = ["ReadCache"]
