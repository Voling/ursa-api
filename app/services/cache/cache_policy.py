from __future__ import annotations

from datetime import datetime, timedelta
from typing import Protocol


class SupportsMetadataAccess(Protocol):
    def get(self, model_id: str): ...


class CachePolicy:
    """Encapsulate caching heuristics such as freshness checks."""

    def __init__(self, metadata_store: SupportsMetadataAccess, clock: type[datetime] = datetime) -> None:
        self._metadata_store = metadata_store
        self._clock = clock

    def is_cached(self, model_id: str) -> bool:
        return self._metadata_store.get(model_id) is not None

    def is_fresh(self, model_id: str, max_age_hours: int = 24) -> bool:
        entry = self._metadata_store.get(model_id)
        if not entry:
            return False
        cached_at = entry.get("cached_at")
        if not cached_at:
            return False
        try:
            cached_time = datetime.fromisoformat(cached_at)
        except ValueError:
            return False
        return self._clock.now() - cached_time < timedelta(hours=max_age_hours)