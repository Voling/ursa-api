from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


class CacheMetadataStore:
    """Persistence helper for cache metadata summary information."""

    def __init__(self, metadata_file: Path) -> None:
        self._metadata_file = metadata_file
        self._metadata_file.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if self._metadata_file.exists():
            try:
                with self._metadata_file.open("r", encoding="utf-8") as handle:
                    raw = json.load(handle)
                if isinstance(raw, dict):
                    return {str(key): dict(value) for key, value in raw.items() if isinstance(value, dict)}
            except json.JSONDecodeError:
                return {}
        return {}

    @property
    def data(self) -> Dict[str, Dict[str, Any]]:
        return self._data

    def get(self, model_id: str) -> Dict[str, Any] | None:
        return self._data.get(model_id)

    def upsert(self, model_id: str, metadata: Dict[str, Any]) -> None:
        self._data[model_id] = dict(metadata)
        self.save()

    def remove(self, model_id: str) -> None:
        if model_id in self._data:
            del self._data[model_id]
            self.save()

    def touch_accessed(self, model_id: str, timestamp: str) -> None:
        entry = self._data.setdefault(model_id, {})
        entry["last_accessed"] = timestamp
        self.save()

    def items(self) -> Iterable[Tuple[str, Dict[str, Any]]]:
        return self._data.items()

    def total_size_bytes(self) -> int:
        return sum(entry.get("size_bytes", 0) for entry in self._data.values())

    def save(self) -> None:
        with self._metadata_file.open("w", encoding="utf-8") as handle:
            json.dump(self._data, handle, indent=2)