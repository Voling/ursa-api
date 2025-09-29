from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class MetadataStore:
    """Simple JSON-backed metadata store for UrsaML aggregates."""

    def __init__(self, metadata_file: Path) -> None:
        self.metadata_file = metadata_file
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.metadata_file.exists():
            try:
                with self.metadata_file.open("r", encoding="utf-8") as handle:
                    raw = json.load(handle)
                if isinstance(raw, dict):
                    # Ensure expected top-level keys exist
                    raw.setdefault("projects", {})
                    raw.setdefault("graphs", {})
                    raw.setdefault("models", {})
                    return raw
            except json.JSONDecodeError:
                pass
        return {"projects": {}, "graphs": {}, "models": {}}

    def save(self) -> None:
        with self.metadata_file.open("w", encoding="utf-8") as handle:
            json.dump(self.data, handle, indent=2)



