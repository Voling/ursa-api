from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional


class LocalCacheRepository:
    """Handle filesystem operations for cached models."""

    def __init__(self, cache_root: Path) -> None:
        self.cache_root = cache_root
        self.models_root = self.cache_root / "models"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.models_root.mkdir(parents=True, exist_ok=True)

    def model_dir(self, model_id: str) -> Path:
        return self.models_root / model_id

    def metadata_path(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "metadata.json"

    def ensure_model_dir(self, model_id: str) -> Path:
        path = self.model_dir(model_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def read_model_metadata(self, model_id: str) -> Dict[str, Any] | None:
        metadata_file = self.metadata_path(model_id)
        if not metadata_file.exists():
            return None
        with metadata_file.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def write_model_metadata(self, model_id: str, metadata: Dict[str, Any]) -> None:
        metadata_file = self.metadata_path(model_id)
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with metadata_file.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    @staticmethod
    def resolve_model_path(metadata: Dict[str, Any], base_dir: Path) -> Optional[Path]:
        """Return a concrete model file path using metadata hints."""
        if "path" in metadata:
            model_path = Path(metadata["path"])
            if model_path.exists():
                return model_path
            candidate = base_dir / model_path.name
            if candidate.exists():
                return candidate

        artifacts = metadata.get("artifacts", {})
        if isinstance(artifacts, dict):
            for artifact in artifacts.values():
                path_hint = artifact.get("path") if isinstance(artifact, dict) else None
                if not path_hint:
                    continue
                artifact_path = Path(path_hint)
                if artifact_path.exists():
                    return artifact_path
                candidate = base_dir / artifact_path.name
                if candidate.exists():
                    return candidate
        return None

    @staticmethod
    def directory_size_bytes(path: Path) -> int:
        return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())

    def copy_from_sdk(self, sdk_model_dir: Path, model_id: str) -> Path:
        cache_path = self.ensure_model_dir(model_id)
        shutil.copytree(sdk_model_dir, cache_path, dirs_exist_ok=True)
        return cache_path

    def remove_model_dir(self, model_id: str) -> None:
        cache_path = self.model_dir(model_id)
        if cache_path.exists():
            shutil.rmtree(cache_path)

    def has_model(self, model_id: str) -> bool:
        cache_path = self.model_dir(model_id)
        metadata_path = self.metadata_path(model_id)
        if not cache_path.exists() or not metadata_path.exists():
            return False
        metadata = self.read_model_metadata(model_id)
        if not metadata:
            return False
        return self.resolve_model_path(metadata, cache_path) is not None