from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

from .local_cache import LocalCacheRepository
from .metadata_store import CacheMetadataStore
from .s3_gateway import ModelS3Gateway, NullModelS3Gateway
from .sdk_workspace import SDKWorkspaceManager
from .cache_policy import CachePolicy


class ModelCacheManager:
    """Orchestrates local cache, remote sync, and SDK workspace preparation.

    Public API mirrors the legacy ModelCacheService for compatibility.
    """

    def __init__(
        self,
        local_cache: LocalCacheRepository,
        metadata_store: CacheMetadataStore,
        sdk_workspace: SDKWorkspaceManager,
        policy: CachePolicy,
        s3_gateway: ModelS3Gateway | NullModelS3Gateway,
        s3_enabled: bool,
    ) -> None:
        self._local = local_cache
        self._meta = metadata_store
        self._sdk = sdk_workspace
        self._policy = policy
        self._s3 = s3_gateway
        self._s3_enabled = s3_enabled

    # --------------- Internal helpers ---------------
    def _resolve_model_path_from_metadata(self, metadata: Dict[str, Any], base_dir: Path) -> Optional[Path]:
        return self._local.resolve_model_path(metadata, base_dir)

    def _refresh_from_s3_if_needed(self, model_id: str, force_refresh: bool) -> None:
        needs_download = force_refresh or not self._policy.is_cached(model_id) or not self._policy.is_fresh(model_id)
        if not needs_download:
            return
        if not self._s3_enabled:
            # If no S3 and not locally cached, this will fail later during get
            return
        cache_dir = self._local.ensure_model_dir(model_id)
        metadata = self._s3.download(model_id, cache_dir)
        # verify at least one model file is present
        if not self._resolve_model_path_from_metadata(metadata, cache_dir):
            raise ValueError("No model file found in metadata after S3 download")

        # update summary metadata
        entry = {
            "cached_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "size_bytes": self._local.directory_size_bytes(cache_dir),
        }
        self._meta.upsert(model_id, entry)

    # --------------- Public API ---------------
    def get_model_for_sdk(self, model_id: str, force_refresh: bool = False) -> Path:
        """Return an SDK-ready temporary workspace containing the model."""
        self._refresh_from_s3_if_needed(model_id, force_refresh)

        # verify local cache
        cache_dir = self._local.model_dir(model_id)
        metadata_path = cache_dir / "metadata.json"
        if not cache_dir.exists() or not metadata_path.exists():
            raise ValueError(f"Model {model_id} not found in cache or remote storage")

        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        model_file = self._resolve_model_path_from_metadata(metadata, cache_dir)
        if not model_file:
            raise ValueError("No model file found in metadata")

        # create workspace and copy into SDK layout
        workspace = self._sdk.create_workspace()
        target_models_dir = workspace / "models"
        target_model_dir = target_models_dir / model_id
        target_model_dir.mkdir(parents=True, exist_ok=True)

        # replicate entire cache directory contents
        # We avoid shutil.copytree to retain control; but cache repo exposes a simple copy capability
        from shutil import copy2
        for file_path in cache_dir.rglob("*"):
            if not file_path.is_file():
                continue
            copy2(str(file_path), str(target_model_dir / file_path.name))

        # rewrite metadata paths to point inside workspace
        updated_metadata = dict(metadata)
        if "path" in updated_metadata:
            updated_metadata["path"] = str(target_model_dir / Path(updated_metadata["path"]).name)
        artifacts = updated_metadata.get("artifacts", {})
        if isinstance(artifacts, dict):
            for value in artifacts.values():
                if isinstance(value, dict) and "path" in value:
                    value["path"] = str(target_model_dir / Path(value["path"]).name)

        with (target_model_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(updated_metadata, handle, indent=2)

        # touch access time
        self._meta.touch_accessed(model_id, datetime.now().isoformat())

        return workspace

    def save_model_from_sdk(self, model_id: str, sdk_dir: Path) -> Path:
        """Persist model from SDK workspace into cache and optionally upload to S3."""
        sdk_model_dir = sdk_dir / "models" / model_id
        if not sdk_model_dir.exists():
            raise ValueError(f"Model {model_id} not found in SDK directory")

        cache_dir = self._local.copy_from_sdk(sdk_model_dir, model_id)

        entry = {
            "cached_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "size_bytes": self._local.directory_size_bytes(cache_dir),
        }
        self._meta.upsert(model_id, entry)

        if self._s3_enabled:
            try:
                self._s3.upload(model_id, cache_dir)
            except Exception:
                # Defer errors; local cache is authoritative for API correctness
                pass

        return cache_dir

    def delete_model(self, model_id: str) -> bool:
        """Delete model from cache and remote if configured."""
        self._local.remove_model_dir(model_id)
        self._meta.remove(model_id)
        if self._s3_enabled:
            try:
                self._s3.delete(model_id)
            except Exception:
                pass
        return True

    def cleanup_old_cache(self, max_age_days: int = 7, max_size_gb: float = 10.0) -> None:
        cutoff = datetime.now() - timedelta(days=max_age_days)
        total_size = self._meta.total_size_bytes()
        max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)

        # Sort by last_accessed ascending
        entries = sorted(self._meta.items(), key=lambda x: x[1].get("last_accessed", ""))
        for model_id, entry in entries:
            should_delete = False
            if max_age_days == 0:
                should_delete = True
            else:
                try:
                    last = datetime.fromisoformat(entry.get("last_accessed", ""))
                    if last < cutoff:
                        should_delete = True
                except ValueError:
                    should_delete = True

            if total_size > max_size_bytes:
                should_delete = True

            if should_delete:
                size = entry.get("size_bytes", 0)
                self.delete_model(model_id)
                total_size -= size

    def get_cache_stats(self) -> Dict[str, Any]:
        total_size = self._meta.total_size_bytes()
        return {
            "total_models": len(list(self._meta.items())),
            "total_size_mb": total_size / (1024 * 1024),
        }


