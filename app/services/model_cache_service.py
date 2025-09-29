"""
ML Model Caching Service for ursa-api.

Manages local cache vs S3 storage for trained ML models (the actual model files).
Minimizes costs and maximizes performance by intelligently caching models locally.

This service handles:
- Trained ML models (scikit-learn, PyTorch, TensorFlow pickled/serialized files)
- NOT database models or API schemas
"""

import json
import shutil
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

import boto3
from app.config import settings, REPO_ROOT
from app.services.cache.cache_manager import ModelCacheManager
from app.services.cache.local_cache import LocalCacheRepository
from app.services.cache.metadata_store import CacheMetadataStore
from app.services.cache.s3_gateway import ModelS3Gateway, NullModelS3Gateway
from app.services.cache.sdk_workspace import SDKWorkspaceManager
from app.services.cache.cache_policy import CachePolicy


class ModelCacheService:
    """Backwards-compatible façade delegating to `ModelCacheManager`."""
    
    def __init__(self):
        self.cache_dir = Path(settings.MODEL_STORAGE_DIR) / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "models").mkdir(parents=True, exist_ok=True)

        # Dependencies
        self.cache_metadata_file = self.cache_dir / "cache_metadata.json"
        metadata_store = CacheMetadataStore(self.cache_metadata_file)
        local_repo = LocalCacheRepository(self.cache_dir)

        # SDK temp workspace
        self.sdk_temp_dir = REPO_ROOT / "storage" / "sdk_temp"
        self.sdk_temp_dir.mkdir(parents=True, exist_ok=True)
        sdk_workspace = SDKWorkspaceManager(self.sdk_temp_dir)

        # Remote gateway
        s3_enabled = settings.STORAGE_TYPE == "s3"
        if s3_enabled:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION
            )
            s3_gateway = ModelS3Gateway(s3_client, settings.S3_BUCKET)
        else:
            s3_gateway = NullModelS3Gateway()

        policy = CachePolicy(metadata_store)
        self._manager = ModelCacheManager(
            local_cache=local_repo,
            metadata_store=metadata_store,
            sdk_workspace=sdk_workspace,
            policy=policy,
            s3_gateway=s3_gateway,
            s3_enabled=s3_enabled,
        )
    
    def _load_cache_metadata(self) -> Dict[str, Any]:
        # Maintained for compatibility; delegate to metadata store
        return {}
    
    def _save_cache_metadata(self):
        return None
    
    def _get_model_cache_path(self, model_id: str) -> Path:
        """Get local cache path for a model."""
        path = self.cache_dir / "models" / model_id
        path.parent.mkdir(parents=True, exist_ok=True)  # Ensure models directory exists
        return path
    
    def _get_sdk_temp_path(self) -> Path:
        """Get a temporary directory for SDK operations."""
        temp_id = str(uuid.uuid4())
        path = self.sdk_temp_dir / temp_id
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def _cleanup_sdk_temp(self, path: Path):
        """Clean up a temporary SDK directory."""
        if path.exists() and path.is_relative_to(self.sdk_temp_dir):
            shutil.rmtree(path)
    
    def _get_model_path_from_metadata(self, metadata: Dict[str, Any], base_dir: Path) -> Optional[Path]:
        """
        Get model file path from metadata, checking both top-level path and artifacts.
        Returns None if no valid path found.
        
        Args:
            metadata: Model metadata dictionary
            base_dir: Base directory to resolve relative paths against
            
        Returns:
            Path to model file if found, None otherwise
        """
        # Check top-level path first
        if "path" in metadata:
            model_path = Path(metadata["path"])
            if model_path.exists():
                return model_path
            relative_path = base_dir / model_path.name
            if relative_path.exists():
                return relative_path
                
        # Check artifacts
        if "artifacts" in metadata:
            for artifact_info in metadata["artifacts"].values():
                if "path" in artifact_info:
                    artifact_path = Path(artifact_info["path"])
                    if artifact_path.exists():
                        return artifact_path
                    relative_path = base_dir / artifact_path.name
                    if relative_path.exists():
                        return relative_path
        
        return None
    
    def _is_model_cached(self, model_id: str) -> bool:
        # Compatibility shim using repository
        try:
            from app.services.cache.local_cache import LocalCacheRepository
            repo = LocalCacheRepository(self.cache_dir)
            return repo.has_model(model_id)
        except Exception:
            return False
    
    def _is_cache_fresh(self, model_id: str, max_age_hours: int = 24) -> bool:
        try:
            from app.services.cache.cache_policy import CachePolicy
            from app.services.cache.metadata_store import CacheMetadataStore
            policy = CachePolicy(CacheMetadataStore(self.cache_metadata_file))
            return policy.is_fresh(model_id, max_age_hours)
        except Exception:
            return False
    
    def _download_from_s3(self, model_id: str) -> Path:
        # Delegate to manager; returns path via side-effect, then ensure dir exists
        self._manager._refresh_from_s3_if_needed(model_id, force_refresh=True)
        return self._get_model_cache_path(model_id)
    
    def _upload_to_s3(self, model_id: str, local_path: Path):
        # Delegated in manager during save; keep method for compatibility
        _ = (model_id, local_path)
        return None
    
    def get_model_for_sdk(self, model_id: str, force_refresh: bool = False) -> Path:
        """
        Get model path for SDK use.
        Returns local cache path, downloading from S3 if necessary.
        
        Args:
            model_id: Model identifier
            force_refresh: Force download from S3 even if cached
            
        Returns:
            Path to cached model directory for SDK to use
        """
        return self._manager.get_model_for_sdk(model_id, force_refresh)
    
    def save_model_from_sdk(self, model_id: str, sdk_dir: Path) -> Path:
        """
        Save model from SDK to cache and optionally S3.
        
        Args:
            model_id: Model identifier  
            sdk_dir: Directory where SDK saved the model
            
        Returns:
            Path to cached model directory
        """
        return self._manager.save_model_from_sdk(model_id, sdk_dir)
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model from cache and optionally S3."""
        _ = cache_path
        return self._manager.delete_model(model_id)
    
    def cleanup_old_cache(self, max_age_days: int = 7, max_size_gb: float = 10.0):
        """Clean up old or oversized cache entries."""
        return self._manager.cleanup_old_cache(max_age_days=max_age_days, max_size_gb=max_size_gb)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._manager.get_cache_stats()