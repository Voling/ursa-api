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
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

import boto3
from app.config import settings


class ModelCacheService:
    """
    Intelligent model caching service.
    
    Strategy:
    - S3: Source of truth, long-term storage, sharing between instances
    - Local cache: Fast access, cost optimization, performance
    - SDK: Always works with local files
    """
    
    def __init__(self):
        self.cache_dir = Path(settings.MODEL_STORAGE_DIR) / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # S3 client (only if using S3 storage)
        self.s3_client = None
        if settings.STORAGE_TYPE == "s3":
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION
            )
        
        # Cache metadata
        self.cache_metadata_file = self.cache_dir / "cache_metadata.json"
        self.cache_metadata = self._load_cache_metadata()
    
    def _load_cache_metadata(self) -> Dict[str, Any]:
        """Load cache metadata (last accessed, sizes, etc.)"""
        if self.cache_metadata_file.exists():
            with open(self.cache_metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache_metadata(self):
        """Save cache metadata to disk."""
        with open(self.cache_metadata_file, 'w') as f:
            json.dump(self.cache_metadata, f, indent=2)
    
    def _get_model_cache_path(self, model_id: str) -> Path:
        """Get local cache path for a model."""
        return self.cache_dir / "models" / model_id
    
    def _is_model_cached(self, model_id: str) -> bool:
        """Check if model is already cached locally."""
        cache_path = self._get_model_cache_path(model_id)
        metadata_path = cache_path / "metadata.json"
        return cache_path.exists() and metadata_path.exists()
    
    def _is_cache_fresh(self, model_id: str, max_age_hours: int = 24) -> bool:
        """Check if cached model is still fresh (not stale)."""
        if model_id not in self.cache_metadata:
            return False
            
        cached_time = datetime.fromisoformat(self.cache_metadata[model_id]["cached_at"])
        age = datetime.now() - cached_time
        return age < timedelta(hours=max_age_hours)
    
    def _download_from_s3(self, model_id: str) -> Path:
        """Download model from S3 to local cache."""
        if not self.s3_client:
            raise ValueError("S3 not configured but S3 download requested")
        
        cache_path = self._get_model_cache_path(model_id)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download metadata first to get artifact list
            metadata_path = cache_path / "metadata.json"
            self.s3_client.download_file(
                settings.S3_BUCKET,
                f"models/{model_id}/metadata.json",
                str(metadata_path)
            )
            
            # Read metadata to find all artifacts
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Download each artifact
            for artifact_name, artifact_info in metadata.get("artifacts", {}).items():
                # Extract filename from stored path
                artifact_filename = Path(artifact_info["path"]).name
                s3_key = f"models/{model_id}/{artifact_filename}"
                local_path = cache_path / artifact_filename
                
                self.s3_client.download_file(
                    settings.S3_BUCKET,
                    s3_key,
                    str(local_path)
                )
            
            # Update cache metadata
            self.cache_metadata[model_id] = {
                "cached_at": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat(),
                "size_bytes": sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
            }
            self._save_cache_metadata()
            
            print(f"✅ Downloaded model {model_id} from S3 to cache")
            return cache_path
            
        except Exception as e:
            # Clean up partial download
            if cache_path.exists():
                shutil.rmtree(cache_path)
            raise Exception(f"Failed to download model {model_id} from S3: {str(e)}")
    
    def _upload_to_s3(self, model_id: str, local_path: Path):
        """Upload model from local path to S3."""
        if not self.s3_client:
            return  # Skip S3 upload if not configured
        
        try:
            # Upload all files in the model directory
            for file_path in local_path.rglob("*"):
                if file_path.is_file():
                    s3_key = f"models/{model_id}/{file_path.name}"
                    self.s3_client.upload_file(
                        str(file_path),
                        settings.S3_BUCKET,
                        s3_key
                    )
            
            print(f"✅ Uploaded model {model_id} to S3")
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to upload model {model_id} to S3: {str(e)}")
            # Don't fail the operation - local save still succeeded
    
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
        # Check if we need to download
        needs_download = (
            force_refresh or 
            not self._is_model_cached(model_id) or 
            not self._is_cache_fresh(model_id)
        )
        
        if needs_download and settings.STORAGE_TYPE == "s3":
            self._download_from_s3(model_id)
        elif not self._is_model_cached(model_id):
            raise ValueError(f"Model {model_id} not found in cache or S3")
        
        # Update access time
        if model_id in self.cache_metadata:
            self.cache_metadata[model_id]["last_accessed"] = datetime.now().isoformat()
            self._save_cache_metadata()
        
        cache_path = self._get_model_cache_path(model_id)
        print(f"📁 Serving model {model_id} from cache: {cache_path}")
        
        # Return the cache directory's parent so SDK can find models/model_id structure
        # The cache_path points to cache/models/model_id, so we need cache/ directory
        return cache_path.parent.parent
    
    def save_model_from_sdk(self, model_id: str, sdk_dir: Path) -> Path:
        """
        Save model from SDK to cache and optionally S3.
        
        Args:
            model_id: Model identifier  
            sdk_dir: Directory where SDK saved the model
            
        Returns:
            Path to cached model directory
        """
        # Copy from SDK temp directory to our cache
        sdk_model_path = sdk_dir / "models" / model_id
        cache_path = self._get_model_cache_path(model_id)
        
        if cache_path.exists():
            shutil.rmtree(cache_path)
        
        shutil.copytree(sdk_model_path, cache_path)
        
        # Fix the metadata paths to point to the cache directory
        metadata_file = cache_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Update artifact paths to be relative to the cache directory
            for artifact_name, artifact_info in metadata.get("artifacts", {}).items():
                # Get the filename from the original path
                original_path = Path(artifact_info["path"])
                filename = original_path.name
                
                # Update to the new cache path
                new_path = cache_path / filename
                artifact_info["path"] = str(new_path)
            
            # Save the updated metadata
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Update cache metadata
        self.cache_metadata[model_id] = {
            "cached_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "size_bytes": sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
        }
        self._save_cache_metadata()
        
        # Upload to S3 if configured
        if settings.STORAGE_TYPE == "s3":
            self._upload_to_s3(model_id, cache_path)
        
        print(f"💾 Cached model {model_id} locally")
        return cache_path
    
    def cleanup_old_cache(self, max_age_days: int = 7, max_size_gb: float = 10.0):
        """
        Clean up old cached models to manage disk space.
        
        Args:
            max_age_days: Remove models older than this
            max_size_gb: If cache exceeds this size, remove least recently used
        """
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        # Remove old models
        for model_id, metadata in list(self.cache_metadata.items()):
            last_accessed = datetime.fromisoformat(metadata["last_accessed"])
            if last_accessed < cutoff_date:
                cache_path = self._get_model_cache_path(model_id)
                if cache_path.exists():
                    shutil.rmtree(cache_path)
                del self.cache_metadata[model_id]
                print(f"🗑️ Removed old cached model: {model_id}")
        
        # Check total cache size
        total_size_bytes = sum(m["size_bytes"] for m in self.cache_metadata.values())
        max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        
        if total_size_bytes > max_size_bytes:
            # Remove least recently used models
            sorted_models = sorted(
                self.cache_metadata.items(),
                key=lambda x: x[1]["last_accessed"]
            )
            
            for model_id, metadata in sorted_models:
                cache_path = self._get_model_cache_path(model_id)
                if cache_path.exists():
                    shutil.rmtree(cache_path)
                total_size_bytes -= metadata["size_bytes"]
                del self.cache_metadata[model_id]
                print(f"🗑️ Removed LRU cached model: {model_id}")
                
                if total_size_bytes <= max_size_bytes:
                    break
        
        self._save_cache_metadata()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_models = len(self.cache_metadata)
        total_size_bytes = sum(m["size_bytes"] for m in self.cache_metadata.values())
        total_size_mb = total_size_bytes / (1024 * 1024)
        
        return {
            "total_models": total_models,
            "total_size_mb": round(total_size_mb, 2),
            "cache_dir": str(self.cache_dir),
            "storage_type": settings.STORAGE_TYPE
        } 