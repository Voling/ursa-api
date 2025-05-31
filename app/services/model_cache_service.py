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


class ModelCacheService:
    """Service for managing model caching.
    Intended for S3 synchronization
    """
    
    def __init__(self):
        self.cache_dir = Path(settings.MODEL_STORAGE_DIR) / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create models subdirectory
        (self.cache_dir / "models").mkdir(parents=True, exist_ok=True)
        
        # Create temp directory for SDK operations
        self.sdk_temp_dir = REPO_ROOT / "storage" / "sdk_temp"
        self.sdk_temp_dir.mkdir(parents=True, exist_ok=True)
        
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
        self._save_cache_metadata()  # Ensure metadata file exists
    
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
        """Check if model is already cached locally."""
        cache_path = self._get_model_cache_path(model_id)
        metadata_path = cache_path / "metadata.json"
        
        # First check if directory and metadata exist
        if not (cache_path.exists() and metadata_path.exists()):
            return False
            
        # Read metadata to find model file path
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Find model path using metadata
            model_path = self._get_model_path_from_metadata(metadata, cache_path)
            return model_path is not None
            
        except Exception as e:
            print(f"⚠️ Warning: Error reading metadata for model {model_id}: {str(e)}")
            return False
    
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
            if "artifacts" in metadata:
                for artifact_info in metadata["artifacts"].values():
                    if "path" in artifact_info:
                        # Extract filename from stored path
                        artifact_filename = Path(artifact_info["path"]).name
                        s3_key = f"models/{model_id}/{artifact_filename}"
                        local_path = cache_path / artifact_filename
                        
                        self.s3_client.download_file(
                            settings.S3_BUCKET,
                            s3_key,
                            str(local_path)
                        )
            
            # Download main model file if specified in top-level path
            if "path" in metadata:
                model_filename = Path(metadata["path"]).name
                s3_key = f"models/{model_id}/{model_filename}"
                local_path = cache_path / model_filename
                
                self.s3_client.download_file(
                    settings.S3_BUCKET,
                    s3_key,
                    str(local_path)
                )
            
            # Verify we have the model file
            model_path = self._get_model_path_from_metadata(metadata, cache_path)
            if not model_path:
                raise ValueError("No model file found in metadata")
            
            # Update cache metadata
            self.cache_metadata[model_id] = {
                "cached_at": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat(),
                "size_bytes": sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
            }
            self._save_cache_metadata()
            
            print(f"Downloaded model {model_id} from S3 to cache")
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
            
            print(f"Uploaded model {model_id} to S3")
            
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
        
        # Create a temporary directory with the SDK structure
        temp_dir = self._get_sdk_temp_path()
        models_dir = temp_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy from cache to temp SDK structure
        cache_path = self._get_model_cache_path(model_id)
        model_dir = models_dir / model_id
        shutil.copytree(cache_path, model_dir, dirs_exist_ok=True)
        
        # Fix the metadata paths to be relative to the model directory
        metadata_file = model_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Get model path from metadata
            model_path = self._get_model_path_from_metadata(metadata, cache_path)
            if model_path:
                # Update metadata to use relative path
                if "path" in metadata:
                    metadata["path"] = str(model_dir / model_path.name)
                
                # Update artifact paths
                if "artifacts" in metadata:
                    for artifact_info in metadata["artifacts"].values():
                        if "path" in artifact_info:
                            artifact_info["path"] = str(model_dir / Path(artifact_info["path"]).name)
                
                # Save updated metadata
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        print(f"📁 Serving model {model_id} from cache via {temp_dir}")
        return temp_dir
    
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
        
        # Verify source exists
        if not sdk_model_path.exists():
            raise ValueError(f"Model {model_id} not found in SDK directory")
        
        # Get cache path and ensure parent exists
        cache_path = self._get_model_cache_path(model_id)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy model files to cache
        shutil.copytree(sdk_model_path, cache_path, dirs_exist_ok=True)
        
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
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model from cache and optionally S3."""
        cache_path = self._get_model_cache_path(model_id)
        
        # Remove from cache
        if cache_path.exists():
            shutil.rmtree(cache_path)
        
        # Remove from cache metadata
        if model_id in self.cache_metadata:
            del self.cache_metadata[model_id]
            self._save_cache_metadata()
        
        # Remove from S3 if configured
        if settings.STORAGE_TYPE == "s3" and self.s3_client:
            try:
                # List and delete all objects with model_id prefix
                response = self.s3_client.list_objects_v2(
                    Bucket=settings.S3_BUCKET,
                    Prefix=f"models/{model_id}/"
                )
                
                for obj in response.get("Contents", []):
                    self.s3_client.delete_object(
                        Bucket=settings.S3_BUCKET,
                        Key=obj["Key"]
                    )
                    
            except Exception as e:
                print(f"⚠️ Warning: Failed to delete model {model_id} from S3: {str(e)}")
        
        return True
    
    def cleanup_old_cache(self, max_age_days: int = 7, max_size_gb: float = 10.0):
        """Clean up old or oversized cache entries."""
        # Convert max size to bytes
        max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        
        # Get current cache size
        total_size = 0
        for model_id in list(self.cache_metadata.keys()):
            total_size += self.cache_metadata[model_id].get("size_bytes", 0)
        
        # Sort models by last access time
        models_by_access = sorted(
            self.cache_metadata.items(),
            key=lambda x: datetime.fromisoformat(x[1]["last_accessed"])
        )
        
        # Remove old models
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        for model_id, metadata in models_by_access:
            should_delete = False
            
            # Check age - if max_age_days is 0, all models are considered old
            if max_age_days == 0:
                should_delete = True
            else:
                last_accessed = datetime.fromisoformat(metadata["last_accessed"])
                if last_accessed < cutoff_date:
                    should_delete = True
            
            # Check size limit
            if total_size > max_size_bytes:
                should_delete = True
            
            # Delete if either condition is met
            if should_delete:
                size = metadata.get("size_bytes", 0)
                self.delete_model(model_id)
                total_size -= size
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = 0
        for metadata in self.cache_metadata.values():
            total_size += metadata.get("size_bytes", 0)
        
        return {
            "total_models": len(self.cache_metadata),
            "total_size_mb": total_size / (1024 * 1024)
        } 