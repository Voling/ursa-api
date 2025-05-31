"""
Health check endpoints for the API.
"""
from fastapi import APIRouter, Depends
from typing import Dict, Any
from datetime import datetime

from app.services.model_cache_service import ModelCacheService
from app.ursaml import UrsaMLStorage
from app.config import settings

router = APIRouter()

def get_storage():
    """Get UrsaML storage instance."""
    return UrsaMLStorage(base_path=settings.URSAML_STORAGE_DIR)

def get_cache_service():
    """Get model cache service instance."""
    return ModelCacheService()

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.
    Returns API status and version information.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT
    }

@router.get("/health/storage")
async def storage_health(
    storage: UrsaMLStorage = Depends(get_storage)
) -> Dict[str, Any]:
    """
    Check UrsaML storage health.
    Verifies storage directories and metadata are accessible.
    """
    try:
        # Check metadata access
        metadata = storage._load_metadata()
        
        # Check directory structure
        dirs_status = {
            "projects": storage.projects_path.exists(),
            "graphs": storage.graphs_path.exists(),
            "models": storage.models_path.exists()
        }
        
        # Get basic stats
        stats = {
            "total_projects": len(metadata.get("projects", {})),
            "total_graphs": len(metadata.get("graphs", {})),
            "total_models": len(metadata.get("models", {}))
        }
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "directories": dirs_status,
            "stats": stats
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.get("/health/cache")
async def cache_health(
    cache_service: ModelCacheService = Depends(get_cache_service)
) -> Dict[str, Any]:
    """
    Check model cache health.
    Verifies cache service is operational and returns cache statistics.
    """
    try:
        # Get cache stats
        stats = cache_service.get_cache_stats()
        
        # Check cache directory structure
        cache_dir = cache_service.cache_dir
        dirs_status = {
            "cache_root": cache_dir.exists(),
            "models": (cache_dir / "models").exists(),
            "metadata": cache_service.cache_metadata_file.exists()
        }
        
        # Check S3 connectivity if configured
        s3_status = "not_configured"
        if settings.STORAGE_TYPE == "s3":
            try:
                # Try to list S3 bucket contents
                cache_service.s3_client.list_objects_v2(
                    Bucket=settings.S3_BUCKET,
                    MaxKeys=1
                )
                s3_status = "connected"
            except Exception as e:
                s3_status = f"error: {str(e)}"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "directories": dirs_status,
            "stats": stats,
            "s3_status": s3_status
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.get("/health/detailed")
async def detailed_health(
    storage: UrsaMLStorage = Depends(get_storage),
    cache_service: ModelCacheService = Depends(get_cache_service)
) -> Dict[str, Any]:
    """
    Detailed health check of all system components.
    """
    # Get component health
    storage_health_check = await storage_health(storage)
    cache_health_check = await cache_health(cache_service)
    basic_health = await health_check()
    
    # Determine overall status
    overall_status = "healthy"
    if (storage_health_check.get("status") == "unhealthy" or 
        cache_health_check.get("status") == "unhealthy"):
        overall_status = "unhealthy"
    
    return {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "api": basic_health,
        "storage": storage_health_check,
        "cache": cache_health_check,
        "config": {
            "storage_type": settings.STORAGE_TYPE,
            "environment": settings.ENVIRONMENT,
            "debug": settings.DEBUG
        }
    } 