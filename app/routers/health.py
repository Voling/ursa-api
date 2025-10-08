"""
Health check endpoints for the API.
"""
from fastapi import APIRouter, Depends
from typing import Dict, Any
from datetime import datetime

from app.config import settings
from app.dependencies import get_cache_manager, get_ursaml_storage
from app.services.cache.cache_manager import ModelCacheManager
from app.domain.ports import StoragePort

router = APIRouter()

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
    storage: StoragePort = Depends(get_ursaml_storage)
) -> Dict[str, Any]:
    """
    Check UrsaML storage health.
    Verifies storage directories and metadata are accessible.
    """
    try:
        storage_stats = storage.get_storage_stats()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            **storage_stats,
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.get("/health/cache")
async def cache_health(
    cache_service: ModelCacheManager = Depends(get_cache_manager)
) -> Dict[str, Any]:
    """
    Check model cache health.
    Verifies cache service is operational and returns cache statistics.
    """
    try:
        # Get cache stats via protocol method
        cache_stats = cache_service.get_cache_stats()
        
        # Check cache directory (still needs some concrete access for health)
        cache_dir = cache_service.cache_root
        dirs_status = {
            "cache_root": cache_dir.exists(),
            "models": (cache_dir / "models").exists(),
        }
        
        # Check S3 connectivity if configured
        s3_status = "not_configured"
        if settings.STORAGE_TYPE == "s3":
            s3_status = "enabled"  # Simplified; detailed S3 checks moved to cache layer
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "directories": dirs_status,
            "cache_stats": cache_stats,
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
    storage: StoragePort = Depends(get_ursaml_storage),
    cache_service: ModelCacheManager = Depends(get_cache_manager)
) -> Dict[str, Any]:
    """
    Detailed health check of all system components.
    """
    # Get component health
    storage_health_check = await storage_health(storage)  # type: ignore[arg-type]
    cache_health_check = await cache_health(cache_service)  # type: ignore[arg-type]
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