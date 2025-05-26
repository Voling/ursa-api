"""
ML Model Management Endpoints with Intelligent Caching.

These endpoints handle actual trained ML models (scikit-learn, PyTorch, TensorFlow, etc.)
using the ursakit SDK and smart caching to minimize S3 costs while maximizing performance.

Note: These are different from:
- Database models (app.db.models) - schema definitions
- API schemas (app.schemas.api_schemas) - request/response structures
"""

import tempfile
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.repositories.models import ModelRepository
from app.services.model_cache_service import ModelCacheService
from app.schemas.api_schemas import ModelUpload, ModelResponse
from ursakit.client import UrsaClient

router = APIRouter()

# Global cache service instance
cache_service = ModelCacheService()


@router.post("/models/", response_model=ModelResponse)
async def create_model(
    request: ModelUpload,
    db: Session = Depends(get_db)
):
    """
    Save a new model.
    Flow: PWA → ursa-api → SDK (temp) → Cache → S3 (async)
    """
    model_repo = ModelRepository(db)
    
    try:
        # 1. Create temporary directory for SDK
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 2. Initialize SDK with temp directory
            sdk_client = UrsaClient(dir=temp_path, use_server=False)
            
            # 3. Deserialize model from request (implementation depends on your format)
            model = deserialize_model_from_request(request)
            
            # 4. Use SDK to save model locally in temp
            model_id = sdk_client.save(model, name=request.name)
            
            # 5. Move from temp to cache and upload to S3
            cache_path = cache_service.save_model_from_sdk(model_id, temp_path)
            
            # 6. Store metadata in database
            db_model = model_repo.create_model({
                "name": request.name,
                "framework": "auto",  # SDK detects this
                "model_type": "auto",  # SDK detects this
                "ursa_model_id": model_id,
                "storage_path": f"models/{model_id}",  # S3 path
                "cache_path": str(cache_path)  # Local cache path
            })
            
            return ModelResponse(
                id=db_model.id,
                name=db_model.name,
                framework=db_model.framework,
                model_type=db_model.model_type,
                ursa_model_id=model_id,
                created_at=db_model.created_at
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save model: {str(e)}")


@router.get("/models/{model_id}", response_model=dict)
async def get_model(
    model_id: str,
    force_refresh: bool = False,
    db: Session = Depends(get_db)
):
    """
    Load a model.
    Flow: PWA → ursa-api → Cache (or S3 if not cached) → SDK → Response
    """
    model_repo = ModelRepository(db)
    
    try:
        # 1. Check if model exists in database
        db_model = model_repo.get_model_by_ursa_id(model_id)
        if not db_model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # 2. Get model from cache (downloads from S3 if needed)
        cache_dir = cache_service.get_model_for_sdk(model_id, force_refresh=force_refresh)
        
        # 3. Use SDK to load model from cache
        sdk_client = UrsaClient(dir=cache_dir, use_server=False)
        model = sdk_client.load(model_id)
        
        # 4. Serialize model for response (implementation depends on your needs)
        model_data = serialize_model_for_response(model)
        
        return {
            "model_id": model_id,
            "name": db_model.name,
            "framework": db_model.framework,
            "model_type": db_model.model_type,
            "model_data": model_data,
            "served_from_cache": True  # Always true due to caching layer
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@router.get("/models/{model_id}/predict")
async def predict_with_model(
    model_id: str,
    input_data: dict,
    db: Session = Depends(get_db)
):
    """
    Make predictions with a cached model.
    This is where caching really shines - frequent predictions with no S3 cost.
    """
    try:
        # Get model from cache (very fast, no S3 download)
        cache_dir = cache_service.get_model_for_sdk(model_id)
        
        # Load model using SDK
        sdk_client = UrsaClient(dir=cache_dir, use_server=False)
        model = sdk_client.load(model_id)
        
        # Make prediction (implementation depends on model type)
        prediction = make_prediction(model, input_data)
        
        return {
            "model_id": model_id,
            "prediction": prediction,
            "served_from_cache": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics for monitoring."""
    return cache_service.get_cache_stats()


@router.post("/cache/cleanup")
async def cleanup_cache():
    """Manual cache cleanup endpoint."""
    try:
        cache_service.cleanup_old_cache(max_age_days=7, max_size_gb=10.0)
        return {"message": "Cache cleanup completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache cleanup failed: {str(e)}")


@router.delete("/cache/models/{model_id}")
async def remove_from_cache(model_id: str):
    """Remove specific model from cache (force S3 download on next access)."""
    try:
        cache_path = cache_service._get_model_cache_path(model_id)
        if cache_path.exists():
            import shutil
            shutil.rmtree(cache_path)
            if model_id in cache_service.cache_metadata:
                del cache_service.cache_metadata[model_id]
                cache_service._save_cache_metadata()
            return {"message": f"Model {model_id} removed from cache"}
        else:
            raise HTTPException(status_code=404, detail="Model not in cache")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove from cache: {str(e)}")


# Helper functions (implementation depends on your specific needs)

def deserialize_model_from_request(request: ModelUpload):
    """Convert API request to model object for SDK."""
    # Implementation depends on how you're sending models from PWA
    # Could be:
    # - Pickled and base64 encoded
    # - JSON representation for simple models
    # - File upload
    pass

def serialize_model_for_response(model):
    """Convert model object to API response format."""
    # Implementation depends on how PWA expects to receive models
    # Could be:
    # - Model summary/metadata only
    # - Pickled and base64 encoded
    # - JSON representation for simple models
    pass

def make_prediction(model, input_data: dict):
    """Make prediction with loaded model."""
    # Implementation depends on model type and input format
    # Examples:
    # - sklearn: model.predict(input_array)
    # - torch: model(input_tensor)
    # - custom: model.predict(input_data)
    pass 