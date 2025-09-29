from fastapi import APIRouter, HTTPException, Path as FastAPIPath, Depends
from app.schemas.api_schemas import ModelUpload, ModelResponse, ModelDetail
from app.ursaml import UrsaMLStorage
from typing import Dict
from datetime import datetime
import base64
import uuid
from pathlib import Path
import pickle
from app.config import settings, REPO_ROOT
import json
from app.dependencies import get_cache_manager, get_ursaml_storage, get_model_app_service
from app.services.cache.cache_manager import ModelCacheManager
from app.services.model_app_service import ModelAppService

router = APIRouter()

def get_storage():
    return get_ursaml_storage()

@router.post("/models/", response_model=ModelResponse, status_code=201)
def save_model(
    model_data: ModelUpload,
    service: ModelAppService = Depends(get_model_app_service)
):
    """
    Upload and save a serialized ML model.
    """
    try:
        result = service.upload_model(model_data.file, model_data.graph_id)
        return ModelResponse(
            model_id=result["model_id"],
            node_id=result["node_id"],
            name=result["name"],
            statistics={
                "framework": "unknown",
                "model_type": "unknown",
                "created_at": result["created_at"],
                "storage_type": "file"
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_id}", response_model=ModelDetail)
def get_model(
    model_id: str = FastAPIPath(..., title="The ID of the model to retrieve"),
    cache_service: ModelCacheManager = Depends(get_cache_manager)
):
    """
    Get model metadata by ID.
    """
    try:
        # Get model from cache
        model_dir = cache_service.get_model_for_sdk(model_id)
        
        # Read metadata
        metadata_filename = "metadata.json"
        with open(model_dir / "models" / model_id / metadata_filename, 'r') as f:
            metadata = json.load(f)
        
        return ModelDetail(
            model_id=model_id,
            framework="unknown",  # Will be detected by SDK
            model_type="unknown",  # Will be detected by SDK
            created_at=datetime.fromisoformat(metadata["created_at"])
        )
    except Exception:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

@router.get("/models/{model_id}/data")
def load_model_data(
    model_id: str = FastAPIPath(..., title="The ID of the model to load"),
    cache_service: ModelCacheManager = Depends(get_cache_manager)
):
    """
    Load model binary data by ID.
    """
    try:
        # Get model from cache
        model_dir = cache_service.get_model_for_sdk(model_id)
        
        # Read metadata to get model path
        metadata_filename = "metadata.json"
        metadata_path = model_dir / "models" / model_id / metadata_filename
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get model path from metadata
        if "path" not in metadata:
            raise HTTPException(status_code=500, detail="Model metadata missing path")
            
        model_path = Path(metadata["path"])
        if not model_path.exists():
            # Try relative to model directory
            model_path = model_dir / "models" / model_id / model_path.name
            if not model_path.exists():
                raise HTTPException(status_code=404, detail="Model file not found")
        
        # Read model file
        with open(model_path, 'rb') as f:
            model_data = f.read()
        
        # Return base64 encoded data
        return {
            "model_id": model_id,
            "data": base64.b64encode(model_data).decode('utf-8'),
            "framework": metadata.get("framework", "unknown"),
            "model_type": metadata.get("model_type", "unknown")
        }
    except Exception:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

@router.delete("/models/{model_id}")
def delete_model(
    model_id: str = FastAPIPath(..., title="The ID of the model to delete"),
    cache_service: ModelCacheManager = Depends(get_cache_manager)
):
    """
    Delete a model and its associated data.
    """
    try:
        # Delete from cache
        success = cache_service.delete_model(model_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete model")
        
        return {"success": True, "model_id": model_id}
    except Exception:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}") 