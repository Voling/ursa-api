from fastapi import APIRouter, Path as FastAPIPath, Depends
from app.schemas.api_schemas import ModelUpload, ModelResponse, ModelDetail
from app.domain.errors import NotFoundError
from typing import Dict
from datetime import datetime
import base64
from pathlib import Path
import json
from ursakit.client import UrsaClient
from app.dependencies import get_cache_manager, get_model_app_service
from app.services.cache.cache_manager import ModelCacheManager
from app.services.model_app_service import ModelAppService
from app.config import settings

router = APIRouter()


@router.post("/models/", response_model=ModelResponse, status_code=201)
def save_model(
    model_data: ModelUpload,
    service: ModelAppService = Depends(get_model_app_service)
):
    """
    Upload and save a serialized ML model.
    """
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

@router.get("/models/{model_id}", response_model=ModelDetail)
def get_model(
    model_id: str = FastAPIPath(..., title="The ID of the model to retrieve"),
    cache_service: ModelCacheManager = Depends(get_cache_manager)
):
    """
    Get model metadata by ID using UrsaSDK.
    """
    try:
        # Get model directory from cache
        model_dir = cache_service.get_model_for_sdk(model_id)
        
        # Use UrsaClient to access metadata
        sdk_client = UrsaClient(dir=model_dir)
        metadata = sdk_client.get_metadata(model_id)
        
        return ModelDetail(
            model_id=model_id,
            framework=metadata.get("framework", "unknown"),
            model_type=metadata.get("model_type", "unknown"),
            created_at=datetime.fromisoformat(metadata["created_at"])
        )
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        raise NotFoundError(f"Model not found: {model_id}") from exc

@router.get("/models/{model_id}/data")
def load_model_data(
    model_id: str = FastAPIPath(..., title="The ID of the model to load"),
    cache_service: ModelCacheManager = Depends(get_cache_manager)
):
    """
    Load model binary data by ID using UrsaSDK.
    """
    try:
        # Get model directory from cache
        model_dir = cache_service.get_model_for_sdk(model_id)
        
        # Use UrsaClient to load the model object
        sdk_client = UrsaClient(dir=model_dir)
        model_obj = sdk_client.load(model_id)
        metadata = sdk_client.get_metadata(model_id)
        
        # Serialize the model object back to bytes using pickle (default)
        import pickle
        model_bytes = pickle.dumps(model_obj)
        
        # Return base64 encoded data
        return {
            "model_id": model_id,
            "data": base64.b64encode(model_bytes).decode('utf-8'),
            "framework": metadata.get("framework", "unknown"),
            "model_type": metadata.get("model_type", "unknown")
        }
    except (FileNotFoundError, KeyError) as exc:
        raise NotFoundError(f"Model not found: {model_id}") from exc

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
        cache_service.delete_model(model_id)
        return {"success": True, "model_id": model_id}
    except (FileNotFoundError, KeyError) as exc:
        raise NotFoundError(f"Model not found: {model_id}") from exc 