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
from app.dependencies import get_cache_manager
from app.services.cache.cache_manager import ModelCacheManager

router = APIRouter()

def get_storage():
    """Get UrsaML storage instance."""
    return UrsaMLStorage(base_path=settings.URSAML_STORAGE_DIR)

@router.post("/models/", response_model=ModelResponse, status_code=201)
def save_model(
    model_data: ModelUpload,
    storage: UrsaMLStorage = Depends(get_storage),
    cache_service: ModelCacheManager = Depends(get_cache_manager)
):
    """
    Upload and save a serialized ML model.
    """
    try:
        # Validate input data
        if not model_data.file:
            raise HTTPException(status_code=400, detail="Model file data is required")
        
        if not model_data.graph_id:
            raise HTTPException(status_code=400, detail="Graph ID is required")
        
        # Validate base64 encoding
        try:
            model_bytes = base64.b64decode(model_data.file)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 model data")
        
        # Validate graph exists
        graph = storage.get_graph(model_data.graph_id)
        if not graph:
            raise HTTPException(status_code=404, detail=f"Graph not found: {model_data.graph_id}")
        
        # Generate model ID and name
        model_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"model_{timestamp}"
        
        # Use repository storage for the model
        sdk_dir = REPO_ROOT / "storage" / "models"
        sdk_dir.mkdir(parents=True, exist_ok=True)
        models_dir = sdk_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model file
        model_dir = models_dir / model_id
        model_dir.mkdir(parents=True)
        
        # Save model metadata
        metadata = {
            "id": model_id,
            "name": model_name,
            "created_at": datetime.now().isoformat(),
            "framework": "unknown",  # Will be detected by SDK
            "model_type": "unknown",  # Will be detected by SDK
            "artifacts": {
                "model": {
                    "path": str(model_dir / "model"),  # Let SDK determine extension
                    "type": "unknown"  # Let SDK determine type
                }
            },
            "serializer": "unknown",  # Let SDK determine serializer
            "path": str(model_dir / "model"),  # Let SDK determine extension
            "metadata": {}
        }
        
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save model file
        model_file = model_dir / "model"  # Let SDK determine extension
        with open(model_file, 'wb') as f:
            f.write(model_bytes)
        
        # Cache the model
        cache_service.save_model_from_sdk(model_id, sdk_dir)
        
        # Create node for the model
        node = storage.create_node(
            graph_id=model_data.graph_id,
            name=model_name,
            model_id=model_id
        )
        
        if not node:
            # Cleanup the model if node creation fails
            cache_service.delete_model(model_id)
            raise HTTPException(status_code=500, detail="Failed to create node for model")
        
        # Return response with complete model information
        return ModelResponse(
            model_id=model_id,
            node_id=node["id"],
            name=model_name,
            statistics={
                "framework": "unknown",  # Will be detected by SDK
                "model_type": "unknown",  # Will be detected by SDK
                "created_at": metadata["created_at"],
                "storage_type": "file"
            }
        )
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