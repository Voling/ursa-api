from fastapi import APIRouter, HTTPException, Path, Depends
from app.schemas.api_schemas import ModelUpload, ModelResponse, ModelDetail
from app.db.database import get_db
from app.db.repositories.models import ModelRepository
from sqlalchemy.orm import Session
from typing import Dict
from datetime import datetime

router = APIRouter()

@router.post("/models/", response_model=ModelResponse, status_code=200)
def save_model(model_data: ModelUpload, db: Session = Depends(get_db)):
    """
    Upload and save a serialized ML model.
    """
    try:
        # Create repository
        model_repo = ModelRepository(db)
        
        # Create model
        model = model_repo.create_model(
            file_base64=model_data.file,
            name=f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            framework="unknown",
            model_type="unknown"
        )
        
        # Create node for the model
        from app.db.repositories.nodes import NodeRepository
        node_repo = NodeRepository(db)
        node = node_repo.create_node(
            graph_id=model_data.graph_id,
            name=model.name,
            model_id=model.id
        )
        
        if not node:
            raise HTTPException(status_code=404, detail=f"Graph not found: {model_data.graph_id}")
        
        # Return response with model information
        return ModelResponse(
            model_id=model.id,
            node_id=node.id,
            name=model.name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_id}", response_model=ModelDetail)
def load_model(model_id: str = Path(..., title="The ID of the model to retrieve"), 
                    db: Session = Depends(get_db)):
    """
    Load a model and return its metadata.
    """
    model_repo = ModelRepository(db)
    model = model_repo.get_model(model_id)
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return ModelDetail(
        model_id=model.id,
        framework=model.framework or "unknown",
        model_type=model.model_type or "unknown",
        created_at=model.created_at
    ) 