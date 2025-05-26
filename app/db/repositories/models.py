from sqlalchemy.orm import Session
from app.db.models import Model, Node
import base64
import pickle
from app.storage import get_storage
from typing import List, Optional
import ursakit

class ModelRepository:
    """Repository for model operations."""
    
    def __init__(self, db: Session):
        self.db = db
        self.storage = get_storage()
    
    def create_model(self, file_base64: str, name: str, framework: str = None, model_type: str = None) -> Model:
        """
        Create a new model by decoding base64 data and storing it.
        
        Args:
            file_base64: Base64 encoded model data
            name: Model name
            framework: ML framework (optional)
            model_type: Model type (optional)
            
        Returns:
            Created model
        """
        # Decode base64 model data
        model_data = base64.b64decode(file_base64)
        
        # Try to detect model type using ursa-kit if not provided
        detected_framework = framework
        detected_model_type = model_type
        
        if not framework or not model_type:
            try:
                # Deserialize the model to detect its type
                model_obj = pickle.loads(model_data)
                
                # Use ursa-kit to detect model type
                client = ursakit.client()
                model_info = client.detect_model_type(model_obj)
                
                if model_info.get("is_model", False):
                    if not detected_framework:
                        metadata = model_info.get("metadata", {})
                        detected_framework = metadata.get("framework", "unknown")
                    if not detected_model_type:
                        metadata = model_info.get("metadata", {})
                        detected_model_type = metadata.get("model_class", "unknown")
            except Exception as e:
                print(f"Warning: Could not detect model type: {e}")
                if not detected_framework:
                    detected_framework = "unknown"
                if not detected_model_type:
                    detected_model_type = "unknown"
        
        # Create model record in database
        model = Model(
            name=name,
            framework=detected_framework,
            model_type=detected_model_type,
            storage_path="",  # Will be updated after saving
            storage_type="unknown"  # Will be updated after saving
        )
        
        self.db.add(model)
        self.db.flush()  # Generate ID without committing
        
        # Save model to storage  
        # Note: Storage operations should be made async in the future
        import asyncio
        storage_path = asyncio.run(self.storage.save_model(model_data, model.id))
        
        # Update storage path and type
        model.storage_path = storage_path
        model.storage_type = "s3" if storage_path.startswith("s3://") else "filesystem"
        
        self.db.commit()
        self.db.refresh(model)
        
        return model
    
    def get_model(self, model_id: str) -> Optional[Model]:
        """
        Get a model by ID.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model if found, None otherwise
        """
        return self.db.query(Model).filter(Model.id == model_id).first()
    
    def get_model_data(self, model_id: str) -> bytes:
        """
        Get model binary data by ID.
        
        Args:
            model_id: Model ID
            
        Returns:
            Binary model data
        
        Raises:
            FileNotFoundError: If model not found
            ValueError: If model ID is invalid
        """
        model = self.get_model(model_id)
        if not model:
            raise ValueError(f"Model not found: {model_id}")
        
        # Load model from storage
        return asyncio.run(self.storage.get_model(model.storage_path))
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model by ID.
        
        Args:
            model_id: Model ID
            
        Returns:
            True if model was deleted, False otherwise
        """
        model = self.get_model(model_id)
        if not model:
            return False
        
        # Delete model from storage
        storage_success = asyncio.run(self.storage.delete_model(model.storage_path))
        
        if storage_success:
            # Delete model from database
            self.db.delete(model)
            self.db.commit()
            
        return storage_success
    
    def link_model_to_node(self, model_id: str, node_id: str) -> bool:
        """
        Link a model to a node.
        
        Args:
            model_id: Model ID
            node_id: Node ID
            
        Returns:
            True if link was created, False otherwise
        """
        node = self.db.query(Node).filter(Node.id == node_id).first()
        if not node:
            return False
        
        node.model_id = model_id
        self.db.commit()
        return True 