import os
import aiofiles
from typing import BinaryIO
from app.storage.interface import ModelStorage

class FilesystemStorage(ModelStorage):
    """
    Implements model storage using the local filesystem.
    """
    
    def __init__(self, base_dir: str = None):
        """
        Initialize filesystem storage.
        
        Args:
            base_dir: Base directory for storing models. 
                      If None, uses 'models' in the current working directory.
        """
        if base_dir is None:
            base_dir = os.path.join(os.getcwd(), "models")
        
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
    
    async def save_model(self, model_data: bytes, model_id: str) -> str:
        """
        Save model data to the filesystem.
        
        Args:
            model_data: Binary model data
            model_id: Unique identifier for the model
            
        Returns:
            Storage path where the model was saved
        """
        # Create directory for this model
        model_dir = os.path.join(self.base_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model file
        file_path = os.path.join(model_dir, "model.bin")
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(model_data)
        
        return file_path
    
    async def get_model(self, storage_path: str) -> bytes:
        """
        Retrieve model data from the filesystem.
        
        Args:
            storage_path: Path to the model in storage
            
        Returns:
            Binary model data
        """
        if not os.path.exists(storage_path):
            raise FileNotFoundError(f"Model not found at path: {storage_path}")
        
        async with aiofiles.open(storage_path, "rb") as f:
            return await f.read()
    
    async def delete_model(self, storage_path: str) -> bool:
        """
        Delete a model from the filesystem.
        
        Args:
            storage_path: Path to the model in storage
            
        Returns:
            True if successfully deleted, False otherwise
        """
        try:
            if os.path.exists(storage_path):
                os.remove(storage_path)
                
                # Try to remove parent directory if it's empty
                parent_dir = os.path.dirname(storage_path)
                if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                    os.rmdir(parent_dir)
                
            return True
        except Exception:
            return False 