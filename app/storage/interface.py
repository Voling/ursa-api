from abc import ABC, abstractmethod
from typing import BinaryIO, Optional

class ModelStorage(ABC):
    """
    Abstract interface for model storage. Supports both S3 and local filesystem.
    """
    
    @abstractmethod
    def save_model(self, model_data: bytes, model_id: str) -> str:
        """
        Save model data to storage and return the path.
        
        Args:
            model_data: Binary model data
            model_id: Unique identifier for the model
            
        Returns:
            Storage path where the model was saved
        """
        pass
    
    @abstractmethod
    def get_model(self, storage_path: str) -> bytes:
        """
        Retrieve model data from storage.
        
        Args:
            storage_path: Path to the model in storage
            
        Returns:
            Binary model data
        """
        pass
    
    @abstractmethod
    def delete_model(self, storage_path: str) -> bool:
        """
        Delete a model from storage.
        
        Args:
            storage_path: Path to the model in storage
            
        Returns:
            True if successfully deleted, False otherwise
        """
        pass 