import os
from app.storage.interface import ModelStorage
from app.storage.filesystem import FilesystemStorage
from app.storage.s3 import S3Storage
from app.config import settings

def get_storage() -> ModelStorage:
    """
    Factory function to create the appropriate storage implementation
    based on environment variables.
    
    Returns:
        A storage implementation (S3 or Filesystem)
    """
    # Determine which storage to use
    storage_type = settings.STORAGE_TYPE.lower()
    
    if storage_type == "s3":
        # Get S3 configuration
        bucket_name = settings.S3_BUCKET_NAME
        aws_access_key_id = settings.AWS_ACCESS_KEY_ID
        aws_secret_access_key = settings.AWS_SECRET_ACCESS_KEY
        region_name = settings.AWS_REGION
        
        if not bucket_name:
            raise ValueError("S3_BUCKET_NAME must be set when using S3 storage")
        
        return S3Storage(
            bucket_name=bucket_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
    else:
        # Use filesystem storage
        base_dir = settings.MODEL_STORAGE_DIR
        return FilesystemStorage(base_dir=base_dir) 