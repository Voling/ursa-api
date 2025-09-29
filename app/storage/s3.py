import os
import boto3
from botocore.exceptions import ClientError
from typing import BinaryIO
import io
from app.storage.interface import ModelStorage

class S3Storage(ModelStorage):
    """
    Implements model storage using AWS S3.
    """
    
    def __init__(self, bucket_name: str, aws_access_key_id: str = None, 
                 aws_secret_access_key: str = None, region_name: str = None):
        """
        Initialize S3 storage.
        
        Args:
            bucket_name: S3 bucket name
            aws_access_key_id: AWS access key ID (if None, uses environment variables)
            aws_secret_access_key: AWS secret access key (if None, uses environment variables)
            region_name: AWS region name (if None, uses environment variables)
        """
        self.bucket_name = bucket_name
        
        # If credentials are not provided, boto3 will look for them in environment variables
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        
        # Ensure bucket exists
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Ensure the S3 bucket exists, create it if it doesn't."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            if error_code == '404':
                # Bucket doesn't exist, create it
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                # Another error occurred
                raise
    
    def save_model(self, model_data: bytes, model_id: str) -> str:
        """
        Save model data to S3.
        
        Args:
            model_data: Binary model data
            model_id: Unique identifier for the model
            
        Returns:
            Storage path (S3 key) where the model was saved
        """
        s3_key = f"models/{model_id}/model.bin"
        
        # Upload to S3
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=s3_key,
            Body=model_data
        )
        
        return f"s3://{self.bucket_name}/{s3_key}"
    
    def get_model(self, storage_path: str) -> bytes:
        """
        Retrieve model data from S3.
        
        Args:
            storage_path: S3 path (s3://bucket-name/key)
            
        Returns:
            Binary model data
        """
        # Parse S3 path
        if not storage_path.startswith("s3://"):
            raise ValueError(f"Invalid S3 path: {storage_path}")
        
        # Extract bucket and key
        path_parts = storage_path[5:].split("/", 1)
        if len(path_parts) < 2 or path_parts[0] != self.bucket_name:
            raise ValueError(f"Invalid S3 path for this storage: {storage_path}")
        
        s3_key = path_parts[1]
        
        try:
            # Download from S3
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            return response['Body'].read()
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') == 'NoSuchKey':
                raise FileNotFoundError(f"Model not found at path: {storage_path}")
            raise
    
    def delete_model(self, storage_path: str) -> bool:
        """
        Delete a model from S3.
        
        Args:
            storage_path: S3 path (s3://bucket-name/key)
            
        Returns:
            True if successfully deleted, False otherwise
        """
        try:
            # Parse S3 path
            if not storage_path.startswith("s3://"):
                raise ValueError(f"Invalid S3 path: {storage_path}")
            
            # Extract bucket and key
            path_parts = storage_path[5:].split("/", 1)
            if len(path_parts) < 2 or path_parts[0] != self.bucket_name:
                raise ValueError(f"Invalid S3 path for this storage: {storage_path}")
            
            s3_key = path_parts[1]
            
            # Delete from S3
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except Exception:
            return False 