from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    """Application settings."""
    
    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    # Storage settings
    STORAGE_TYPE: str = "filesystem"  # "filesystem" or "s3"
    URSAML_STORAGE_DIR: str = str(Path.home() / ".ursa" / "storage")
    MODEL_STORAGE_DIR: str = str(Path.home() / ".ursa" / "models")
    
    # S3 settings (only used if STORAGE_TYPE = "s3")
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-1"
    S3_BUCKET: str = "ursa-models"
    
    class Config:
        env_file = ".env"

settings = Settings() 