from pydantic_settings import BaseSettings
from pathlib import Path

# Get the repository root directory (parent of app directory)
REPO_ROOT = Path(__file__).parent.parent.absolute()

class Settings(BaseSettings):
    """Application settings."""
    
    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    # Version and environment
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # Storage settings
    STORAGE_TYPE: str = "filesystem"  # "filesystem" or "s3"
    URSAML_STORAGE_DIR: str = str(REPO_ROOT / "storage" / "ursaml")
    MODEL_STORAGE_DIR: str = str(REPO_ROOT / "storage" / "models")
    
    # S3 settings (only used if STORAGE_TYPE = "s3")
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-1"
    S3_BUCKET: str = "ursa-models"
    
    class Config:
        env_file = ".env"

settings = Settings() 