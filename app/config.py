import os
from typing import Optional
from pydantic import PostgresDsn
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class Settings(BaseSettings):
    """Centralized application settings, loaded from environment variables."""
    
    # Database settings
    DATABASE_URL: PostgresDsn = "postgresql://postgres:postgres@localhost:5432/ursa"
    
    # API settings
    API_PORT: int = 6422
    API_HOST: str = "0.0.0.0"
    API_RELOAD: bool = True
    
    # Storage settings
    STORAGE_TYPE: str = "filesystem"  # "s3" or "filesystem"
    MODEL_STORAGE_DIR: Optional[str] = "./models"  # For filesystem storage
    
    # S3 settings
    S3_BUCKET_NAME: Optional[str] = None
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True
settings = Settings()

def get_db_components():
    """Parse database URL into components."""
    db_url = str(settings.DATABASE_URL)
    db_components = db_url.split("/")
    db_name = db_components[-1]
    # Create URL to postgres db (without specific database name)
    db_url_without_name = "/".join(db_components[:-1]) + "/postgres"
    
    return {
        "db_name": db_name,
        "db_url_without_name": db_url_without_name,
        "db_url": db_url
    } 