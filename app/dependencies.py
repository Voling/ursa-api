from __future__ import annotations

import boto3
from app.config import settings, REPO_ROOT
from pathlib import Path

from app.services.cache.cache_manager import ModelCacheManager
from app.services.cache.local_cache import LocalCacheRepository
from app.services.cache.metadata_store import CacheMetadataStore
from app.services.cache.s3_gateway import ModelS3Gateway, NullModelS3Gateway
from app.services.cache.sdk_workspace import SDKWorkspaceManager
from app.services.cache.cache_policy import CachePolicy
from app.ursaml.storage import UrsaMLStorage
from app.services.model_app_service import ModelAppService
from app.application.graph_access_service import GraphAccessService
from app.application.metrics_service import MetricsService
from app.application.project_validation_service import ProjectValidationService
from app.application.graph_validation_service import GraphValidationService
from app.infrastructure.model_ingestion_adapter import ModelIngestionAdapter


def get_cache_manager() -> ModelCacheManager:
    cache_root = Path(settings.MODEL_STORAGE_DIR) / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    metadata_store = CacheMetadataStore(cache_root / "cache_metadata.json")
    local_repo = LocalCacheRepository(cache_root)
    sdk_workspace = SDKWorkspaceManager(REPO_ROOT / "storage" / "sdk_temp")

    s3_enabled = settings.STORAGE_TYPE == "s3"
    if s3_enabled:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        gateway = ModelS3Gateway(s3_client, settings.S3_BUCKET)
    else:
        gateway = NullModelS3Gateway()

    policy = CachePolicy(metadata_store)
    return ModelCacheManager(
        local_cache=local_repo,
        metadata_store=metadata_store,
        sdk_workspace=sdk_workspace,
        policy=policy,
        s3_gateway=gateway,
        s3_enabled=s3_enabled,
    )


def get_ursaml_storage() -> UrsaMLStorage:
    return UrsaMLStorage(base_path=str(settings.URSAML_STORAGE_DIR))


def get_model_app_service() -> ModelAppService:
    sdk_dir = Path(settings.MODEL_STORAGE_DIR)
    return ModelAppService(
        storage=get_ursaml_storage(),
        cache=get_cache_manager(),
        ingestion=ModelIngestionAdapter(sdk_dir=sdk_dir, framework="pickle"),
    )


def get_graph_access_service() -> GraphAccessService:
    return GraphAccessService(storage=get_ursaml_storage())


def get_metrics_service() -> MetricsService:
    return MetricsService(storage=get_ursaml_storage())


def get_project_validation_service() -> ProjectValidationService:
    return ProjectValidationService(storage=get_ursaml_storage())


def get_graph_validation_service() -> GraphValidationService:
    return GraphValidationService(storage=get_ursaml_storage())
