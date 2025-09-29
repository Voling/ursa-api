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


