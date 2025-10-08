"""Adapter handling SDK layout preparation using UrsaClient."""
from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from ursakit.client import UrsaClient

from app.domain.errors import ValidationError
from app.domain.strategies import SerializationStrategyFactory


class ModelIngestionResult(NamedTuple):
    """Result of model ingestion with SDK layout prepared."""

    model_id: str
    model_name: str
    created_at: str
    sdk_dir: Path
    framework: str


class ModelIngestionAdapter:
    """Prepares model artifacts using UrsaSDK with pluggable serialization."""

    def __init__(self, sdk_dir: Path, framework: str = "pickle"):
        """
        Args:
            sdk_dir: Root directory for UrsaSDK storage
            framework: Default serialization framework (pickle, pytorch, tensorflow, onnx)
        """
        self.sdk_client = UrsaClient(dir=sdk_dir, use_server=False)
        self.default_framework = framework

    def prepare(self, file_b64: str, framework: str | None = None) -> ModelIngestionResult:
        """
        Decode model data, deserialize using strategy, save via UrsaClient.
        
        Args:
            file_b64: Base64-encoded model data
            framework: Serialization framework (uses default if None)
        
        Returns:
            ModelIngestionResult with model_id and metadata
        """
        # Decode base64
        try:
            model_bytes = base64.b64decode(file_b64)
        except Exception as exc:  # noqa: BLE001
            raise ValidationError("Invalid base64 model data") from exc

        # Determine serialization strategy
        framework = framework or self.default_framework
        serializer = SerializationStrategyFactory.get_strategy(framework)

        # Deserialize model object
        try:
            model_obj = serializer.deserialize(model_bytes)
        except Exception as exc:  # noqa: BLE001
            raise ValidationError(f"Failed to deserialize model with {framework} strategy") from exc

        # Generate model name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"model_{timestamp}"
        created_at = datetime.now().isoformat()

        # Use UrsaSDK to save model - it handles directory structure, metadata, framework detection
        try:
            model_id = self.sdk_client.save(model_obj, name=model_name)
        except Exception as exc:  # noqa: BLE001
            raise ValidationError(f"Failed to save model via UrsaSDK: {exc}") from exc

        return ModelIngestionResult(
            model_id=model_id,
            model_name=model_name,
            created_at=created_at,
            sdk_dir=self.sdk_client.get_ursa_dir(),
            framework=serializer.get_framework_name(),
        )

