from __future__ import annotations

from typing import Any, Dict

from app.domain.ports import StoragePort, CachePort
from app.domain.errors import ValidationError, NotFoundError
from app.domain.entities import ModelUploadResult
from app.domain.events import event_publisher, ModelUploaded, ModelDeleted
from app.infrastructure.model_ingestion_adapter import ModelIngestionAdapter


class ModelAppService:
    """Application service orchestrating model upload and graph registration.

    Now depends on abstractions (ports) and delegates infrastructure to adapter.
    """

    def __init__(
        self, storage: StoragePort, cache: CachePort, ingestion: ModelIngestionAdapter
    ) -> None:
        self._storage = storage
        self._cache = cache
        self._ingestion = ingestion

    def upload_model(self, file_b64: str, graph_id: str) -> ModelUploadResult:
        if not file_b64:
            raise ValidationError("Model file data is required")
        if not graph_id:
            raise ValidationError("Graph ID is required")

        # Validate graph exists
        graph = self._storage.get_graph(graph_id)
        if not graph:
            raise NotFoundError(f"Graph not found: {graph_id}")

        # Prepare model artifact
        result = self._ingestion.prepare(file_b64)

        # Cache persist
        self._cache.save_model_from_sdk(result.model_id, result.sdk_dir)

        # Graph node registration
        node = self._storage.create_node(
            graph_id=graph_id,
            name=result.model_name,
            model_id=result.model_id,
        )

        if not node:
            # Rollback cached model if node creation fails
            self._cache.delete_model(result.model_id)
            raise RuntimeError("Failed to create node for model")

        # Publish domain event
        event_publisher.publish(ModelUploaded(
            event_id="",
            timestamp=None,
            aggregate_id=result.model_id,
            model_id=result.model_id,
            node_id=node["id"],
            graph_id=graph_id,
            name=result.model_name,
            framework=result.framework,
        ))

        return ModelUploadResult(
            model_id=result.model_id,
            node_id=node["id"],
            name=result.model_name,
            created_at=result.created_at,
        )


