from __future__ import annotations

import base64
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from app.config import REPO_ROOT
from app.ursaml.storage import UrsaMLStorage
from app.services.cache.cache_manager import ModelCacheManager


class ModelAppService:
    """Application service orchestrating model upload and graph registration.

    Responsibilities:
      - Decode and persist inbound model artifact into SDK-compatible layout
      - Delegate caching/persistence to ModelCacheManager
      - Register the model as a node in the UrsaML graph
    """

    def __init__(self, storage: UrsaMLStorage, cache: ModelCacheManager) -> None:
        self._storage = storage
        self._cache = cache

    def upload_model(self, file_b64: str, graph_id: str) -> Dict[str, Any]:
        if not file_b64:
            raise ValueError("Model file data is required")
        if not graph_id:
            raise ValueError("Graph ID is required")

        # Validate graph exists
        graph = self._storage.get_graph(graph_id)
        if not graph:
            raise KeyError(f"Graph not found: {graph_id}")

        # Decode
        try:
            model_bytes = base64.b64decode(file_b64)
        except Exception as exc:  # noqa: BLE001 - map all decode errors uniformly
            raise ValueError("Invalid base64 model data") from exc

        # Prepare SDK layout
        model_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"model_{timestamp}"

        sdk_dir = REPO_ROOT / "storage" / "models"
        models_dir = sdk_dir / "models"
        model_dir = models_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "id": model_id,
            "name": model_name,
            "created_at": datetime.now().isoformat(),
            "framework": "unknown",
            "model_type": "unknown",
            "artifacts": {
                "model": {
                    "path": str(model_dir / "model"),
                    "type": "unknown"
                }
            },
            "serializer": "unknown",
            "path": str(model_dir / "model"),
            "metadata": {}
        }
        with (model_dir / "metadata.json").open('w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        with (model_dir / "model").open('wb') as f:
            f.write(model_bytes)

        # Cache persist
        self._cache.save_model_from_sdk(model_id, sdk_dir)

        # Graph node registration
        node = self._storage.create_node(
            graph_id=graph_id,
            name=model_name,
            model_id=model_id,
        )

        if not node:
            # Rollback cached model if node creation fails
            self._cache.delete_model(model_id)
            raise RuntimeError("Failed to create node for model")

        return {
            "model_id": model_id,
            "node_id": node["id"],
            "name": model_name,
            "created_at": metadata["created_at"],
        }


