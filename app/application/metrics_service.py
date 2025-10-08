"""Service for metrics operations, extracted from storage layer."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from app.domain.ports import StoragePort
from app.domain.errors import NotFoundError
from app.domain.events import event_publisher, MetricsRecorded


class MetricsService:
    """Encapsulates metrics domain logic."""

    def __init__(self, storage: StoragePort) -> None:
        self._storage = storage

    def add_node_metrics(
        self, graph_id: str, node_id: str, metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add metrics to a node in a graph."""
        ursaml_data = self._storage.load_graph_ursaml(graph_id)
        if not ursaml_data or node_id not in ursaml_data["nodes"]:
            raise NotFoundError(f"Node {node_id} not found in graph {graph_id}")

        # Update score column if present
        if "accuracy" in metrics and "score" in ursaml_data["nodes"][node_id]["columns"]:
            ursaml_data["nodes"][node_id]["columns"]["score"] = metrics["accuracy"]

        # Add metrics to detailed metadata
        if "meta" not in ursaml_data["nodes"][node_id]["detailed"]:
            ursaml_data["nodes"][node_id]["detailed"]["meta"] = {}

        ursaml_data["nodes"][node_id]["detailed"]["meta"].update(
            {
                "score": metrics.get("accuracy", 0.0),
                "loss": metrics.get("loss", 0.0),
                "epochs": metrics.get("epochs", 0),
                "metrics_timestamp": datetime.now().isoformat(),
            }
        )

        # Add additional metrics
        for key, value in metrics.items():
            if key not in ["accuracy", "loss", "epochs"]:
                ursaml_data["nodes"][node_id]["detailed"]["meta"][key] = value

        self._storage.save_graph_ursaml(graph_id, ursaml_data)
        
        # Publish domain event
        event_publisher.publish(MetricsRecorded(
            event_id="",
            timestamp=None,
            aggregate_id=node_id,
            graph_id=graph_id,
            node_id=node_id,
            metrics=metrics,
        ))
        
        return metrics

