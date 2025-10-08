"""Service for graph access validation and ownership checks."""
from __future__ import annotations

from app.domain.ports import StoragePort
from app.domain.errors import NotFoundError, ValidationError


class GraphAccessService:
    """Centralizes graph/project validation to avoid controller duplication."""

    def __init__(self, storage: StoragePort) -> None:
        self._storage = storage

    def require_project_exists(self, project_id: str) -> None:
        """Raise NotFoundError if project doesn't exist."""
        project = self._storage.get_project(project_id)
        if not project:
            raise NotFoundError(f"Project not found: {project_id}")

    def require_graph_exists(self, graph_id: str) -> None:
        """Raise NotFoundError if graph doesn't exist."""
        graph = self._storage.get_graph(graph_id)
        if not graph:
            raise NotFoundError(f"Graph not found: {graph_id}")

    def require_graph_in_project(self, project_id: str, graph_id: str) -> None:
        """Raise NotFoundError if graph doesn't exist, ValidationError if wrong project."""
        self.require_project_exists(project_id)
        graph = self._storage.get_graph(graph_id)
        if not graph:
            raise NotFoundError(f"Graph not found: {graph_id}")
        if graph["project_id"] != project_id:
            raise ValidationError("Graph does not belong to specified project")

    def require_node_exists(self, graph_id: str, node_id: str) -> None:
        """Raise NotFoundError if node doesn't exist in graph."""
        node = self._storage.get_node(graph_id, node_id)
        if not node:
            raise NotFoundError(f"Node not found: {node_id}")

