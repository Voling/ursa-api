"""Service for graph validation logic."""
from __future__ import annotations

from typing import List, Dict, Any

from app.domain.ports import StoragePort
from app.domain.errors import ValidationError, ConflictError


class GraphValidationService:
    """Validates graph operations."""

    def __init__(self, storage: StoragePort) -> None:
        self._storage = storage

    def validate_name(self, name: str) -> str:
        """Validate and normalize graph name."""
        if not name or not name.strip():
            raise ValidationError("Graph name is required and cannot be empty")
        return name.strip()

    def check_duplicate_name_in_project(
        self, project_id: str, name: str, exclude_id: str | None = None
    ) -> None:
        """Check if graph name already exists in project."""
        existing_graphs = self._storage.get_project_graphs(project_id)
        for graph in existing_graphs:
            if graph["name"].lower() == name.lower():
                if exclude_id is None or graph["id"] != exclude_id:
                    raise ConflictError(
                        f"Graph with name '{name}' already exists in this project"
                    )

