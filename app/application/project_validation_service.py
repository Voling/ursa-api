"""Service for project validation logic."""
from __future__ import annotations

from typing import List, Dict, Any

from app.domain.ports import StoragePort
from app.domain.errors import ValidationError, ConflictError


class ProjectValidationService:
    """Validates project operations."""

    def __init__(self, storage: StoragePort) -> None:
        self._storage = storage

    def validate_name(self, name: str) -> str:
        """Validate and normalize project name."""
        if not name or not name.strip():
            raise ValidationError("Project name is required and cannot be empty")
        return name.strip()

    def check_duplicate_name(self, name: str, exclude_id: str | None = None) -> None:
        """Check if project name already exists."""
        existing_projects = self._storage.get_all_projects()
        for project in existing_projects:
            if project["name"].lower() == name.lower():
                if exclude_id is None or project["id"] != exclude_id:
                    raise ConflictError(f"Project with name '{name}' already exists")

