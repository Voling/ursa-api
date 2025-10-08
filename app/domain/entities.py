"""Internal domain entities as TypedDicts for type safety at boundaries."""
from __future__ import annotations

from typing import Any, Dict, TypedDict


class ProjectEntity(TypedDict):
    id: str
    project_id: str
    name: str
    description: str
    created_at: str
    graphs: list[str]


class GraphEntity(TypedDict):
    id: str
    graph_id: str
    project_id: str
    name: str
    description: str
    created_at: str


class NodeEntity(TypedDict):
    id: str
    graph_id: str
    name: str
    model_id: str | None
    metadata: Dict[str, Any]


class ModelUploadResult(TypedDict):
    model_id: str
    node_id: str
    name: str
    created_at: str

