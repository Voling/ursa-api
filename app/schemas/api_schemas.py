"""
API Request/Response Schemas using Pydantic.

These schemas define the structure of HTTP requests and responses for the ursa-api.
They are NOT related to:
- Database models (see app.db.models)
- ML models managed by ursakit (the actual trained models we store/cache)
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

# Model schemas
class ModelUpload(BaseModel):
    file: str  # base64 encoded model
    project_id: str
    graph_id: str

class ModelResponse(BaseModel):
    model_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    node_id: str
    name: str
    statistics: Dict[str, Any] = Field(default_factory=lambda: {"metrics": "{\"accuracy\": 0.94, \"loss\": 0.03}"})

class ModelDetail(BaseModel):
    model_id: str
    framework: str
    model_type: str
    created_at: datetime

# Metrics schemas
class MetricsUpload(BaseModel):
    model_id: str
    metrics: str

class MetricsResponse(BaseModel):
    success: bool = True

class NodeMetrics(BaseModel):
    accuracy: float
    loss: float
    epochs: int
    timestamp: str

class AllNodeMetricsResponse(BaseModel):
    graph_id: str
    metrics: Dict[str, Dict[str, Any]]

# Node schemas
class NodeDelete(BaseModel):
    model_id: str

class NodeUpdate(BaseModel):
    node_id: str
    metadata: Dict[str, Any]

class NodeModelUpdate(BaseModel):
    node_id: str
    model_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class NodeResponse(BaseModel):
    success: bool = True

class Node(BaseModel):
    id: str
    name: str
    model_id: str

class Edge(BaseModel):
    source: str
    target: str
    type: str

class GraphStructure(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

# Project schemas
class ProjectCreate(BaseModel):
    name: str
    description: str

class ProjectResponse(BaseModel):
    project_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class ProjectDetail(BaseModel):
    project_id: str
    name: str
    created_at: str

class ProjectDeleteResponse(BaseModel):
    success: bool = True

# Graph schemas
class GraphCreate(BaseModel):
    name: str
    description: str

class GraphResponse(BaseModel):
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4())) 