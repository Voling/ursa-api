"""
API Request/Response Schemas using Pydantic.

Structure of HTTP requests and responses for Ursa API.
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime

# Model schemas
class ModelUpload(BaseModel):
    file: str = Field(..., description="Base64 encoded model data")
    project_id: str = Field(..., description="ID of the project")
    graph_id: str = Field(..., description="ID of the graph to add the model to")

class ModelResponse(BaseModel):
    model_id: str = Field(..., description="Unique identifier for the created model")
    node_id: str = Field(..., description="ID of the node created for this model")
    name: str = Field(..., description="Generated name for the model")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Model metadata and statistics")

class ModelDetail(BaseModel):
    model_id: str = Field(..., description="Unique identifier for the model")
    framework: str = Field(..., description="ML framework used (e.g., sklearn, pytorch, tensorflow)")
    model_type: str = Field(..., description="Type of model (e.g., classifier, regressor)")
    created_at: datetime = Field(..., description="Timestamp when model was created")

# Metrics schemas
class MetricsUpload(BaseModel):
    model_id: str = Field(..., description="ID of the model/node to associate metrics with")
    graph_id: str = Field(..., description="ID of the graph containing the node")
    metrics: str = Field(..., description="JSON string containing metrics data")

class MetricsResponse(BaseModel):
    success: bool = Field(default=True, description="Whether the operation was successful")

class NodeMetrics(BaseModel):
    accuracy: Optional[float] = Field(None, description="Model accuracy score")
    loss: Optional[float] = Field(None, description="Model loss value")
    epochs: Optional[int] = Field(None, description="Number of training epochs")
    timestamp: Optional[str] = Field(None, description="ISO timestamp of metrics recording")
    additional_metrics: Dict[str, Any] = Field(default_factory=dict, description="Additional custom metrics")

class AllNodeMetricsResponse(BaseModel):
    graph_id: str = Field(..., description="ID of the graph")
    metrics: Dict[str, Dict[str, Any]] = Field(..., description="Metrics for each node in the graph")

# Node schemas
class NodeDelete(BaseModel):
    model_id: str = Field(..., description="ID of the model associated with the node")

class NodeUpdate(BaseModel):
    node_id: str = Field(..., description="ID of the node to update")
    metadata: Dict[str, Any] = Field(..., description="Metadata to update for the node")

class NodeModelUpdate(BaseModel):
    node_id: str = Field(..., description="ID of the node")
    model_id: str = Field(..., description="ID of the new model to associate")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class NodeResponse(BaseModel):
    success: bool = Field(default=True, description="Whether the operation was successful")

class Node(BaseModel):
    id: str = Field(..., description="Unique identifier for the node")
    name: str = Field(..., description="Display name of the node")
    model_id: str = Field(..., description="ID of the associated model, empty string if none")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Node metadata including metrics")

class Edge(BaseModel):
    source: str = Field(..., description="ID of the source node")
    target: str = Field(..., description="ID of the target node")
    type: str = Field(..., description="Type/label of the edge relationship")
    weight: float = Field(default=1.0, description="Edge weight")

class GraphStructure(BaseModel):
    nodes: List[Node] = Field(..., description="List of nodes in the graph")
    edges: List[Edge] = Field(..., description="List of edges connecting nodes")

# Project schemas
class ProjectCreate(BaseModel):
    name: str = Field(..., description="Name of the project", min_length=1, max_length=255)
    description: str = Field("", description="Optional description of the project", max_length=1000)

class ProjectResponse(BaseModel):
    project_id: str = Field(..., description="Unique identifier for the created project")

class ProjectDetail(BaseModel):
    project_id: str = Field(..., description="Unique identifier for the project")
    name: str = Field(..., description="Name of the project")
    created_at: str = Field(..., description="ISO format creation timestamp")
    description: Optional[str] = Field(None, description="Project description")

class ProjectDeleteResponse(BaseModel):
    success: bool = Field(default=True, description="Whether the deletion was successful")

# Graph schemas
class GraphCreate(BaseModel):
    name: str = Field(..., description="Name of the graph", min_length=1, max_length=255)
    description: str = Field("", description="Optional description of the graph", max_length=1000)

class GraphResponse(BaseModel):
    graph_id: str = Field(..., description="Unique identifier for the created graph")
    name: str = Field(..., description="Name of the graph")
    description: Optional[str] = Field(None, description="Graph description")
    project_id: str = Field(..., description="ID of the project containing this graph")
    created_at: str = Field(..., description="ISO format creation timestamp") 