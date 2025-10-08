from fastapi import APIRouter, Path, Depends
from app.schemas.api_schemas import NodeUpdate, NodeResponse, GraphStructure, Node as NodeSchema, Edge
from app.dependencies import get_ursaml_storage, get_graph_access_service
from app.domain.ports import StoragePort
from app.application.graph_access_service import GraphAccessService
from app.domain.errors import NotFoundError, ValidationError
from typing import List

router = APIRouter()

@router.delete("/projects/{project_id}/graphs/{graph_id}/nodes/{node_id}")
def delete_node(
    project_id: str,
    graph_id: str,
    node_id: str,
    access_svc: GraphAccessService = Depends(get_graph_access_service),
    storage: StoragePort = Depends(get_ursaml_storage)
):
    """
    Delete a node from the knowledge graph.
    """
    # Validate graph exists and belongs to project
    access_svc.require_graph_in_project(project_id, graph_id)
    
    # Validate node exists
    access_svc.require_node_exists(graph_id, node_id)
    
    # Delete the node
    storage.delete_node(graph_id, node_id)
    
    return NodeResponse(success=True)

@router.put("/projects/{project_id}/graphs/{graph_id}/nodes/{node_id}")
def update_node(
    project_id: str,
    graph_id: str,
    node_id: str,
    node_data: NodeUpdate,
    storage: StoragePort = Depends(get_ursaml_storage)
):
    """
    Update node attributes in the knowledge graph.
    """
    # Validate node exists
    access_svc.require_node_exists(graph_id, node_id)
    
    # Validate metadata is provided
    if not node_data.metadata:
        raise ValidationError("Metadata is required for node update")
    
    # Update the node
    storage.update_node(graph_id, node_id, node_data.metadata)
    
    return NodeResponse(success=True)

@router.put("/projects/{project_id}/graphs/{graph_id}/nodes/{node_id}/model")
def replace_node_model(
    project_id: str,
    graph_id: str,
    node_id: str,
    node_data: NodeUpdate,
    access_svc: GraphAccessService = Depends(get_graph_access_service),
    storage: StoragePort = Depends(get_ursaml_storage)
):
    """
    Swap model within node in knowledge graph.
    """
    # Validate graph exists and belongs to project
    access_svc.require_graph_in_project(project_id, graph_id)
    
    # Validate node exists
    access_svc.require_node_exists(graph_id, node_id)
    
    # Validate model_id is provided in metadata
    if not node_data.metadata or "model_id" not in node_data.metadata:
        raise ValidationError("model_id is required in metadata")
    
    model_id = node_data.metadata["model_id"]
    
    # Update node with new model
    storage.update_node(graph_id, node_id, {"model_id": model_id})
    
    return NodeResponse(success=True)

@router.get("/projects/{project_id}/graphs/{graph_id}/nodes", response_model=GraphStructure)
def get_nodes(
    project_id: str,
    graph_id: str,
    access_svc: GraphAccessService = Depends(get_graph_access_service),
    storage: StoragePort = Depends(get_ursaml_storage)
):
    """
    Retrieve full information of nodes and edges of knowledge graph.
    """
    # Validate graph exists and belongs to project
    access_svc.require_graph_in_project(project_id, graph_id)
    
    # Get nodes and edges for the graph
    nodes = storage.get_graph_nodes(graph_id)
    edges = storage.get_graph_edges(graph_id)
    
    # Convert to schema format
    node_schemas = [
        NodeSchema(
            id=node["id"], 
            name=node["name"], 
            model_id=node["model_id"] or "",
            metadata=node["metadata"]
        ) 
        for node in nodes
    ]
    
    edge_schemas = [
        Edge(
            source=edge["source_id"], 
            target=edge["target_id"], 
            type=edge["type"] or "default",
            weight=edge["weight"]
        ) 
        for edge in edges
    ]
    
    return GraphStructure(
        nodes=node_schemas,
        edges=edge_schemas
    )

@router.post("/projects/{project_id}/graphs/{graph_id}/nodes")
def create_node(
    project_id: str,
    graph_id: str,
    node_data: dict,
    access_svc: GraphAccessService = Depends(get_graph_access_service),
    storage: StoragePort = Depends(get_ursaml_storage)
):
    """
    Create a new node in the knowledge graph.
    """
    # Validate graph exists and belongs to project
    access_svc.require_graph_in_project(project_id, graph_id)
    
    # Validate required fields
    if "name" not in node_data:
        raise ValidationError("Node name is required")
    
    # Create the node
    node = storage.create_node(
        graph_id=graph_id,
        name=node_data["name"],
        model_id=node_data.get("model_id")
    )
    
    return {
        "success": True,
        "node_id": node["id"],
        "name": node["name"],
        "model_id": node["model_id"]
    }

@router.get("/projects/{project_id}/graphs/{graph_id}/nodes/{node_id}")
def get_node_detail(
    project_id: str,
    graph_id: str,
    node_id: str,
    storage: StoragePort = Depends(get_ursaml_storage)
):
    """
    Get detailed information about a specific node.
    """
    # Validate node exists
    node = storage.get_node(graph_id, node_id)
    if not node:
        raise NotFoundError(f"Node not found: {node_id}")
    
    return {
        "id": node["id"],
        "name": node["name"],
        "model_id": node["model_id"],
        "graph_id": node["graph_id"],
        "metadata": node["metadata"]
    } 