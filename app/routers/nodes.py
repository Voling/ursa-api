from fastapi import APIRouter, HTTPException, Path, Depends
from app.schemas.api_schemas import NodeUpdate, NodeResponse, GraphStructure, Node as NodeSchema, Edge
from app.ursaml import UrsaMLStorage
from app.dependencies import get_ursaml_storage
from app.config import settings
from typing import List

router = APIRouter()

def get_storage():
    return get_ursaml_storage()

@router.delete("/projects/{project_id}/graphs/{graph_id}/nodes/{node_id}")
def delete_node(
    project_id: str,
    graph_id: str,
    node_id: str,
    storage: UrsaMLStorage = Depends(get_storage)
):
    """
    Delete a node from the knowledge graph.
    """
    # Validate project exists
    project = storage.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    
    # Validate graph exists and belongs to project
    graph = storage.get_graph(graph_id)
    if not graph:
        raise HTTPException(status_code=404, detail=f"Graph not found: {graph_id}")
    
    if graph["project_id"] != project_id:
        raise HTTPException(status_code=400, detail="Graph does not belong to specified project")
    
    # Validate node exists
    node = storage.get_node(graph_id, node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    
    # Delete the node
    success = storage.delete_node(graph_id, node_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete node")
    
    return NodeResponse(success=True)

@router.put("/projects/{project_id}/graphs/{graph_id}/nodes/{node_id}")
def update_node(
    project_id: str,
    graph_id: str,
    node_id: str,
    node_data: NodeUpdate,
    storage: UrsaMLStorage = Depends(get_storage)
):
    """
    Update node attributes in the knowledge graph.
    """
    # Validate node exists and belongs to the correct graph
    node = storage.get_node(graph_id, node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    
    # Validate metadata is provided
    if not node_data.metadata:
        raise HTTPException(status_code=400, detail="Metadata is required for node update")
    
    # Update the node
    updated_node = storage.update_node(graph_id, node_id, node_data.metadata)
    if not updated_node:
        raise HTTPException(status_code=500, detail="Failed to update node")
    
    return NodeResponse(success=True)

@router.put("/projects/{project_id}/graphs/{graph_id}/nodes/{node_id}/model")
def replace_node_model(
    project_id: str,
    graph_id: str,
    node_id: str,
    node_data: NodeUpdate,
    storage: UrsaMLStorage = Depends(get_storage)
):
    """
    Swap model within node in knowledge graph.
    """
    # Validate project exists
    project = storage.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    
    # Validate graph exists and belongs to project
    graph = storage.get_graph(graph_id)
    if not graph:
        raise HTTPException(status_code=404, detail=f"Graph not found: {graph_id}")
    
    if graph["project_id"] != project_id:
        raise HTTPException(status_code=400, detail="Graph does not belong to specified project")
    
    # Validate node exists
    node = storage.get_node(graph_id, node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    
    # Validate model_id is provided in metadata
    if not node_data.metadata or "model_id" not in node_data.metadata:
        raise HTTPException(status_code=400, detail="model_id is required in metadata")
    
    model_id = node_data.metadata["model_id"]
    
    # Update node with new model
    updated_node = storage.update_node(graph_id, node_id, {"model_id": model_id})
    if not updated_node:
        raise HTTPException(status_code=500, detail="Failed to update node model")
    
    return NodeResponse(success=True)

@router.get("/projects/{project_id}/graphs/{graph_id}/nodes", response_model=GraphStructure)
def get_nodes(
    project_id: str,
    graph_id: str,
    storage: UrsaMLStorage = Depends(get_storage)
):
    """
    Retrieve full information of nodes and edges of knowledge graph.
    """
    # Validate graph exists
    graph = storage.get_graph(graph_id)
    if not graph:
        raise HTTPException(status_code=404, detail=f"Graph not found: {graph_id}")
    
    if graph["project_id"] != project_id:
        raise HTTPException(status_code=400, detail="Graph does not belong to specified project")
    
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
    storage: UrsaMLStorage = Depends(get_storage)
):
    """
    Create a new node in the knowledge graph.
    """
    # Validate graph exists
    graph = storage.get_graph(graph_id)
    if not graph:
        raise HTTPException(status_code=404, detail=f"Graph not found: {graph_id}")
    
    if graph["project_id"] != project_id:
        raise HTTPException(status_code=400, detail="Graph does not belong to specified project")
    
    # Validate required fields
    if "name" not in node_data:
        raise HTTPException(status_code=400, detail="Node name is required")
    
    # Create the node
    node = storage.create_node(
        graph_id=graph_id,
        name=node_data["name"],
        model_id=node_data.get("model_id")
    )
    
    if not node:
        raise HTTPException(status_code=500, detail="Failed to create node")
    
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
    storage: UrsaMLStorage = Depends(get_storage)
):
    """
    Get detailed information about a specific node.
    """
    # Validate node exists and belongs to the correct graph
    node = storage.get_node(graph_id, node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    
    return {
        "id": node["id"],
        "name": node["name"],
        "model_id": node["model_id"],
        "graph_id": node["graph_id"],
        "metadata": node["metadata"]
    } 