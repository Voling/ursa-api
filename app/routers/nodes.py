from fastapi import APIRouter, HTTPException, Path, Depends
from app.schemas.api_schemas import NodeDelete, NodeUpdate, NodeResponse, GraphStructure, Node, Edge
from app.db.database import get_db
from app.db.repositories.nodes import NodeRepository
from sqlalchemy.orm import Session
from typing import List

router = APIRouter()

@router.delete("/projects/{project_id}/graphs/{graph_id}/nodes/{node_id}")
def delete_node(
    project_id: str,
    graph_id: str,
    node_id: str,
    node_data: NodeDelete,
    db: Session = Depends(get_db)
):
    """
    Delete a node from the knowledge graph.
    """
    node_repo = NodeRepository(db)
    success = node_repo.delete_node(node_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Node not found")
    
    return NodeResponse(success=True)

@router.put("/projects/{project_id}/graphs/{graph_id}/nodes/{node_id}")
def update_node(
    project_id: str,
    graph_id: str,
    node_id: str,
    node_data: NodeUpdate,
    db: Session = Depends(get_db)
):
    """
    Update node attributes in the knowledge graph.
    """
    node_repo = NodeRepository(db)
    node = node_repo.update_node(node_id, node_data.metadata)
    
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    
    return NodeResponse(success=True)

@router.put("/projects/{project_id}/graphs/{graph_id}/nodes/{node_id}/model")
def replace_node_model(
    project_id: str,
    graph_id: str,
    node_id: str,
    node_data: NodeUpdate,
    db: Session = Depends(get_db)
):
    """
    Swap model within node in knowledge graph.
    """
    node_repo = NodeRepository(db)
    from app.db.repositories.models import ModelRepository
    model_repo = ModelRepository(db)
    
    # Check if model exists
    model = model_repo.get_model(node_data.metadata.get("model_id"))
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Update node with new model
    node = node_repo.replace_node_model(node_id, model.id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    
    return NodeResponse(success=True)

@router.get("/projects/{project_id}/graphs/{graph_id}/nodes", response_model=GraphStructure)
def get_nodes(
    project_id: str,
    graph_id: str,
    db: Session = Depends(get_db)
):
    """
    Retrieve full information of nodes and edges of knowledge graph.
    """
    node_repo = NodeRepository(db)
    
    # Get nodes and edges for the graph
    nodes = node_repo.get_graph_nodes(graph_id)
    edges = node_repo.get_graph_edges(graph_id)
    
    # Convert to schema format
    node_schemas = [
        Node(
            id=node.id, 
            name=node.name, 
            model_id=node.model_id or ""
        ) 
        for node in nodes
    ]
    
    edge_schemas = [
        Edge(
            source=edge.source_id, 
            target=edge.target_id, 
            type=edge.type
        ) 
        for edge in edges
    ]
    
    return GraphStructure(
        nodes=node_schemas,
        edges=edge_schemas
    ) 