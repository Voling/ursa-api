from fastapi import APIRouter, Path, Depends
from app.schemas.api_schemas import GraphCreate, GraphResponse
from app.dependencies import (
    get_ursaml_storage,
    get_graph_access_service,
    get_graph_validation_service,
)
from app.domain.ports import StoragePort
from app.application.graph_access_service import GraphAccessService
from app.application.graph_validation_service import GraphValidationService
from app.domain.errors import NotFoundError
from typing import List, Dict, Any

router = APIRouter()

@router.post("/projects/{project_id}/graphs", response_model=GraphResponse, status_code=201)
def create_graph(
    project_id: str,
    graph_data: GraphCreate,
    storage: StoragePort = Depends(get_ursaml_storage),
    access_svc: GraphAccessService = Depends(get_graph_access_service),
    validator: GraphValidationService = Depends(get_graph_validation_service),
):
    """
    Create a new graph in a project.
    """
    # Validate project exists
    access_svc.require_project_exists(project_id)
    
    # Validate and normalize
    name = validator.validate_name(graph_data.name)
    validator.check_duplicate_name_in_project(project_id, name)
    description = graph_data.description.strip() if graph_data.description else ""
    
    # Create the graph
    graph = storage.create_graph(project_id=project_id, name=name, description=description)
    
    return GraphResponse(
        graph_id=graph["id"],
        name=graph["name"],
        description=graph.get("description", ""),
        project_id=graph["project_id"],
        created_at=graph["created_at"]
    )

@router.get("/projects/{project_id}/graphs")
def get_project_graphs(
    project_id: str,
    storage: StoragePort = Depends(get_ursaml_storage),
    access_svc: GraphAccessService = Depends(get_graph_access_service),
) -> List[Dict[str, Any]]:
    """
    Retrieve all graphs in a project with detailed information.
    """
    # Validate project exists
    access_svc.require_project_exists(project_id)
    
    graphs = storage.get_project_graphs(project_id)
    
    # Return detailed graph information
    return [
        {
            "graph_id": graph["id"],
            "name": graph["name"],
            "description": graph.get("description", ""),
            "project_id": graph["project_id"],
            "created_at": graph["created_at"]
        }
        for graph in graphs
    ]

@router.get("/projects/{project_id}/graphs/{graph_id}")
def get_graph(
    project_id: str,
    graph_id: str,
    access_svc: GraphAccessService = Depends(get_graph_access_service),
    storage: StoragePort = Depends(get_ursaml_storage)
) -> Dict[str, Any]:
    """
    Get detailed information about a specific graph.
    """
    access_svc.require_graph_in_project(project_id, graph_id)
    graph = storage.get_graph(graph_id)
    
    return {
        "graph_id": graph["id"],
        "name": graph["name"],
        "description": graph.get("description", ""),
        "project_id": graph["project_id"],
        "created_at": graph["created_at"]
    }

@router.put("/projects/{project_id}/graphs/{graph_id}")
def update_graph(
    project_id: str,
    graph_id: str,
    graph_data: GraphCreate,
    access_svc: GraphAccessService = Depends(get_graph_access_service),
    storage: StoragePort = Depends(get_ursaml_storage),
    validator: GraphValidationService = Depends(get_graph_validation_service),
):
    """
    Update a graph's name and description.
    """
    # Validate graph exists and belongs to project
    access_svc.require_graph_in_project(project_id, graph_id)
    
    # Validate and normalize
    name = validator.validate_name(graph_data.name)
    validator.check_duplicate_name_in_project(project_id, name, exclude_id=graph_id)
    description = graph_data.description.strip() if graph_data.description else ""
    
    # Update the graph
    updated_graph = storage.update_graph(graph_id, name, description)
    
    return {
        "success": True,
        "graph_id": graph_id,
        "name": updated_graph["name"],
        "description": updated_graph.get("description", "")
    }

@router.delete("/projects/{project_id}/graphs/{graph_id}")
def delete_graph(
    project_id: str,
    graph_id: str,
    access_svc: GraphAccessService = Depends(get_graph_access_service),
    storage: StoragePort = Depends(get_ursaml_storage)
):
    """
    Delete a graph and all its associated nodes and edges.
    """
    # Validate graph exists and belongs to project
    access_svc.require_graph_in_project(project_id, graph_id)
    
    # Delete the graph (this will cascade to nodes, edges, etc.)
    storage.delete_graph(graph_id)
    return {"success": True, "graph_id": graph_id} 