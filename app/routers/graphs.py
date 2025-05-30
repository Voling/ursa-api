from fastapi import APIRouter, HTTPException, Path, Depends
from app.schemas.api_schemas import GraphCreate, GraphResponse
from app.ursaml import UrsaMLStorage
from typing import List, Dict, Any
from app.config import settings

router = APIRouter()

def get_storage():
    """Get UrsaML storage instance."""
    return UrsaMLStorage(base_path=settings.URSAML_STORAGE_DIR)

@router.post("/projects/{project_id}/graphs", response_model=GraphResponse, status_code=201)
def create_graph(project_id: str, graph_data: GraphCreate, storage: UrsaMLStorage = Depends(get_storage)):
    """
    Create a new graph in a project.
    """
    try:
        # Validate input data
        if not graph_data.name or not graph_data.name.strip():
            raise HTTPException(status_code=400, detail="Graph name is required and cannot be empty")
        
        # Validate project exists
        project = storage.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
        
        # Trim whitespace
        name = graph_data.name.strip()
        description = graph_data.description.strip() if graph_data.description else ""
        
        # Check for duplicate graph names within the project
        existing_graphs = storage.get_project_graphs(project_id)
        if any(g["name"].lower() == name.lower() for g in existing_graphs):
            raise HTTPException(status_code=400, detail=f"Graph with name '{name}' already exists in this project")
        
        # Create the graph
        graph = storage.create_graph(
            project_id=project_id,
            name=name,
            description=description
        )
        
        if not graph:
            raise HTTPException(status_code=500, detail="Failed to create graph")
        
        return GraphResponse(
            graph_id=graph["id"],
            name=graph["name"],
            description=graph.get("description", ""),
            project_id=graph["project_id"],
            created_at=graph["created_at"]
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/projects/{project_id}/graphs")
def get_project_graphs(project_id: str, storage: UrsaMLStorage = Depends(get_storage)) -> List[Dict[str, Any]]:
    """
    Retrieve all graphs in a project with detailed information.
    """
    # Validate project exists
    project = storage.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    
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
    storage: UrsaMLStorage = Depends(get_storage)
) -> Dict[str, Any]:
    """
    Get detailed information about a specific graph.
    """
    graph = storage.get_graph(graph_id)
    
    if not graph:
        raise HTTPException(status_code=404, detail=f"Graph not found: {graph_id}")
    
    if graph["project_id"] != project_id:
        raise HTTPException(status_code=400, detail="Graph does not belong to specified project")
    
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
    storage: UrsaMLStorage = Depends(get_storage)
):
    """
    Update a graph's name and description.
    """
    # Validate graph exists and belongs to project
    graph = storage.get_graph(graph_id)
    if not graph:
        raise HTTPException(status_code=404, detail=f"Graph not found: {graph_id}")
    
    if graph["project_id"] != project_id:
        raise HTTPException(status_code=400, detail="Graph does not belong to specified project")
    
    # Validate input data
    if not graph_data.name or not graph_data.name.strip():
        raise HTTPException(status_code=400, detail="Graph name is required and cannot be empty")
    
    # Trim whitespace
    name = graph_data.name.strip()
    description = graph_data.description.strip() if graph_data.description else ""
    
    # Check for duplicate names within the project (excluding current graph)
    existing_graphs = storage.get_project_graphs(project_id)
    if any(g["name"].lower() == name.lower() and g["id"] != graph_id for g in existing_graphs):
        raise HTTPException(status_code=400, detail=f"Graph with name '{name}' already exists in this project")
    
    # Update the graph
    updated_graph = storage.update_graph(graph_id, name, description)
    if not updated_graph:
        raise HTTPException(status_code=500, detail="Failed to update graph")
    
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
    storage: UrsaMLStorage = Depends(get_storage)
):
    """
    Delete a graph and all its associated nodes and edges.
    """
    # Validate graph exists and belongs to project
    graph = storage.get_graph(graph_id)
    if not graph:
        raise HTTPException(status_code=404, detail=f"Graph not found: {graph_id}")
    
    if graph["project_id"] != project_id:
        raise HTTPException(status_code=400, detail="Graph does not belong to specified project")
    
    # Delete the graph (this will cascade to nodes, edges, etc.)
    success = storage.delete_graph(graph_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete graph")
    
    return {"success": True, "graph_id": graph_id} 