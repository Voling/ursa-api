from fastapi import APIRouter, HTTPException, Path, Depends
from app.schemas.api_schemas import GraphCreate, GraphResponse
from app.db.database import get_db
from app.db.repositories.graphs import GraphRepository
from sqlalchemy.orm import Session
from typing import List

router = APIRouter()

@router.post("/projects/{project_id}/graphs", response_model=GraphResponse)
def create_graph(
    project_id: str,
    graph_data: GraphCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new graph in a project.
    """
    graph_repo = GraphRepository(db)
    graph = graph_repo.create_graph(
        project_id=project_id,
        name=graph_data.name,
        description=graph_data.description
    )
    
    if not graph:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    
    return GraphResponse(graph_id=graph.id)

@router.get("/projects/{project_id}/graphs", response_model=List[str])
def get_project_graphs(
    project_id: str,
    db: Session = Depends(get_db)
):
    """
    Retrieve all graphs in a project.
    """
    graph_repo = GraphRepository(db)
    graphs = graph_repo.get_project_graphs(project_id)
    
    # Return list of graph IDs
    return [graph.id for graph in graphs] 