from fastapi import APIRouter, HTTPException, Path, Depends
from app.schemas.api_schemas import ProjectCreate, ProjectResponse, ProjectDetail, ProjectDeleteResponse
from app.ursaml import UrsaMLStorage
from typing import List
from app.config import settings

router = APIRouter()

def get_storage():
    """Get UrsaML storage instance."""
    return UrsaMLStorage(base_path=settings.URSAML_STORAGE_DIR)

@router.post("/projects/", response_model=ProjectResponse, status_code=201)
def create_project(project_data: ProjectCreate, storage: UrsaMLStorage = Depends(get_storage)):
    """
    Create a project that contains one or more graphs.
    """
    try:
        # Validate input data
        if not project_data.name or not project_data.name.strip():
            raise HTTPException(status_code=400, detail="Project name is required and cannot be empty")
        
        # Trim whitespace from name and description
        name = project_data.name.strip()
        description = project_data.description.strip() if project_data.description else ""
        
        # Create the project
        project = storage.create_project(name=name, description=description)
        if not project:
            raise HTTPException(status_code=500, detail="Failed to create project")
        
        return ProjectResponse(project_id=project["id"])
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/projects", response_model=List[ProjectDetail])
def get_all_projects(storage: UrsaMLStorage = Depends(get_storage)):
    """
    Retrieve all available projects.
    """
    projects = storage.get_all_projects()
    
    return [
        ProjectDetail(
            project_id=project["id"],
            name=project["name"],
            created_at=project["created_at"],
            description=project.get("description", "")
        )
        for project in projects
    ]

@router.get("/projects/{project_id}", response_model=ProjectDetail)
def get_project(
    project_id: str = Path(..., title="The ID of the project to retrieve"),
    storage: UrsaMLStorage = Depends(get_storage)
):
    """
    Get a specific project by ID.
    """
    project = storage.get_project(project_id)
    
    if not project:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    
    return ProjectDetail(
        project_id=project["id"],
        name=project["name"],
        created_at=project["created_at"],
        description=project.get("description", "")
    )

@router.put("/projects/{project_id}")
def update_project(
    project_id: str = Path(..., title="The ID of the project to update"),
    project_data: ProjectCreate = None,
    storage: UrsaMLStorage = Depends(get_storage)
):
    """
    Update a project's name and description.
    """
    # Validate project exists
    project = storage.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    
    # Validate input data
    if not project_data or not project_data.name or not project_data.name.strip():
        raise HTTPException(status_code=400, detail="Project name is required and cannot be empty")
    
    # Trim whitespace
    name = project_data.name.strip()
    description = project_data.description.strip() if project_data.description else ""
    
    # Check for duplicate names (excluding current project)
    existing_projects = storage.get_all_projects()
    if any(p["name"].lower() == name.lower() and p["id"] != project_id for p in existing_projects):
        raise HTTPException(status_code=400, detail=f"Project with name '{name}' already exists")
    
    # Update the project
    updated_project = storage.update_project(project_id, name, description)
    if not updated_project:
        raise HTTPException(status_code=500, detail="Failed to update project")
    
    return {
        "success": True,
        "project_id": project_id,
        "name": updated_project["name"],
        "description": updated_project["description"]
    }

@router.delete("/projects/{project_id}", response_model=ProjectDeleteResponse)
def delete_project(
    project_id: str = Path(..., title="The ID of the project to delete"),
    storage: UrsaMLStorage = Depends(get_storage)
):
    """
    Delete a project and all its associated graphs, nodes, and models.
    """
    # Validate project exists
    project = storage.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    
    # Delete the project (this will cascade to graphs, nodes, etc.)
    success = storage.delete_project(project_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete project")
    
    return ProjectDeleteResponse(success=True) 