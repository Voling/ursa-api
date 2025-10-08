from fastapi import APIRouter, Path, Depends
from app.schemas.api_schemas import ProjectCreate, ProjectResponse, ProjectDetail, ProjectDeleteResponse
from app.dependencies import get_ursaml_storage, get_project_validation_service
from app.domain.ports import StoragePort
from app.application.project_validation_service import ProjectValidationService
from app.domain.errors import NotFoundError
from typing import List

router = APIRouter()

@router.post("/projects/", response_model=ProjectResponse, status_code=201)
def create_project(
    project_data: ProjectCreate,
    storage: StoragePort = Depends(get_ursaml_storage),
    validator: ProjectValidationService = Depends(get_project_validation_service)
):
    """
    Create a project that contains one or more graphs.
    """
    # Validate and normalize
    name = validator.validate_name(project_data.name)
    validator.check_duplicate_name(name)
    description = project_data.description.strip() if project_data.description else ""
    
    # Create the project
    project = storage.create_project(name=name, description=description)
    return ProjectResponse(project_id=project["id"])

@router.get("/projects", response_model=List[ProjectDetail])
def get_all_projects(storage: StoragePort = Depends(get_ursaml_storage)):
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
    storage: StoragePort = Depends(get_ursaml_storage)
):
    """
    Get a specific project by ID.
    """
    project = storage.get_project(project_id)
    if not project:
        raise NotFoundError(f"Project not found: {project_id}")
    
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
    storage: StoragePort = Depends(get_ursaml_storage),
    validator: ProjectValidationService = Depends(get_project_validation_service)
):
    """
    Update a project's name and description.
    """
    # Validate project exists
    project = storage.get_project(project_id)
    if not project:
        raise NotFoundError(f"Project not found: {project_id}")
    
    # Validate and normalize
    name = validator.validate_name(project_data.name)
    validator.check_duplicate_name(name, exclude_id=project_id)
    description = project_data.description.strip() if project_data.description else ""
    
    # Update the project
    updated_project = storage.update_project(project_id, name, description)
    
    return {
        "success": True,
        "project_id": project_id,
        "name": updated_project["name"],
        "description": updated_project["description"]
    }

@router.delete("/projects/{project_id}", response_model=ProjectDeleteResponse)
def delete_project(
    project_id: str = Path(..., title="The ID of the project to delete"),
    storage: StoragePort = Depends(get_ursaml_storage)
):
    """
    Delete a project and all its associated graphs, nodes, and models.
    """
    # Validate project exists
    project = storage.get_project(project_id)
    if not project:
        raise NotFoundError(f"Project not found: {project_id}")
    
    # Delete the project (this will cascade to graphs, nodes, etc.)
    storage.delete_project(project_id)
    return ProjectDeleteResponse(success=True) 