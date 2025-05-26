from fastapi import APIRouter, HTTPException, Path, Depends
from app.schemas.api_schemas import ProjectCreate, ProjectResponse, ProjectDetail, ProjectDeleteResponse
from app.db.database import get_db
from app.db.repositories.projects import ProjectRepository
from sqlalchemy.orm import Session
from typing import List

router = APIRouter()

@router.post("/projects/", response_model=ProjectResponse)
def create_project(project_data: ProjectCreate, db: Session = Depends(get_db)):
    """
    Create a project that contains one or more graphs.
    """
    project_repo = ProjectRepository(db)
    project = project_repo.create_project(
        name=project_data.name,
        description=project_data.description
    )
    
    return ProjectResponse(project_id=project.id)

@router.get("/projects", response_model=List[ProjectDetail])
def get_all_projects(db: Session = Depends(get_db)):
    """
    Retrieve all available projects.
    """
    project_repo = ProjectRepository(db)
    projects = project_repo.get_all_projects()
    
    return [
        ProjectDetail(
            project_id=project.id,
            name=project.name,
            created_at=project.created_at.strftime("%m/%d/%Y")
        )
        for project in projects
    ]

@router.delete("/projects/{project_id}", response_model=ProjectDeleteResponse)
def delete_project(project_id: str = Path(..., title="The ID of the project to delete"),
                         db: Session = Depends(get_db)):
    """
    Delete a project.
    """
    project_repo = ProjectRepository(db)
    success = project_repo.delete_project(project_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Project not found")
    
    return ProjectDeleteResponse(success=True) 