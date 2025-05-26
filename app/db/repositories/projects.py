from sqlalchemy.orm import Session
from app.db.models import Project
from typing import List, Optional

class ProjectRepository:
    """Repository for project operations."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_project(self, name: str, description: str = None) -> Project:
        """
        Create a new project.
        
        Args:
            name: Project name
            description: Project description (optional)
            
        Returns:
            Created project
        """
        project = Project(name=name, description=description)
        self.db.add(project)
        self.db.commit()
        self.db.refresh(project)
        return project
    
    def get_project(self, project_id: str) -> Optional[Project]:
        """
        Get a project by ID.
        
        Args:
            project_id: Project ID
            
        Returns:
            Project if found, None otherwise
        """
        return self.db.query(Project).filter(Project.id == project_id).first()
    
    def get_all_projects(self) -> List[Project]:
        """
        Get all projects.
        
        Returns:
            List of all projects
        """
        return self.db.query(Project).all()
    
    def delete_project(self, project_id: str) -> bool:
        """
        Delete a project by ID.
        
        Args:
            project_id: Project ID
            
        Returns:
            True if project was deleted, False otherwise
        """
        project = self.get_project(project_id)
        if not project:
            return False
        
        self.db.delete(project)
        self.db.commit()
        return True 