from sqlalchemy.orm import Session
from app.db.models import Graph, Project
from typing import List, Optional

class GraphRepository:
    """Repository for graph operations."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_graph(self, project_id: str, name: str, description: str = None) -> Optional[Graph]:
        """
        Create a new graph in a project.
        
        Args:
            project_id: Project ID
            name: Graph name
            description: Graph description (optional)
            
        Returns:
            Created graph or None if project not found
        """
        # Verify project exists
        project = self.db.query(Project).filter(Project.id == project_id).first()
        if not project:
            return None
        
        graph = Graph(project_id=project_id, name=name, description=description)
        self.db.add(graph)
        self.db.commit()
        self.db.refresh(graph)
        return graph
    
    def get_graph(self, graph_id: str) -> Optional[Graph]:
        """
        Get a graph by ID.
        
        Args:
            graph_id: Graph ID
            
        Returns:
            Graph if found, None otherwise
        """
        return self.db.query(Graph).filter(Graph.id == graph_id).first()
    
    def get_project_graphs(self, project_id: str) -> List[Graph]:
        """
        Get all graphs in a project.
        
        Args:
            project_id: Project ID
            
        Returns:
            List of graphs in the project
        """
        return self.db.query(Graph).filter(Graph.project_id == project_id).all()
    
    def delete_graph(self, graph_id: str) -> bool:
        """
        Delete a graph by ID.
        
        Args:
            graph_id: Graph ID
            
        Returns:
            True if graph was deleted, False otherwise
        """
        graph = self.get_graph(graph_id)
        if not graph:
            return False
        
        self.db.delete(graph)
        self.db.commit()
        return True 