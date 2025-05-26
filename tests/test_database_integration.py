"""
Database integration tests using PostgreSQL.
"""
import pytest
from sqlalchemy import text
from app.db.repositories.projects import ProjectRepository
from app.db.repositories.graphs import GraphRepository
from app.db.repositories.nodes import NodeRepository
from app.db.repositories.models import ModelRepository


class TestDatabaseIntegration:
    """Test that the database actually works with real PostgreSQL."""
    
    def test_database_connection(self, isolated_db):
        """Test that we can connect to PostgreSQL."""
        with isolated_db() as db:
            # Execute a simple query
            result = db.execute(text("SELECT 1 as test_value")).fetchone()
            assert result.test_value == 1
    
    def test_project_repository_crud(self, isolated_db):
        """Test project repository CRUD operations."""
        with isolated_db() as db:
            project_repo = ProjectRepository(db)
            
            # Create project
            project = project_repo.create_project("Test Project", "Test Description")
            assert project.name == "Test Project"
            assert project.id is not None
            
            # Read project
            fetched_project = project_repo.get_project(project.id)
            assert fetched_project.name == "Test Project"
            
            # List projects
            all_projects = project_repo.get_all_projects()
            assert len(all_projects) == 1
            assert all_projects[0].name == "Test Project"
            
            # Delete project
            success = project_repo.delete_project(project.id)
            assert success is True
            
            # Verify deletion
            deleted_project = project_repo.get_project(project.id)
            assert deleted_project is None
    
    def test_graph_repository_crud(self, isolated_db):
        """Test graph repository CRUD operations."""
        with isolated_db() as db:
            project_repo = ProjectRepository(db)
            graph_repo = GraphRepository(db)
            
            # Create project first
            project = project_repo.create_project("Test Project", "Test Description")
            
            # Create graph
            graph = graph_repo.create_graph(project.id, "Test Graph", "Test Graph Description")
            assert graph.name == "Test Graph"
            assert graph.project_id == project.id
            
            # Get graphs for project
            graphs = graph_repo.get_project_graphs(project.id)
            assert len(graphs) == 1
            assert graphs[0].name == "Test Graph"
    
    def test_model_repository_crud(self, isolated_db):
        """Test model repository CRUD operations."""
        with isolated_db() as db:
            model_repo = ModelRepository(db)
            
            # Create model with proper signature
            import base64
            test_data = b"fake_model_data_for_testing"
            file_base64 = base64.b64encode(test_data).decode('utf-8')
            
            model = model_repo.create_model(
                file_base64=file_base64,
                name="Test Model",
                framework="scikit-learn",
                model_type="RandomForestClassifier"
            )
            assert model.name == "Test Model"
            assert model.framework == "scikit-learn"
            
            # Get model
            fetched_model = model_repo.get_model(model.id)
            assert fetched_model.name == "Test Model"
    
    def test_node_repository_with_model(self, isolated_db):
        """Test node repository with model relationships."""
        with isolated_db() as db:
            project_repo = ProjectRepository(db)
            graph_repo = GraphRepository(db)
            node_repo = NodeRepository(db)
            model_repo = ModelRepository(db)
            
            # Create dependencies
            project = project_repo.create_project("Test Project", "Test Description")
            graph = graph_repo.create_graph(project.id, "Test Graph", "Test Graph Description")
            
            # Create model with proper signature
            import base64
            test_data = b"fake_model_data_for_testing"
            file_base64 = base64.b64encode(test_data).decode('utf-8')
            
            model = model_repo.create_model(
                file_base64=file_base64,
                name="Test Model",
                framework="scikit-learn",
                model_type="RandomForestClassifier"
            )
            
            # Create node with model
            node = node_repo.create_node(graph.id, "Test Node", model.id)
            assert node.name == "Test Node"
            assert node.model_id == model.id
            
            # Verify relationship
            assert node.model.name == "Test Model"
            assert node.graph.name == "Test Graph"
    
    def test_metrics_repository(self, isolated_db):
        """Test metrics repository operations."""
        with isolated_db() as db:
            project_repo = ProjectRepository(db)
            graph_repo = GraphRepository(db)
            node_repo = NodeRepository(db)
            
            # Create dependencies
            project = project_repo.create_project("Test Project", "Test Description")
            graph = graph_repo.create_graph(project.id, "Test Graph", "Test Graph Description")
            node = node_repo.create_node(graph.id, "Test Node")
            
            # Add metrics
            node_repo.add_metrics(
                node_id=node.id,
                accuracy=0.95,
                loss=0.05,
                epochs=10,
                additional_metrics={"precision": 0.94, "recall": 0.96}
            )
            
            # Get metrics
            metrics = node_repo.get_node_metrics(node.id)
            assert len(metrics) == 1
            assert metrics[0].accuracy == 0.95
            assert metrics[0].additional_metrics["precision"] == 0.94
    
    def test_database_constraints(self, isolated_db):
        """Test database constraints and foreign key relationships."""
        with isolated_db() as db:
            project_repo = ProjectRepository(db)
            graph_repo = GraphRepository(db)
            
            # Create project
            project = project_repo.create_project("Test Project", "Test Description")
            
            # Try to create graph with invalid project_id
            invalid_graph = graph_repo.create_graph("invalid-project-id", "Test Graph", "Description")
            assert invalid_graph is None  # Should return None for invalid project_id
            
            # Delete project should cascade to graphs
            graph = graph_repo.create_graph(project.id, "Test Graph", "Description")
            project_repo.delete_project(project.id)
            
            # Graph should be deleted due to cascade
            graphs = graph_repo.get_project_graphs(project.id)
            assert len(graphs) == 0 