"""Tests for repositories and metadata store."""
from __future__ import annotations

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

from app.ursaml.repositories import (
    ProjectsRepository, GraphsRepository, NodesRepository, ModelsRepository
)
from app.ursaml.metadata import MetadataStore


class TestMetadataStore:
    """Test metadata store functionality."""

    def test_metadata_store_initialization(self):
        """Test metadata store initialization."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            metadata_file = Path(f.name)
        
        try:
            store = MetadataStore(metadata_file)
            
            assert store.metadata_file == metadata_file
            assert store.data == {}
            
        finally:
            metadata_file.unlink(missing_ok=True)

    def test_metadata_store_load_existing_file(self):
        """Test loading existing metadata file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            metadata_file = Path(f.name)
            test_data = {"projects": ["proj1", "proj2"], "graphs": ["graph1"]}
            json.dump(test_data, f)
        
        try:
            store = MetadataStore(metadata_file)
            assert store.data == test_data
            
        finally:
            metadata_file.unlink(missing_ok=True)

    def test_metadata_store_load_nonexistent_file(self):
        """Test loading nonexistent metadata file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_file = Path(temp_dir) / "nonexistent.json"
            
            store = MetadataStore(metadata_file)
            assert store.data == {}

    def test_metadata_store_save(self):
        """Test saving metadata to file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            metadata_file = Path(f.name)
        
        try:
            store = MetadataStore(metadata_file)
            test_data = {"projects": ["proj1"], "graphs": ["graph1"]}
            store.data = test_data
            store.save()
            
            # Verify file was written
            assert metadata_file.exists()
            
            with open(metadata_file, 'r') as f:
                loaded_data = json.load(f)
            
            assert loaded_data == test_data
            
        finally:
            metadata_file.unlink(missing_ok=True)

    def test_metadata_store_data_property(self):
        """Test metadata data property getter and setter."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            metadata_file = Path(f.name)
        
        try:
            store = MetadataStore(metadata_file)
            
            # Test getter
            assert store.data == {}
            
            # Test setter
            test_data = {"test": "data"}
            store.data = test_data
            assert store.data == test_data
            
        finally:
            metadata_file.unlink(missing_ok=True)


class TestProjectsRepository:
    """Test projects repository functionality."""

    def test_projects_repository_initialization(self):
        """Test projects repository initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            metadata_store = Mock()
            
            repo = ProjectsRepository(base_path, metadata_store)
            
            assert repo._base_path == base_path
            assert repo._projects_path == base_path / "projects"
            assert repo._metadata == metadata_store

    def test_create_project(self):
        """Test creating a new project."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            metadata_store = Mock()
            metadata_store.data = {"projects": {}}
            
            repo = ProjectsRepository(base_path, metadata_store)
            
            result = repo.create("Test Project", "Test description")
            
            assert result["name"] == "Test Project"
            assert result["description"] == "Test description"
            assert "id" in result
            assert "created_at" in result
            
            # Verify project directory was created
            project_dir = repo._projects_path / result["id"]
            assert project_dir.exists()
            
            # Verify info.json was created
            info_file = project_dir / "info.json"
            assert info_file.exists()
            
            # Verify info.json content
            with open(info_file, 'r') as f:
                info_data = json.load(f)
            
            assert info_data["name"] == "Test Project"
            assert info_data["description"] == "Test description"

    def test_get_project(self):
        """Test getting an existing project."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            metadata_store = Mock()
            metadata_store.data = {"projects": {}}
            
            repo = ProjectsRepository(base_path, metadata_store)
            
            # Create a project first
            created = repo.create("Test Project", "Test description")
            project_id = created["id"]
            
            # Get the project
            result = repo.get(project_id)
            
            assert result is not None
            assert result["id"] == project_id
            assert result["name"] == "Test Project"
            assert result["description"] == "Test description"

    def test_get_nonexistent_project(self):
        """Test getting a nonexistent project."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            metadata_store = Mock()
            metadata_store.data = {"projects": {}}
            
            repo = ProjectsRepository(base_path, metadata_store)
            
            result = repo.get("nonexistent-id")
            assert result is None

    def test_get_all_projects(self):
        """Test getting all projects."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            metadata_store = Mock()
            metadata_store.data = {"projects": {}}
            
            repo = ProjectsRepository(base_path, metadata_store)
            
            # Create multiple projects
            project1 = repo.create("Project 1", "Description 1")
            project2 = repo.create("Project 2", "Description 2")
            
            # Get all projects
            all_projects = repo.get_all()
            
            assert len(all_projects) == 2
            project_ids = [p["id"] for p in all_projects]
            assert project1["id"] in project_ids
            assert project2["id"] in project_ids

    def test_update_project(self):
        """Test updating an existing project."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            metadata_store = Mock()
            metadata_store.data = {"projects": {}}
            
            repo = ProjectsRepository(base_path, metadata_store)
            
            # Create a project first
            created = repo.create("Original Name", "Original description")
            project_id = created["id"]
            
            # Update the project
            result = repo.update(project_id, "Updated Name", "Updated description")
            
            assert result is not None
            assert result["name"] == "Updated Name"
            assert result["description"] == "Updated description"
            assert result["id"] == project_id
            
            # Verify the update was persisted
            retrieved = repo.get(project_id)
            assert retrieved["name"] == "Updated Name"
            assert retrieved["description"] == "Updated description"

    def test_delete_project(self):
        """Test deleting a project."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            metadata_store = Mock()
            metadata_store.data = {"projects": {}}
            
            repo = ProjectsRepository(base_path, metadata_store)
            
            # Create a project first
            created = repo.create("Test Project", "Test description")
            project_id = created["id"]
            project_dir = repo._projects_path / project_id
            
            assert project_dir.exists()
            
            # Delete the project
            result = repo.delete(project_id)
            
            assert result is True
            assert not project_dir.exists()
            
            # Verify project is no longer retrievable
            retrieved = repo.get(project_id)
            assert retrieved is None


class TestGraphsRepository:
    """Test graphs repository functionality."""

    def test_graphs_repository_initialization(self):
        """Test graphs repository initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            metadata_store = Mock()
            
            repo = GraphsRepository(base_path, metadata_store)
            
            assert repo._base_path == base_path
            assert repo._graphs_path == base_path / "graphs"
            assert repo._metadata == metadata_store

    def test_create_graph(self):
        """Test creating a new graph."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            metadata_store = Mock()
            metadata_store.data = {"projects": {}, "graphs": {}}
            
            repo = GraphsRepository(base_path, metadata_store)
            
            result = repo.create("proj-123", "Test Graph", "Test description")
            
            assert result["name"] == "Test Graph"
            assert result["description"] == "Test description"
            assert result["project_id"] == "proj-123"
            assert "id" in result
            assert "created_at" in result
            
            # Verify graph directory was created
            graph_dir = repo._graphs_path / result["id"]
            assert graph_dir.exists()

    def test_get_graph(self):
        """Test getting an existing graph."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            metadata_store = Mock()
            metadata_store.data = {"projects": {}, "graphs": {}}
            
            repo = GraphsRepository(base_path, metadata_store)
            
            # Create a graph first
            created = repo.create("proj-123", "Test Graph", "Test description")
            graph_id = created["id"]
            
            # Get the graph
            result = repo.get(graph_id)
            
            assert result is not None
            assert result["id"] == graph_id
            assert result["name"] == "Test Graph"
            assert result["project_id"] == "proj-123"

    def test_get_project_graphs(self):
        """Test getting graphs for a project."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            metadata_store = Mock()
            metadata_store.data = {"projects": {}, "graphs": {}}
            
            repo = GraphsRepository(base_path, metadata_store)
            
            # Create graphs for different projects
            graph1 = repo.create("proj-123", "Graph 1", "Description 1")
            graph2 = repo.create("proj-123", "Graph 2", "Description 2")
            graph3 = repo.create("proj-456", "Graph 3", "Description 3")
            
            # Get graphs for proj-123
            project_graphs = repo.get_by_project("proj-123")
            
            assert len(project_graphs) == 2
            graph_ids = [g["id"] for g in project_graphs]
            assert graph1["id"] in graph_ids
            assert graph2["id"] in graph_ids
            assert graph3["id"] not in graph_ids


class TestNodesRepository:
    """Test nodes repository functionality."""

    def test_nodes_repository_initialization(self):
        """Test nodes repository initialization."""
        graphs_repo = Mock()
        
        repo = NodesRepository(graphs_repo)
        
        assert repo._graphs_repo == graphs_repo

    def test_create_node(self):
        """Test creating a new node."""
        graphs_repo = Mock()
        graphs_repo.get.return_value = {
            "id": "graph-123",
            "name": "Test Graph",
            "project_id": "proj-456"
        }
        
        repo = NodesRepository(graphs_repo)
        
        result = repo.create("graph-123", "Test Node", "model-789")
        
        assert result["name"] == "Test Node"
        assert result["model_id"] == "model-789"
        assert result["graph_id"] == "graph-123"
        assert "id" in result
        assert "created_at" in result

    def test_create_node_graph_not_found(self):
        """Test creating node for nonexistent graph."""
        graphs_repo = Mock()
        graphs_repo.get.return_value = None
        
        repo = NodesRepository(graphs_repo)
        
        result = repo.create("nonexistent-graph", "Test Node", "model-789")
        assert result is None

    def test_get_node(self):
        """Test getting an existing node."""
        graphs_repo = Mock()
        graphs_repo.get.return_value = {
            "id": "graph-123",
            "name": "Test Graph",
            "project_id": "proj-456"
        }
        
        repo = NodesRepository(graphs_repo)
        
        # Create a node first
        created = repo.create("graph-123", "Test Node", "model-789")
        node_id = created["id"]
        
        # Get the node
        result = repo.get("graph-123", node_id)
        
        assert result is not None
        assert result["id"] == node_id
        assert result["name"] == "Test Node"
        assert result["model_id"] == "model-789"

    def test_get_graph_nodes(self):
        """Test getting all nodes for a graph."""
        graphs_repo = Mock()
        graphs_repo.get.return_value = {
            "id": "graph-123",
            "name": "Test Graph",
            "project_id": "proj-456"
        }
        
        repo = NodesRepository(graphs_repo)
        
        # Create multiple nodes
        node1 = repo.create("graph-123", "Node 1", "model-1")
        node2 = repo.create("graph-123", "Node 2", "model-2")
        
        # Get all nodes for the graph
        all_nodes = repo.get_by_graph("graph-123")
        
        assert len(all_nodes) == 2
        node_ids = [n["id"] for n in all_nodes]
        assert node1["id"] in node_ids
        assert node2["id"] in node_ids


class TestModelsRepository:
    """Test models repository functionality."""

    def test_models_repository_initialization(self):
        """Test models repository initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            
            repo = ModelsRepository(base_path)
            
            assert repo._base_path == base_path
            assert repo._models_path == base_path / "models"

    def test_create_model(self):
        """Test creating a new model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            repo = ModelsRepository(base_path)
            
            model_data = {
                "name": "test-model",
                "framework": "scikit-learn",
                "metadata": {"type": "classifier"}
            }
            
            result = repo.create("model-123", model_data)
            
            assert result["id"] == "model-123"
            assert result["name"] == "test-model"
            assert result["framework"] == "scikit-learn"
            
            # Verify model directory was created
            model_dir = repo._models_path / "model-123"
            assert model_dir.exists()
            
            # Verify metadata.json was created
            metadata_file = model_dir / "metadata.json"
            assert metadata_file.exists()

    def test_get_model(self):
        """Test getting an existing model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            repo = ModelsRepository(base_path)
            
            # Create a model first
            model_data = {
                "name": "test-model",
                "framework": "scikit-learn",
                "metadata": {"type": "classifier"}
            }
            created = repo.create("model-123", model_data)
            
            # Get the model
            result = repo.get("model-123")
            
            assert result is not None
            assert result["id"] == "model-123"
            assert result["name"] == "test-model"

    def test_get_nonexistent_model(self):
        """Test getting a nonexistent model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            repo = ModelsRepository(base_path)
            
            result = repo.get("nonexistent-model")
            assert result is None
