import pytest
from fastapi.testclient import TestClient
from app.main import app
import shutil
from pathlib import Path
from app.ursaml import UrsaMLStorage
from app.config import settings, REPO_ROOT

@pytest.fixture(autouse=True)
def clean_storage():
    """Ensure clean storage before each test."""
    # Use repository storage directory
    storage_dir = REPO_ROOT / "storage" / "ursaml"
    storage_dir.mkdir(parents=True, exist_ok=True)
    
    # Override the UrsaMLStorage to use repo directory
    original_init = UrsaMLStorage.__init__
    
    def mock_init(self, base_path=None):
        self.base_path = storage_dir
        self.projects_path = self.base_path / "projects"
        self.graphs_path = self.base_path / "graphs"
        self.models_path = self.base_path / "models"
        self.metadata_file = self.base_path / "metadata.json"
        
        # Create directories
        self.projects_path.mkdir(parents=True, exist_ok=True)
        self.graphs_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Load or create metadata
        self.metadata = self._load_metadata()
    
    UrsaMLStorage.__init__ = mock_init
    
    yield
    
    # Restore original init and cleanup but keep structure
    UrsaMLStorage.__init__ = original_init
    for item in storage_dir.glob("*"):
        if item.name != ".gitkeep":
            if item.is_file():
                item.unlink()
            else:
                shutil.rmtree(item)

client = TestClient(app)

def test_create_project():
    """Test creating a project."""
    response = client.post("/projects/", json={
        "name": "Test Project",
        "description": "A test project"
    })
    assert response.status_code == 201
    data = response.json()
    assert "project_id" in data

def test_get_projects():
    """Test getting all projects."""
    # First create a project
    create_response = client.post("/projects/", json={
        "name": "Test Project for Get",
        "description": "A test project"
    })
    assert create_response.status_code == 201
    project_id = create_response.json()["project_id"]
    
    # Get all projects
    response = client.get("/projects")
    assert response.status_code == 200
    projects = response.json()
    assert len(projects) > 0
    assert any(p["project_id"] == project_id for p in projects)

def test_create_graph():
    """Test creating a graph in a project."""
    # Create a project first
    project_response = client.post("/projects/", json={
        "name": "Graph Test Project",
        "description": "Project for graph testing"
    })
    assert project_response.status_code == 201
    project_id = project_response.json()["project_id"]
    
    # Create a graph
    response = client.post(f"/projects/{project_id}/graphs", json={
        "name": "Test Graph",
        "description": "A test knowledge graph"
    })
    assert response.status_code == 201
    data = response.json()
    assert "graph_id" in data

def test_save_model():
    """Test saving a model."""
    import base64
    import pickle
    
    # Create project
    project_response = client.post("/projects/", json={
        "name": "Model Test Project",
        "description": "Project for model testing"
    })
    assert project_response.status_code == 201
    project_id = project_response.json()["project_id"]
    
    # Create graph
    graph_response = client.post(f"/projects/{project_id}/graphs", json={
        "name": "Model Test Graph",
        "description": "Graph for model testing"
    })
    assert graph_response.status_code == 201
    graph_id = graph_response.json()["graph_id"]
    
    # Create a test model
    model = {"type": "test_model", "data": [1, 2, 3]}
    model_bytes = pickle.dumps(model)
    model_base64 = base64.b64encode(model_bytes).decode('utf-8')
    
    # Save the model
    response = client.post("/models/", json={
        "file": model_base64,
        "project_id": project_id,
        "graph_id": graph_id
    })
    assert response.status_code == 201
    data = response.json()
    assert "model_id" in data
    assert "node_id" in data

def test_get_graph_nodes():
    """Test getting nodes and edges from a graph."""
    # Create project
    project_response = client.post("/projects/", json={
        "name": "Node Test Project",
        "description": "Project for node testing"
    })
    assert project_response.status_code == 201
    project_id = project_response.json()["project_id"]
    
    # Create graph
    graph_response = client.post(f"/projects/{project_id}/graphs", json={
        "name": "Node Test Graph",
        "description": "Graph for node testing"
    })
    assert graph_response.status_code == 201
    graph_id = graph_response.json()["graph_id"]
    
    # Create a node directly
    node_response = client.post(f"/projects/{project_id}/graphs/{graph_id}/nodes", json={
        "name": "Test Node"
    })
    assert node_response.status_code == 200
    
    # Get all nodes
    response = client.get(f"/projects/{project_id}/graphs/{graph_id}/nodes")
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) > 0

def test_log_metrics():
    """Test logging metrics for a node."""
    # Create project, graph, and node
    project_response = client.post("/projects/", json={
        "name": "Metrics Test Project",
        "description": "Project for metrics testing"
    })
    assert project_response.status_code == 201
    project_id = project_response.json()["project_id"]
    
    graph_response = client.post(f"/projects/{project_id}/graphs", json={
        "name": "Metrics Test Graph",
        "description": "Graph for metrics testing"
    })
    assert graph_response.status_code == 201
    graph_id = graph_response.json()["graph_id"]
    
    node_response = client.post(f"/projects/{project_id}/graphs/{graph_id}/nodes", json={
        "name": "Metrics Test Node"
    })
    assert node_response.status_code == 200
    node_id = node_response.json()["node_id"]
    
    # Log metrics
    response = client.post("/metrics/", json={
        "model_id": node_id,
        "graph_id": graph_id,
        "metrics": '{"accuracy": 0.95, "loss": 0.05, "epochs": 10}'
    })
    assert response.status_code == 200
    assert response.json()["success"] == True
    
    # Get metrics
    metrics_response = client.get(f"/projects/{project_id}/graphs/{graph_id}/nodes/{node_id}/metrics")
    assert metrics_response.status_code == 200
    metrics = metrics_response.json()
    assert metrics["accuracy"] == 0.95
    assert metrics["loss"] == 0.05
    assert metrics["epochs"] == 10 