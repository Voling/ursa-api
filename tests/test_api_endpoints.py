"""
Tests for FastAPI endpoints.
"""
import json
import base64
import pickle
from unittest.mock import patch, Mock
import pytest

from app.ursaml import UrsaMLStorage


class TestProjectEndpoints:
    """Test project-related API endpoints."""
    
    def test_create_project(self, client):
        """Test creating a new project."""
        project_data = {
            "name": "Test Project",
            "description": "A test project"
        }
        
        response = client.post("/projects/", json=project_data)
        assert response.status_code in [200, 201]  # Both are valid for creation
        
        data = response.json()
        assert "project_id" in data
        assert data["project_id"] is not None
    
    def test_get_projects(self, client):
        """Test getting list of projects."""
        # Create a project first
        project_data = {
            "name": "Test Project",
            "description": "A test project"
        }
        create_response = client.post("/projects/", json=project_data)
        assert create_response.status_code in [200, 201]
        
        # Get projects
        response = client.get("/projects")
        assert response.status_code == 200
        
        projects = response.json()
        assert isinstance(projects, list)
        assert len(projects) >= 1
        
        # Verify project structure
        project = projects[0]
        assert "project_id" in project
        assert "name" in project
        assert "created_at" in project
    
    def test_delete_project(self, client):
        """Test deleting a project."""
        # Create a project first
        project_data = {
            "name": "Test Project to Delete",
            "description": "Will be deleted"
        }
        create_response = client.post("/projects/", json=project_data)
        project_id = create_response.json()["project_id"]
        
        # Delete the project
        response = client.delete(f"/projects/{project_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True


class TestGraphEndpoints:
    """Test graph-related API endpoints."""
    
    def test_create_graph(self, client, sample_project):
        """Test creating a new graph."""
        graph_data = {
            "name": "Test Graph",
            "description": "A test graph"
        }
        
        response = client.post(f"/projects/{sample_project['project_id']}/graphs", json=graph_data)
        assert response.status_code in [200, 201]
        
        data = response.json()
        assert "graph_id" in data
        assert data["graph_id"] is not None
    
    def test_get_graphs(self, client, sample_project, sample_graph):
        """Test getting graphs for a project."""
        # Get graphs
        response = client.get(f"/projects/{sample_project['project_id']}/graphs")
        assert response.status_code == 200
        
        graphs = response.json()
        assert isinstance(graphs, list)
        assert len(graphs) >= 1
        
        # Verify graph structure
        graph = graphs[0]
        assert "graph_id" in graph
        assert "name" in graph
        assert "project_id" in graph
        assert "created_at" in graph


class TestNodeEndpoints:
    """Test node-related API endpoints."""
    
    def test_update_node_model(self, client, sample_project, sample_graph, sample_sklearn_model):
        """Test updating a node with a model."""
        model, X, y = sample_sklearn_model
        
        # Create a model
        model_bytes = pickle.dumps(model)
        model_b64 = base64.b64encode(model_bytes).decode('utf-8')
        
        model_upload_data = {
            "file": model_b64,
            "project_id": sample_project["project_id"],
            "graph_id": sample_graph["graph_id"]
        }
        
        model_response = client.post("/models/", json=model_upload_data)
        assert model_response.status_code in [200, 201]
        model_data = model_response.json()
        model_id = model_data["model_id"]
        node_id = model_data["node_id"]  # This creates a node automatically
        
        # Now update the node with metadata
        update_data = {
            "node_id": node_id,
            "metadata": {"model_id": model_id, "type": "classifier"}
        }
        
        response = client.put(
            f"/projects/{sample_project['project_id']}/graphs/{sample_graph['graph_id']}/nodes/{node_id}/model",
            json=update_data
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
    
    def test_delete_node(self, client, sample_project, sample_graph, sample_sklearn_model):
        """Test deleting a node."""
        model, X, y = sample_sklearn_model
        
        # Create a model which will create a node
        model_bytes = pickle.dumps(model)
        model_b64 = base64.b64encode(model_bytes).decode('utf-8')
        
        model_upload_data = {
            "file": model_b64,
            "project_id": sample_project["project_id"],
            "graph_id": sample_graph["graph_id"]
        }
        
        model_response = client.post("/models/", json=model_upload_data)
        assert model_response.status_code in [200, 201]
        model_data = model_response.json()
        node_id = model_data["node_id"]
        response = client.delete(
            f"/projects/{sample_project['project_id']}/graphs/{sample_graph['graph_id']}/nodes/{node_id}"
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True

    def test_model_swap_comprehensive(self, client, sample_project, sample_graph, sample_sklearn_model):
        """Test comprehensive model swapping functionality."""
        model, X, y = sample_sklearn_model
        
        # Create first model
        model_bytes = pickle.dumps(model)
        model_b64 = base64.b64encode(model_bytes).decode('utf-8')
        
        first_model_data = {
            "file": model_b64,
            "project_id": sample_project["project_id"],
            "graph_id": sample_graph["graph_id"]
        }
        
        # Upload first model
        first_response = client.post("/models/", json=first_model_data)
        assert first_response.status_code in [200, 201]
        first_model = first_response.json()
        first_model_id = first_model["model_id"]
        node_id = first_model["node_id"]
        
        # Verify node has first model
        node_response = client.get(
            f"/projects/{sample_project['project_id']}/graphs/{sample_graph['graph_id']}/nodes/{node_id}"
        )
        assert node_response.status_code == 200
        node_data = node_response.json()
        assert node_data["model_id"] == first_model_id
        
        # Create second model (using same model for simplicity)
        second_model_data = {
            "file": model_b64,
            "project_id": sample_project["project_id"],
            "graph_id": sample_graph["graph_id"]
        }
        
        # Upload second model
        second_response = client.post("/models/", json=second_model_data)
        assert second_response.status_code in [200, 201]
        second_model = second_response.json()
        second_model_id = second_model["model_id"]
        
        # Swap model in node
        update_data = {
            "node_id": node_id,
            "metadata": {
                "model_id": second_model_id,
                "type": "classifier"
            }
        }
        
        swap_response = client.put(
            f"/projects/{sample_project['project_id']}/graphs/{sample_graph['graph_id']}/nodes/{node_id}/model",
            json=update_data
        )
        assert swap_response.status_code == 200
        assert swap_response.json()["success"] is True
        
        # Verify node now has second model
        node_response = client.get(
            f"/projects/{sample_project['project_id']}/graphs/{sample_graph['graph_id']}/nodes/{node_id}"
        )
        assert node_response.status_code == 200
        node_data = node_response.json()
        assert node_data["model_id"] == second_model_id
        assert node_data["model_id"] != first_model_id
        
        # Verify both models still exist and are accessible
        first_model_response = client.get(f"/models/{first_model_id}")
        assert first_model_response.status_code == 200
        second_model_response = client.get(f"/models/{second_model_id}")
        assert second_model_response.status_code == 200


class TestModelEndpoints:
    """Test model-related API endpoints."""
    
    def test_upload_model_basic(self, client, sample_project, sample_graph, sample_sklearn_model):
        """Test basic model upload."""
        model, X, y = sample_sklearn_model
        
        # Create model upload data
        model_bytes = pickle.dumps(model)
        model_b64 = base64.b64encode(model_bytes).decode('utf-8')
        
        model_upload_data = {
            "file": model_b64,
            "project_id": sample_project["project_id"],
            "graph_id": sample_graph["graph_id"]
        }
        
        response = client.post("/models/", json=model_upload_data)
        assert response.status_code in [200, 201]
        
        data = response.json()
        assert "model_id" in data
        assert "node_id" in data
        assert data["model_id"] is not None
        assert data["node_id"] is not None
    
    def test_get_model_details(self, client, sample_project, sample_graph, sample_sklearn_model):
        """Test getting model details."""
        model, X, y = sample_sklearn_model
        
        # Upload a model first
        model_bytes = pickle.dumps(model)
        model_b64 = base64.b64encode(model_bytes).decode('utf-8')
        
        model_upload_data = {
            "file": model_b64,
            "project_id": sample_project["project_id"],
            "graph_id": sample_graph["graph_id"]
        }
        
        upload_response = client.post("/models/", json=model_upload_data)
        model_id = upload_response.json()["model_id"]
        
        # Get model details
        response = client.get(f"/models/{model_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert "model_id" in data
        assert data["model_id"] == model_id


class TestMetricEndpoints:
    """Test metric-related API endpoints."""
    
    def test_upload_metrics(self, client, sample_project, sample_graph, sample_node):
        """Test uploading metrics."""
        metrics_data = {
            "model_id": sample_node["id"],
            "graph_id": sample_graph["graph_id"],
            "metrics": json.dumps({
                "accuracy": 0.95,
                "loss": 0.05,
                "epochs": 10,
                "custom_metric": "test"
            })
        }
        
        response = client.post("/metrics/", json=metrics_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
    
    def test_get_all_metrics(self, client, sample_project, sample_graph, sample_node):
        """Test getting all metrics for a graph."""
        # Upload some metrics first
        metrics_data = {
            "model_id": sample_node["id"],
            "graph_id": sample_graph["graph_id"],
            "metrics": json.dumps({
                "accuracy": 0.95,
                "loss": 0.05,
                "epochs": 10
            })
        }
        
        client.post("/metrics/", json=metrics_data)
        
        # Get all metrics
        response = client.get(f"/projects/{sample_project['project_id']}/graphs/{sample_graph['graph_id']}/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "graph_id" in data
        assert "metrics" in data
        assert isinstance(data["metrics"], dict)


class TestHealthAndStatus:
    """Test health check and status endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"


class TestStorageIntegration:
    """Test integration with UrsaML storage."""
    
    def test_project_persists_in_storage(self, client):
        """Test that created projects persist in storage."""
        project_data = {
            "name": "Storage Test Project",
            "description": "Testing storage persistence"
        }
        
        # Create project
        create_response = client.post("/projects/", json=project_data)
        assert create_response.status_code == 201
        project_id = create_response.json()["project_id"]
        
        # Verify project exists by getting it through the API
        get_response = client.get(f"/projects/{project_id}")
        assert get_response.status_code == 200
        
        project = get_response.json()
        assert project["name"] == project_data["name"]
        assert project["description"] == project_data["description"]
        
        # Verify project appears in the list of all projects
        list_response = client.get("/projects")
        assert list_response.status_code == 200
        projects = list_response.json()
        
        found = False
        for p in projects:
            if p["project_id"] == project_id:
                found = True
                assert p["name"] == project_data["name"]
                assert p["description"] == project_data["description"]
                break
        
        assert found, "Project not found in list of all projects"
    
    def test_graph_persists_in_storage(self, client):
        """Test that created graphs persist in storage."""
        # First create a project
        project_data = {
            "name": "Graph Test Project",
            "description": "Testing graph persistence"
        }
        project_response = client.post("/projects/", json=project_data)
        assert project_response.status_code == 201
        project_id = project_response.json()["project_id"]
        
        # Create a graph in the project
        graph_data = {
            "name": "Storage Test Graph",
            "description": "Testing graph storage"
        }
        create_response = client.post(f"/projects/{project_id}/graphs", json=graph_data)
        assert create_response.status_code in [200, 201]
        graph_id = create_response.json()["graph_id"]
        
        # Verify graph exists by getting project's graphs
        graphs_response = client.get(f"/projects/{project_id}/graphs")
        assert graphs_response.status_code == 200
        graphs = graphs_response.json()
        
        found = False
        for graph in graphs:
            if graph["graph_id"] == graph_id:
                found = True
                assert graph["name"] == graph_data["name"]
                assert graph["description"] == graph_data["description"]
                assert graph["project_id"] == project_id
                break
        
        assert found, "Graph not found in project's graphs"


class TestErrorHandling:
    """Test error handling in API endpoints."""
    
    def test_create_graph_with_invalid_project(self, client):
        """Test creating a graph with invalid project ID."""
        graph_data = {
            "name": "Test Graph",
            "description": "A test graph"
        }
        
        response = client.post("/projects/invalid-id/graphs", json=graph_data)
        assert response.status_code == 404
    
    def test_invalid_json_data(self, client):
        """Test handling invalid JSON data."""
        response = client.post("/projects/", data="invalid json")
        assert response.status_code == 422  # FastAPI validation error
    
    def test_missing_required_fields(self, client):
        """Test handling missing required fields."""
        response = client.post("/projects/", json={})
        assert response.status_code == 422  # FastAPI validation error 