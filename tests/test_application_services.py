"""Tests for application services."""
from __future__ import annotations

import pytest
from unittest.mock import Mock, MagicMock
from pathlib import Path

from app.services.model_app_service import ModelAppService
from app.application.project_validation_service import ProjectValidationService
from app.application.graph_validation_service import GraphValidationService
from app.application.graph_access_service import GraphAccessService
from app.application.metrics_service import MetricsService
from app.domain.errors import ValidationError, NotFoundError, ConflictError
from app.domain.events import ModelUploaded, MetricsRecorded
from app.infrastructure.model_ingestion_adapter import ModelIngestionAdapter


class TestModelAppService:
    """Test ModelAppService orchestration logic."""

    def test_upload_model_validation_errors(self):
        """Test validation errors for missing data."""
        # Setup
        mock_storage = Mock()
        mock_cache = Mock()
        mock_ingestion = Mock()
        
        service = ModelAppService(mock_storage, mock_cache, mock_ingestion)
        
        # Test empty file data
        with pytest.raises(ValidationError, match="Model file data is required"):
            service.upload_model("", "graph-123")
        
        # Test empty graph ID
        with pytest.raises(ValidationError, match="Graph ID is required"):
            service.upload_model("base64data", "")

    def test_upload_model_graph_not_found(self):
        """Test error when graph doesn't exist."""
        # Setup
        mock_storage = Mock()
        mock_storage.get_graph.return_value = None
        mock_cache = Mock()
        mock_ingestion = Mock()
        
        service = ModelAppService(mock_storage, mock_cache, mock_ingestion)
        
        # Test
        with pytest.raises(NotFoundError, match="Graph not found"):
            service.upload_model("base64data", "nonexistent-graph")

    def test_upload_model_success(self):
        """Test successful model upload flow."""
        # Setup
        mock_storage = Mock()
        mock_storage.get_graph.return_value = {"id": "graph-123", "name": "test-graph"}
        mock_storage.create_node.return_value = {"id": "node-456", "name": "test-model"}
        
        mock_cache = Mock()
        
        mock_ingestion = Mock()
        mock_ingestion.prepare.return_value = Mock(
            model_id="model-789",
            model_name="test-model",
            created_at="2024-01-01T00:00:00",
            sdk_dir=Path("/tmp/sdk"),
        )
        
        service = ModelAppService(mock_storage, mock_cache, mock_ingestion)
        
        # Test
        result = service.upload_model("base64data", "graph-123")
        
        # Verify
        assert result["model_id"] == "model-789"
        assert result["node_id"] == "node-456"
        assert result["name"] == "test-model"
        
        # Verify calls
        mock_storage.get_graph.assert_called_once_with("graph-123")
        mock_ingestion.prepare.assert_called_once_with("base64data")
        mock_cache.save_model_from_sdk.assert_called_once()
        mock_storage.create_node.assert_called_once()

    def test_upload_model_node_creation_rollback(self):
        """Test rollback when node creation fails."""
        # Setup
        mock_storage = Mock()
        mock_storage.get_graph.return_value = {"id": "graph-123"}
        mock_storage.create_node.return_value = None  # Simulate failure
        
        mock_cache = Mock()
        
        mock_ingestion = Mock()
        mock_ingestion.prepare.return_value = Mock(
            model_id="model-789",
            model_name="test-model",
            created_at="2024-01-01T00:00:00",
            sdk_dir=Path("/tmp/sdk"),
        )
        
        service = ModelAppService(mock_storage, mock_cache, mock_ingestion)
        
        # Test
        with pytest.raises(RuntimeError, match="Failed to create node"):
            service.upload_model("base64data", "graph-123")
        
        # Verify rollback
        mock_cache.delete_model.assert_called_once_with("model-789")


class TestProjectValidationService:
    """Test ProjectValidationService validation logic."""

    def test_validate_name_success(self):
        """Test successful name validation."""
        mock_storage = Mock()
        service = ProjectValidationService(mock_storage)
        
        result = service.validate_name("  Test Project  ")
        assert result == "Test Project"

    def test_validate_name_empty(self):
        """Test validation error for empty name."""
        mock_storage = Mock()
        service = ProjectValidationService(mock_storage)
        
        with pytest.raises(ValidationError, match="Project name is required"):
            service.validate_name("")
        
        with pytest.raises(ValidationError, match="Project name is required"):
            service.validate_name("   ")

    def test_check_duplicate_name_no_duplicates(self):
        """Test no duplicate names found."""
        mock_storage = Mock()
        mock_storage.get_all_projects.return_value = [
            {"id": "proj-1", "name": "Project A"},
            {"id": "proj-2", "name": "Project B"},
        ]
        service = ProjectValidationService(mock_storage)
        
        # Should not raise
        service.check_duplicate_name("Project C")

    def test_check_duplicate_name_conflict(self):
        """Test duplicate name conflict."""
        mock_storage = Mock()
        mock_storage.get_all_projects.return_value = [
            {"id": "proj-1", "name": "Project A"},
            {"id": "proj-2", "name": "Project B"},
        ]
        service = ProjectValidationService(mock_storage)
        
        with pytest.raises(ConflictError, match="Project with name 'Project A' already exists"):
            service.check_duplicate_name("Project A")
        
        # Case insensitive
        with pytest.raises(ConflictError, match="Project with name 'project a' already exists"):
            service.check_duplicate_name("project a")

    def test_check_duplicate_name_exclude_id(self):
        """Test duplicate check with exclude ID."""
        mock_storage = Mock()
        mock_storage.get_all_projects.return_value = [
            {"id": "proj-1", "name": "Project A"},
            {"id": "proj-2", "name": "Project B"},
        ]
        service = ProjectValidationService(mock_storage)
        
        # Should not raise when excluding the same ID
        service.check_duplicate_name("Project A", exclude_id="proj-1")


class TestGraphValidationService:
    """Test GraphValidationService validation logic."""

    def test_validate_name_success(self):
        """Test successful name validation."""
        mock_storage = Mock()
        service = GraphValidationService(mock_storage)
        
        result = service.validate_name("  Test Graph  ")
        assert result == "Test Graph"

    def test_validate_name_empty(self):
        """Test validation error for empty name."""
        mock_storage = Mock()
        service = GraphValidationService(mock_storage)
        
        with pytest.raises(ValidationError, match="Graph name is required"):
            service.validate_name("")

    def test_check_duplicate_name_in_project_no_duplicates(self):
        """Test no duplicate names in project."""
        mock_storage = Mock()
        mock_storage.get_project.return_value = {"id": "proj-1", "name": "Test Project"}
        mock_storage.get_project_graphs.return_value = [
            {"id": "graph-1", "name": "Graph A"},
            {"id": "graph-2", "name": "Graph B"},
        ]
        service = GraphValidationService(mock_storage)
        
        # Should not raise
        service.check_duplicate_name_in_project("proj-1", "Graph C")

    def test_check_duplicate_name_in_project_conflict(self):
        """Test duplicate name conflict in project."""
        mock_storage = Mock()
        mock_storage.get_project.return_value = {"id": "proj-1", "name": "Test Project"}
        mock_storage.get_project_graphs.return_value = [
            {"id": "graph-1", "name": "Graph A"},
            {"id": "graph-2", "name": "Graph B"},
        ]
        service = GraphValidationService(mock_storage)
        
        with pytest.raises(ConflictError, match="Graph with name 'Graph A' already exists"):
            service.check_duplicate_name_in_project("proj-1", "Graph A")

    def test_check_duplicate_name_project_not_found(self):
        """Test error when project doesn't exist."""
        mock_storage = Mock()
        mock_storage.get_project.return_value = None
        service = GraphValidationService(mock_storage)
        
        with pytest.raises(NotFoundError, match="Project not found"):
            service.check_duplicate_name_in_project("nonexistent", "Graph A")


class TestGraphAccessService:
    """Test GraphAccessService access control logic."""

    def test_require_graph_in_project_success(self):
        """Test successful graph ownership validation."""
        mock_storage = Mock()
        mock_storage.get_graph.return_value = {
            "id": "graph-123",
            "project_id": "proj-456",
            "name": "Test Graph"
        }
        service = GraphAccessService(mock_storage)
        
        # Should not raise
        service.require_graph_in_project("proj-456", "graph-123")

    def test_require_graph_in_project_graph_not_found(self):
        """Test error when graph doesn't exist."""
        mock_storage = Mock()
        mock_storage.get_graph.return_value = None
        service = GraphAccessService(mock_storage)
        
        with pytest.raises(NotFoundError, match="Graph not found"):
            service.require_graph_in_project("proj-456", "nonexistent-graph")

    def test_require_graph_in_project_wrong_ownership(self):
        """Test error when graph belongs to different project."""
        mock_storage = Mock()
        mock_storage.get_graph.return_value = {
            "id": "graph-123",
            "project_id": "proj-456",  # Different project
            "name": "Test Graph"
        }
        service = GraphAccessService(mock_storage)
        
        with pytest.raises(ValidationError, match="Graph does not belong to specified project"):
            service.require_graph_in_project("proj-789", "graph-123")


class TestMetricsService:
    """Test MetricsService metrics logic."""

    def test_add_node_metrics_success(self):
        """Test successful metrics recording."""
        mock_storage = Mock()
        mock_storage.add_metrics.return_value = {"status": "success"}
        service = MetricsService(mock_storage)
        
        metrics = {"accuracy": 0.95, "loss": 0.05}
        result = service.add_node_metrics("graph-123", "node-456", metrics)
        
        assert result == {"status": "success"}
        mock_storage.add_metrics.assert_called_once_with("graph-123", "node-456", metrics)
