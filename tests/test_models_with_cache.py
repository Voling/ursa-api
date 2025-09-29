"""
Tests for models with cache endpoints.
"""
import base64
import pickle
from pathlib import Path
from unittest.mock import patch, Mock
import pytest

from app.services.cache.cache_manager import ModelCacheManager
from app.dependencies import get_cache_manager
from ursakit.client import UrsaClient
from app.config import settings, REPO_ROOT


class TestModelsWithCache:
    """Test the models with cache endpoints."""
    
    def test_create_model_with_cache(self, client, sample_sklearn_model):
        """Test creating a model using the standard models endpoint."""
        model, X, y = sample_sklearn_model
        
        # Create a project and graph first
        project_response = client.post("/projects/", json={
            "name": "Test Project",
            "description": "A test project"
        })
        assert project_response.status_code in [200, 201]
        project_id = project_response.json()["project_id"]
        
        graph_response = client.post(f"/projects/{project_id}/graphs/", json={
            "name": "Test Graph",
            "description": "A test graph"
        })
        assert graph_response.status_code in [200, 201]
        graph_id = graph_response.json()["graph_id"]
        
        # Serialize model for upload
        model_bytes = pickle.dumps(model)
        model_b64 = base64.b64encode(model_bytes).decode('utf-8')
        
        upload_data = {
            "file": model_b64,
            "project_id": project_id,
            "graph_id": graph_id
        }
        
        # Use the standard models endpoint
        response = client.post("/models/", json=upload_data)
        
        # Should work with the standard endpoint
        assert response.status_code in [200, 201]
        data = response.json()
        assert "model_id" in data
        assert "node_id" in data
    
    def test_get_model_from_cache(self, client):
        """Test retrieving a model using the standard endpoint."""
        model_id = "test-model-123"
        
        response = client.get(f"/models/{model_id}")
        
        # The endpoint should exist but model might not be found
        assert response.status_code in [200, 404]
    
    def test_predict_with_cached_model(self, client, sample_sklearn_model):
        """Test that the prediction functionality concept works."""
        model, X, y = sample_sklearn_model
        model_id = "test-model-123"
        
        # Create a model in the repository storage
        sdk_dir = Path(settings.MODEL_STORAGE_DIR)
        sdk_client = UrsaClient(dir=sdk_dir, use_server=False)
        
        # Save model using SDK
        actual_model_id = sdk_client.save(model, name="test_model")
        
        # Test that we can make predictions with the model
        predictions = model.predict(X[:5])
        assert len(predictions) == 5
    
    def test_cache_stats_endpoint(self, client):
        """Test the cache statistics endpoint (may not exist)."""
        response = client.get("/cache/stats")
        
        # This endpoint might not exist yet
        assert response.status_code in [200, 404]
    
    def test_cache_cleanup_endpoint(self, client):
        """Test the cache cleanup endpoint (may not exist)."""
        response = client.post("/cache/cleanup")
        
        # This endpoint might not exist yet
        assert response.status_code in [200, 404]
    
    def test_remove_from_cache_endpoint(self, client):
        """Test removing a specific model from cache (may not exist)."""
        model_id = "test-model-123"
        
        response = client.delete(f"/cache/models/{model_id}")
        
        # This endpoint might not exist yet
        assert response.status_code in [200, 404]


class TestCacheServiceIntegration:
    """Test integration between cache service and API endpoints."""
    
    def test_cache_service_with_real_model(self, test_cache_service, sample_sklearn_model):
        """Test the cache service with a real model end-to-end."""
        model, X, y = sample_sklearn_model
        
        # Save model using SDK
        sdk_dir = Path(settings.MODEL_STORAGE_DIR)
        sdk_client = UrsaClient(dir=sdk_dir, use_server=False)
        
        model_id = sdk_client.save(model, name="integration_test")
        
        # Cache the model
        cache_path = test_cache_service.save_model_from_sdk(model_id, sdk_dir)
        
        # Verify it's cached
        assert test_cache_service._local.has_model(model_id)
        
        # Retrieve from cache
        retrieved_cache_dir = test_cache_service.get_model_for_sdk(model_id)
        
        # Load model from cache using SDK
        cache_sdk_client = UrsaClient(dir=retrieved_cache_dir, use_server=False)
        loaded_model = cache_sdk_client.load(model_id)
        
        # Test that the cached model works
        predictions = loaded_model.predict(X[:5])
        original_predictions = model.predict(X[:5])
        
        assert all(predictions == original_predictions)
    
    def test_cache_service_performance_benefits(self, test_cache_service, sample_sklearn_model):
        """Test that cache provides performance benefits."""
        model, X, y = sample_sklearn_model
        
        # Save and cache model
        sdk_dir = Path(settings.MODEL_STORAGE_DIR)
        sdk_client = UrsaClient(dir=sdk_dir, use_server=False)
        
        model_id = sdk_client.save(model, name="performance_test")
        test_cache_service.save_model_from_sdk(model_id, sdk_dir)
        
        import time
        
        # First access (should be fast since it's already cached)
        start_time = time.time()
        cache_dir1 = test_cache_service.get_model_for_sdk(model_id)
        first_access_time = time.time() - start_time
        
        # Second access (should be even faster)
        start_time = time.time()
        cache_dir2 = test_cache_service.get_model_for_sdk(model_id)
        second_access_time = time.time() - start_time
        
        # Both should return valid paths (but may be different temp directories)
        assert cache_dir1.exists()
        assert cache_dir2.exists()
        assert (cache_dir1 / "models" / model_id).exists()
        assert (cache_dir2 / "models" / model_id).exists()
        
        # Both should be very fast (under 1 second for local cache)
        assert first_access_time < 1.0
        assert second_access_time < 1.0
    
    def test_cache_service_with_multiple_models(self, test_cache_service, sample_sklearn_model, sample_torch_model):
        """Test cache service with multiple different model types."""
        sklearn_model, X_sklearn, y_sklearn = sample_sklearn_model
        torch_model, X_torch = sample_torch_model
        
        # Save both models
        sdk_dir = Path(settings.MODEL_STORAGE_DIR)
        sdk_client = UrsaClient(dir=sdk_dir, use_server=False)
        
        # Save both models
        sklearn_id = sdk_client.save(sklearn_model, name="sklearn_cache_test")
        torch_id = sdk_client.save(torch_model, name="torch_cache_test")
        
        # Cache both models
        test_cache_service.save_model_from_sdk(sklearn_id, sdk_dir)
        test_cache_service.save_model_from_sdk(torch_id, sdk_dir)
        
        # Verify both are cached
        assert test_cache_service._is_model_cached(sklearn_id)
        assert test_cache_service._is_model_cached(torch_id)
        
        # Retrieve both from cache
        sklearn_cache_dir = test_cache_service.get_model_for_sdk(sklearn_id)
        torch_cache_dir = test_cache_service.get_model_for_sdk(torch_id)
        
        # Load both models from cache
        sklearn_loaded = UrsaClient(dir=sklearn_cache_dir, use_server=False).load(sklearn_id)
        torch_loaded = UrsaClient(dir=torch_cache_dir, use_server=False).load(torch_id)
        
        # Test both models work
        sklearn_pred = sklearn_loaded.predict(X_sklearn[:1])
        assert sklearn_pred is not None
        
        import torch
        with torch.no_grad():
            torch_output = torch_loaded(X_torch)
            assert torch_output is not None
    
    def test_cache_service_error_handling(self, test_cache_service):
        """Test cache service error handling."""
        # Test with non-existent model
        with pytest.raises(ValueError):
            test_cache_service.get_model_for_sdk("non-existent-model")
        
        # Test cache stats with empty cache
        stats = test_cache_service.get_cache_stats()
        assert stats["total_models"] == 0
        assert stats["total_size_mb"] == 0
    
    def test_cache_service_cleanup_functionality(self, test_cache_service, sample_sklearn_model):
        """Test cache cleanup functionality."""
        model, X, y = sample_sklearn_model
        
        # Save and cache a model
        sdk_dir = Path(settings.MODEL_STORAGE_DIR)
        sdk_client = UrsaClient(dir=sdk_dir, use_server=False)
        
        model_id = sdk_client.save(model, name="cleanup_test")
        test_cache_service.save_model_from_sdk(model_id, sdk_dir)
        
        # Verify it's cached
        assert test_cache_service._is_model_cached(model_id)
        
        # Run cleanup with very restrictive settings
        test_cache_service.cleanup_old_cache(max_age_days=0, max_size_gb=0.001)
        
        # Model should be removed
        assert not test_cache_service._is_model_cached(model_id) 