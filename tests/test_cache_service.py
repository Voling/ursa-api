"""
Tests for ModelCacheService.
"""
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
import pytest

from app.services.model_cache_service import ModelCacheService
from ursakit.client import UrsaClient


class TestModelCacheService:
    """Test the ModelCacheService functionality."""
    
    def test_cache_service_initialization(self, test_cache_service):
        """Test that cache service initializes correctly."""
        assert test_cache_service.cache_dir.exists()
        assert test_cache_service.cache_metadata_file.exists() or not test_cache_service.cache_metadata_file.exists()
        assert isinstance(test_cache_service.cache_metadata, dict)
    
    def test_get_model_cache_path(self, test_cache_service):
        """Test that cache path generation works correctly."""
        model_id = "test-model-123"
        cache_path = test_cache_service._get_model_cache_path(model_id)
        
        expected_path = test_cache_service.cache_dir / "models" / model_id
        assert cache_path == expected_path
    
    def test_is_model_cached_false_when_not_cached(self, test_cache_service):
        """Test that is_model_cached returns False for uncached models."""
        model_id = "non-existent-model"
        assert not test_cache_service._is_model_cached(model_id)
    
    def test_is_model_cached_true_when_cached(self, test_cache_service, sample_sklearn_model):
        """Test that is_model_cached returns True for cached models."""
        model, X, y = sample_sklearn_model
        
        # Save a model using SDK first
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sdk_client = UrsaClient(dir=temp_path, use_server=False)
            
            model_id = sdk_client.save(model, name="test_model")
            
            # Cache the model
            test_cache_service.save_model_from_sdk(model_id, temp_path)
            
            # Check if it's cached
            assert test_cache_service._is_model_cached(model_id)
    
    def test_save_model_from_sdk(self, test_cache_service, sample_sklearn_model):
        """Test saving a model from SDK to cache."""
        model, X, y = sample_sklearn_model
        
        # Save model using SDK first
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sdk_client = UrsaClient(dir=temp_path, use_server=False)
            
            model_id = sdk_client.save(model, name="test_model")
            
            # Cache the model
            cache_path = test_cache_service.save_model_from_sdk(model_id, temp_path)
            
            # Verify cache structure
            assert cache_path.exists()
            assert (cache_path / "metadata.json").exists()
            
            # Verify metadata was updated
            assert model_id in test_cache_service.cache_metadata
            assert "cached_at" in test_cache_service.cache_metadata[model_id]
            assert "size_bytes" in test_cache_service.cache_metadata[model_id]
    
    def test_get_model_for_sdk_from_cache(self, test_cache_service, sample_sklearn_model):
        """Test retrieving a cached model for SDK use."""
        model, X, y = sample_sklearn_model
        
        # Save and cache model first
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sdk_client = UrsaClient(dir=temp_path, use_server=False)
            
            model_id = sdk_client.save(model, name="test_model")
            test_cache_service.save_model_from_sdk(model_id, temp_path)
            
            # Verify the model was cached
            assert test_cache_service._is_model_cached(model_id)
            
            # Get model from cache
            cache_dir = test_cache_service.get_model_for_sdk(model_id)
            
            # Verify the returned path exists
            assert cache_dir.exists()
            
            # Verify the models subdirectory exists with our model
            models_dir = cache_dir / "models" / model_id
            assert models_dir.exists()
            assert (models_dir / "metadata.json").exists()
            
            # Verify we can load the model using SDK from cache
            sdk_client = UrsaClient(dir=cache_dir, use_server=False)
            loaded_model = sdk_client.load(model_id)
            
            # Test that the loaded model works
            predictions = loaded_model.predict(X[:5])
            assert predictions is not None
            assert len(predictions) == 5
    
    def test_get_model_for_sdk_not_found(self, test_cache_service):
        """Test that get_model_for_sdk raises error for non-existent models."""
        with pytest.raises(ValueError):
            test_cache_service.get_model_for_sdk("non-existent")
    
    def test_cache_metadata_persistence(self, test_cache_service, sample_sklearn_model):
        """Test that cache metadata persists across service instances."""
        model, X, y = sample_sklearn_model
        
        # Save a model
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sdk_client = UrsaClient(dir=temp_path, use_server=False)
            
            model_id = sdk_client.save(model, name="test_model")
            test_cache_service.save_model_from_sdk(model_id, temp_path)
        
        # Create new cache service instance with same directory
        new_service = ModelCacheService()
        new_service.cache_dir = test_cache_service.cache_dir
        new_service.cache_metadata_file = test_cache_service.cache_metadata_file
        new_service.cache_metadata = new_service._load_cache_metadata()
        
        # Should have the same metadata
        assert model_id in new_service.cache_metadata
    
    def test_cleanup_old_cache_by_age(self, test_cache_service, sample_sklearn_model):
        """Test cache cleanup by age."""
        model, X, y = sample_sklearn_model
        
        # Save a model
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sdk_client = UrsaClient(dir=temp_path, use_server=False)
            
            model_id = sdk_client.save(model, name="test_model")
            test_cache_service.save_model_from_sdk(model_id, temp_path)
        
        # Artificially age the cache entry
        from datetime import datetime, timedelta
        old_date = (datetime.now() - timedelta(days=10)).isoformat()
        test_cache_service.cache_metadata[model_id]["last_accessed"] = old_date
        test_cache_service._save_cache_metadata()
        
        # Run cleanup with 5 day max age
        test_cache_service.cleanup_old_cache(max_age_days=5)
        
        # Model should be removed
        assert model_id not in test_cache_service.cache_metadata
        assert not test_cache_service._is_model_cached(model_id)
    
    def test_cleanup_old_cache_by_size(self, test_cache_service, sample_sklearn_model):
        """Test cache cleanup by size (LRU eviction)."""
        model, X, y = sample_sklearn_model
        
        # Save a model
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sdk_client = UrsaClient(dir=temp_path, use_server=False)
            
            model_id = sdk_client.save(model, name="test_model")
            test_cache_service.save_model_from_sdk(model_id, temp_path)
        
        # Artificially set large size and old access time
        from datetime import datetime, timedelta
        old_date = (datetime.now() - timedelta(hours=1)).isoformat()
        test_cache_service.cache_metadata[model_id]["size_bytes"] = 11 * 1024 * 1024 * 1024  # 11GB
        test_cache_service.cache_metadata[model_id]["last_accessed"] = old_date
        test_cache_service._save_cache_metadata()
        
        # Run cleanup with 10GB max size
        test_cache_service.cleanup_old_cache(max_size_gb=10.0)
        
        # Model should be removed due to size limit
        assert model_id not in test_cache_service.cache_metadata
        assert not test_cache_service._is_model_cached(model_id)
    
    def test_get_cache_stats(self, test_cache_service, sample_sklearn_model):
        """Test cache statistics reporting."""
        model, X, y = sample_sklearn_model
        
        # Initially empty
        stats = test_cache_service.get_cache_stats()
        assert stats["total_models"] == 0
        assert stats["total_size_mb"] == 0
        
        # Save a model
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sdk_client = UrsaClient(dir=temp_path, use_server=False)
            
            model_id = sdk_client.save(model, name="test_model")
            test_cache_service.save_model_from_sdk(model_id, temp_path)
        
        # Check updated stats
        stats = test_cache_service.get_cache_stats()
        assert stats["total_models"] == 1
        assert stats["total_size_mb"] > 0
    
    def test_force_refresh(self, test_cache_service, sample_sklearn_model):
        """Test force refreshing a cached model."""
        model, X, y = sample_sklearn_model
        
        # Save a model
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sdk_client = UrsaClient(dir=temp_path, use_server=False)
            
            model_id = sdk_client.save(model, name="test_model")
            test_cache_service.save_model_from_sdk(model_id, temp_path)
            
            # Get model with force refresh
            cache_dir = test_cache_service.get_model_for_sdk(model_id, force_refresh=True)
            
            # Should still work
            assert cache_dir.exists()
            assert (cache_dir / "models" / model_id).exists()
            
            # Load and test model
            sdk_client = UrsaClient(dir=cache_dir, use_server=False)
            loaded_model = sdk_client.load(model_id)
            predictions = loaded_model.predict(X[:5])
            assert predictions is not None
    
    def test_cache_freshness_check(self, test_cache_service, sample_sklearn_model):
        """Test cache freshness checking."""
        model, X, y = sample_sklearn_model
        
        # Save a model
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sdk_client = UrsaClient(dir=temp_path, use_server=False)
            
            model_id = sdk_client.save(model, name="test_model")
            test_cache_service.save_model_from_sdk(model_id, temp_path)
            
            # Should be fresh initially
            assert test_cache_service._is_cache_fresh(model_id)
            
            # Make it stale
            from datetime import datetime, timedelta
            old_date = (datetime.now() - timedelta(days=2)).isoformat()
            test_cache_service.cache_metadata[model_id]["cached_at"] = old_date
            test_cache_service._save_cache_metadata()
            
            # Should be stale now
            assert not test_cache_service._is_cache_fresh(model_id, max_age_hours=24)
    
    def test_multiple_model_types(self, test_cache_service, sample_sklearn_model, sample_torch_model):
        """Test caching different types of models."""
        sklearn_model, X_sklearn, y_sklearn = sample_sklearn_model
        torch_model, X_torch = sample_torch_model
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sdk_client = UrsaClient(dir=temp_path, use_server=False)
            
            # Save both models
            sklearn_id = sdk_client.save(sklearn_model, name="sklearn_test")
            torch_id = sdk_client.save(torch_model, name="torch_test")
            
            # Cache both models
            test_cache_service.save_model_from_sdk(sklearn_id, temp_path)
            test_cache_service.save_model_from_sdk(torch_id, temp_path)
            
            # Both should be cached
            assert test_cache_service._is_model_cached(sklearn_id)
            assert test_cache_service._is_model_cached(torch_id)
            
            # Get both from cache
            sklearn_dir = test_cache_service.get_model_for_sdk(sklearn_id)
            torch_dir = test_cache_service.get_model_for_sdk(torch_id)
            
            # Load and test both models
            sklearn_sdk = UrsaClient(dir=sklearn_dir, use_server=False)
            torch_sdk = UrsaClient(dir=torch_dir, use_server=False)
            
            sklearn_loaded = sklearn_sdk.load(sklearn_id)
            torch_loaded = torch_sdk.load(torch_id)
            
            # Test sklearn model
            sklearn_pred = sklearn_loaded.predict(X_sklearn[:5])
            assert sklearn_pred is not None
            
            # Test torch model
            import torch
            with torch.no_grad():
                torch_output = torch_loaded(X_torch)
                assert torch_output is not None 