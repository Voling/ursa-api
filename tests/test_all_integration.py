"""
Comprehensive integration test showing the full ursa-api + ursakit + cache flow.
"""
import json
import base64
import pickle
from pathlib import Path
import pytest

from app.services.model_cache_service import ModelCacheService
from ursakit.client import UrsaClient
from app.config import settings


class TestAllIntegration:
    """Test the complete integration flow."""
    
    def test_model_flow(self, sample_sklearn_model):
        """Test the complete end-to-end model flow with caching."""
        model, X, y = sample_sklearn_model
        
        # Use the actual model storage directory
        cache_service = ModelCacheService()
        
        # Step 1: Save model using SDK (simulating PWA → ursa-api → SDK)
        model_id = None
        sdk_dir = Path(settings.MODEL_STORAGE_DIR)
        sdk_client = UrsaClient(dir=sdk_dir, use_server=False)
        
        model_id = sdk_client.save(model, name="integration_test")
        print(f"Model saved with ID: {model_id}")
        
        # Debug: Print SDK directory structure
        print("\nSDK Directory Structure:")
        for path in sdk_dir.rglob("*"):
            if path.is_file():
                print(f"SDK File: {path}")
        
        # Step 2: Cache the model (simulating ursa-api caching)
        cache_path = cache_service.save_model_from_sdk(model_id, sdk_dir)
        print(f"Model cached at: {cache_path}")
        
        # Debug: Print cache directory structure
        print("\nCache Directory Structure:")
        for path in cache_path.rglob("*"):
            if path.is_file():
                print(f"Cache File: {path}")
                if path.name == "metadata.json":
                    with open(path, 'r') as f:
                        print(f"Metadata content: {f.read()}")
        
        # Verify caching worked
        assert cache_service._is_model_cached(model_id)
        assert cache_path.exists()
        assert (cache_path / "metadata.json").exists()
        
        # Read metadata to find model file
        with open(cache_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        assert "path" in metadata, "Metadata missing path"
        model_path = cache_path / Path(metadata["path"]).name
        assert model_path.exists(), f"Model file not found at {model_path}"
        
        print("Cache path:", cache_path)
        
        # Step 3: Retrieve from cache and use (simulating later API call)
        # Get the cache directory for SDK use
        sdk_cache_dir = cache_service.get_model_for_sdk(model_id)
        print(f"Retrieved cache directory: {sdk_cache_dir}")
        
        # Debug: Print retrieved cache directory structure
        print("\nRetrieved Cache Directory Structure:")
        for path in sdk_cache_dir.rglob("*"):
            if path.is_file():
                print(f"Retrieved File: {path}")
                if path.name == "metadata.json":
                    with open(path, 'r') as f:
                        print(f"Retrieved metadata content: {f.read()}")
        
        # Verify cache structure
        assert sdk_cache_dir.exists()
        models_dir = sdk_cache_dir / "models" / model_id
        assert models_dir.exists()
        assert (models_dir / "metadata.json").exists()
        
        # Read metadata to find model file
        with open(models_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        assert "path" in metadata, "Metadata missing path"
        model_path = models_dir / Path(metadata["path"]).name
        assert model_path.exists(), f"Model file not found at {model_path}"
        
        # Create new SDK client pointing to cache
        cache_sdk_client = UrsaClient(dir=sdk_cache_dir, use_server=False)
        
        # Load model from cache
        loaded_model = cache_sdk_client.load(model_id)
        print(f"Model loaded from cache")
        
        # Step 4: Test that cached model works correctly
        original_predictions = model.predict(X[:5])
        cached_predictions = loaded_model.predict(X[:5])
        
        assert all(original_predictions == cached_predictions)
        print(f"Predictions match: {original_predictions}")
        
        # Step 5: Test cache statistics
        stats = cache_service.get_cache_stats()
        assert stats["total_models"] == 1
        assert stats["total_size_mb"] > 0
        print(f"Cache stats: {stats}")
        
    
    def test_multiple_models_caching(self, sample_sklearn_model, sample_torch_model):
        """Test caching multiple models simultaneously."""
        sklearn_model, X_sklearn, y_sklearn = sample_sklearn_model
        torch_model, X_torch = sample_torch_model
        
        # Use the actual model storage directory
        cache_service = ModelCacheService()
        sdk_dir = Path(settings.MODEL_STORAGE_DIR)
        sdk_client = UrsaClient(dir=sdk_dir, use_server=False)
        
        # Save both models
        sklearn_id = sdk_client.save(sklearn_model, name="sklearn_test")
        torch_id = sdk_client.save(torch_model, name="torch_test")
        
        # Cache both models
        cache_service.save_model_from_sdk(sklearn_id, sdk_dir)
        cache_service.save_model_from_sdk(torch_id, sdk_dir)
        
        # Verify both are cached
        assert cache_service._is_model_cached(sklearn_id)
        assert cache_service._is_model_cached(torch_id)
        
        # Retrieve both from cache
        sklearn_cache_dir = cache_service.get_model_for_sdk(sklearn_id)
        torch_cache_dir = cache_service.get_model_for_sdk(torch_id)
        
        # Verify cache directories exist
        assert sklearn_cache_dir.exists()
        assert torch_cache_dir.exists()
        assert (sklearn_cache_dir / "models" / sklearn_id).exists()
        assert (torch_cache_dir / "models" / torch_id).exists()
        
        # Check model files exist by reading metadata
        sklearn_metadata = json.load(open(sklearn_cache_dir / "models" / sklearn_id / "metadata.json"))
        torch_metadata = json.load(open(torch_cache_dir / "models" / torch_id / "metadata.json"))
        
        # Print metadata for debugging
        print("\nScikit-learn metadata:", json.dumps(sklearn_metadata, indent=2))
        print("\nPyTorch metadata:", json.dumps(torch_metadata, indent=2))
        
        # Print directory contents for debugging
        print("\nScikit-learn directory contents:")
        for path in (sklearn_cache_dir / "models" / sklearn_id).rglob("*"):
            if path.is_file():
                print(f"  {path.name}")
                
        print("\nPyTorch directory contents:")
        for path in (torch_cache_dir / "models" / torch_id).rglob("*"):
            if path.is_file():
                print(f"  {path.name}")
        
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
        
        print("Multiple models cached and loaded successfully")
    
    def test_cache_cleanup_functionality(self, sample_sklearn_model):
        """Test cache cleanup functionality."""
        model, X, y = sample_sklearn_model
        
        # Use the actual model storage directory
        cache_service = ModelCacheService()
        sdk_dir = Path(settings.MODEL_STORAGE_DIR)
        sdk_client = UrsaClient(dir=sdk_dir, use_server=False)
        
        # Save a model
        model_id = sdk_client.save(model, name="cleanup_test")
        cache_service.save_model_from_sdk(model_id, sdk_dir)
        
        # Verify model is cached
        assert cache_service._is_model_cached(model_id)
        
        # Run cleanup with very restrictive settings
        cache_service.cleanup_old_cache(max_age_days=0, max_size_gb=0.001)
        
        # Model should be removed
        assert not cache_service._is_model_cached(model_id)
        
        # Cache stats should show 0 models
        stats = cache_service.get_cache_stats()
        assert stats["total_models"] == 0
        assert stats["total_size_mb"] == 0
        print("Cache cleanup working correctly") 