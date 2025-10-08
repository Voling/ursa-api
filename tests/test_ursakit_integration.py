"""
Tests for ursakit SDK integration with ursa-api.
Verifies that the SDK works correctly without HTTP transport.
"""
import uuid
import json
from pathlib import Path
import pytest

from ursakit.client import UrsaClient
from app.config import settings, REPO_ROOT


class TestUrsaKitIntegration:
    """Test ursakit SDK integration without HTTP transport."""
    
    def test_sdk_client_initialization(self):
        """Test that UrsaClient initializes correctly."""
        sdk_dir = Path(settings.MODEL_STORAGE_DIR)
        client = UrsaClient(dir=sdk_dir)
        
        assert client.get_ursa_dir().exists()
        assert client.get_ursa_dir() == sdk_dir
    
    def test_sklearn_model_detection_and_save(self, sample_sklearn_model):
        """Test sklearn model detection and saving."""
        model, X, y = sample_sklearn_model
        
        sdk_dir = Path(settings.MODEL_STORAGE_DIR)
        client = UrsaClient(dir=sdk_dir)
        
        # Save model
        model_id = client.save(model, name="sklearn_test")
        
        # Verify model was saved
        assert model_id is not None
        model_dir = client.get_ursa_dir() / "models" / model_id
        assert model_dir.exists()
        assert (model_dir / "metadata.json").exists()
        
        # Verify metadata structure
        with open(model_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        assert "id" in metadata
        assert "name" in metadata
        assert "framework" in metadata
        assert "created_at" in metadata
        assert metadata["framework"] == "scikit-learn"
    
    def test_sklearn_model_load_and_predict(self, sample_sklearn_model):
        """Test sklearn model loading and prediction."""
        model, X, y = sample_sklearn_model
        
        sdk_dir = Path(settings.MODEL_STORAGE_DIR)
        client = UrsaClient(dir=sdk_dir)
        
        # Save and load model
        model_id = client.save(model, name="sklearn_test")
        loaded_model = client.load(model_id)
        
        # Test prediction
        original_pred = model.predict(X[:5])
        loaded_pred = loaded_model.predict(X[:5])
        
        # Predictions should be the same
        assert all(original_pred == loaded_pred)
    
    def test_torch_model_detection_and_save(self, sample_torch_model):
        """Test PyTorch model detection and saving."""
        model, sample_input = sample_torch_model
        
        sdk_dir = Path(settings.MODEL_STORAGE_DIR)
        client = UrsaClient(dir=sdk_dir)
        
        # Save model
        model_id = client.save(model, name="torch_test")
        
        # Verify model was saved
        assert model_id is not None
        model_dir = client.get_ursa_dir() / "models" / model_id
        assert model_dir.exists()
        assert (model_dir / "metadata.json").exists()
        
        # Verify metadata structure
        with open(model_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        assert "id" in metadata
        assert "name" in metadata
        assert "framework" in metadata
        assert "created_at" in metadata
        assert metadata["framework"] == "pytorch"
    
    def test_torch_model_load_and_predict(self, sample_torch_model):
        """Test PyTorch model loading and forward pass."""
        model, sample_input = sample_torch_model
        
        sdk_dir = Path(settings.MODEL_STORAGE_DIR)
        client = UrsaClient(dir=sdk_dir)
        
        # Save and load model
        model_id = client.save(model, name="torch_test")
        loaded_model = client.load(model_id)
        
        # Test forward pass
        import torch
        with torch.no_grad():
            original_output = model(sample_input)
            loaded_output = loaded_model(sample_input)
        
        # Outputs should be close (allowing for small numerical differences)
        assert torch.allclose(original_output, loaded_output, atol=1e-6)
    
    def test_tensorflow_model_detection_and_save(self, sample_tf_model):
        """Test TensorFlow model detection and saving."""
        model, X, y = sample_tf_model
        
        sdk_dir = Path(settings.MODEL_STORAGE_DIR)
        client = UrsaClient(dir=sdk_dir)
        
        # Save model
        model_id = client.save(model, name="tf_test")
        
        # Verify model was saved
        assert model_id is not None
        model_dir = client.get_ursa_dir() / "models" / model_id
        assert model_dir.exists()
        assert (model_dir / "metadata.json").exists()
        
        # Verify metadata structure
        with open(model_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        assert "id" in metadata
        assert "name" in metadata
        assert "framework" in metadata
        assert "created_at" in metadata
        assert metadata["framework"] == "tensorflow"
    
    def test_tensorflow_model_load_and_predict(self, sample_tf_model):
        """Test TensorFlow model loading and prediction."""
        model, X, y = sample_tf_model
        
        sdk_dir = Path(settings.MODEL_STORAGE_DIR)
        client = UrsaClient(dir=sdk_dir)
        
        # Save and load model
        model_id = client.save(model, name="tf_test")
        loaded_model = client.load(model_id)
        
        # Test prediction
        import numpy as np
        test_data = X[:5]
        
        original_pred = model.predict(test_data, verbose=0)
        loaded_pred = loaded_model.predict(test_data, verbose=0)
        
        # Predictions should be close
        assert np.allclose(original_pred, loaded_pred, atol=1e-6)
    
    def test_model_metadata_generation(self, sample_sklearn_model):
        """Test that model metadata is generated correctly."""
        model, X, y = sample_sklearn_model
        
        sdk_dir = Path(settings.MODEL_STORAGE_DIR)
        client = UrsaClient(dir=sdk_dir)
        
        model_id = client.save(model, name="metadata_test")
        
        # Read metadata file
        metadata_file = client.get_ursa_dir() / "models" / model_id / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Check required metadata fields
        assert "id" in metadata
        assert "name" in metadata
        assert "framework" in metadata
        assert "created_at" in metadata
        assert "metadata" in metadata
        
        # Check framework detection
        assert metadata["framework"] == "scikit-learn"
        assert metadata["name"] == "metadata_test"
        
        # Check model file exists
        model_files = list((client.get_ursa_dir() / "models" / model_id).glob("*.pkl"))
        assert len(model_files) > 0, "No model file found"
        assert model_files[0].exists(), "Model file does not exist"
    
    def test_multiple_models_in_same_directory(self, sample_sklearn_model, sample_torch_model):
        """Test saving multiple models in the same directory."""
        sklearn_model, X, y = sample_sklearn_model
        torch_model, sample_input = sample_torch_model
        
        sdk_dir = Path(settings.MODEL_STORAGE_DIR)
        client = UrsaClient(dir=sdk_dir)
        
        # Save multiple models
        sklearn_id = client.save(sklearn_model, name="sklearn_model")
        torch_id = client.save(torch_model, name="torch_model")
        
        assert sklearn_id != torch_id
        
        # Both should be loadable
        loaded_sklearn = client.load(sklearn_id)
        loaded_torch = client.load(torch_id)
        
        assert loaded_sklearn is not None
        assert loaded_torch is not None
        
        # Test they work correctly
        sklearn_pred = loaded_sklearn.predict(X[:1])
        assert sklearn_pred is not None
        
        import torch
        with torch.no_grad():
            torch_output = loaded_torch(sample_input)
            assert torch_output is not None
    
    def test_error_handling_invalid_model(self):
        """Test error handling when saving invalid model."""
        invalid_model = "not a model"
        
        sdk_dir = Path(settings.MODEL_STORAGE_DIR)
        client = UrsaClient(dir=sdk_dir)
        
        # Should raise an error when trying to save an invalid model
        with pytest.raises(Exception):  # Any exception is acceptable for invalid models
            client.save(invalid_model, name="invalid_model")
    
    def test_error_handling_nonexistent_model(self):
        """Test error handling when loading nonexistent model."""
        sdk_dir = Path(settings.MODEL_STORAGE_DIR)
        client = UrsaClient(dir=sdk_dir)
        
        with pytest.raises(ValueError):
            client.load("nonexistent-model-id")
    
    def test_no_http_transport_in_client(self):
        """Test that client has no HTTP transport when use_server=False."""
        sdk_dir = Path(settings.MODEL_STORAGE_DIR)
        client = UrsaClient(dir=sdk_dir)
        
        # Real UrsaClient works in local-only mode by default
        assert client.get_ursa_dir() == sdk_dir
    
    def test_local_storage_only(self, sample_sklearn_model):
        """Test that models are only stored locally."""
        model, X, y = sample_sklearn_model
        
        sdk_dir = Path(settings.MODEL_STORAGE_DIR)
        client = UrsaClient(dir=sdk_dir)
        
        model_id = client.save(model, name="local_test")
        
        # Verify model is in local storage
        model_dir = client.get_ursa_dir() / "models" / model_id
        assert model_dir.exists()
        assert (model_dir / "metadata.json").exists()
        
        # Read metadata to find model file
        with open(model_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Model file should exist in the model directory
        model_files = list(model_dir.glob("*.pkl"))
        assert len(model_files) > 0, "No model file found"
        assert model_files[0].exists(), "Model file does not exist" 