"""
Test configuration and fixtures for ursa-api tests.
"""
import pytest
from pathlib import Path
from contextlib import contextmanager
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import shutil

from app.main import app
from app.ursaml import UrsaMLStorage
from app.config import Settings, REPO_ROOT
from app.services.cache.cache_manager import ModelCacheManager
from app.dependencies import get_cache_manager


def clean_dir_keep_gitkeep(directory: Path) -> None:
    """Remove all contents except .gitkeep within a directory."""
    for item in directory.glob("*"):
        if item.name == ".gitkeep":
            continue
        if item.is_file():
            item.unlink()
        else:
            shutil.rmtree(item)

@pytest.fixture(autouse=True)
def test_settings():
    """Override settings for testing."""
    with patch("app.config.settings") as mock_settings:
        # Create test settings using repository storage
        test_settings = Settings(
            STORAGE_TYPE="filesystem",
            MODEL_STORAGE_DIR=str(REPO_ROOT / "storage" / "models"),
            URSAML_STORAGE_DIR=str(REPO_ROOT / "storage" / "ursaml")
        )
        
        # Update the mock to use our test settings
        for key, value in test_settings.model_dump().items():
            setattr(mock_settings, key, value)
        
        # Ensure storage directories exist
        Path(test_settings.MODEL_STORAGE_DIR).mkdir(parents=True, exist_ok=True)
        Path(test_settings.URSAML_STORAGE_DIR).mkdir(parents=True, exist_ok=True)
        
        yield test_settings
        
        # Clean storage directories but keep structure
        clean_dir_keep_gitkeep(Path(test_settings.MODEL_STORAGE_DIR))
        clean_dir_keep_gitkeep(Path(test_settings.URSAML_STORAGE_DIR))


@pytest.fixture
def test_storage_dir():
    """Create a storage directory for testing."""
    storage_dir = REPO_ROOT / "storage" / "ursaml"
    storage_dir.mkdir(parents=True, exist_ok=True)
    yield str(storage_dir)
    # Clean directory but keep structure
    clean_dir_keep_gitkeep(storage_dir)


@pytest.fixture
def storage(test_storage_dir):
    """Create a test UrsaML storage instance."""
    return UrsaMLStorage(base_path=test_storage_dir)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_project(storage):
    """Create a sample project for testing."""
    project_data = {
        "name": "Test Project",
        "description": "A test project"
    }
    project = storage.create_project(project_data["name"], project_data["description"])
    project["project_id"] = project["id"]  # Add project_id for API compatibility
    return project


@pytest.fixture
def sample_graph(storage, sample_project):
    """Create a sample graph for testing."""
    graph_data = {
        "name": "Test Graph",
        "description": "A test graph"
    }
    graph = storage.create_graph(sample_project["id"], graph_data["name"], graph_data["description"])
    graph["graph_id"] = graph["id"]  # Add graph_id for API compatibility
    return graph


@pytest.fixture
def sample_node(storage, sample_graph):
    """Create a sample node for testing."""
    node = storage.create_node(sample_graph['id'], "Test Node")
    return node


@pytest.fixture
def base64_model():
    """Create a base64 encoded test model."""
    import pickle
    import base64
    
    # Create a simple mock model
    model = {"type": "test_model", "data": [1, 2, 3]}
    model_bytes = pickle.dumps(model)
    return base64.b64encode(model_bytes).decode('utf-8')


@pytest.fixture(scope="function")
def test_cache_service(test_settings):
    """Create a test cache manager."""
    manager = get_cache_manager()
    # Clean any existing cache root contents
    cache_root = manager.cache_root
    if cache_root.exists():
        clean_dir_keep_gitkeep(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    yield manager
    # Cleanup after
    clean_dir_keep_gitkeep(cache_root)


@pytest.fixture
def sample_sklearn_model():
    """Create a simple sklearn model for testing."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    import numpy as np
    
    # Create sample data
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=3, random_state=42)
    model.fit(X, y)
    
    return model, X, y


@pytest.fixture 
def sample_torch_model():
    """Create a simple PyTorch model for testing."""
    import torch
    import torch.nn as nn
    
    # Define the model class at module level to avoid pickling issues
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    )
    sample_input = torch.randn(1, 4)
    
    return model, sample_input


@pytest.fixture
def sample_tf_model():
    """Create a simple TensorFlow model for testing."""
    import tensorflow as tf
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    # Create sample data
    import numpy as np
    X = np.random.randn(100, 4)
    y = np.random.randint(0, 2, 100)
    
    return model, X, y 