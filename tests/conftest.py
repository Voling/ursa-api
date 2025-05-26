"""
Test configuration and fixtures for ursa-api tests.
"""
import pytest
import tempfile
from pathlib import Path
from contextlib import contextmanager
from fastapi.testclient import TestClient

from app.main import app
from app.db.database import get_db
from app.services.model_cache_service import ModelCacheService


# Test database setup - PostgreSQL on port 5432
@pytest.fixture(scope="function")
def test_db():
    """Create a test database session using PostgreSQL on port 5432."""
    from app.db.database import SessionLocal
    from app.config import settings
    from sqlalchemy import text
    
    # Verify we're using PostgreSQL on port 5432
    db_url = str(settings.DATABASE_URL)
    assert "postgresql://" in db_url, f"Expected PostgreSQL, got: {db_url}"
    assert ":5432/" in db_url, f"Expected port 5432, got: {db_url}"
    
    # For API tests, we need to use the regular SessionLocal without cleanup per session
    # Cleanup will happen at the end of the test
    yield SessionLocal
    
    # Clean up all test data after the entire test
    cleanup_session = SessionLocal()
    try:
        # Delete all data from tables in proper order (respecting foreign keys)
        cleanup_session.execute(text("DELETE FROM metrics"))
        cleanup_session.execute(text("DELETE FROM edges"))
        cleanup_session.execute(text("DELETE FROM nodes"))
        cleanup_session.execute(text("DELETE FROM models"))
        cleanup_session.execute(text("DELETE FROM graphs"))
        cleanup_session.execute(text("DELETE FROM projects"))
        cleanup_session.commit()
    except Exception:
        cleanup_session.rollback()
    finally:
        cleanup_session.close()


@pytest.fixture(scope="function") 
def db_session(test_db):
    """Create a database session for testing."""
    session = test_db()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="function")
def isolated_db():
    """Create an isolated database session with cleanup for integration tests."""
    from app.db.database import SessionLocal
    from app.config import settings
    from sqlalchemy import text
    
    # Verify we're using PostgreSQL on port 5432
    db_url = str(settings.DATABASE_URL)
    assert "postgresql://" in db_url, f"Expected PostgreSQL, got: {db_url}"
    assert ":5432/" in db_url, f"Expected port 5432, got: {db_url}"
    
    @contextmanager 
    def get_clean_session():
        """Get a session with cleanup after each use."""
        session = SessionLocal()
        try:
            yield session
        finally:
            # Clean up all test data after each session
            try:
                # Delete all data from tables in proper order (respecting foreign keys)
                session.execute(text("DELETE FROM metrics"))
                session.execute(text("DELETE FROM edges"))
                session.execute(text("DELETE FROM nodes"))
                session.execute(text("DELETE FROM models"))
                session.execute(text("DELETE FROM graphs"))
                session.execute(text("DELETE FROM projects"))
                session.commit()
            except Exception:
                session.rollback()
            finally:
                session.close()
    
    # Return the session factory
    yield get_clean_session


@pytest.fixture(scope="function")
def client(test_db):
    """Create a test client with test database."""
    def override_get_db():
        db = test_db()
        try:
            yield db
        finally:
            db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    # Clean up dependency override
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def temp_cache_dir():
    """Create a temporary directory for cache testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(scope="function")
def test_cache_service():
    """Create a test cache service with temporary directory."""
    # Create our own temporary directory that persists for the test
    import tempfile
    temp_dir = tempfile.mkdtemp()
    cache_dir = Path(temp_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Temporarily override the cache directory
    original_init = ModelCacheService.__init__
    
    def mock_init(self):
        self.cache_dir = cache_dir
        self.s3_client = None  # No S3 for testing
        self.cache_metadata_file = self.cache_dir / "cache_metadata.json"
        self.cache_metadata = self._load_cache_metadata()
    
    ModelCacheService.__init__ = mock_init
    
    service = ModelCacheService()
    
    yield service
    
    # Restore original __init__
    ModelCacheService.__init__ = original_init
    
    # Clean up the temporary directory
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


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