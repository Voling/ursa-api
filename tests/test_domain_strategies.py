"""Tests for domain strategies (serialization)."""
from __future__ import annotations

import pytest
import pickle
from unittest.mock import Mock

from app.domain.strategies import (
    ModelSerializationStrategy, PickleSerializationStrategy,
    SerializationStrategyFactory
)
from app.domain.errors import ValidationError


class TestModelSerializationStrategy:
    """Test serialization strategy protocol."""

    def test_pickle_strategy_implements_protocol(self):
        """Test that PickleSerializationStrategy implements the protocol."""
        strategy = PickleSerializationStrategy()
        
        # Test that it has required methods
        assert hasattr(strategy, 'serialize')
        assert hasattr(strategy, 'deserialize')
        assert hasattr(strategy, 'get_framework_name')
        
        # Test method signatures work
        test_data = {"test": "data"}
        serialized = strategy.serialize(test_data)
        assert isinstance(serialized, bytes)
        
        deserialized = strategy.deserialize(serialized)
        assert deserialized == test_data
        
        framework_name = strategy.get_framework_name()
        assert isinstance(framework_name, str)


class TestPickleSerializationStrategy:
    """Test pickle serialization strategy implementation."""

    def test_serialize_basic_data(self):
        """Test serializing basic Python data structures."""
        strategy = PickleSerializationStrategy()
        
        # Test various data types
        test_cases = [
            "simple string",
            42,
            3.14,
            [1, 2, 3],
            {"key": "value"},
            (1, 2, 3),
            {"nested": {"data": [1, 2, 3]}},
        ]
        
        for test_data in test_cases:
            serialized = strategy.serialize(test_data)
            assert isinstance(serialized, bytes)
            
            deserialized = strategy.deserialize(serialized)
            assert deserialized == test_data

    def test_serialize_complex_objects(self):
        """Test serializing complex objects."""
        strategy = PickleSerializationStrategy()
        
        # Create a mock model-like object
        class MockModel:
            def __init__(self, name, params):
                self.name = name
                self.params = params
            
            def predict(self, data):
                return [1, 2, 3]  # Mock prediction
        
        model = MockModel("test_model", {"learning_rate": 0.01})
        
        serialized = strategy.serialize(model)
        assert isinstance(serialized, bytes)
        
        deserialized = strategy.deserialize(serialized)
        assert isinstance(deserialized, MockModel)
        assert deserialized.name == "test_model"
        assert deserialized.params == {"learning_rate": 0.01}
        assert deserialized.predict([1, 2, 3]) == [1, 2, 3]

    def test_deserialize_invalid_data(self):
        """Test deserializing invalid pickle data."""
        strategy = PickleSerializationStrategy()
        
        invalid_data = b"not pickle data"
        
        with pytest.raises((pickle.UnpicklingError, EOFError)):
            strategy.deserialize(invalid_data)

    def test_get_framework_name(self):
        """Test getting framework name."""
        strategy = PickleSerializationStrategy()
        framework_name = strategy.get_framework_name()
        
        assert framework_name == "sklearn"  # Actual implementation returns "sklearn"

    def test_roundtrip_serialization(self):
        """Test complete roundtrip serialization."""
        strategy = PickleSerializationStrategy()
        
        original_data = {
            "model_type": "sklearn_classifier",
            "parameters": {"n_estimators": 100, "max_depth": 5},
            "training_data": [[1, 2], [3, 4], [5, 6]],
            "labels": [0, 1, 0],
        }
        
        # Serialize
        serialized = strategy.serialize(original_data)
        
        # Deserialize
        deserialized = strategy.deserialize(serialized)
        
        # Verify
        assert deserialized == original_data
        assert deserialized["model_type"] == "sklearn_classifier"
        assert deserialized["parameters"]["n_estimators"] == 100


class TestSerializationStrategyFactory:
    """Test serialization strategy factory."""

    def test_get_pickle_strategy(self):
        """Test getting pickle strategy."""
        strategy = SerializationStrategyFactory.get_strategy("pickle")
        
        assert isinstance(strategy, PickleSerializationStrategy)
        assert strategy.get_framework_name() == "sklearn"

    def test_get_strategy_case_insensitive(self):
        """Test getting strategy with different case."""
        strategy = SerializationStrategyFactory.get_strategy("PICKLE")
        
        assert isinstance(strategy, PickleSerializationStrategy)
        assert strategy.get_framework_name() == "sklearn"

    def test_get_strategy_unknown_framework(self):
        """Test getting strategy for unknown framework defaults to pickle."""
        strategy = SerializationStrategyFactory.get_strategy("unknown_framework")
        assert isinstance(strategy, PickleSerializationStrategy)

    def test_get_strategy_empty_framework(self):
        """Test getting strategy for empty framework name defaults to pickle."""
        strategy = SerializationStrategyFactory.get_strategy("")
        assert isinstance(strategy, PickleSerializationStrategy)

    def test_get_strategy_none_framework(self):
        """Test getting strategy for None framework defaults to pickle."""
        strategy = SerializationStrategyFactory.get_strategy(None)
        assert isinstance(strategy, PickleSerializationStrategy)

    def test_factory_singleton_behavior(self):
        """Test that factory returns new instances (not singleton)."""
        strategy1 = SerializationStrategyFactory.get_strategy("pickle")
        strategy2 = SerializationStrategyFactory.get_strategy("pickle")
        
        # Factory creates new instances each time
        assert strategy1 is not strategy2
        assert isinstance(strategy1, PickleSerializationStrategy)
        assert isinstance(strategy2, PickleSerializationStrategy)

    def test_factory_multiple_strategies(self):
        """Test factory with multiple strategy types."""
        # Test known strategy
        pickle_strategy = SerializationStrategyFactory.get_strategy("pickle")
        assert isinstance(pickle_strategy, PickleSerializationStrategy)
        
        # Test unknown strategy defaults to pickle
        strategy = SerializationStrategyFactory.get_strategy("torch")
        assert isinstance(strategy, TorchSerializationStrategy)

    def test_factory_strategy_usage(self):
        """Test that factory-created strategies work correctly."""
        strategy = SerializationStrategyFactory.get_strategy("pickle")
        
        test_data = {"test": "data", "number": 42}
        serialized = strategy.serialize(test_data)
        deserialized = strategy.deserialize(serialized)
        
        assert deserialized == test_data


class TestSerializationIntegration:
    """Test serialization strategy integration scenarios."""

    def test_strategy_with_model_like_objects(self):
        """Test serialization with model-like objects."""
        strategy = SerializationStrategyFactory.get_strategy("pickle")
        
        # Simulate a trained model
        class TrainedModel:
            def __init__(self):
                self.weights = [0.1, 0.2, 0.3, 0.4]
                self.bias = 0.5
                self.trained = True
            
            def predict(self, X):
                return sum(x * w for x, w in zip(X, self.weights)) + self.bias
        
        model = TrainedModel()
        test_input = [1.0, 2.0, 3.0, 4.0]
        original_prediction = model.predict(test_input)
        
        # Serialize model
        serialized_model = strategy.serialize(model)
        
        # Deserialize model
        deserialized_model = strategy.deserialize(serialized_model)
        
        # Test that deserialized model works
        assert deserialized_model.trained is True
        assert deserialized_model.weights == [0.1, 0.2, 0.3, 0.4]
        assert deserialized_model.bias == 0.5
        
        # Test prediction
        new_prediction = deserialized_model.predict(test_input)
        assert new_prediction == original_prediction

    def test_strategy_with_large_data(self):
        """Test serialization with large data structures."""
        strategy = SerializationStrategyFactory.get_strategy("pickle")
        
        # Create large data structure
        large_data = {
            "features": [[i, i*2, i*3] for i in range(1000)],
            "labels": [i % 2 for i in range(1000)],
            "metadata": {
                "dataset_size": 1000,
                "feature_count": 3,
                "classes": [0, 1],
            }
        }
        
        # Serialize
        serialized = strategy.serialize(large_data)
        
        # Verify it's reasonable size (not too large)
        assert len(serialized) > 1000  # Should be substantial
        assert len(serialized) < 100000  # But not huge
        
        # Deserialize
        deserialized = strategy.deserialize(serialized)
        
        # Verify
        assert deserialized == large_data
        assert len(deserialized["features"]) == 1000
        assert len(deserialized["labels"]) == 1000
