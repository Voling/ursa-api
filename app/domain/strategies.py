"""Strategy pattern for model serialization."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol
import pickle


class ModelSerializationStrategy(Protocol):
    """Protocol for model serialization strategies."""
    
    def serialize(self, model: Any) -> bytes:
        """Serialize a model to bytes."""
        ...
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to a model."""
        ...
    
    def get_framework_name(self) -> str:
        """Return the framework name."""
        ...


class PickleSerializationStrategy:
    """Pickle-based serialization (default for scikit-learn)."""
    
    def serialize(self, model: Any) -> bytes:
        return pickle.dumps(model)
    
    def deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)
    
    def get_framework_name(self) -> str:
        return "sklearn"


class TorchSerializationStrategy:
    """PyTorch model serialization."""
    
    def serialize(self, model: Any) -> bytes:
        try:
            import torch
            import io
            buffer = io.BytesIO()
            torch.save(model, buffer)
            return buffer.getvalue()
        except ImportError:
            raise RuntimeError("PyTorch not installed")
    
    def deserialize(self, data: bytes) -> Any:
        try:
            import torch
            import io
            buffer = io.BytesIO(data)
            return torch.load(buffer)
        except ImportError:
            raise RuntimeError("PyTorch not installed")
    
    def get_framework_name(self) -> str:
        return "pytorch"


class TensorFlowSerializationStrategy:
    """TensorFlow/Keras model serialization."""
    
    def serialize(self, model: Any) -> bytes:
        try:
            import tensorflow as tf
            import tempfile
            import shutil
            from pathlib import Path
            
            # Save to temp directory
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = Path(tmpdir) / "model"
                tf.keras.models.save_model(model, str(model_path))
                
                # Create tar archive
                import tarfile
                import io
                buffer = io.BytesIO()
                with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
                    tar.add(model_path, arcname='model')
                return buffer.getvalue()
        except ImportError:
            raise RuntimeError("TensorFlow not installed")
    
    def deserialize(self, data: bytes) -> Any:
        try:
            import tensorflow as tf
            import tempfile
            import tarfile
            import io
            from pathlib import Path
            
            # Extract from tar archive
            with tempfile.TemporaryDirectory() as tmpdir:
                buffer = io.BytesIO(data)
                with tarfile.open(fileobj=buffer, mode='r:gz') as tar:
                    tar.extractall(tmpdir)
                
                model_path = Path(tmpdir) / "model"
                return tf.keras.models.load_model(str(model_path))
        except ImportError:
            raise RuntimeError("TensorFlow not installed")
    
    def get_framework_name(self) -> str:
        return "tensorflow"


class ONNXSerializationStrategy:
    """ONNX model serialization."""
    
    def serialize(self, model: Any) -> bytes:
        try:
            import onnx
            return model.SerializeToString()
        except ImportError:
            raise RuntimeError("ONNX not installed")
    
    def deserialize(self, data: bytes) -> Any:
        try:
            import onnx
            model = onnx.ModelProto()
            model.ParseFromString(data)
            return model
        except ImportError:
            raise RuntimeError("ONNX not installed")
    
    def get_framework_name(self) -> str:
        return "onnx"


class SerializationStrategyFactory:
    """Factory to select serialization strategy based on framework."""
    
    _strategies = {
        "sklearn": PickleSerializationStrategy,
        "scikit-learn": PickleSerializationStrategy,
        "pickle": PickleSerializationStrategy,
        "pytorch": TorchSerializationStrategy,
        "torch": TorchSerializationStrategy,
        "tensorflow": TensorFlowSerializationStrategy,
        "keras": TensorFlowSerializationStrategy,
        "onnx": ONNXSerializationStrategy,
    }
    
    @classmethod
    def get_strategy(cls, framework: str) -> ModelSerializationStrategy:
        """Get serialization strategy for a framework."""
        strategy_class = cls._strategies.get(framework.lower())
        if strategy_class is None:
            # Default to pickle for unknown frameworks
            return PickleSerializationStrategy()
        return strategy_class()
    
    @classmethod
    def detect_framework(cls, model: Any) -> str:
        """Attempt to detect framework from model type."""
        model_type = type(model).__module__
        
        if "sklearn" in model_type or "scikit" in model_type:
            return "sklearn"
        elif "torch" in model_type:
            return "pytorch"
        elif "tensorflow" in model_type or "keras" in model_type:
            return "tensorflow"
        elif "onnx" in model_type:
            return "onnx"
        else:
            return "unknown"
    
    @classmethod
    def register_strategy(cls, framework: str, strategy_class: type) -> None:
        """Register a new serialization strategy."""
        cls._strategies[framework.lower()] = strategy_class

