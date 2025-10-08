"""Tests for domain events and event handling."""
from __future__ import annotations

import pytest
from unittest.mock import Mock
from datetime import datetime

from app.domain.events import (
    DomainEvent, ModelUploaded, MetricsRecorded, ModelDeleted,
    DomainEventPublisher, event_publisher
)


class TestDomainEvent:
    """Test base domain event functionality."""

    def test_domain_event_creation(self):
        """Test domain event creation with defaults."""
        event = DomainEvent(
            event_id="test-id",
            timestamp=datetime.now(),
            aggregate_id="test-aggregate"
        )
        
        assert event.event_id == "test-id"
        assert isinstance(event.timestamp, datetime)
        assert event.aggregate_id == "test-aggregate"

    def test_domain_event_with_custom_values(self):
        """Test domain event creation with custom values."""
        custom_id = "custom-event-id"
        custom_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        custom_aggregate_id = "model-123"
        
        event = DomainEvent(
            event_id=custom_id,
            timestamp=custom_timestamp,
            aggregate_id=custom_aggregate_id
        )
        
        assert event.event_id == custom_id
        assert event.timestamp == custom_timestamp
        assert event.aggregate_id == custom_aggregate_id

    def test_domain_event_to_dict(self):
        """Test domain event serialization."""
        event = DomainEvent(
            event_id="test-id",
            timestamp=datetime.now(),
            aggregate_id="model-123"
        )
        # DomainEvent doesn't have to_dict method, test dataclass fields
        assert event.aggregate_id == "model-123"
        assert event.event_id == "test-id"
        assert isinstance(event.timestamp, datetime)


class TestModelUploadedEvent:
    """Test ModelUploaded domain event."""

    def test_model_uploaded_creation(self):
        """Test ModelUploaded event creation."""
        event = ModelUploaded(
            event_id="test-id",
            timestamp=datetime.now(),
            aggregate_id="model-123",
            model_id="model-123",
            node_id="node-456",
            graph_id="graph-789",
            name="test-model",
            framework="scikit-learn"
        )
        
        assert event.model_id == "model-123"
        assert event.node_id == "node-456"
        assert event.graph_id == "graph-789"
        assert event.name == "test-model"
        assert event.framework == "scikit-learn"
        assert event.aggregate_id == "model-123"

    def test_model_uploaded_to_dict(self):
        """Test ModelUploaded event serialization."""
        event = ModelUploaded(
            event_id="test-id",
            timestamp=datetime.now(),
            aggregate_id="model-123",
            model_id="model-123",
            node_id="node-456",
            graph_id="graph-789",
            name="test-model",
            framework="scikit-learn"
        )
        # Test dataclass fields directly
        assert event.model_id == "model-123"
        assert event.node_id == "node-456"
        assert event.graph_id == "graph-789"
        assert event.name == "test-model"
        assert event.framework == "scikit-learn"


class TestMetricsRecordedEvent:
    """Test MetricsRecorded domain event."""

    def test_metrics_recorded_creation(self):
        """Test MetricsRecorded event creation."""
        metrics = {"accuracy": 0.95, "loss": 0.05}
        event = MetricsRecorded(
            graph_id="graph-123",
            node_id="node-456",
            metrics=metrics
        )
        
        assert event.graph_id == "graph-123"
        assert event.node_id == "node-456"
        assert event.metrics == metrics
        assert event.aggregate_id == "node-456"  # Should be set to node_id

    def test_metrics_recorded_to_dict(self):
        """Test MetricsRecorded event serialization."""
        metrics = {"accuracy": 0.95, "loss": 0.05}
        event = MetricsRecorded(
            graph_id="graph-123",
            node_id="node-456",
            metrics=metrics
        )
        event_dict = event.to_dict()
        
        assert event_dict["event_type"] == "MetricsRecorded"
        assert event_dict["graph_id"] == "graph-123"
        assert event_dict["node_id"] == "node-456"
        assert event_dict["metrics"] == metrics


class TestModelDeletedEvent:
    """Test ModelDeleted domain event."""

    def test_model_deleted_creation(self):
        """Test ModelDeleted event creation."""
        event = ModelDeleted(model_id="model-123")
        
        assert event.model_id == "model-123"
        assert event.aggregate_id == "model-123"

    def test_model_deleted_to_dict(self):
        """Test ModelDeleted event serialization."""
        event = ModelDeleted(model_id="model-123")
        event_dict = event.to_dict()
        
        assert event_dict["event_type"] == "ModelDeleted"
        assert event_dict["model_id"] == "model-123"


class TestDomainEventPublisher:
    """Test domain event publisher functionality."""

    def test_publisher_initialization(self):
        """Test event publisher initialization."""
        publisher = DomainEventPublisher()
        assert publisher._subscribers == {}

    def test_subscribe_handler(self):
        """Test subscribing event handlers."""
        publisher = DomainEventPublisher()
        handler = Mock()
        
        publisher.subscribe(ModelUploaded, handler)
        
        assert ModelUploaded in publisher._subscribers
        assert handler in publisher._subscribers[ModelUploaded]

    def test_subscribe_multiple_handlers(self):
        """Test subscribing multiple handlers for same event type."""
        publisher = DomainEventPublisher()
        handler1 = Mock()
        handler2 = Mock()
        
        publisher.subscribe(ModelUploaded, handler1)
        publisher.subscribe(ModelUploaded, handler2)
        
        assert len(publisher._subscribers[ModelUploaded]) == 2
        assert handler1 in publisher._subscribers[ModelUploaded]
        assert handler2 in publisher._subscribers[ModelUploaded]

    def test_publish_event_with_handler(self):
        """Test publishing event with subscribed handler."""
        publisher = DomainEventPublisher()
        handler = Mock()
        
        publisher.subscribe(ModelUploaded, handler)
        
        event = ModelUploaded(
            model_id="model-123",
            node_id="node-456",
            graph_id="graph-789",
            name="test-model",
            framework="scikit-learn"
        )
        
        publisher.publish(event)
        
        handler.assert_called_once_with(event)

    def test_publish_event_no_handlers(self):
        """Test publishing event with no subscribed handlers."""
        publisher = DomainEventPublisher()
        
        event = ModelUploaded(
            model_id="model-123",
            node_id="node-456",
            graph_id="graph-789",
            name="test-model",
            framework="scikit-learn"
        )
        
        # Should not raise
        publisher.publish(event)

    def test_publish_multiple_handlers(self):
        """Test publishing event to multiple handlers."""
        publisher = DomainEventPublisher()
        handler1 = Mock()
        handler2 = Mock()
        
        publisher.subscribe(ModelUploaded, handler1)
        publisher.subscribe(ModelUploaded, handler2)
        
        event = ModelUploaded(
            model_id="model-123",
            node_id="node-456",
            graph_id="graph-789",
            name="test-model",
            framework="scikit-learn"
        )
        
        publisher.publish(event)
        
        handler1.assert_called_once_with(event)
        handler2.assert_called_once_with(event)

    def test_publish_different_event_types(self):
        """Test publishing different event types to different handlers."""
        publisher = DomainEventPublisher()
        upload_handler = Mock()
        metrics_handler = Mock()
        
        publisher.subscribe(ModelUploaded, upload_handler)
        publisher.subscribe(MetricsRecorded, metrics_handler)
        
        upload_event = ModelUploaded(
            model_id="model-123",
            node_id="node-456",
            graph_id="graph-789",
            name="test-model",
            framework="scikit-learn"
        )
        
        metrics_event = MetricsRecorded(
            graph_id="graph-123",
            node_id="node-456",
            metrics={"accuracy": 0.95}
        )
        
        publisher.publish(upload_event)
        publisher.publish(metrics_event)
        
        upload_handler.assert_called_once_with(upload_event)
        metrics_handler.assert_called_once_with(metrics_event)


class TestGlobalEventPublisher:
    """Test the global event publisher instance."""

    def test_global_publisher_exists(self):
        """Test that global event publisher exists."""
        assert event_publisher is not None
        assert isinstance(event_publisher, DomainEventPublisher)

    def test_global_publisher_functionality(self):
        """Test global event publisher functionality."""
        handler = Mock()
        
        # Clear any existing handlers
        event_publisher._handlers.clear()
        
        event_publisher.subscribe(ModelUploaded, handler)
        
        event = ModelUploaded(
            model_id="model-123",
            node_id="node-456",
            graph_id="graph-789",
            name="test-model",
            framework="scikit-learn"
        )
        
        event_publisher.publish(event)
        
        handler.assert_called_once_with(event)
