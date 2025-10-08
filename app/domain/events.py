"""Domain events for decoupled side effects and integrations."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List
from uuid import uuid4


@dataclass
class DomainEvent:
    """Base class for all domain events."""
    event_id: str
    timestamp: datetime
    aggregate_id: str
    
    def __post_init__(self):
        if not hasattr(self, 'event_id') or not self.event_id:
            object.__setattr__(self, 'event_id', str(uuid4()))
        if not hasattr(self, 'timestamp') or not self.timestamp:
            object.__setattr__(self, 'timestamp', datetime.now())


@dataclass
class ProjectCreated(DomainEvent):
    """Raised when a new project is created."""
    name: str
    description: str


@dataclass
class ProjectDeleted(DomainEvent):
    """Raised when a project is deleted."""
    name: str


@dataclass
class GraphCreated(DomainEvent):
    """Raised when a new graph is created."""
    project_id: str
    name: str
    description: str


@dataclass
class GraphDeleted(DomainEvent):
    """Raised when a graph is deleted."""
    project_id: str
    name: str


@dataclass
class ModelUploaded(DomainEvent):
    """Raised when a model is uploaded."""
    model_id: str
    node_id: str
    graph_id: str
    name: str
    framework: str


@dataclass
class ModelDeleted(DomainEvent):
    """Raised when a model is deleted."""
    model_id: str


@dataclass
class MetricsRecorded(DomainEvent):
    """Raised when metrics are recorded for a node."""
    graph_id: str
    node_id: str
    metrics: Dict[str, Any]


class DomainEventPublisher:
    """Singleton publisher for domain events."""
    
    _instance: DomainEventPublisher | None = None
    _subscribers: Dict[type, List[Callable[[DomainEvent], None]]]
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._subscribers = {}
        return cls._instance
    
    def subscribe(self, event_type: type[DomainEvent], handler: Callable[[DomainEvent], None]) -> None:
        """Subscribe a handler to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
    
    def publish(self, event: DomainEvent) -> None:
        """Publish an event to all subscribers."""
        event_type = type(event)
        if event_type in self._subscribers:
            for handler in self._subscribers[event_type]:
                try:
                    handler(event)
                except Exception as e:
                    # Log error but don't fail the main operation
                    print(f"Event handler error: {e}")
    
    def clear_subscribers(self) -> None:
        """Clear all subscribers (useful for testing)."""
        self._subscribers = {}


# Singleton instance
event_publisher = DomainEventPublisher()

