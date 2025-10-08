"""Event handlers for domain events."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.domain.events import (
        ProjectCreated,
        ProjectDeleted,
        GraphCreated,
        GraphDeleted,
        ModelUploaded,
        ModelDeleted,
        MetricsRecorded,
    )

logger = logging.getLogger(__name__)


class AuditLogHandler:
    """Logs all domain events for audit trail."""
    
    def handle_project_created(self, event: ProjectCreated) -> None:
        logger.info(f"[AUDIT] Project created: {event.aggregate_id} - {event.name}")
    
    def handle_project_deleted(self, event: ProjectDeleted) -> None:
        logger.info(f"[AUDIT] Project deleted: {event.aggregate_id} - {event.name}")
    
    def handle_graph_created(self, event: GraphCreated) -> None:
        logger.info(f"[AUDIT] Graph created: {event.aggregate_id} in project {event.project_id}")
    
    def handle_graph_deleted(self, event: GraphDeleted) -> None:
        logger.info(f"[AUDIT] Graph deleted: {event.aggregate_id}")
    
    def handle_model_uploaded(self, event: ModelUploaded) -> None:
        logger.info(f"[AUDIT] Model uploaded: {event.model_id} - {event.name} ({event.framework})")
    
    def handle_model_deleted(self, event: ModelDeleted) -> None:
        logger.info(f"[AUDIT] Model deleted: {event.model_id}")
    
    def handle_metrics_recorded(self, event: MetricsRecorded) -> None:
        logger.info(f"[AUDIT] Metrics recorded for node {event.node_id} in graph {event.graph_id}")


class CacheWarmingHandler:
    """Warms cache when models are uploaded."""
    
    def handle_model_uploaded(self, event: ModelUploaded) -> None:
        # In a real implementation, this would trigger background cache warming
        logger.info(f"[CACHE] Warming cache for model {event.model_id}")


class NotificationHandler:
    """Sends notifications for important events."""
    
    def handle_project_created(self, event: ProjectCreated) -> None:
        # In a real implementation, send email/webhook
        logger.info(f"[NOTIFICATION] New project: {event.name}")
    
    def handle_model_uploaded(self, event: ModelUploaded) -> None:
        logger.info(f"[NOTIFICATION] New model uploaded: {event.name} in graph {event.graph_id}")
    
    def handle_metrics_recorded(self, event: MetricsRecorded) -> None:
        # Could trigger alerts if metrics are below threshold
        accuracy = event.metrics.get("accuracy")
        if accuracy and accuracy < 0.5:
            logger.warning(f"[NOTIFICATION] Low accuracy ({accuracy}) for node {event.node_id}")


def register_event_handlers():
    """Register all event handlers with the publisher."""
    from app.domain.events import (
        event_publisher,
        ProjectCreated,
        ProjectDeleted,
        GraphCreated,
        GraphDeleted,
        ModelUploaded,
        ModelDeleted,
        MetricsRecorded,
    )
    
    audit = AuditLogHandler()
    cache = CacheWarmingHandler()
    notification = NotificationHandler()
    
    # Audit handlers (all events)
    event_publisher.subscribe(ProjectCreated, audit.handle_project_created)
    event_publisher.subscribe(ProjectDeleted, audit.handle_project_deleted)
    event_publisher.subscribe(GraphCreated, audit.handle_graph_created)
    event_publisher.subscribe(GraphDeleted, audit.handle_graph_deleted)
    event_publisher.subscribe(ModelUploaded, audit.handle_model_uploaded)
    event_publisher.subscribe(ModelDeleted, audit.handle_model_deleted)
    event_publisher.subscribe(MetricsRecorded, audit.handle_metrics_recorded)
    
    # Cache warming
    event_publisher.subscribe(ModelUploaded, cache.handle_model_uploaded)
    
    # Notifications
    event_publisher.subscribe(ProjectCreated, notification.handle_project_created)
    event_publisher.subscribe(ModelUploaded, notification.handle_model_uploaded)
    event_publisher.subscribe(MetricsRecorded, notification.handle_metrics_recorded)

