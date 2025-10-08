from fastapi import APIRouter, Path, Depends
from app.schemas.api_schemas import MetricsUpload, MetricsResponse, AllNodeMetricsResponse
from app.dependencies import get_ursaml_storage, get_metrics_service
from app.domain.ports import StoragePort
from app.application.metrics_service import MetricsService
from app.domain.errors import ValidationError
from typing import Dict, Any
import json

router = APIRouter()

@router.post("/metrics/", response_model=MetricsResponse)
def log_metrics(
    metrics_data: MetricsUpload,
    metrics_svc: MetricsService = Depends(get_metrics_service)
):
    """
    Upload metrics of model
    Stored in UrsaML graph
    """
    # Parse metrics JSON
    try:
        metrics = json.loads(metrics_data.metrics)
    except json.JSONDecodeError as exc:
        raise ValidationError("Invalid JSON in metrics field") from exc
    
    # Store metrics (service will validate node existence)
    metrics_svc.add_node_metrics(
        graph_id=metrics_data.graph_id,
        node_id=metrics_data.model_id,
        metrics={
            "accuracy": metrics.get("accuracy"),
            "loss": metrics.get("loss"),
            "epochs": metrics.get("epochs", 1),
            **metrics  # Include all metrics
        }
    )
    
    return MetricsResponse(success=True)

@router.get("/projects/{project_id}/graphs/{graph_id}/nodes/{node_id}/metrics")
def get_node_metrics(
    project_id: str,
    graph_id: str,
    node_id: str,
    storage: StoragePort = Depends(get_ursaml_storage)
) -> Dict[str, Any]:
    """
    Get metrics for a specific node.
    """
    # Get node from graph
    nodes = storage.get_graph_nodes(graph_id)
    node = next((n for n in nodes if n["id"] == node_id), None)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    
    # Get metrics from node metadata
    metrics = node.get("metadata", {}).get("meta", {})
    if not metrics:
        return {
            "accuracy": None,
            "loss": None,
            "epochs": None,
            "timestamp": None,
            "additional_metrics": {}
        }
    
    # Return metrics
    result = {
        "accuracy": metrics.get("score"),
        "loss": metrics.get("loss"),
        "epochs": metrics.get("epochs"),
        "timestamp": metrics.get("metrics_timestamp")
    }
    
    # Include additional metrics
    additional_metrics = {k: v for k, v in metrics.items() 
                        if k not in ["score", "loss", "epochs", "metrics_timestamp"]}
    result["additional_metrics"] = additional_metrics
    
    return result

@router.get("/projects/{project_id}/graphs/{graph_id}/metrics", response_model=AllNodeMetricsResponse)
def get_all_node_metrics(
    project_id: str,
    graph_id: str,
    storage: StoragePort = Depends(get_ursaml_storage)
):
    """
    Get metrics for all nodes in a graph.
    """
    # Get all nodes in the graph
    nodes = storage.get_graph_nodes(graph_id)
    
    # If no nodes in graph, return empty metrics
    if not nodes:
        return AllNodeMetricsResponse(
            graph_id=graph_id,
            metrics={}
        )
    
    # Format the metrics for the response
    formatted_metrics = {}
    for node in nodes:
        metrics = node.get("metadata", {}).get("meta", {})
        if not metrics:
            # Include nodes with no metrics
            formatted_metrics[node["id"]] = {
                "accuracy": None,
                "loss": None,
                "epochs": None,
                "timestamp": None,
                "additional_metrics": {}
            }
            continue
        
        # Format metrics
        formatted_metrics[node["id"]] = {
            "accuracy": metrics.get("score"),
            "loss": metrics.get("loss"),
            "epochs": metrics.get("epochs"),
            "timestamp": metrics.get("metrics_timestamp"),
            "additional_metrics": {
                k: v for k, v in metrics.items() 
                if k not in ["score", "loss", "epochs", "metrics_timestamp"]
            }
        }
    
    return AllNodeMetricsResponse(
        graph_id=graph_id,
        metrics=formatted_metrics
    ) 