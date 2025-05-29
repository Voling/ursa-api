from fastapi import APIRouter, HTTPException, Path, Depends
from app.schemas.api_schemas import MetricsUpload, MetricsResponse, AllNodeMetricsResponse
from app.db.database import get_db
from app.db.repositories.nodes import NodeRepository
from sqlalchemy.orm import Session
from typing import Dict, Any
import json

router = APIRouter()

@router.post("/metrics/", response_model=MetricsResponse)
def log_metrics(metrics_data: MetricsUpload, db: Session = Depends(get_db)):
    """
    Upload metrics of a model.
    """
    try:
        node_repo = NodeRepository(db)
        metrics = json.loads(metrics_data.metrics)
        
        # Add metrics to node (would need to map model_id to node_id in a real implementation)
        # This is simplified for demo purposes
        node_repo.add_metrics(
            node_id=metrics_data.model_id,  # Simplified mapping
            accuracy=metrics.get("accuracy"),
            loss=metrics.get("loss"),
            epochs=metrics.get("epochs", 1),
            additional_metrics=metrics
        )
        
        return MetricsResponse(success=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/projects/{project_id}/graphs/{graph_id}/nodes/{node_id}/metrics")
def get_node_metrics(
    project_id: str,
    graph_id: str,
    node_id: str,
    db: Session = Depends(get_db)
):
    """
    Get metrics for a specific node.
    """
    node_repo = NodeRepository(db)
    node = node_repo.get_node(node_id)
    
    if not node:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    
    # Get the most recent metric for the node
    metrics = node_repo.get_node_metrics(node_id)
    if not metrics:
        return {"val_accuracy": 0}  # Default if no metrics found
    
    # Convert the most recent metric to a dictionary
    latest_metric = metrics[-1]
    result = {
        "accuracy": latest_metric.accuracy,
        "loss": latest_metric.loss,
        "epochs": latest_metric.epochs,
        "timestamp": latest_metric.timestamp.isoformat()
    }
    
    # Add additional metrics if available
    if latest_metric.additional_metrics:
        result.update(latest_metric.additional_metrics)
    
    return result

@router.get("/projects/{project_id}/graphs/{graph_id}/metrics", response_model=AllNodeMetricsResponse)
def get_all_node_metrics(
    project_id: str,
    graph_id: str,
    db: Session = Depends(get_db)
):
    """
    Get metrics for all nodes in a graph.
    """
    node_repo = NodeRepository(db)
    metrics_by_node = node_repo.get_graph_metrics(graph_id)
    
    # Format the metrics for the response
    formatted_metrics = {}
    for node_id, metrics_list in metrics_by_node.items():
        if not metrics_list:
            continue
        
        # Use the most recent metric for each node
        latest_metric = metrics_list[-1]
        formatted_metrics[node_id] = {
            "accuracy": latest_metric.accuracy,
            "loss": latest_metric.loss,
            "epochs": latest_metric.epochs,
            "timestamp": latest_metric.timestamp.isoformat()
        }
        
        # Add additional metrics if available
        if latest_metric.additional_metrics:
            formatted_metrics[node_id].update(latest_metric.additional_metrics)
    
    return AllNodeMetricsResponse(
        graph_id=graph_id,
        metrics=formatted_metrics
    ) 