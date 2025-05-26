from sqlalchemy.orm import Session
from app.db.models import Node, Edge, Graph, Metric
from typing import List, Optional, Dict, Any

class NodeRepository:
    """Repository for node and edge operations."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_node(self, graph_id: str, name: str, model_id: str = None, metadata: Dict = None) -> Optional[Node]:
        """
        Create a new node in a graph.
        
        Args:
            graph_id: Graph ID
            name: Node name
            model_id: Model ID (optional)
            metadata: Node metadata (optional)
            
        Returns:
            Created node or None if graph not found
        """
        # Verify graph exists
        graph = self.db.query(Graph).filter(Graph.id == graph_id).first()
        if not graph:
            return None
        
        node = Node(
            graph_id=graph_id,
            name=name,
            model_id=model_id,
            meta_data=metadata or {}
        )
        self.db.add(node)
        self.db.commit()
        self.db.refresh(node)
        return node
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """
        Get a node by ID.
        
        Args:
            node_id: Node ID
            
        Returns:
            Node if found, None otherwise
        """
        return self.db.query(Node).filter(Node.id == node_id).first()
    
    def update_node(self, node_id: str, metadata: Dict[str, Any]) -> Optional[Node]:
        """
        Update node metadata.
        
        Args:
            node_id: Node ID
            metadata: Node metadata
            
        Returns:
            Updated node or None if node not found
        """
        node = self.get_node(node_id)
        if not node:
            return None
        
        node.meta_data = metadata
        self.db.commit()
        self.db.refresh(node)
        return node
    
    def replace_node_model(self, node_id: str, model_id: str) -> Optional[Node]:
        """
        Replace the model associated with a node.
        
        Args:
            node_id: Node ID
            model_id: New model ID
            
        Returns:
            Updated node or None if node not found
        """
        node = self.get_node(node_id)
        if not node:
            return None
        
        node.model_id = model_id
        self.db.commit()
        self.db.refresh(node)
        return node
    
    def delete_node(self, node_id: str) -> bool:
        """
        Delete a node by ID.
        
        Args:
            node_id: Node ID
            
        Returns:
            True if node was deleted, False otherwise
        """
        node = self.get_node(node_id)
        if not node:
            return False
        
        self.db.delete(node)
        self.db.commit()
        return True
    
    def get_graph_nodes(self, graph_id: str) -> List[Node]:
        """
        Get all nodes in a graph.
        
        Args:
            graph_id: Graph ID
            
        Returns:
            List of nodes in the graph
        """
        return self.db.query(Node).filter(Node.graph_id == graph_id).all()
    
    def create_edge(self, graph_id: str, source_id: str, target_id: str, 
                    edge_type: str, metadata: Dict = None) -> Optional[Edge]:
        """
        Create a new edge between nodes.
        
        Args:
            graph_id: Graph ID
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Edge type (e.g., 'difference')
            metadata: Edge metadata (optional)
            
        Returns:
            Created edge or None if nodes not found
        """
        # Verify nodes exist
        source = self.db.query(Node).filter(Node.id == source_id).first()
        target = self.db.query(Node).filter(Node.id == target_id).first()
        
        if not source or not target:
            return None
        
        edge = Edge(
            graph_id=graph_id,
            source_id=source_id,
            target_id=target_id,
            type=edge_type,
            meta_data=metadata or {}
        )
        self.db.add(edge)
        self.db.commit()
        self.db.refresh(edge)
        return edge
    
    def get_graph_edges(self, graph_id: str) -> List[Edge]:
        """
        Get all edges in a graph.
        
        Args:
            graph_id: Graph ID
            
        Returns:
            List of edges in the graph
        """
        return self.db.query(Edge).filter(Edge.graph_id == graph_id).all()
    
    def add_metrics(self, node_id: str, accuracy: float = None, loss: float = None, 
                    epochs: int = None, additional_metrics: Dict = None) -> Optional[Metric]:
        """
        Add metrics for a node.
        
        Args:
            node_id: Node ID
            accuracy: Accuracy value (optional)
            loss: Loss value (optional)
            epochs: Number of epochs (optional)
            additional_metrics: Additional metrics (optional)
            
        Returns:
            Created metric or None if node not found
        """
        # Verify node exists
        node = self.db.query(Node).filter(Node.id == node_id).first()
        if not node:
            return None
        
        metric = Metric(
            node_id=node_id,
            accuracy=accuracy,
            loss=loss,
            epochs=epochs,
            additional_metrics=additional_metrics or {}
        )
        self.db.add(metric)
        self.db.commit()
        self.db.refresh(metric)
        return metric
    
    def get_node_metrics(self, node_id: str) -> List[Metric]:
        """
        Get all metrics for a node.
        
        Args:
            node_id: Node ID
            
        Returns:
            List of metrics for the node
        """
        return self.db.query(Metric).filter(Metric.node_id == node_id).all()
    
    def get_graph_metrics(self, graph_id: str) -> Dict[str, List[Metric]]:
        """
        Get all metrics for all nodes in a graph.
        
        Args:
            graph_id: Graph ID
            
        Returns:
            Dictionary mapping node IDs to lists of metrics
        """
        # Get all nodes in the graph
        nodes = self.get_graph_nodes(graph_id)
        
        # Get metrics for each node
        metrics = {}
        for node in nodes:
            metrics[node.id] = self.get_node_metrics(node.id)
        
        return metrics 