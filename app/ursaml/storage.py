"""
UrsaML Storage - Composed façade delegating to repositories for projects, graphs, nodes, models.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import shutil

from .repositories import ProjectsRepository, GraphsRepository, NodesRepository, ModelsRepository
from .metadata import MetadataStore
from .parser import parse_ursaml, serialize_ursaml


class UrsaMLStorage:
    """File-based storage using UrsaML format, composed of repositories."""
    
    def __init__(self, base_path: str = "data/ursaml"):
        self.base_path = Path(base_path)
        self.projects_path = self.base_path / "projects"
        self.graphs_path = self.base_path / "graphs"
        self.models_path = self.base_path / "models"
        self.metadata_file = self.base_path / "metadata.json"

        # Ensure directories
        self.projects_path.mkdir(parents=True, exist_ok=True)
        self.graphs_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)

        # Compose repositories
        self._metadata = MetadataStore(self.metadata_file)
        self._projects = ProjectsRepository(self.base_path, self._metadata)
        self._graphs = GraphsRepository(self.base_path, self._metadata)
        self._nodes = NodesRepository(self._graphs)
        self._models = ModelsRepository(self.base_path)

    # Compatibility helpers for health checks/tests
    def _load_metadata(self) -> Dict[str, Any]:
        return self._metadata.data

    def _save_metadata(self, metadata: Dict[str, Any] = None):
        if metadata is not None:
            self._metadata.data = metadata
        self._metadata.save()

    # Project operations
    def create_project(self, name: str, description: str = "") -> Dict[str, Any]:
        return self._projects.create(name, description)

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        return self._projects.get(project_id)

    def get_all_projects(self) -> List[Dict[str, Any]]:
        return self._projects.all()

    def update_project(self, project_id: str, name: str, description: str) -> Optional[Dict[str, Any]]:
        return self._projects.update(project_id, name, description)

    def delete_project(self, project_id: str) -> bool:
        project = self._projects.get(project_id)
        if not project:
            return False
        for graph_id in project.get('graphs', []):
            self.delete_graph(graph_id)
        return self._projects.delete(project_id)

    # Graph operations
    def create_graph(self, project_id: str, name: str, description: str = "") -> Optional[Dict[str, Any]]:
        return self._graphs.create(project_id, name, description)

    def get_graph(self, graph_id: str) -> Optional[Dict[str, Any]]:
        return self._graphs.get(graph_id)

    def get_project_graphs(self, project_id: str) -> List[Dict[str, Any]]:
        return self._graphs.list_for_project(project_id)

    def update_graph(self, graph_id: str, name: str, description: str) -> Optional[Dict[str, Any]]:
        return self._graphs.update(graph_id, name, description)

    def delete_graph(self, graph_id: str) -> bool:
        return self._graphs.delete(graph_id)

    def load_graph_ursaml(self, graph_id: str) -> Optional[Dict[str, Any]]:
        return self._graphs.load_ursaml(graph_id)

    def save_graph_ursaml(self, graph_id: str, ursaml_data: Dict[str, Any]):
        return self._graphs.save_ursaml(graph_id, ursaml_data)

    # Node operations
    def create_node(self, graph_id: str, name: str, model_id: str = None) -> Optional[Dict[str, Any]]:
        return self._nodes.create(graph_id, name, model_id)

    def get_node(self, graph_id: str, node_id: str) -> Optional[Dict[str, Any]]:
        return self._nodes.get(graph_id, node_id)

    def update_node(self, graph_id: str, node_id: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self._nodes.update(graph_id, node_id, metadata)

    def delete_node(self, graph_id: str, node_id: str) -> bool:
        return self._nodes.delete(graph_id, node_id)

    def get_graph_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        return self._nodes.list_for_graph(graph_id)

    def create_edge(self, graph_id: str, source_id: str, target_id: str, 
                    edge_type: str = "default", weight: float = 1.0) -> bool:
        return self._nodes.create_edge(graph_id, source_id, target_id, edge_type, weight)

    def get_graph_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        return self._nodes.list_edges(graph_id)

    # Model operations
    def save_model(self, model_data: bytes, model_id: str) -> str:
        return self._models.save(model_data, model_id)

    def get_model(self, model_id: str) -> Optional[bytes]:
        return self._models.get(model_id)

    def delete_model(self, model_id: str) -> bool:
        return self._models.delete(model_id)

    # Metrics operations
    def add_metrics(self, graph_id: str, node_id: str, metrics: Dict[str, Any]):
        ursaml_data = self.load_graph_ursaml(graph_id)
        if not ursaml_data or node_id not in ursaml_data['nodes']:
            return None
        if 'accuracy' in metrics and 'score' in ursaml_data['nodes'][node_id]['columns']:
            ursaml_data['nodes'][node_id]['columns']['score'] = metrics['accuracy']
        if 'meta' not in ursaml_data['nodes'][node_id]['detailed']:
            ursaml_data['nodes'][node_id]['detailed']['meta'] = {}
        ursaml_data['nodes'][node_id]['detailed']['meta'].update({
            'score': metrics.get('accuracy', 0.0),
            'loss': metrics.get('loss', 0.0),
            'epochs': metrics.get('epochs', 0),
            'metrics_timestamp': datetime.now().isoformat()
        })
        for key, value in metrics.items():
            if key not in ['accuracy', 'loss', 'epochs']:
                ursaml_data['nodes'][node_id]['detailed']['meta'][key] = value
        self.save_graph_ursaml(graph_id, ursaml_data)
        return metrics