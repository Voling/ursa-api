"""
UrsaML Storage - File-based storage system for graphs, projects, and models
"""
import os
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from .parser import parse_ursaml, serialize_ursaml
import shutil

class UrsaMLStorage:
    """File-based storage using UrsaML format."""
    
    def __init__(self, base_path: str = "data/ursaml"):
        self.base_path = Path(base_path)
        self.projects_path = self.base_path / "projects"
        self.graphs_path = self.base_path / "graphs"
        self.models_path = self.base_path / "models"
        self.metadata_file = self.base_path / "metadata.json"
        
        # Create directories
        self.projects_path.mkdir(parents=True, exist_ok=True)
        self.graphs_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Load or create metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load system metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        else:
            metadata = {
                'projects': {},
                'graphs': {},
                'models': {}
            }
            self._save_metadata(metadata)
            return metadata
    
    def _save_metadata(self, metadata: Dict[str, Any] = None):
        """Save system metadata."""
        if metadata:
            self.metadata = metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _generate_id(self) -> str:
        """Generate a unique ID."""
        return str(uuid.uuid4())
    
    # Project operations
    def create_project(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new project."""
        project_id = self._generate_id()
        project = {
            'id': project_id,
            'project_id': project_id,
            'name': name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'graphs': []
        }
        
        # Save to metadata
        self.metadata['projects'][project_id] = project
        self._save_metadata()
        
        # Create project directory
        project_dir = self.projects_path / project_id
        project_dir.mkdir(exist_ok=True)
        
        # Save project info
        with open(project_dir / 'info.json', 'w') as f:
            json.dump(project, f, indent=2)
        
        return project
    
    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get a project by ID."""
        project = self.metadata['projects'].get(project_id)
        if project:
            project['project_id'] = project['id']
        return project
    
    def get_all_projects(self) -> List[Dict[str, Any]]:
        """Get all projects."""
        projects = list(self.metadata['projects'].values())
        for project in projects:
            project['project_id'] = project['id']
        return projects
    
    def update_project(self, project_id: str, name: str, description: str) -> Optional[Dict[str, Any]]:
        """Update a project."""
        if project_id not in self.metadata['projects']:
            return None
        
        project = self.metadata['projects'][project_id]
        project['name'] = name
        project['description'] = description
        
        self._save_metadata()
        
        # Update project info file
        project_dir = self.projects_path / project_id
        with open(project_dir / 'info.json', 'w') as f:
            json.dump(project, f, indent=2)
        
        return project
    
    def delete_project(self, project_id: str) -> bool:
        """Delete a project and all its graphs."""
        if project_id not in self.metadata['projects']:
            return False
        
        # Delete all graphs in the project
        project = self.metadata['projects'][project_id]
        for graph_id in project.get('graphs', []):
            self.delete_graph(graph_id)
        
        # Delete project
        del self.metadata['projects'][project_id]
        self._save_metadata()
        
        # Delete project directory
        project_dir = self.projects_path / project_id
        if project_dir.exists():
            shutil.rmtree(project_dir)
        
        return True
    
    # Graph operations
    def create_graph(self, project_id: str, name: str, description: str = "") -> Optional[Dict[str, Any]]:
        """Create a new graph in a project."""
        # Get project to validate it exists
        project = self.get_project(project_id)
        if not project:
            return None
        
        graph_id = self._generate_id()
        graph = {
            'id': graph_id,
            'graph_id': graph_id,
            'project_id': project_id,
            'name': name,
            'description': description,
            'created_at': datetime.now().isoformat()
        }
        
        # Save to metadata
        self.metadata['graphs'][graph_id] = graph
        if 'graphs' not in self.metadata['projects'][project_id]:
            self.metadata['projects'][project_id]['graphs'] = []
        self.metadata['projects'][project_id]['graphs'].append(graph_id)
        self._save_metadata()
        
        # Create empty UrsaML file for the graph
        ursaml_data = {
            'version': '0.1',
            'identifier': f"{name.replace(' ', '_')}_{graph_id[:8]}",
            'columns': ['score', 'name'],
            'column_values': {
                'score': [],
                'name': []
            },
            'structure': [],
            'nodes': {}
        }
        
        graph_file = self.graphs_path / f"{graph_id}.ursaml"
        with open(graph_file, 'w') as f:
            f.write(serialize_ursaml(ursaml_data))
        
        return graph
    
    def get_graph(self, graph_id: str) -> Optional[Dict[str, Any]]:
        """Get a graph by ID."""
        graph = self.metadata['graphs'].get(graph_id)
        if graph:
            graph['graph_id'] = graph['id']
        return graph
    
    def get_project_graphs(self, project_id: str) -> List[Dict[str, Any]]:
        """Get all graphs in a project."""
        if project_id not in self.metadata['projects']:
            return []
        
        graph_ids = self.metadata['projects'][project_id].get('graphs', [])
        graphs = [self.metadata['graphs'][gid] for gid in graph_ids if gid in self.metadata['graphs']]
        for graph in graphs:
            graph['graph_id'] = graph['id']
        return graphs
    
    def update_graph(self, graph_id: str, name: str, description: str) -> Optional[Dict[str, Any]]:
        """Update a graph."""
        if graph_id not in self.metadata['graphs']:
            return None
        
        graph = self.metadata['graphs'][graph_id]
        graph['name'] = name
        graph['description'] = description
        
        self._save_metadata()
        return graph
    
    def delete_graph(self, graph_id: str) -> bool:
        """Delete a graph."""
        if graph_id not in self.metadata['graphs']:
            return False
        
        graph = self.metadata['graphs'][graph_id]
        project_id = graph['project_id']
        
        # Remove from project
        if project_id in self.metadata['projects']:
            graphs = self.metadata['projects'][project_id].get('graphs', [])
            if graph_id in graphs:
                graphs.remove(graph_id)
        
        # Delete graph metadata
        del self.metadata['graphs'][graph_id]
        self._save_metadata()
        
        # Delete UrsaML file
        graph_file = self.graphs_path / f"{graph_id}.ursaml"
        if graph_file.exists():
            graph_file.unlink()
        
        return True
    
    def load_graph_ursaml(self, graph_id: str) -> Optional[Dict[str, Any]]:
        """Load and parse a graph's UrsaML file."""
        graph_file = self.graphs_path / f"{graph_id}.ursaml"
        if not graph_file.exists():
            return None
        
        with open(graph_file, 'r') as f:
            content = f.read()
        
        return parse_ursaml(content)
    
    def save_graph_ursaml(self, graph_id: str, ursaml_data: Dict[str, Any]):
        """Save a graph's UrsaML data."""
        graph_file = self.graphs_path / f"{graph_id}.ursaml"
        with open(graph_file, 'w') as f:
            f.write(serialize_ursaml(ursaml_data))
    
    # Node operations (work with UrsaML data)
    def create_node(self, graph_id: str, name: str, model_id: str = None) -> Optional[Dict[str, Any]]:
        """Create a node in a graph."""
        ursaml_data = self.load_graph_ursaml(graph_id)
        if not ursaml_data:
            return None
        
        node_id = f"n{len(ursaml_data['nodes']) + 1}"
        
        # Create node data
        node = {
            'id': node_id,
            'graph_id': graph_id,
            'name': name,
            'model_id': model_id,
            'created_at': datetime.now().isoformat()
        }
        
        # Add to UrsaML data
        ursaml_data['nodes'][node_id] = {
            'columns': {
                'score': 0.0,
                'name': name
            },
            'detailed': {
                'id': node_id,
                'name': name,
                'model_id': model_id or "",
                'created_at': node['created_at']
            }
        }
        
        # Update column values if needed
        if 'score' in ursaml_data['column_values']:
            if isinstance(ursaml_data['column_values']['score'], list):
                ursaml_data['column_values']['score'].append(0.0)
        if 'name' in ursaml_data['column_values']:
            if isinstance(ursaml_data['column_values']['name'], list):
                ursaml_data['column_values']['name'].append(name)
        
        self.save_graph_ursaml(graph_id, ursaml_data)
        return node
    
    def get_node(self, graph_id: str, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node from a graph."""
        ursaml_data = self.load_graph_ursaml(graph_id)
        if not ursaml_data or node_id not in ursaml_data['nodes']:
            return None
        
        node_data = ursaml_data['nodes'][node_id]
        return {
            'id': node_id,
            'graph_id': graph_id,
            'name': node_data['columns'].get('name', ''),
            'model_id': node_data['detailed'].get('model_id'),
            'metadata': node_data['detailed']
        }
    
    def update_node(self, graph_id: str, node_id: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update node metadata."""
        ursaml_data = self.load_graph_ursaml(graph_id)
        if not ursaml_data or node_id not in ursaml_data['nodes']:
            return None
        
        # Update detailed metadata
        ursaml_data['nodes'][node_id]['detailed'].update(metadata)
        
        self.save_graph_ursaml(graph_id, ursaml_data)
        return self.get_node(graph_id, node_id)
    
    def delete_node(self, graph_id: str, node_id: str) -> bool:
        """Delete a node from a graph."""
        ursaml_data = self.load_graph_ursaml(graph_id)
        if not ursaml_data or node_id not in ursaml_data['nodes']:
            return False
        
        # Remove node
        del ursaml_data['nodes'][node_id]
        
        # Remove edges involving this node
        ursaml_data['structure'] = [
            edge for edge in ursaml_data['structure']
            if edge[0] != node_id and edge[1] != node_id
        ]
        
        self.save_graph_ursaml(graph_id, ursaml_data)
        return True
    
    def get_graph_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        """Get all nodes in a graph."""
        ursaml_data = self.load_graph_ursaml(graph_id)
        if not ursaml_data:
            return []
        
        nodes = []
        for node_id, node_data in ursaml_data['nodes'].items():
            nodes.append({
                'id': node_id,
                'graph_id': graph_id,
                'name': node_data['columns'].get('name', ''),
                'model_id': node_data['detailed'].get('model_id'),
                'metadata': node_data['detailed']
            })
        
        return nodes
    
    def create_edge(self, graph_id: str, source_id: str, target_id: str, 
                    edge_type: str = "default", weight: float = 1.0) -> bool:
        """Create an edge between nodes."""
        ursaml_data = self.load_graph_ursaml(graph_id)
        if not ursaml_data:
            return False
        
        # Verify nodes exist
        if source_id not in ursaml_data['nodes'] or target_id not in ursaml_data['nodes']:
            return False
        
        # Add edge
        ursaml_data['structure'].append((source_id, target_id, weight, edge_type))
        
        self.save_graph_ursaml(graph_id, ursaml_data)
        return True
    
    def get_graph_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        """Get all edges in a graph."""
        ursaml_data = self.load_graph_ursaml(graph_id)
        if not ursaml_data:
            return []
        
        edges = []
        for edge in ursaml_data['structure']:
            source, target, weight, edge_type = edge
            edges.append({
                'source_id': source,
                'target_id': target,
                'weight': weight,
                'type': edge_type,
                'graph_id': graph_id
            })
        
        return edges
    
    # Model operations
    def save_model(self, model_data: bytes, model_id: str) -> str:
        """Save model binary data."""
        model_path = self.models_path / f"{model_id}.pkl"
        with open(model_path, 'wb') as f:
            f.write(model_data)
        return str(model_path)
    
    def get_model(self, model_id: str) -> Optional[bytes]:
        """Get model binary data."""
        model_path = self.models_path / f"{model_id}.pkl"
        if not model_path.exists():
            return None
        
        with open(model_path, 'rb') as f:
            return f.read()
    
    def delete_model(self, model_id: str) -> bool:
        """Delete model file."""
        model_path = self.models_path / f"{model_id}.pkl"
        if model_path.exists():
            model_path.unlink()
            return True
        return False
    
    # Metrics operations
    def add_metrics(self, graph_id: str, node_id: str, metrics: Dict[str, Any]):
        """Add metrics to a node."""
        ursaml_data = self.load_graph_ursaml(graph_id)
        if not ursaml_data or node_id not in ursaml_data['nodes']:
            return None
        
        # Update score column if present
        if 'accuracy' in metrics and 'score' in ursaml_data['nodes'][node_id]['columns']:
            ursaml_data['nodes'][node_id]['columns']['score'] = metrics['accuracy']
        
        # Add metrics to detailed metadata
        if 'meta' not in ursaml_data['nodes'][node_id]['detailed']:
            ursaml_data['nodes'][node_id]['detailed']['meta'] = {}
        
        ursaml_data['nodes'][node_id]['detailed']['meta'].update({
            'score': metrics.get('accuracy', 0.0),
            'loss': metrics.get('loss', 0.0),
            'epochs': metrics.get('epochs', 0),
            'metrics_timestamp': datetime.now().isoformat()
        })
        
        # Add additional metrics
        for key, value in metrics.items():
            if key not in ['accuracy', 'loss', 'epochs']:
                ursaml_data['nodes'][node_id]['detailed']['meta'][key] = value
        
        self.save_graph_ursaml(graph_id, ursaml_data)
        return metrics 