from __future__ import annotations

import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metadata import MetadataStore
from .parser import parse_ursaml, serialize_ursaml


class ProjectsRepository:
    def __init__(self, base_path: Path, metadata: MetadataStore) -> None:
        self.projects_path = base_path / "projects"
        self.projects_path.mkdir(parents=True, exist_ok=True)
        self._metadata = metadata

    def create(self, name: str, description: str = "") -> Dict[str, Any]:
        project_id = str(uuid.uuid4())
        project = {
            'id': project_id,
            'project_id': project_id,
            'name': name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'graphs': []
        }
        self._metadata.data['projects'][project_id] = project
        self._metadata.save()
        (self.projects_path / project_id).mkdir(exist_ok=True)
        with (self.projects_path / project_id / 'info.json').open('w', encoding='utf-8') as f:
            import json
            json.dump(project, f, indent=2)
        return project

    def get(self, project_id: str) -> Optional[Dict[str, Any]]:
        project = self._metadata.data['projects'].get(project_id)
        if project:
            project['project_id'] = project['id']
        return project

    def all(self) -> List[Dict[str, Any]]:
        items = list(self._metadata.data['projects'].values())
        for p in items:
            p['project_id'] = p['id']
        return items

    def update(self, project_id: str, name: str, description: str) -> Optional[Dict[str, Any]]:
        if project_id not in self._metadata.data['projects']:
            return None
        project = self._metadata.data['projects'][project_id]
        project['name'] = name
        project['description'] = description
        self._metadata.save()
        with (self.projects_path / project_id / 'info.json').open('w', encoding='utf-8') as f:
            import json
            json.dump(project, f, indent=2)
        return project

    def delete(self, project_id: str) -> bool:
        if project_id not in self._metadata.data['projects']:
            return False
        project = self._metadata.data['projects'][project_id]
        for _ in project.get('graphs', []):
            # graphs repo should handle cascade; here we just remove metadata link
            pass
        del self._metadata.data['projects'][project_id]
        self._metadata.save()
        project_dir = self.projects_path / project_id
        if project_dir.exists():
            shutil.rmtree(project_dir)
        return True


class GraphsRepository:
    def __init__(self, base_path: Path, metadata: MetadataStore) -> None:
        self.graphs_path = base_path / "graphs"
        self.graphs_path.mkdir(parents=True, exist_ok=True)
        self._metadata = metadata

    def create(self, project_id: str, name: str, description: str = "") -> Optional[Dict[str, Any]]:
        if project_id not in self._metadata.data['projects']:
            return None
        graph_id = str(uuid.uuid4())
        graph = {
            'id': graph_id,
            'graph_id': graph_id,
            'project_id': project_id,
            'name': name,
            'description': description,
            'created_at': datetime.now().isoformat()
        }
        self._metadata.data['graphs'][graph_id] = graph
        self._metadata.data['projects'][project_id].setdefault('graphs', []).append(graph_id)
        self._metadata.save()

        ursaml_data = {
            'version': '0.1',
            'identifier': f"{name.replace(' ', '_')}_{graph_id[:8]}",
            'columns': ['score', 'name'],
            'column_values': {'score': [], 'name': []},
            'structure': [],
            'nodes': {}
        }
        graph_file = self.graphs_path / f"{graph_id}.ursaml"
        with graph_file.open('w', encoding='utf-8') as f:
            f.write(serialize_ursaml(ursaml_data))
        return graph

    def get(self, graph_id: str) -> Optional[Dict[str, Any]]:
        graph = self._metadata.data['graphs'].get(graph_id)
        if graph:
            graph['graph_id'] = graph['id']
        return graph

    def list_for_project(self, project_id: str) -> List[Dict[str, Any]]:
        graph_ids = self._metadata.data['projects'].get(project_id, {}).get('graphs', [])
        graphs = [self._metadata.data['graphs'][gid] for gid in graph_ids if gid in self._metadata.data['graphs']]
        for g in graphs:
            g['graph_id'] = g['id']
        return graphs

    def update(self, graph_id: str, name: str, description: str) -> Optional[Dict[str, Any]]:
        if graph_id not in self._metadata.data['graphs']:
            return None
        graph = self._metadata.data['graphs'][graph_id]
        graph['name'] = name
        graph['description'] = description
        self._metadata.save()
        return graph

    def delete(self, graph_id: str) -> bool:
        if graph_id not in self._metadata.data['graphs']:
            return False
        graph = self._metadata.data['graphs'][graph_id]
        project_id = graph['project_id']
        if project_id in self._metadata.data['projects']:
            graphs = self._metadata.data['projects'][project_id].get('graphs', [])
            if graph_id in graphs:
                graphs.remove(graph_id)
        del self._metadata.data['graphs'][graph_id]
        self._metadata.save()
        graph_file = self.graphs_path / f"{graph_id}.ursaml"
        if graph_file.exists():
            graph_file.unlink()
        return True

    def load_ursaml(self, graph_id: str) -> Optional[Dict[str, Any]]:
        graph_file = self.graphs_path / f"{graph_id}.ursaml"
        if not graph_file.exists():
            return None
        with graph_file.open('r', encoding='utf-8') as f:
            content = f.read()
        return parse_ursaml(content)

    def save_ursaml(self, graph_id: str, ursaml_data: Dict[str, Any]) -> None:
        graph_file = self.graphs_path / f"{graph_id}.ursaml"
        with graph_file.open('w', encoding='utf-8') as f:
            f.write(serialize_ursaml(ursaml_data))


class NodesRepository:
    def __init__(self, graphs_repo: GraphsRepository) -> None:
        self._graphs = graphs_repo

    def create(self, graph_id: str, name: str, model_id: str | None = None) -> Optional[Dict[str, Any]]:
        ursaml = self._graphs.load_ursaml(graph_id)
        if not ursaml:
            return None
        node_id = f"n{len(ursaml['nodes']) + 1}"
        node = {
            'id': node_id,
            'graph_id': graph_id,
            'name': name,
            'model_id': model_id,
            'created_at': datetime.now().isoformat()
        }
        ursaml['nodes'][node_id] = {
            'columns': {'score': 0.0, 'name': name},
            'detailed': {'id': node_id, 'name': name, 'model_id': model_id or "", 'created_at': node['created_at']}
        }
        if isinstance(ursaml.get('column_values', {}).get('score'), list):
            ursaml['column_values']['score'].append(0.0)
        if isinstance(ursaml.get('column_values', {}).get('name'), list):
            ursaml['column_values']['name'].append(name)
        self._graphs.save_ursaml(graph_id, ursaml)
        return node

    def get(self, graph_id: str, node_id: str) -> Optional[Dict[str, Any]]:
        ursaml = self._graphs.load_ursaml(graph_id)
        if not ursaml or node_id not in ursaml['nodes']:
            return None
        node_data = ursaml['nodes'][node_id]
        return {
            'id': node_id,
            'graph_id': graph_id,
            'name': node_data['columns'].get('name', ''),
            'model_id': node_data['detailed'].get('model_id'),
            'metadata': node_data['detailed']
        }

    def update(self, graph_id: str, node_id: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        ursaml = self._graphs.load_ursaml(graph_id)
        if not ursaml or node_id not in ursaml['nodes']:
            return None
        ursaml['nodes'][node_id]['detailed'].update(metadata)
        self._graphs.save_ursaml(graph_id, ursaml)
        return self.get(graph_id, node_id)

    def delete(self, graph_id: str, node_id: str) -> bool:
        ursaml = self._graphs.load_ursaml(graph_id)
        if not ursaml or node_id not in ursaml['nodes']:
            return False
        del ursaml['nodes'][node_id]
        ursaml['structure'] = [edge for edge in ursaml['structure'] if edge[0] != node_id and edge[1] != node_id]
        self._graphs.save_ursaml(graph_id, ursaml)
        return True

    def list_for_graph(self, graph_id: str) -> List[Dict[str, Any]]:
        ursaml = self._graphs.load_ursaml(graph_id)
        if not ursaml:
            return []
        nodes: List[Dict[str, Any]] = []
        for node_id, node_data in ursaml['nodes'].items():
            nodes.append({
                'id': node_id,
                'graph_id': graph_id,
                'name': node_data['columns'].get('name', ''),
                'model_id': node_data['detailed'].get('model_id'),
                'metadata': node_data['detailed']
            })
        return nodes

    def create_edge(self, graph_id: str, source_id: str, target_id: str, edge_type: str = "default", weight: float = 1.0) -> bool:
        ursaml = self._graphs.load_ursaml(graph_id)
        if not ursaml:
            return False
        if source_id not in ursaml['nodes'] or target_id not in ursaml['nodes']:
            return False
        ursaml['structure'].append((source_id, target_id, weight, edge_type))
        self._graphs.save_ursaml(graph_id, ursaml)
        return True

    def list_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        ursaml = self._graphs.load_ursaml(graph_id)
        if not ursaml:
            return []
        edges: List[Dict[str, Any]] = []
        for source, target, weight, edge_type in ursaml['structure']:
            edges.append({'source_id': source, 'target_id': target, 'weight': weight, 'type': edge_type, 'graph_id': graph_id})
        return edges


class ModelsRepository:
    def __init__(self, base_path: Path) -> None:
        self.models_path = base_path / "models"
        self.models_path.mkdir(parents=True, exist_ok=True)

    def save(self, model_data: bytes, model_id: str) -> str:
        model_dir = self.models_path / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        file_path = model_dir / "model"
        with file_path.open('wb') as f:
            f.write(model_data)
        metadata = {
            "id": model_id,
            "created_at": datetime.now().isoformat(),
            "path": str(file_path),
            "artifacts": {"model": {"path": str(file_path), "type": "unknown"}},
        }
        with (model_dir / "metadata.json").open('w', encoding='utf-8') as f:
            import json
            json.dump(metadata, f, indent=2)
        return str(file_path)

    def get(self, model_id: str) -> Optional[bytes]:
        model_dir = self.models_path / model_id
        if not model_dir.exists():
            return None
        try:
            import json
            with (model_dir / "metadata.json").open('r', encoding='utf-8') as f:
                metadata = json.load(f)
            if "path" in metadata:
                model_path = Path(metadata["path"])
                if not model_path.exists():
                    model_path = model_dir / model_path.name
            elif "artifacts" in metadata and "model" in metadata["artifacts"]:
                model_path = Path(metadata["artifacts"]["model"]["path"])
                if not model_path.exists():
                    model_path = model_dir / model_path.name
            else:
                return None
            if not model_path.exists():
                return None
            with model_path.open('rb') as f:
                return f.read()
        except Exception:
            return None

    def delete(self, model_id: str) -> bool:
        model_dir = self.models_path / model_id
        if model_dir.exists():
            shutil.rmtree(model_dir)
            return True
        return False


