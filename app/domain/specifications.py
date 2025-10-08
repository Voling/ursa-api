"""Specification pattern for reusable query logic."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Specification(ABC):
    """Abstract base for specifications (query filters)."""
    
    @abstractmethod
    def is_satisfied_by(self, candidate: Dict[str, Any]) -> bool:
        """Check if candidate satisfies this specification."""
        pass
    
    def and_(self, other: Specification) -> Specification:
        """Combine with AND logic."""
        return AndSpecification(self, other)
    
    def or_(self, other: Specification) -> Specification:
        """Combine with OR logic."""
        return OrSpecification(self, other)
    
    def not_(self) -> Specification:
        """Negate this specification."""
        return NotSpecification(self)


class AndSpecification(Specification):
    """AND composite specification."""
    
    def __init__(self, left: Specification, right: Specification):
        self.left = left
        self.right = right
    
    def is_satisfied_by(self, candidate: Dict[str, Any]) -> bool:
        return self.left.is_satisfied_by(candidate) and self.right.is_satisfied_by(candidate)


class OrSpecification(Specification):
    """OR composite specification."""
    
    def __init__(self, left: Specification, right: Specification):
        self.left = left
        self.right = right
    
    def is_satisfied_by(self, candidate: Dict[str, Any]) -> bool:
        return self.left.is_satisfied_by(candidate) or self.right.is_satisfied_by(candidate)


class NotSpecification(Specification):
    """NOT specification."""
    
    def __init__(self, spec: Specification):
        self.spec = spec
    
    def is_satisfied_by(self, candidate: Dict[str, Any]) -> bool:
        return not self.spec.is_satisfied_by(candidate)


# Project Specifications

class ProjectByName(Specification):
    """Finds projects by name (case-insensitive contains)."""
    
    def __init__(self, name_pattern: str):
        self.pattern = name_pattern.lower()
    
    def is_satisfied_by(self, project: Dict[str, Any]) -> bool:
        return self.pattern in project.get("name", "").lower()


class ProjectByDescription(Specification):
    """Finds projects by description keyword."""
    
    def __init__(self, keyword: str):
        self.keyword = keyword.lower()
    
    def is_satisfied_by(self, project: Dict[str, Any]) -> bool:
        return self.keyword in project.get("description", "").lower()


class ProjectHasGraphs(Specification):
    """Projects that have at least one graph."""
    
    def __init__(self, storage_graphs_getter):
        """
        Args:
            storage_graphs_getter: Callable that returns list of graphs for a project_id
        """
        self.get_graphs = storage_graphs_getter
    
    def is_satisfied_by(self, project: Dict[str, Any]) -> bool:
        graphs = self.get_graphs(project["id"])
        return len(graphs) > 0


# Graph Specifications

class GraphByName(Specification):
    """Finds graphs by name (case-insensitive contains)."""
    
    def __init__(self, name_pattern: str):
        self.pattern = name_pattern.lower()
    
    def is_satisfied_by(self, graph: Dict[str, Any]) -> bool:
        return self.pattern in graph.get("name", "").lower()


class GraphInProject(Specification):
    """Graphs belonging to a specific project."""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
    
    def is_satisfied_by(self, graph: Dict[str, Any]) -> bool:
        return graph.get("project_id") == self.project_id


class GraphHasNodes(Specification):
    """Graphs that have at least one node."""
    
    def __init__(self, storage_nodes_getter):
        """
        Args:
            storage_nodes_getter: Callable that returns list of nodes for a graph_id
        """
        self.get_nodes = storage_nodes_getter
    
    def is_satisfied_by(self, graph: Dict[str, Any]) -> bool:
        nodes = self.get_nodes(graph["id"])
        return len(nodes) > 0


# Node Specifications

class NodeWithModel(Specification):
    """Nodes that have an associated model."""
    
    def is_satisfied_by(self, node: Dict[str, Any]) -> bool:
        model_id = node.get("model_id")
        return model_id is not None and model_id != ""


class NodeHasMetrics(Specification):
    """Nodes that have recorded metrics."""
    
    def is_satisfied_by(self, node: Dict[str, Any]) -> bool:
        metadata = node.get("metadata", {})
        meta = metadata.get("meta", {})
        return "score" in meta or "accuracy" in meta or "loss" in meta


class NodeInGraph(Specification):
    """Nodes belonging to a specific graph."""
    
    def __init__(self, graph_id: str):
        self.graph_id = graph_id
    
    def is_satisfied_by(self, node: Dict[str, Any]) -> bool:
        return node.get("graph_id") == self.graph_id


# Helper function to filter collections

def filter_by_specification(items: List[Dict[str, Any]], spec: Specification) -> List[Dict[str, Any]]:
    """Filter a collection using a specification."""
    return [item for item in items if spec.is_satisfied_by(item)]

