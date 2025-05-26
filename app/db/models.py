"""
Database Models using SQLAlchemy.

These define the database schema for projects, graphs, nodes, and metadata about ML models.
They are NOT related to:
- API schemas (see app.schemas.api_schemas) 
- The actual ML models themselves (which are stored via ursakit and cached)
"""
from sqlalchemy import Column, ForeignKey, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime
import uuid

Base = declarative_base()

def generate_uuid():
    return str(uuid.uuid4())

class Project(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    graphs = relationship("Graph", back_populates="project", cascade="all, delete-orphan")

class Graph(Base):
    __tablename__ = "graphs"

    id = Column(String, primary_key=True, default=generate_uuid)
    project_id = Column(String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="graphs")
    nodes = relationship("Node", back_populates="graph", cascade="all, delete-orphan")
    edges = relationship("Edge", back_populates="graph", cascade="all, delete-orphan")

class Node(Base):
    __tablename__ = "nodes"

    id = Column(String, primary_key=True, default=generate_uuid)
    graph_id = Column(String, ForeignKey("graphs.id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)
    model_id = Column(String, ForeignKey("models.id", ondelete="SET NULL"), nullable=True)
    meta_data = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    graph = relationship("Graph", back_populates="nodes")
    model = relationship("Model", back_populates="nodes")
    metrics = relationship("Metric", back_populates="node", cascade="all, delete-orphan")
    source_edges = relationship("Edge", foreign_keys="Edge.source_id", back_populates="source")
    target_edges = relationship("Edge", foreign_keys="Edge.target_id", back_populates="target")

class Edge(Base):
    __tablename__ = "edges"

    id = Column(String, primary_key=True, default=generate_uuid)
    graph_id = Column(String, ForeignKey("graphs.id", ondelete="CASCADE"), nullable=False)
    source_id = Column(String, ForeignKey("nodes.id", ondelete="CASCADE"), nullable=False)
    target_id = Column(String, ForeignKey("nodes.id", ondelete="CASCADE"), nullable=False)
    type = Column(String, nullable=False)
    meta_data = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    graph = relationship("Graph", back_populates="edges")
    source = relationship("Node", foreign_keys=[source_id], back_populates="source_edges")
    target = relationship("Node", foreign_keys=[target_id], back_populates="target_edges")

class Model(Base):
    __tablename__ = "models"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    framework = Column(String)
    model_type = Column(String)
    storage_path = Column(String, nullable=False)  # Path to model in storage (S3 or filesystem)
    storage_type = Column(String, nullable=False)  # "s3" or "filesystem"
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    nodes = relationship("Node", back_populates="model")

class Metric(Base):
    __tablename__ = "metrics"

    id = Column(String, primary_key=True, default=generate_uuid)
    node_id = Column(String, ForeignKey("nodes.id", ondelete="CASCADE"), nullable=False)
    accuracy = Column(Float, nullable=True)
    loss = Column(Float, nullable=True)
    epochs = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    additional_metrics = Column(JSON, default={})
    
    # Relationships
    node = relationship("Node", back_populates="metrics") 