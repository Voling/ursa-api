"""
Basic PostgreSQL integration tests.
"""
import pytest
from sqlalchemy import text
from app.db.database import SessionLocal
from app.db.repositories.projects import ProjectRepository
from app.config import settings


class TestPostgreSQLBasic:
    """Basic tests to verify PostgreSQL integration."""
    
    def test_database_url_is_postgres(self):
        """Verify we're using PostgreSQL on port 5432."""
        db_url = str(settings.DATABASE_URL)
        assert "postgresql://" in db_url
        assert ":5432/" in db_url
        assert "ursa" in db_url
        print(f"Using PostgreSQL URL: {db_url}")
    
    def test_direct_connection(self):
        """Test direct database connection."""
        db = SessionLocal()
        try:
            result = db.execute(text("SELECT current_database(), version()"))
            row = result.fetchone()
            db_name, version = row
            
            assert db_name == "ursa"
            assert "PostgreSQL" in version
            print(f"Connected to PostgreSQL database: {db_name}")
            print(f"Version: {version}")
        finally:
            db.close()
    
    def test_tables_exist(self):
        """Verify all required tables exist."""
        db = SessionLocal()
        try:
            result = db.execute(text("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name
            """))
            tables = [row[0] for row in result.fetchall()]
            
            required_tables = ['projects', 'graphs', 'nodes', 'models', 'edges', 'metrics']
            for table in required_tables:
                assert table in tables, f"Table {table} not found"
            
            print(f"All required tables exist: {required_tables}")
        finally:
            db.close()
    
    def test_project_repository_basic(self):
        """Test basic project repository functionality."""
        # Use the test_db fixture for proper session management
        # The test fixture returns a SessionLocal factory
        db = SessionLocal()
        
        project_repo = ProjectRepository(db)
        
        # Create a uniquely named project
        import uuid
        unique_name = f"Test Project {uuid.uuid4().hex[:8]}"
        
        try:
            project = project_repo.create_project(unique_name, "Test Description")
            assert project.name == unique_name
            assert project.id is not None
            
            # Read project back
            fetched_project = project_repo.get_project(project.id)
            assert fetched_project is not None
            assert fetched_project.name == unique_name
            
            print(f"Created and retrieved project: {unique_name}")
            
            # Clean up this specific project
            project_repo.delete_project(project.id)
        finally:
            db.close()
    
    def test_constraint_and_relationships(self):
        """Test that foreign key constraints work."""
        db = SessionLocal()
        try:
            from app.db.repositories.graphs import GraphRepository
            from app.db.repositories.projects import ProjectRepository
            
            project_repo = ProjectRepository(db)
            graph_repo = GraphRepository(db)
            
            # Create project
            import uuid
            project_name = f"Constraint Test {uuid.uuid4().hex[:8]}"
            project = project_repo.create_project(project_name, "Constraint test")
            
            # Create graph in project
            graph = graph_repo.create_graph(project.id, "Test Graph", "Test graph description")
            assert graph is not None
            assert graph.project_id == project.id
            
            print("Foreign key relationships working")
            
            # Clean up
            project_repo.delete_project(project.id)  # Should cascade delete graph
        finally:
            db.close()
    
    def test_concurrent_sessions(self):
        """Test that multiple sessions can work simultaneously."""
        db1 = SessionLocal()
        db2 = SessionLocal()
        try:
            # Both sessions should be able to query
            result1 = db1.execute(text("SELECT 1 as test"))
            result2 = db2.execute(text("SELECT 2 as test"))
            
            assert result1.fetchone()[0] == 1
            assert result2.fetchone()[0] == 2
            
            print("Concurrent sessions working")
        finally:
            db1.close()
            db2.close() 