"""
Database initialization and migration utilities.
"""
import logging
from sqlalchemy import create_engine, text
from app.db.models import Base
from app.config import settings, get_db_components

logger = logging.getLogger(__name__)


def create_database_if_not_exists():
    """Create the database if it doesn't exist."""
    db_components = get_db_components()
    db_name = db_components["db_name"]
    db_url_without_name = db_components["db_url_without_name"]
    
    engine = create_engine(db_url_without_name)
    
    with engine.connect() as conn:
        conn.execute(text("COMMIT"))
        
        # Check if database exists
        result = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
            {"db_name": db_name}
        )
        
        if not result.fetchone():
            logger.info(f"Creating database: {db_name}")
            # Note: Database names cannot be parameterized in PostgreSQL
            # db_name is validated through get_db_components() parsing
            conn.execute(text(f"CREATE DATABASE {db_name}"))
            logger.info(f"Database {db_name} created successfully")
        else:
            logger.info(f"Database {db_name} already exists")
    
    engine.dispose()


def create_tables():
    """Create all tables defined in models."""
    db_components = get_db_components()
    engine = create_engine(db_components["db_url"])
    
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("All tables created successfully")
    
    engine.dispose()


def drop_all_tables():
    """Drop all tables (useful for testing)."""
    db_components = get_db_components()
    engine = create_engine(db_components["db_url"])
    
    logger.info("Dropping all database tables...")
    Base.metadata.drop_all(bind=engine)
    logger.info("All tables dropped successfully")
    
    engine.dispose()


def reset_database():
    """Drop and recreate all tables."""
    logger.info("Resetting database...")
    drop_all_tables()
    create_tables()
    logger.info("Database reset complete")


def init_database():
    """Complete database initialization."""
    logger.info("Initializing database...")
    create_database_if_not_exists()
    create_tables()
    logger.info("Database initialization complete")


def test_connection():
    """Test database connection."""
    db_components = get_db_components()
    engine = create_engine(db_components["db_url"])
    
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        assert result.fetchone()[0] == 1
    
    logger.info("Database connection test successful")
    engine.dispose()
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_database() 