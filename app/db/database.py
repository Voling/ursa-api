from sqlalchemy import create_engine, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging

from app.config import settings, get_db_components

# Get database components
db_components = get_db_components()
db_name = db_components["db_name"]
db_url_without_name = db_components["db_url_without_name"]

def create_database_if_not_exists():
    """Create the database if it doesn't exist."""
    # Connect to default postgres database to check if our db exists
    conn = psycopg2.connect(db_url_without_name)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Check if database exists
    cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (db_name,))
    exists = cursor.fetchone()
    
    if not exists:
        print(f"Database '{db_name}' does not exist. Creating...")
        # Note: Database names cannot be parameterized in PostgreSQL CREATE DATABASE
        # db_name is validated through get_db_components() parsing
        cursor.execute(f'CREATE DATABASE "{db_name}"')
        print(f"Database '{db_name}' created successfully!")
    else:
        print(f"Database '{db_name}' already exists.")
        
    cursor.close()
    conn.close()

# Create SQLAlchemy engine
engine = create_engine(str(settings.DATABASE_URL))

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a database session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 