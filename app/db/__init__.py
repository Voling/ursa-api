from app.db.models import Base
from app.db.database import engine, create_database_if_not_exists

# Import the comprehensive initialization function
from app.db.init_db import init_database

# Create database and tables if they don't exist
def init_db():
    """Initialize the database - create both the database and tables if needed."""
    init_database() 