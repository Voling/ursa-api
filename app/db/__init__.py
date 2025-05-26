from app.db.models import Base
from app.db.database import engine, create_database_if_not_exists

# Create database and tables if they don't exist
def init_db():
    """Initialize the database - create both the database and tables if needed."""
    # First create the database if it doesn't exist
    create_database_if_not_exists()
    
    # Then create all tables
    Base.metadata.create_all(bind=engine)
    print("All database tables initialized successfully!") 