import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from app.config import get_db_components

def reset_database():
    """Drop and recreate the database."""
    # Get database components
    db_components = get_db_components()
    db_name = db_components["db_name"]
    db_url_without_name = db_components["db_url_without_name"]
    
    # Connect to default postgres database
    print(f"Connecting to PostgreSQL to drop database '{db_name}'...")
    conn = psycopg2.connect(db_url_without_name)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Drop connections
    print("Closing all connections to the database...")
    cursor.execute("""
        SELECT pg_terminate_backend(pg_stat_activity.pid)
        FROM pg_stat_activity
        WHERE pg_stat_activity.datname = %s
        AND pid <> pg_backend_pid();
    """, (db_name,))
    
    print(f"Dropping database '{db_name}'...")
    # Note: Database names cannot be parameterized in PostgreSQL DDL
    # db_name is validated through get_db_components() parsing
    cursor.execute(f'DROP DATABASE IF EXISTS "{db_name}"')
    print(f"Creating database '{db_name}'...")
    cursor.execute(f'CREATE DATABASE "{db_name}"')
    
    cursor.close()
    conn.close()
    
    print(f"Database '{db_name}' has been reset successfully!")
    print("Run 'python run.py' to initialize tables and start the application.")

if __name__ == "__main__":
    confirm = input("This will DELETE ALL DATA in the database. Are you sure? (y/n): ")
    if confirm.lower() == 'y':
        reset_database()
    else:
        print("Operation cancelled.") 