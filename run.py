import uvicorn
from app.db import init_db
from app.config import settings

if __name__ == "__main__":
    # Initialize database
    print("Initializing database...")
    try:
        init_db()
        print("Database initialization completed successfully!")
    except Exception as e:
        print(f"Error initializing database: {e}")
        print("Please check your database configuration in .env file")
        print("You may need to create the database manually or check PostgreSQL is running")
        exit(1)
    
    # Start the API server
    print(f"Starting API server on {settings.API_HOST}:{settings.API_PORT}...")
    uvicorn.run(
        "app.main:app", 
        host=settings.API_HOST, 
        port=settings.API_PORT, 
        reload=settings.API_RELOAD
    )