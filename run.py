import uvicorn
from app.config import settings

if __name__ == "__main__":
    # Start the API server
    print(f"Starting API server on {settings.API_HOST}:{settings.API_PORT}...")
    uvicorn.run(
        "app.main:app", 
        host=settings.API_HOST, 
        port=settings.API_PORT, 
        reload=settings.API_RELOAD
    )