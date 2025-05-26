from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db import init_db
from app.storage import get_storage
from app.routers import models, metrics, nodes, projects, graphs

# Database will be initialized when the server starts

app = FastAPI(
    title="Ursa API",
    description="API for Ursa SDK's functions to utilize the web app's functionalities",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(models.router, tags=["Models"])
app.include_router(metrics.router, tags=["Metrics"])
app.include_router(nodes.router, tags=["Nodes"])
app.include_router(projects.router, tags=["Projects"])
app.include_router(graphs.router, tags=["Graphs"])

@app.get("/")
async def root():
    return {"message": "Welcome to Ursa API. See /docs for API documentation"} 