from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routers import models, metrics, nodes, projects, graphs, health
from app.domain.errors import DomainError, NotFoundError, ValidationError, ConflictError
from app.application.event_handlers import register_event_handlers

app = FastAPI(
    title="Ursa API",
    description="API for Ursa SDK web availability",
    version="1.0.0",
)

# Register domain event handlers on startup
@app.on_event("startup")
async def startup_event():
    register_event_handlers()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Domain error handlers
@app.exception_handler(NotFoundError)
async def not_found_handler(request: Request, exc: NotFoundError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(ConflictError)
async def conflict_error_handler(request: Request, exc: ConflictError):
    return JSONResponse(status_code=409, content={"detail": str(exc)})

# Include routers
app.include_router(health.router, tags=["Health"])  # Health check endpoints first
app.include_router(models.router, tags=["Models"])
app.include_router(metrics.router, tags=["Metrics"])
app.include_router(nodes.router, tags=["Nodes"])
app.include_router(projects.router, tags=["Projects"])
app.include_router(graphs.router, tags=["Graphs"])

@app.get("/")
async def root():
    return {"message": "Welcome to Ursa API. See /docs for API documentation"} 