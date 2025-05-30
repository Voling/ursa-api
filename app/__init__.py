"""
ursa-api Application Package

Directory Structure:
├── routers/           # FastAPI route handlers
├── schemas/           # Pydantic models for API requests/responses
│   └── api_schemas.py # HTTP request/response structures
├── services/          # Business logic services
│   └── model_cache_service.py  # ML model caching service
├── storage/           # Storage implementations
│   ├── filesystem.py  # Local filesystem storage
│   └── s3.py         # S3 storage
└── config.py         # Application configuration

Model Types Clarification:
1. **API Schemas** (app.schemas.api_schemas): Pydantic models for HTTP requests/responses
2. **ML Models** (managed by ursakit): Actual trained models (scikit-learn, PyTorch, etc.)

The ursa-api manages metadata about ML models using UrsaML file-based storage,
while the actual ML model files are handled by ursakit SDK and cached by the
model cache service.
""" 