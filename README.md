# Ursa API

FastAPI implementation of the Ursa API for machine learning model management and knowledge graph operations.

## Overview

The Ursa API provides endpoints for:
- Model management (save/load) with ursakit integration
- Metrics tracking
- Node operations in the knowledge graph
- Project management
- Graph management

## Installation

1. Clone repository
2. Install dependencies
   ```
   pip install -r requirements.txt
   ```
3. Set up PostgreSQL:
   - Make sure PostgreSQL is installed and running
   - Create `.env` file with your configuration
   - The application will create the database and tables automatically

## Configuration

Create a `.env` file with the following settings:

```
# Database Configuration
DATABASE_URL=

# Storage Configuration
# Options: "s3" or "filesystem"
STORAGE_TYPE=filesystem

# For filesystem storage
MODEL_STORAGE_DIR=./models

# For S3 storage (uncomment when using S3)
# S3_BUCKET_NAME=ursa-models
# AWS_ACCESS_KEY_ID=your_access_key
# AWS_SECRET_ACCESS_KEY=your_secret_key
# AWS_REGION=us-east-1

# Server Configuration
API_PORT=6422
```

## Running the API

```
python run.py
```

The API will be available at http://localhost:6422

API documentation is available at:
- http://localhost:6422/docs (Swagger UI)
- http://localhost:6422/redoc (ReDoc)

## Testing

Run the test suite:

```
python -m pytest tests/ -v
```

Tests cover:
- API endpoints functionality
- Database integration
- Model caching service
- ursakit integration
- PostgreSQL operations
- Error handling

## Database Management

- **Automatic Initialization**: The database and tables are created automatically on startup
- **Reset Database**: To reset the database completely, run:
  ```
  python reset_db.py
  ```

## Endpoints

The API implements the following endpoints:

### Models
- `POST /models/` - Save a model
- `GET /models/{model_id}` - Load a model

### Metrics
- `POST /metrics/` - Log metrics
- `GET /projects/{project_id}/graphs/{graph_id}/nodes/{node_id}/metrics` - Get node metrics
- `GET /projects/{project_id}/graphs/{graph_id}/metrics` - Get all node metrics in a graph

### Nodes
- `DELETE /projects/{project_id}/graphs/{graph_id}/nodes/{node_id}` - Delete a node
- `PUT /projects/{project_id}/graphs/{graph_id}/nodes/{node_id}` - Update a node
- `PUT /projects/{project_id}/graphs/{graph_id}/nodes/{node_id}/model` - Replace node model
- `GET /projects/{project_id}/graphs/{graph_id}/nodes` - Get nodes

### Projects
- `POST /projects/` - Create a project
- `GET /projects` - Get all projects
- `DELETE /projects/{project_id}` - Delete a project

### Graphs
- `POST /projects/{project_id}/graphs` - Create a graph
- `GET /projects/{project_id}/graphs` - Get all graphs in a project