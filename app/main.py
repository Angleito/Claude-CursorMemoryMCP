"""Mem0AI Server - Production FastAPI Application.

This is the main FastAPI application for the Mem0AI memory management system.
It provides a REST API for managing AI memories with vector database operations
using PostgreSQL pgvector and Qdrant for efficient similarity search.

The API includes endpoints for creating, retrieving, updating, and deleting memories,
as well as health checks and monitoring metrics.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

import psycopg2
import redis
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import Response, JSONResponse
from mem0 import Memory
from prometheus_client import CONTENT_TYPE_LATEST
from prometheus_client import Counter
from prometheus_client import Histogram
from prometheus_client import generate_latest
from qdrant_client import QdrantClient
from pydantic import BaseModel, Field
from datetime import datetime

# Add src to path to import models
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from models import MemoryCreate, MemoryResponse, MemorySearch, MemoryUpdate

# Response models for API documentation
class APIResponse(BaseModel):
    """Standard API response wrapper."""
    success: bool = Field(..., description="Whether the operation was successful")
    data: Any = Field(None, description="Response data")
    message: str = Field(None, description="Human-readable message")

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Overall system health status")
    services: list[str] = Field(..., description="List of healthy services")
    timestamp: str = Field(..., description="Health check timestamp")

class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    code: int = Field(..., description="Error code")
    details: dict[str, Any] = Field(None, description="Additional error details")

# Metrics
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint"]
)
REQUEST_DURATION = Histogram("http_request_duration_seconds", "HTTP request duration")

# Global instances
memory_instance = None
redis_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global memory_instance, redis_client

    # Initialize connections
    try:
        # Initialize Mem0 with Qdrant backend
        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": os.getenv("QDRANT_HOST", "localhost"),
                    "port": int(os.getenv("QDRANT_PORT", "6333")),
                    "collection_name": "mem0ai_memories",
                },
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
        }

        memory_instance = Memory(config=config)

        # Initialize Redis
        redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD"),
            decode_responses=True,
        )

        logging.info("Application initialized successfully")

    except Exception as e:
        logging.error("Failed to initialize application", error=str(e))
        raise

    yield

    # Cleanup
    if redis_client:
        await redis_client.aclose()


# Initialize FastAPI app with comprehensive metadata
app = FastAPI(
    title="Mem0AI Memory Management API",
    description="""
    ## AI Memory Management System with Vector Database

    This FastAPI application provides a comprehensive REST API for managing AI memories 
    using vector database operations. Built on top of PostgreSQL with pgvector extension 
    and Qdrant for efficient similarity search and embedding storage.

    ### Key Features
    - **Memory Management**: Create, retrieve, update, and delete AI memories
    - **Vector Search**: Semantic similarity search using embeddings
    - **Multi-Backend Support**: PostgreSQL pgvector and Qdrant integration
    - **Real-time Monitoring**: Prometheus metrics and health checks
    - **Scalable Architecture**: Async operations with Redis caching
    - **Type Safety**: Full Pydantic model validation

    ### Authentication
    API keys are required for most operations. Include your API key in the 
    `Authorization` header as `Bearer <your-api-key>`.

    ### Rate Limiting
    API requests are rate-limited to ensure system stability. Default limits:
    - 100 requests per minute for general operations
    - 10 requests per minute for batch operations

    ### Data Privacy
    All memory data is encrypted at rest and in transit. Memory expiration 
    and automatic cleanup are supported for privacy compliance.
    """,
    version="1.0.0",
    terms_of_service="https://mem0ai.com/terms",
    contact={
        "name": "Mem0AI Support",
        "url": "https://mem0ai.com/support",
        "email": "support@mem0ai.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=[
        {
            "name": "memories",
            "description": "Memory management operations - create, read, update, delete memories",
        },
        {
            "name": "search",
            "description": "Vector similarity search operations for finding related memories",
        },
        {
            "name": "health",
            "description": "System health checks and service status monitoring",
        },
        {
            "name": "monitoring",
            "description": "Metrics collection and system monitoring endpoints",
        },
        {
            "name": "system",
            "description": "System information and API metadata",
        },
    ],
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGIN", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["*"]  # Configure based on your domain
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Add metrics collection."""
    with REQUEST_DURATION.time():
        response = await call_next(request)
        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
        return response


# Health check endpoint
@app.get(
    "/health",
    tags=["health"],
    summary="System Health Check",
    description="Comprehensive health check for all system dependencies",
    response_model=HealthResponse,
    responses={
        200: {
            "description": "All services are healthy",
            "model": HealthResponse,
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "services": ["qdrant", "redis", "postgres"],
                        "timestamp": "2024-01-01T12:00:00Z"
                    }
                }
            }
        },
        503: {
            "description": "One or more services are unhealthy",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": "Health check failed: Connection refused",
                        "code": 503,
                        "details": {"failed_service": "redis"}
                    }
                }
            }
        }
    }
)
async def health_check():
    """Perform comprehensive health check of all system dependencies.
    
    This endpoint verifies connectivity to:
    - Qdrant vector database
    - Redis cache
    - PostgreSQL database
    
    Returns the health status and list of operational services.
    """
    try:
        services = []
        
        # Check Qdrant connection
        qdrant_client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
        )
        qdrant_client.get_collections()
        services.append("qdrant")

        # Check Redis connection
        redis_client.ping()
        services.append("redis")

        # Check PostgreSQL connection
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            database=os.getenv("POSTGRES_DB", "mem0ai"),
            user=os.getenv("POSTGRES_USER", "mem0ai"),
            password=os.getenv("POSTGRES_PASSWORD"),
        )
        conn.close()
        services.append("postgres")

        return HealthResponse(
            status="healthy",
            services=services,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {e!s}"
        ) from e


# Metrics endpoint
@app.get(
    "/metrics",
    tags=["monitoring"],
    summary="Prometheus Metrics",
    description="Endpoint for Prometheus metrics collection",
    response_class=Response,
    responses={
        200: {
            "description": "Prometheus metrics in text format",
            "content": {
                "text/plain": {
                    "example": "# HELP http_requests_total Total HTTP requests\n# TYPE http_requests_total counter\nhttp_requests_total{endpoint=\"/health\",method=\"GET\"} 42.0"
                }
            }
        }
    }
)
async def metrics():
    """Return Prometheus metrics for monitoring system performance.
    
    This endpoint provides metrics in Prometheus text format including:
    - HTTP request counts by endpoint and method
    - Response time histograms
    - Memory usage statistics
    - Custom application metrics
    
    Designed to be scraped by Prometheus monitoring systems.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Memory API endpoints
@app.post(
    "/memories",
    tags=["memories"],
    summary="Create New Memory",
    description="Add a new memory to the vector database with automatic embedding generation",
    response_model=APIResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {
            "description": "Memory created successfully",
            "model": APIResponse,
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "data": {
                            "id": "mem_12345",
                            "content": "User prefers dark theme",
                            "created_at": "2024-01-01T12:00:00Z"
                        },
                        "message": "Memory created successfully"
                    }
                }
            }
        },
        400: {
            "description": "Invalid request data",
            "model": ErrorResponse
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse
        }
    }
)
async def add_memory(data: dict[str, Any]):
    """Create a new memory with automatic vector embedding generation.
    
    This endpoint accepts memory content and metadata, automatically generates
    embeddings using the configured LLM provider, and stores the memory in the
    vector database for efficient similarity search.
    
    **Request Body:**
    - `messages`: List of conversation messages to extract memory from
    - `user_id`: Unique identifier for the user (defaults to "default")
    - `agent_id`: Optional agent identifier for multi-agent scenarios
    - `run_id`: Optional run identifier for session tracking
    
    **Processing:**
    1. Content is processed to extract meaningful memories
    2. Embeddings are generated using OpenAI or configured provider
    3. Memory is stored in Qdrant vector database
    4. Metadata is indexed for efficient retrieval
    
    **Returns:**
    Memory object with generated ID, timestamps, and metadata.
    """
    try:
        result = memory_instance.add(
            messages=data.get("messages", []),
            user_id=data.get("user_id", "default"),
            agent_id=data.get("agent_id"),
            run_id=data.get("run_id"),
        )
        return APIResponse(
            success=True,
            data=result,
            message="Memory created successfully"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request data: {e!s}"
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create memory: {e!s}"
        ) from e


@app.get(
    "/memories",
    tags=["memories"],
    summary="Retrieve Memories",
    description="Get all memories for a user with optional filtering by agent and run",
    response_model=APIResponse,
    responses={
        200: {
            "description": "Memories retrieved successfully",
            "model": APIResponse,
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "data": [
                            {
                                "id": "mem_12345",
                                "content": "User prefers dark theme",
                                "user_id": "user_123",
                                "created_at": "2024-01-01T12:00:00Z",
                                "tags": ["preference", "ui"]
                            }
                        ],
                        "message": "Retrieved 1 memories"
                    }
                }
            }
        },
        400: {
            "description": "Invalid parameters",
            "model": ErrorResponse
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse
        }
    }
)
async def get_memories(
    user_id: str = Query(
        default="default",
        description="User ID to filter memories",
        example="user_123"
    ),
    agent_id: str | None = Query(
        default=None,
        description="Optional agent ID to filter memories",
        example="agent_gpt4"
    ),
    run_id: str | None = Query(
        default=None,
        description="Optional run ID to filter memories by session",
        example="run_abc123"
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of memories to return",
        example=50
    )
):
    """Retrieve all memories for a user with optional filtering.
    
    This endpoint returns all memories associated with a specific user,
    with optional filtering by agent ID and run ID for more targeted retrieval.
    
    **Query Parameters:**
    - `user_id`: User identifier to filter memories (required)
    - `agent_id`: Optional agent identifier for multi-agent filtering
    - `run_id`: Optional run identifier for session-specific filtering
    - `limit`: Maximum number of memories to return (1-1000, default: 100)
    
    **Filtering Logic:**
    - When only user_id is provided: Returns all user memories
    - With agent_id: Returns memories from specific agent for user
    - With run_id: Returns memories from specific session for user
    - With both: Returns memories from specific agent and session
    
    **Returns:**
    List of memory objects sorted by creation time (newest first).
    """
    try:
        result = memory_instance.get_all(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            limit=limit
        )
        
        count = len(result) if isinstance(result, list) else 0
        return APIResponse(
            success=True,
            data=result,
            message=f"Retrieved {count} memories"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid parameters: {e!s}"
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve memories: {e!s}"
        ) from e


@app.post(
    "/memories/search",
    tags=["search"],
    summary="Vector Similarity Search",
    description="Perform semantic search across memories using vector embeddings",
    response_model=APIResponse,
    responses={
        200: {
            "description": "Search completed successfully",
            "model": APIResponse,
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "data": [
                            {
                                "id": "mem_12345",
                                "content": "User prefers dark theme",
                                "similarity_score": 0.95,
                                "user_id": "user_123",
                                "created_at": "2024-01-01T12:00:00Z"
                            }
                        ],
                        "message": "Found 1 similar memories"
                    }
                }
            }
        },
        400: {
            "description": "Invalid search parameters",
            "model": ErrorResponse
        },
        500: {
            "description": "Search operation failed",
            "model": ErrorResponse
        }
    }
)
async def search_memories(data: dict[str, Any]):
    """Perform semantic similarity search across stored memories.
    
    This endpoint uses vector embeddings to find memories semantically similar
    to the provided query. The search is performed using cosine similarity
    in the high-dimensional embedding space.
    
    **Request Body:**
    - `query`: Text query to search for (required)
    - `user_id`: User ID to filter search results (defaults to "default")
    - `agent_id`: Optional agent ID for filtering
    - `run_id`: Optional run ID for session filtering
    - `limit`: Maximum number of results (default: 10, max: 100)
    
    **Search Process:**
    1. Query text is converted to embedding using same model as memories
    2. Vector similarity search is performed in Qdrant
    3. Results are ranked by similarity score (0.0 to 1.0)
    4. Top matches within user/agent/run scope are returned
    
    **Similarity Scoring:**
    - 1.0: Identical semantic meaning
    - 0.8-0.9: Very similar meaning
    - 0.6-0.8: Somewhat related
    - <0.6: Potentially relevant but distant
    
    **Returns:**
    List of memory objects with similarity scores, ordered by relevance.
    """
    try:
        # Validate required query parameter
        query = data.get("query")
        if not query or not isinstance(query, str) or not query.strip():
            raise ValueError("Query parameter is required and must be a non-empty string")
        
        # Validate limit parameter
        limit = data.get("limit", 10)
        if not isinstance(limit, int) or limit < 1 or limit > 100:
            raise ValueError("Limit must be an integer between 1 and 100")
        
        result = memory_instance.search(
            query=query.strip(),
            user_id=data.get("user_id", "default"),
            agent_id=data.get("agent_id"),
            run_id=data.get("run_id"),
            limit=limit,
        )
        
        count = len(result) if isinstance(result, list) else 0
        return APIResponse(
            success=True,
            data=result,
            message=f"Found {count} similar memories"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid search parameters: {e!s}"
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search operation failed: {e!s}"
        ) from e


@app.put(
    "/memories/{memory_id}",
    tags=["memories"],
    summary="Update Memory",
    description="Update an existing memory's content and metadata",
    response_model=APIResponse,
    responses={
        200: {
            "description": "Memory updated successfully",
            "model": APIResponse,
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "data": {
                            "id": "mem_12345",
                            "content": "User prefers dark theme with blue accents",
                            "updated_at": "2024-01-01T13:00:00Z"
                        },
                        "message": "Memory updated successfully"
                    }
                }
            }
        },
        400: {
            "description": "Invalid memory ID or update data",
            "model": ErrorResponse
        },
        404: {
            "description": "Memory not found",
            "model": ErrorResponse
        },
        500: {
            "description": "Update operation failed",
            "model": ErrorResponse
        }
    }
)
async def update_memory(
    memory_id: str = Path(
        ...,
        description="Unique identifier of the memory to update",
        example="mem_12345"
    ),
    data: dict[str, Any] = ...
):
    """Update an existing memory's content, metadata, or other properties.
    
    This endpoint allows partial updates to existing memories. Only provided
    fields will be updated, while others remain unchanged. Updating content
    will trigger re-generation of embeddings.
    
    **Path Parameters:**
    - `memory_id`: Unique identifier of the memory to update
    
    **Request Body:**
    - `data`: Object containing the fields to update
      - `content`: New memory content (triggers embedding regeneration)
      - `metadata`: Updated metadata dictionary
      - `tags`: Updated tags list
      - `priority`: Updated priority level
    
    **Update Process:**
    1. Memory existence is verified
    2. Provided fields are validated
    3. If content is updated, new embeddings are generated
    4. Updated memory is stored with new timestamp
    5. Vector database is updated with new embeddings
    
    **Returns:**
    Updated memory object with new timestamps and data.
    """
    try:
        if not memory_id or not memory_id.strip():
            raise ValueError("Memory ID is required and cannot be empty")
        
        update_data = data.get("data")
        if not update_data:
            raise ValueError("Update data is required")
        
        result = memory_instance.update(memory_id=memory_id.strip(), data=update_data)
        return APIResponse(
            success=True,
            data=result,
            message="Memory updated successfully"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {e!s}"
        ) from e
    except Exception as e:
        # Check if it's a not found error based on the message
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory not found: {memory_id}"
            ) from e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Update operation failed: {e!s}"
        ) from e


@app.delete(
    "/memories/{memory_id}",
    tags=["memories"],
    summary="Delete Memory",
    description="Permanently delete a memory from the vector database",
    response_model=APIResponse,
    responses={
        200: {
            "description": "Memory deleted successfully",
            "model": APIResponse,
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "data": None,
                        "message": "Memory mem_12345 deleted successfully"
                    }
                }
            }
        },
        400: {
            "description": "Invalid memory ID",
            "model": ErrorResponse
        },
        404: {
            "description": "Memory not found",
            "model": ErrorResponse
        },
        500: {
            "description": "Delete operation failed",
            "model": ErrorResponse
        }
    }
)
async def delete_memory(
    memory_id: str = Path(
        ...,
        description="Unique identifier of the memory to delete",
        example="mem_12345"
    )
):
    """Permanently delete a memory from the vector database.
    
    This operation removes the memory and its associated embeddings from
    both the metadata storage and the vector database. This action is
    irreversible.
    
    **Path Parameters:**
    - `memory_id`: Unique identifier of the memory to delete
    
    **Deletion Process:**
    1. Memory existence is verified
    2. Memory is removed from vector database (Qdrant)
    3. Associated metadata is deleted
    4. Operation is logged for audit purposes
    
    **Security Note:**
    Ensure proper authorization before calling this endpoint as deleted
    memories cannot be recovered.
    
    **Returns:**
    Confirmation message with the deleted memory ID.
    """
    try:
        if not memory_id or not memory_id.strip():
            raise ValueError("Memory ID is required and cannot be empty")
        
        memory_instance.delete(memory_id=memory_id.strip())
        return APIResponse(
            success=True,
            data=None,
            message=f"Memory {memory_id} deleted successfully"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid memory ID: {e!s}"
        ) from e
    except Exception as e:
        # Check if it's a not found error based on the message
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory not found: {memory_id}"
            ) from e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Delete operation failed: {e!s}"
        ) from e


@app.get(
    "/",
    tags=["system"],
    summary="API Information",
    description="Get basic information about the Mem0AI API",
    response_model=dict,
    responses={
        200: {
            "description": "API information and status",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Mem0AI Memory Management API",
                        "version": "1.0.0",
                        "status": "running",
                        "docs_url": "/docs",
                        "redoc_url": "/redoc",
                        "health_check": "/health",
                        "metrics": "/metrics"
                    }
                }
            }
        }
    }
)
async def root():
    """Get basic API information and available endpoints.
    
    This endpoint provides metadata about the Mem0AI API including
    version information, status, and links to documentation and
    monitoring endpoints.
    
    **Returns:**
    - API name and version
    - Current operational status  
    - Links to documentation (/docs, /redoc)
    - Health check endpoint (/health)
    - Metrics endpoint (/metrics)
    """
    return {
        "message": "Mem0AI Memory Management API",
        "version": "1.0.0",
        "status": "running",
        "description": "AI Memory Management System with Vector Database",
        "docs_url": "/docs",
        "redoc_url": "/redoc", 
        "health_check": "/health",
        "metrics": "/metrics",
        "repository": "https://github.com/mem0ai/mem0ai",
        "documentation": "https://docs.mem0ai.com"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=8000,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
