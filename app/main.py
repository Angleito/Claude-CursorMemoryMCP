"""Mem0AI Server - Production FastAPI Application."""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any
from typing import Dict
from typing import Optional

import psycopg2
import redis
import uvicorn
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import Response
from mem0 import Memory
from prometheus_client import CONTENT_TYPE_LATEST
from prometheus_client import Counter
from prometheus_client import Histogram
from prometheus_client import generate_latest
from qdrant_client import QdrantClient

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
        logging.error(f"Failed to initialize application: {e}")
        raise

    yield

    # Cleanup
    if redis_client:
        await redis_client.aclose()


# Initialize FastAPI app
app = FastAPI(
    title="Mem0AI Server",
    description="Open Memory Vector Database Server",
    version="1.0.0",
    lifespan=lifespan,
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
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check Qdrant connection
        qdrant_client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
        )
        qdrant_client.get_collections()

        # Check Redis connection
        redis_client.ping()

        # Check PostgreSQL connection
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            database=os.getenv("POSTGRES_DB", "mem0ai"),
            user=os.getenv("POSTGRES_USER", "mem0ai"),
            password=os.getenv("POSTGRES_PASSWORD"),
        )
        conn.close()

        return {"status": "healthy", "services": ["qdrant", "redis", "postgres"]}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {e!s}")


# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Memory API endpoints
@app.post("/memories")
async def add_memory(data: Dict[str, Any]):
    """Add new memory."""
    try:
        result = memory_instance.add(
            messages=data.get("messages", []),
            user_id=data.get("user_id", "default"),
            agent_id=data.get("agent_id"),
            run_id=data.get("run_id"),
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories")
async def get_memories(
    user_id: str = "default", agent_id: Optional[str] = None, run_id: Optional[str] = None, limit: int = 100
):
    """Get memories."""
    try:
        result = memory_instance.get_all(
            user_id=user_id, agent_id=agent_id, run_id=run_id, limit=limit
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memories/search")
async def search_memories(data: Dict[str, Any]):
    """Search memories."""
    try:
        result = memory_instance.search(
            query=data.get("query"),
            user_id=data.get("user_id", "default"),
            agent_id=data.get("agent_id"),
            run_id=data.get("run_id"),
            limit=data.get("limit", 10),
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/memories/{memory_id}")
async def update_memory(memory_id: str, data: Dict[str, Any]):
    """Update memory."""
    try:
        result = memory_instance.update(memory_id=memory_id, data=data.get("data"))
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete memory."""
    try:
        memory_instance.delete(memory_id=memory_id)
        return {"success": True, "message": "Memory deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Mem0AI Server", "version": "1.0.0", "status": "running"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
