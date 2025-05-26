#!/usr/bin/env python3
"""Mem0 AI MCP Server - Production-ready memory vector database server
Integrates with Claude Code and Cursor through Model Context Protocol.
"""

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from typing import Optional

import structlog
import uvicorn
from fastapi import Depends
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette import EventSourceResponse

from src.auth import create_access_token
from src.auth import get_current_user
from src.config import Settings
from src.mcp import MCPServer
from src.memory import MemoryManager
from src.metrics import setup_metrics
from src.models import MemoryCreate
from src.models import MemoryResponse
from src.models import MemorySearch
from src.models import Token
from src.models import UserCreate
from src.models import UserResponse
from src.plugins import PluginManager
from src.websocket import ConnectionManager

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Global instances
settings = Settings()
memory_manager: Optional[MemoryManager] = None
mcp_server: Optional[MCPServer] = None
connection_manager = ConnectionManager()
plugin_manager: Optional[PluginManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    global memory_manager, mcp_server, plugin_manager

    try:
        logger.info("Starting Mem0 AI MCP Server")

        # Initialize components
        memory_manager = MemoryManager(settings)
        await memory_manager.initialize()

        mcp_server = MCPServer(memory_manager, settings)
        await mcp_server.initialize()

        plugin_manager = PluginManager(settings)
        await plugin_manager.load_plugins()

        # Setup metrics
        setup_metrics(app)

        logger.info("Server initialization complete")
        yield

    except Exception as e:
        logger.error("Failed to initialize server", error=str(e))
        raise
    finally:
        logger.info("Shutting down server")
        if memory_manager:
            await memory_manager.close()
        if mcp_server:
            await mcp_server.close()
        if plugin_manager:
            await plugin_manager.close()


# Create FastAPI app
app = FastAPI(
    title="Mem0 AI MCP Server",
    description="Memory Vector Database with Model Context Protocol Support",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "memory_count": (
            await memory_manager.get_memory_count() if memory_manager else 0
        ),
    }


@app.post("/auth/register", response_model=UserResponse)
async def register(user_data: UserCreate):
    """Register a new user."""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")

    user = await memory_manager.create_user(user_data)
    return user


@app.post("/auth/token", response_model=Token)
async def login(form_data: dict):
    """Login and get access token."""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")

    user = await memory_manager.authenticate_user(
        form_data.get("username"), form_data.get("password")
    )
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(data={"sub": user.id})
    return {"access_token": token, "token_type": "bearer"}


@app.post("/memories", response_model=MemoryResponse)
async def create_memory(
    memory_data: MemoryCreate, current_user: dict = Depends(get_current_user)
):
    """Create a new memory."""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")

    memory = await memory_manager.create_memory(memory_data, current_user["id"])

    # Broadcast to connected clients
    await connection_manager.broadcast(
        {"type": "memory_created", "data": memory.dict()}
    )

    return memory


@app.get("/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(memory_id: str, current_user: dict = Depends(get_current_user)):
    """Get a specific memory."""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")

    memory = await memory_manager.get_memory(memory_id, current_user["id"])
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")

    return memory


@app.post("/memories/search")
async def search_memories(
    search_data: MemorySearch, current_user: dict = Depends(get_current_user)
):
    """Search memories using semantic similarity."""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")

    results = await memory_manager.search_memories(search_data, current_user["id"])
    return {"results": results}


@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a memory."""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")

    await memory_manager.delete_memory(memory_id, current_user["id"])

    # Broadcast to connected clients
    await connection_manager.broadcast(
        {"type": "memory_deleted", "data": {"memory_id": memory_id}}
    )

    return {"message": "Memory deleted successfully"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await connection_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle different message types
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)


@app.get("/mcp/sse")
async def mcp_sse_endpoint(current_user: dict = Depends(get_current_user)):
    """Server-Sent Events endpoint for MCP communication."""
    if not mcp_server:
        raise HTTPException(status_code=503, detail="MCP Server not initialized")

    async def event_generator():
        async for event in mcp_server.sse_stream(current_user["id"]):
            yield event

    return EventSourceResponse(event_generator())


@app.post("/mcp/stdio")
async def mcp_stdio_endpoint(
    request: dict, current_user: dict = Depends(get_current_user)
):
    """Standard I/O endpoint for MCP communication."""
    if not mcp_server:
        raise HTTPException(status_code=503, detail="MCP Server not initialized")

    response = await mcp_server.handle_request(request, current_user["id"])
    return response


@app.get("/plugins")
async def list_plugins(current_user: dict = Depends(get_current_user)):
    """List available plugins."""
    if not plugin_manager:
        raise HTTPException(status_code=503, detail="Plugin manager not initialized")

    return await plugin_manager.list_plugins()


@app.post("/plugins/{plugin_name}/execute")
async def execute_plugin(
    plugin_name: str, data: dict, current_user: dict = Depends(get_current_user)
):
    """Execute a plugin."""
    if not plugin_manager:
        raise HTTPException(status_code=503, detail="Plugin manager not initialized")

    result = await plugin_manager.execute_plugin(plugin_name, data, current_user["id"])
    return result


async def stdio_main():
    """Main function for stdio transport."""
    global memory_manager, mcp_server

    try:
        # Initialize without FastAPI
        memory_manager = MemoryManager(settings)
        await memory_manager.initialize()

        mcp_server = MCPServer(memory_manager, settings)
        await mcp_server.initialize()

        logger.info("MCP Server started in stdio mode")

        # Handle stdio communication
        while True:
            line = await asyncio.get_event_loop().run_in_executor(
                None, sys.stdin.readline
            )
            if not line:
                break

            try:
                request = json.loads(line.strip())
                await mcp_server.handle_request(request, "stdio_user")
                sys.stdout.flush()
            except Exception as e:
                logger.error("Error handling stdio request", error=str(e))
                {"error": {"code": -32603, "message": str(e)}}
                sys.stdout.flush()

    except Exception as e:
        logger.error("Failed to start stdio server", error=str(e))
        sys.exit(1)
    finally:
        if memory_manager:
            await memory_manager.close()
        if mcp_server:
            await mcp_server.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--stdio":
        asyncio.run(stdio_main())
    else:
        uvicorn.run(
            "main:app",
            host=settings.host,
            port=settings.port,
            reload=settings.debug,
            log_level=settings.log_level,
        )
