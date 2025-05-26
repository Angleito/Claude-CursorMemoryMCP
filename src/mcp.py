"""Model Context Protocol (MCP) server implementation.

This module implements the Model Context Protocol server for the Mem0 AI system,
providing standardized interfaces for memory operations, tools, resources, and
real-time communication via Server-Sent Events.
"""

from __future__ import annotations

import asyncio
import json
import weakref
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Dict, Optional

import structlog

from .config import Settings
from .memory import MemoryManager
from .models import MCPRequest, MCPResponse, MemoryCreate, MemorySearch

logger = structlog.get_logger()


class MCPErrorCode(Enum):
    """MCP protocol error codes."""

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    SERVER_ERROR_START = -32099
    SERVER_ERROR_END = -32000
    INITIALIZATION_FAILED = -32001
    UNAUTHORIZED = -32002
    FORBIDDEN = -32003
    NOT_FOUND = -32004
    TIMEOUT = -32005


class MCPProtocolVersion:
    """MCP protocol version management."""

    SUPPORTED_VERSIONS = ["2024-11-05"]
    DEFAULT_VERSION = "2024-11-05"

    @classmethod
    def is_supported(cls, version: str) -> bool:
        return version in cls.SUPPORTED_VERSIONS

    @classmethod
    def get_latest(cls) -> str:
        return cls.DEFAULT_VERSION


class MCPServer:
    """MCP protocol server for memory operations.
    
    This class implements a complete MCP server with support for:
    - Memory management tools and operations
    - Resource serving for search and statistics
    - Prompt templates for natural language interaction
    - Real-time notifications via Server-Sent Events
    - Protocol version negotiation and validation
    """

    def __init__(self, memory_manager: MemoryManager, settings: Settings) -> None:
        self.memory_manager = memory_manager
        self.settings = settings
        self._initialized = False
        self._protocol_version = MCPProtocolVersion.DEFAULT_VERSION
        self._client_info: Optional[Dict[str, Any]] = None

        self.capabilities = {
            "logging": {},
            "prompts": {"listChanged": True},
            "resources": {"subscribe": True, "listChanged": True},
            "tools": {"listChanged": True},
        }

        self.server_info = {
            "name": "mem0-ai-mcp-server",
            "version": "1.0.0",
            "protocolVersion": self._protocol_version,
        }

        self.active_streams: Dict[str, asyncio.Queue] = {}
        self._cleanup_tasks: weakref.WeakSet = weakref.WeakSet()
        self._closed = False

    async def initialize(self, client_info: Optional[Dict[str, Any]] = None) -> None:
        """Initialize MCP server."""
        if self._initialized:
            raise ValueError("Server already initialized")

        self._client_info = client_info or {}
        self._initialized = True

        # Validate protocol version if provided by client
        if client_info and "protocolVersion" in client_info:
            client_version = client_info["protocolVersion"]
            if not MCPProtocolVersion.is_supported(client_version):
                raise ValueError(f"Unsupported protocol version: {client_version}")
            self._protocol_version = client_version

        logger.info(
            "MCP server initialized",
            capabilities=self.capabilities,
            protocol_version=self._protocol_version,
            client_info=self._client_info,
        )

    async def handle_request(
        self, request: Dict[str, Any], user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle MCP protocol requests.
        
        Args:
            request: MCP request dictionary
            user_id: Optional user ID for authorization
            
        Returns:
            MCP response dictionary
        """
        request_id = request.get("id")

        try:
            # Validate request structure
            if not isinstance(request, dict):
                return self._create_error_response(
                    request_id, MCPErrorCode.PARSE_ERROR, "Invalid request format"
                )

            if "method" not in request:
                return self._create_error_response(
                    request_id, MCPErrorCode.INVALID_REQUEST, "Missing 'method' field"
                )

            mcp_request = MCPRequest(**request)
            method = mcp_request.method
            params = mcp_request.params or {}

            logger.info(
                "Handling MCP request",
                method=method,
                user_id=user_id,
                request_id=request_id,
            )

            # Check if server is initialized for methods other than initialize
            if method != "initialize" and not self._initialized:
                return self._create_error_response(
                    request_id,
                    MCPErrorCode.INITIALIZATION_FAILED,
                    "Server not initialized",
                )

            # Route to appropriate handler
            if method == "initialize":
                result = await self._handle_initialize(params)
            elif method == "tools/list":
                result = await self._handle_list_tools()
            elif method == "tools/call":
                result = await self._handle_tool_call(params, user_id)
            elif method == "resources/list":
                result = await self._handle_list_resources()
            elif method == "resources/read":
                result = await self._handle_read_resource(params, user_id)
            elif method == "prompts/list":
                result = await self._handle_list_prompts()
            elif method == "prompts/get":
                result = await self._handle_get_prompt(params)
            elif method == "logging/setLevel":
                result = await self._handle_set_log_level(params)
            elif method == "notifications/subscribe":
                result = await self._handle_subscribe_notifications(params, user_id)
            # Legacy memory operations (for backwards compatibility)
            elif method == "memories/create":
                result = await self._handle_memory_create(params, user_id)
            elif method == "memories/search":
                result = await self._handle_memory_search(params, user_id)
            elif method == "memories/get":
                result = await self._handle_memory_get(params, user_id)
            elif method == "memories/list":
                result = await self._handle_memory_list(params, user_id)
            elif method == "memories/update":
                result = await self._handle_memory_update(params, user_id)
            elif method == "memories/delete":
                result = await self._handle_memory_delete(params, user_id)
            else:
                return self._create_error_response(
                    request_id,
                    MCPErrorCode.METHOD_NOT_FOUND,
                    f"Unknown method: {method}",
                )

            return MCPResponse(id=request_id, result=result).dict()

        except ValueError as e:
            logger.error(
                "Validation error in MCP request",
                error=str(e),
                user_id=user_id,
                request_id=request_id,
            )
            return self._create_error_response(
                request_id, MCPErrorCode.INVALID_PARAMS, str(e)
            )
        except Exception as e:
            logger.error(
                "Error handling MCP request",
                error=str(e),
                user_id=user_id,
                request_id=request_id,
            )
            return self._create_error_response(
                request_id,
                MCPErrorCode.INTERNAL_ERROR,
                "Internal server error",
                {"type": type(e).__name__},
            )

    def _create_error_response(
        self,
        request_id: Optional[str],
        error_code: MCPErrorCode,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create standardized error response."""
        return MCPResponse(
            id=request_id,
            error={"code": error_code.value, "message": message, "data": data},
        ).dict()

    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialization."""
        protocol_version = params.get(
            "protocolVersion", MCPProtocolVersion.DEFAULT_VERSION
        )
        client_info = params.get("clientInfo", {})

        if not MCPProtocolVersion.is_supported(protocol_version):
            raise ValueError(f"Unsupported protocol version: {protocol_version}")

        await self.initialize({"protocolVersion": protocol_version, **client_info})

        return {
            "capabilities": self.capabilities,
            "serverInfo": self.server_info,
            "protocolVersion": self._protocol_version,
        }

    async def _handle_list_tools(self) -> Dict[str, Any]:
        """List available MCP tools."""
        tools = [
            {
                "name": "memory_search",
                "description": "Search memories using semantic similarity",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {
                            "type": "integer",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 100,
                        },
                        "threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "memory_types": {"type": "array", "items": {"type": "string"}},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "memory_create",
                "description": "Create a new memory",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Memory content"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "memory_type": {
                            "type": "string",
                            "enum": [
                                "fact",
                                "conversation",
                                "task",
                                "preference",
                                "skill",
                                "context",
                            ],
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "critical"],
                        },
                        "source": {"type": "string"},
                        "context": {"type": "string"},
                        "metadata": {"type": "object"},
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "memory_get",
                "description": "Get a specific memory by ID",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string", "description": "Memory ID"}
                    },
                    "required": ["memory_id"],
                },
            },
            {
                "name": "memory_list",
                "description": "List user's memories with optional filtering",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "default": 20,
                            "minimum": 1,
                            "maximum": 100,
                        },
                        "offset": {"type": "integer", "default": 0, "minimum": 0},
                        "memory_type": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "sort_by": {
                            "type": "string",
                            "enum": ["created_at", "updated_at", "access_count"],
                            "default": "created_at",
                        },
                        "sort_order": {
                            "type": "string",
                            "enum": ["asc", "desc"],
                            "default": "desc",
                        },
                    },
                },
            },
            {
                "name": "memory_update",
                "description": "Update an existing memory",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string", "description": "Memory ID"},
                        "content": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "memory_type": {"type": "string"},
                        "priority": {"type": "string"},
                        "context": {"type": "string"},
                        "metadata": {"type": "object"},
                    },
                    "required": ["memory_id"],
                },
            },
            {
                "name": "memory_delete",
                "description": "Delete a memory",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string", "description": "Memory ID"}
                    },
                    "required": ["memory_id"],
                },
            },
        ]

        return {"tools": tools}

    async def _handle_list_resources(self) -> Dict[str, Any]:
        """List available MCP resources."""
        resources = [
            {
                "uri": "memory://search",
                "name": "Memory Search",
                "description": "Search through stored memories",
                "mimeType": "application/json",
            },
            {
                "uri": "memory://stats",
                "name": "Memory Statistics",
                "description": "Get memory usage statistics",
                "mimeType": "application/json",
            },
        ]
        return {"resources": resources}

    async def _handle_read_resource(
        self, params: Dict[str, Any], user_id: str
    ) -> Dict[str, Any]:
        """Read a specific resource."""
        uri = params.get("uri")

        if uri == "memory://search":
            # Return search interface
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(
                            {
                                "interface": "search",
                                "description": "Search memories using semantic similarity",
                                "parameters": ["query", "limit", "threshold"],
                            }
                        ),
                    }
                ]
            }
        elif uri == "memory://stats":
            # Return memory statistics
            try:
                stats = await self.memory_manager.get_stats(user_id)
                return {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": json.dumps(stats),
                        }
                    ]
                }
            except Exception as e:
                raise ValueError(f"Failed to get memory stats: {e!s}")
        else:
            raise ValueError(f"Unknown resource URI: {uri}")

    async def _handle_list_prompts(self) -> Dict[str, Any]:
        """List available prompts."""
        prompts = [
            {
                "name": "memory_search",
                "description": "Search memories with natural language",
                "arguments": [
                    {"name": "query", "description": "Search query", "required": True},
                    {
                        "name": "limit",
                        "description": "Maximum number of results",
                        "required": False,
                    },
                ],
            }
        ]
        return {"prompts": prompts}

    async def _handle_get_prompt(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get a specific prompt."""
        name = params.get("name")

        if name == "memory_search":
            arguments = params.get("arguments", {})
            query = arguments.get("query", "")
            limit = arguments.get("limit", 10)

            return {
                "description": "Search memories with natural language",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": f"Search for memories related to: {query} (limit: {limit})",
                        },
                    }
                ],
            }
        else:
            raise ValueError(f"Unknown prompt: {name}")

    async def _handle_set_log_level(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Set logging level."""
        level = params.get("level", "info").upper()

        if level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"Invalid log level: {level}")

        # Update logger level
        import logging

        logging.getLogger().setLevel(getattr(logging, level))

        logger.info(f"Log level set to {level}")
        return {"success": True, "level": level}

    async def _handle_tool_call(
        self, params: Dict[str, Any], user_id: str
    ) -> Dict[str, Any]:
        """Handle tool execution calls."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name == "memory_search":
            return await self._tool_memory_search(arguments, user_id)
        elif tool_name == "memory_create":
            return await self._tool_memory_create(arguments, user_id)
        elif tool_name == "memory_get":
            return await self._tool_memory_get(arguments, user_id)
        elif tool_name == "memory_list":
            return await self._tool_memory_list(arguments, user_id)
        elif tool_name == "memory_update":
            return await self._tool_memory_update(arguments, user_id)
        elif tool_name == "memory_delete":
            return await self._tool_memory_delete(arguments, user_id)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    async def _tool_memory_search(
        self, args: Dict[str, Any], user_id: str
    ) -> Dict[str, Any]:
        """Execute memory search tool."""
        try:
            search_data = MemorySearch(**args)
        except Exception as e:
            raise ValueError(f"Invalid search parameters: {e!s}")
        results = await self.memory_manager.search_memories(search_data, user_id)

        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Found {len(results)} memories matching '{search_data.query}'",
                }
            ],
            "isError": False,
            "data": {
                "results": [r.dict() for r in results],
                "query": search_data.query,
                "count": len(results),
            },
        }

    async def _tool_memory_create(
        self, args: Dict[str, Any], user_id: str
    ) -> Dict[str, Any]:
        """Execute memory create tool."""
        try:
            memory_data = MemoryCreate(**args)
        except Exception as e:
            raise ValueError(f"Invalid memory creation parameters: {e!s}")
        memory = await self.memory_manager.create_memory(memory_data, user_id)

        return {
            "content": [
                {"type": "text", "text": f"Created memory: {memory.content[:100]}..."}
            ],
            "isError": False,
            "data": {"memory": memory.dict()},
        }

    async def _tool_memory_get(
        self, args: Dict[str, Any], user_id: str
    ) -> Dict[str, Any]:
        """Execute memory get tool."""
        memory_id = args.get("memory_id")
        memory = await self.memory_manager.get_memory(memory_id, user_id)

        if not memory:
            return {
                "content": [{"type": "text", "text": f"Memory {memory_id} not found"}],
                "isError": True,
            }

        return {
            "content": [{"type": "text", "text": f"Memory: {memory.content}"}],
            "isError": False,
            "data": {"memory": memory.dict()},
        }

    async def _tool_memory_list(
        self, args: Dict[str, Any], user_id: str
    ) -> Dict[str, Any]:
        """Execute memory list tool."""
        # Implementation for listing memories with pagination
        limit = args.get("limit", 20)
        offset = args.get("offset", 0)

        # Build query based on filters
        query = """
            SELECT id, content, metadata, tags, memory_type, priority,
                   created_at, updated_at, access_count
            FROM memories
            WHERE user_id = $1
        """
        params = [user_id]
        param_index = 2  # Start from $2 since $1 is user_id

        if args.get("memory_type"):
            query += f" AND memory_type = ${param_index}"
            params.append(args["memory_type"])
            param_index += 1

        if args.get("tags"):
            query += f" AND tags && ${param_index}"
            params.append(args["tags"])
            param_index += 1

        # Add sorting
        sort_by = args.get("sort_by", "created_at")
        sort_order = args.get("sort_order", "desc")
        query += f" ORDER BY {sort_by} {sort_order.upper()}"

        # Add pagination
        query += f" LIMIT ${param_index} OFFSET ${param_index + 1}"
        params.extend([limit, offset])

        try:
            results = await self.memory_manager.db.execute_query(query, *params)
        except Exception as e:
            logger.error("Database query failed", error=str(e), query=query)
            raise ValueError(f"Failed to list memories: {e!s}")
        memories = [dict(row) for row in results]

        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Found {len(memories)} memories (showing {offset + 1}-{offset + len(memories)})",
                }
            ],
            "isError": False,
            "data": {
                "memories": memories,
                "count": len(memories),
                "limit": limit,
                "offset": offset,
            },
        }

    async def _tool_memory_update(
        self, args: Dict[str, Any], user_id: str
    ) -> Dict[str, Any]:
        """Execute memory update tool."""
        memory_id = args.pop("memory_id")

        from .models import MemoryUpdate

        try:
            update_data = MemoryUpdate(**args)
        except Exception as e:
            raise ValueError(f"Invalid update parameters: {e!s}")

        memory = await self.memory_manager.update_memory(
            memory_id, update_data, user_id
        )
        if not memory:
            return {
                "content": [{"type": "text", "text": f"Memory {memory_id} not found"}],
                "isError": True,
            }

        return {
            "content": [
                {"type": "text", "text": f"Updated memory: {memory.content[:100]}..."}
            ],
            "isError": False,
            "data": {"memory": memory.dict()},
        }

    async def _tool_memory_delete(
        self, args: Dict[str, Any], user_id: str
    ) -> Dict[str, Any]:
        """Execute memory delete tool."""
        memory_id = args.get("memory_id")
        success = await self.memory_manager.delete_memory(memory_id, user_id)

        if not success:
            return {
                "content": [{"type": "text", "text": f"Memory {memory_id} not found"}],
                "isError": True,
            }

        return {
            "content": [{"type": "text", "text": f"Deleted memory {memory_id}"}],
            "isError": False,
            "data": {"deleted": True, "memory_id": memory_id},
        }

    async def _handle_memory_create(
        self, params: Dict[str, Any], user_id: str
    ) -> Dict[str, Any]:
        """Direct memory create handler."""
        try:
            memory_data = MemoryCreate(**params)
        except Exception as e:
            raise ValueError(f"Invalid memory creation parameters: {e!s}")
        memory = await self.memory_manager.create_memory(memory_data, user_id)
        return {"memory": memory.dict()}

    async def _handle_memory_search(
        self, params: Dict[str, Any], user_id: str
    ) -> Dict[str, Any]:
        """Direct memory search handler."""
        try:
            search_data = MemorySearch(**params)
        except Exception as e:
            raise ValueError(f"Invalid search parameters: {e!s}")
        results = await self.memory_manager.search_memories(search_data, user_id)
        return {"results": [r.dict() for r in results]}

    async def _handle_memory_get(
        self, params: Dict[str, Any], user_id: str
    ) -> Dict[str, Any]:
        """Direct memory get handler."""
        memory_id = params.get("memory_id")
        memory = await self.memory_manager.get_memory(memory_id, user_id)
        if not memory:
            raise ValueError(f"Memory {memory_id} not found")
        return {"memory": memory.dict()}

    async def _handle_memory_list(
        self, params: Dict[str, Any], user_id: str
    ) -> Dict[str, Any]:
        """Direct memory list handler."""
        return await self._tool_memory_list(params, user_id)

    async def _handle_memory_update(
        self, params: Dict[str, Any], user_id: str
    ) -> Dict[str, Any]:
        """Direct memory update handler."""
        memory_id = params.pop("memory_id")

        from .models import MemoryUpdate

        try:
            update_data = MemoryUpdate(**params)
        except Exception as e:
            raise ValueError(f"Invalid update parameters: {e!s}")

        memory = await self.memory_manager.update_memory(
            memory_id, update_data, user_id
        )
        if not memory:
            raise ValueError(f"Memory {memory_id} not found")
        return {"memory": memory.dict()}

    async def _handle_memory_delete(
        self, params: Dict[str, Any], user_id: str
    ) -> Dict[str, Any]:
        """Direct memory delete handler."""
        memory_id = params.get("memory_id")
        success = await self.memory_manager.delete_memory(memory_id, user_id)
        if not success:
            raise ValueError(f"Memory {memory_id} not found")
        return {"deleted": True}

    async def _handle_subscribe_notifications(
        self, params: Dict[str, Any], user_id: str
    ) -> Dict[str, Any]:
        """Subscribe to real-time notifications."""
        stream_id = f"{user_id}_{datetime.now().isoformat()}"
        self.active_streams[stream_id] = asyncio.Queue()

        return {
            "stream_id": stream_id,
            "subscribed": True,
            "events": ["memory_created", "memory_updated", "memory_deleted"],
        }

    async def sse_stream(self, user_id: str) -> AsyncGenerator[str, None]:
        """Server-Sent Events stream for real-time updates.
        
        Args:
            user_id: User ID for stream authorization and filtering
            
        Yields:
            SSE-formatted event strings
            
        Raises:
            RuntimeError: If server is closed
        """
        if self._closed:
            raise RuntimeError("Server is closed")

        stream_id = f"{user_id}_{datetime.now().isoformat()}"
        queue = asyncio.Queue(maxsize=1000)  # Prevent memory leaks
        self.active_streams[stream_id] = queue

        # Create cleanup task
        cleanup_task = asyncio.create_task(
            self._cleanup_stream_after_timeout(stream_id, 3600)
        )  # 1 hour timeout
        self._cleanup_tasks.add(cleanup_task)

        try:
            # Send initial connection event with protocol version
            initial_event = {
                "type": "connected",
                "stream_id": stream_id,
                "protocolVersion": self._protocol_version,
                "serverInfo": self.server_info,
            }
            yield f"data: {json.dumps(initial_event)}\n\n"

            while not self._closed:
                try:
                    # Wait for events with timeout
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)

                    # Validate event structure
                    if not isinstance(event, dict):
                        logger.warning("Invalid event format, skipping", event=event)
                        continue

                    yield f"data: {json.dumps(event)}\n\n"

                except asyncio.TimeoutError:
                    # Send keepalive ping
                    ping_event = {
                        "type": "ping",
                        "timestamp": datetime.now().isoformat(),
                        "stream_id": stream_id,
                    }
                    yield f"data: {json.dumps(ping_event)}\n\n"
                except asyncio.QueueFull:
                    logger.warning(
                        "Stream queue full, dropping oldest events", stream_id=stream_id
                    )
                    # Clear some old events
                    for _ in range(min(100, queue.qsize())):
                        try:
                            queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break

        except asyncio.CancelledError:
            logger.info("SSE stream cancelled", stream_id=stream_id)
        except Exception as e:
            logger.error("Error in SSE stream", error=str(e), stream_id=stream_id)
        finally:
            # Cleanup stream
            cleanup_task.cancel()
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
            logger.info("SSE stream closed", stream_id=stream_id)

    async def _cleanup_stream_after_timeout(self, stream_id: str, timeout_seconds: int):
        """Cleanup stream after timeout."""
        try:
            await asyncio.sleep(timeout_seconds)
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
                logger.info("Stream cleaned up after timeout", stream_id=stream_id)
        except asyncio.CancelledError:
            pass

    async def broadcast_event(
        self, event: Dict[str, Any], user_id: Optional[str] = None
    ) -> None:
        """Broadcast event to active streams.
        
        Args:
            event: Event dictionary to broadcast
            user_id: Optional user ID to filter recipients
        """
        if self._closed:
            logger.warning("Cannot broadcast event: server is closed")
            return

        # Validate event structure
        if not isinstance(event, dict) or "type" not in event:
            logger.error("Invalid event structure", event=event)
            return

        # Add timestamp if not present
        if "timestamp" not in event:
            event["timestamp"] = datetime.now().isoformat()

        failed_streams = []

        for stream_id, queue in self.active_streams.items():
            if user_id is None or user_id in stream_id:
                try:
                    # Use put_nowait to avoid blocking
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    logger.warning(
                        "Stream queue full, dropping event", stream_id=stream_id
                    )
                    failed_streams.append(stream_id)
                except Exception as e:
                    logger.error(
                        "Failed to send event to stream",
                        error=str(e),
                        stream_id=stream_id,
                    )
                    failed_streams.append(stream_id)

        # Clean up failed streams
        for stream_id in failed_streams:
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
                logger.info("Removed failed stream", stream_id=stream_id)

    async def close(self) -> None:
        """Close MCP server and cleanup resources.
        
        This method gracefully shuts down the server, cancels all tasks,
        and cleans up all active streams and resources.
        """
        if self._closed:
            return

        self._closed = True

        # Cancel all cleanup tasks
        for task in list(self._cleanup_tasks):
            if not task.done():
                task.cancel()

        # Wait for cleanup tasks to complete
        if self._cleanup_tasks:
            try:
                await asyncio.wait(list(self._cleanup_tasks), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Some cleanup tasks did not complete in time")

        # Cancel all active streams
        for stream_id, queue in list(self.active_streams.items()):
            try:
                # Send close event
                close_event = {
                    "type": "close",
                    "reason": "server_shutdown",
                    "timestamp": datetime.now().isoformat(),
                }
                await queue.put(close_event)
            except Exception as e:
                logger.warning(
                    "Failed to send close event to stream",
                    stream_id=stream_id,
                    error=str(e),
                )
            finally:
                del self.active_streams[stream_id]

        self._initialized = False
        logger.info("MCP server closed successfully")

    def is_closed(self) -> bool:
        """Check if server is closed."""
        return self._closed

    def get_protocol_version(self) -> str:
        """Get current protocol version."""
        return self._protocol_version

    def get_client_info(self) -> Optional[Dict[str, Any]]:
        """Get client information."""
        return self._client_info.copy() if self._client_info else None
