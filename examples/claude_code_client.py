#!/usr/bin/env python3
"""Claude Code MCP Client Example.

Demonstrates how to interact with Mem0 AI MCP Server from Claude Code.
This example shows best practices for error handling, type annotations,
and working with MCP (Model Context Protocol) servers.

Usage:
    python claude_code_client.py

Requirements:
    - Python 3.11+
    - Mem0 AI MCP Server running locally
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Set up logging
logger = logging.getLogger(__name__)

# Claude Code client constants
MAX_SEARCH_LIMIT = 1000
MAX_CONTENT_LENGTH = 100000


class ClaudeCodeMCPClient:
    """Client for interacting with Mem0 AI MCP server via Claude Code.

    This client manages the lifecycle of an MCP server process and provides
    high-level methods for common memory operations.

    Args:
        server_command: Command to start the MCP server. Defaults to starting
                       the local main.py with --stdio flag.

    Example:
        >>> client = ClaudeCodeMCPClient()
        >>> await client.start_server()
        >>> await client.initialize()
        >>> result = await client.create_memory("My first memory")
        >>> await client.stop_server()
    """

    def __init__(self, server_command: list[str] | None = None) -> None:
        """Initialize the MCP client."""
        self.server_command = server_command or [
            "python",
            str(Path(__file__).parent.parent / "main.py"),
            "--stdio",
        ]
        self.process: asyncio.subprocess.Process | None = None
        self.request_id: int = 0
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialized = False

    async def start_server(self) -> None:
        """Start the MCP server process.

        Raises:
            RuntimeError: If the server fails to start or command is invalid.
            FileNotFoundError: If the server executable is not found.
        """
        if self.process is not None:
            self.logger.warning("Server already started")
            return

        try:
            # Validate server command
            if not self.server_command:
                raise ValueError("Server command cannot be empty")

            self.logger.info("Starting MCP server with command: %s", ' '.join(self.server_command))

            self.process = await asyncio.create_subprocess_exec(
                *self.server_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Give the server a moment to start
            await asyncio.sleep(0.1)

            # Check if process is still running
            if self.process.returncode is not None:
                stderr_output = ""
                if self.process.stderr:
                    stderr_output = (await self.process.stderr.read()).decode()
                raise RuntimeError(f"Server process exited immediately. stderr: {stderr_output}")

            self.logger.info("MCP server started successfully")

        except FileNotFoundError as e:
            self.logger.error("Server executable not found: %s", e)
            raise FileNotFoundError(f"Server executable not found: {self.server_command[0]}") from e
        except Exception as e:
            self.logger.error("Failed to start MCP server: %s", e)
            if self.process:
                self.process.terminate()
                self.process = None
            raise RuntimeError(f"Failed to start MCP server: {e}") from e

    async def stop_server(self) -> None:
        """Stop the MCP server process."""
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
                self.logger.info("MCP server stopped")
            except TimeoutError:
                self.logger.warning("Server didn't stop gracefully, killing process")
                self.process.kill()
                await self.process.wait()
            except Exception as e:
                self.logger.error("Error stopping server: %s", e)
            finally:
                self.process = None

    def _validate_request_input(self, method: str) -> None:
        """Validate request inputs."""
        if not method or not method.strip():
            raise ValueError("Method cannot be empty")

        if not self.process or not self.process.stdin or not self.process.stdout:
            raise RuntimeError("Server not started or streams not available")

        # Check if process is still alive
        if self.process.returncode is not None:
            raise RuntimeError(f"Server process has exited with code {self.process.returncode}")

    async def _send_and_receive(
        self, request: dict[str, Any], method: str
    ) -> dict[str, Any]:
        """Send request and receive response."""
        # Send request
        request_json = json.dumps(request, ensure_ascii=False) + "\n"
        self.process.stdin.write(request_json.encode('utf-8'))
        await self.process.stdin.drain()

        self.logger.debug("Sent request: %s with params: %s", method, request.get("params"))

        # Read response with timeout
        response_line = await asyncio.wait_for(
            self.process.stdout.readline(), timeout=30.0
        )

        if not response_line:
            raise RuntimeError("Empty response from server")

        response_text = response_line.decode('utf-8').strip()
        if not response_text:
            raise RuntimeError("Empty response text from server")

        response = json.loads(response_text)
        self.logger.debug("Received response: %s", response)

        return response

    def _handle_error_response(self, response: dict[str, Any]) -> None:
        """Handle error in response."""
        if "error" not in response:
            return

        error_info = response["error"]
        if isinstance(error_info, dict):
            message = error_info.get("message", "Unknown error")
            code = error_info.get("code", "Unknown")
            data = error_info.get("data", "")
            error_msg = f"MCP Error [{code}]: {message}"
            if data:
                error_msg += f" - {data}"
            raise Exception(error_msg)
        else:
            raise Exception(f"MCP Error: {error_info}")

    async def send_request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send a request to the MCP server.

        Args:
            method: The MCP method to call (e.g., 'initialize', 'tools/list')
            params: Parameters for the method call

        Returns:
            The result from the MCP server

        Raises:
            RuntimeError: If server is not started or streams unavailable
            ValueError: If method is empty or response is invalid JSON
            TimeoutError: If request times out
            Exception: For MCP protocol errors
        """
        # Validate inputs
        self._validate_request_input(method)

        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method.strip(),
            "params": params or {},
        }

        try:
            # Send and receive
            response = await self._send_and_receive(request, method)

            # Check for errors
            self._handle_error_response(response)

            # Validate response structure
            if "id" not in response or response["id"] != self.request_id:
                raise ValueError(f"Invalid response ID: expected {self.request_id}, got {response.get('id')}")

            return response.get("result", {})

        except TimeoutError as e:
            self.logger.error("Request timeout for method: %s", method)
            raise TimeoutError(f"Request timed out for method: {method}") from e
        except json.JSONDecodeError as e:
            self.logger.error("Invalid JSON response: %s", e)
            raise ValueError(f"Invalid JSON response: {e}") from e
        except Exception as e:
            self.logger.error("Request failed for method %s: %s", method, e)
            raise

    async def initialize(self) -> dict[str, Any]:
        """Initialize the MCP connection.

        This must be called after starting the server and before using other methods.

        Returns:
            Server capabilities and protocol information

        Raises:
            RuntimeError: If initialization fails or server is not compatible
        """
        if self._initialized:
            self.logger.warning("Client already initialized")
            return {}

        try:
            result = await self.send_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "clientInfo": {"name": "claude-code-client", "version": "1.0.0"},
                },
            )

            # Send initialized notification
            await self.send_request("initialized")

            self._initialized = True
            self.logger.info("MCP client initialized successfully")
            return result

        except Exception as e:
            self.logger.error("Failed to initialize MCP client: %s", e)
            raise RuntimeError(f"Failed to initialize MCP client: {e}") from e

    async def list_tools(self) -> dict[str, Any]:
        """List available MCP tools.

        Returns:
            Dictionary containing available tools and their schemas

        Raises:
            RuntimeError: If client is not initialized
        """
        if not self._initialized:
            raise RuntimeError("Client must be initialized before listing tools")

        return await self.send_request("tools/list")

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Call a specific MCP tool.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool name is empty or arguments invalid
            RuntimeError: If client is not initialized
        """
        if not self._initialized:
            raise RuntimeError("Client must be initialized before calling tools")

        if not tool_name or not tool_name.strip():
            raise ValueError("Tool name cannot be empty")
        if not isinstance(arguments, dict):
            raise ValueError("Arguments must be a dictionary")

        return await self.send_request(
            "tools/call", {"name": tool_name.strip(), "arguments": arguments}
        )

    async def search_memories(
        self, query: str, limit: int = 10, **kwargs
    ) -> dict[str, Any]:
        """Search memories using semantic similarity.

        Args:
            query: Search query text
            limit: Maximum number of results (1-1000)
            **kwargs: Additional search parameters (tags, memory_type, etc.)

        Returns:
            Search results with memories and similarity scores

        Raises:
            ValueError: If query is empty or limit is invalid
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if not isinstance(limit, int) or limit <= 0 or limit > MAX_SEARCH_LIMIT:
            raise ValueError(f"Limit must be an integer between 1 and {MAX_SEARCH_LIMIT}")

        args = {"query": query.strip(), "limit": limit}
        # Validate and add optional parameters
        for key, value in kwargs.items():
            if value is not None:
                args[key] = value

        return await self.call_tool("memory_search", args)

    async def create_memory(self, content: str, **kwargs) -> dict[str, Any]:
        """Create a new memory.

        Args:
            content: Memory content text
            **kwargs: Additional memory metadata (tags, memory_type, priority, etc.)

        Returns:
            Created memory information including ID

        Raises:
            ValueError: If content is empty or too long
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")
        if len(content) > MAX_CONTENT_LENGTH:
            raise ValueError(f"Content too long (max {MAX_CONTENT_LENGTH:,} characters)")

        args = {"content": content.strip()}
        # Validate and add optional parameters
        for key, value in kwargs.items():
            if value is not None:
                args[key] = value

        return await self.call_tool("memory_create", args)

    async def get_memory(self, memory_id: str) -> dict[str, Any]:
        """Get a specific memory by ID.

        Args:
            memory_id: Unique identifier of the memory

        Returns:
            Memory details including content and metadata

        Raises:
            ValueError: If memory ID is empty
        """
        if not memory_id or not memory_id.strip():
            raise ValueError("Memory ID cannot be empty")

        return await self.call_tool("memory_get", {"memory_id": memory_id.strip()})

    async def list_memories(self, **kwargs) -> dict[str, Any]:
        """List memories with optional filtering.

        Args:
            **kwargs: Filter parameters (memory_type, tags, limit, offset, etc.)

        Returns:
            List of memories matching the filter criteria
        """
        # Filter out None values
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return await self.call_tool("memory_list", filtered_kwargs)

    async def update_memory(self, memory_id: str, **kwargs) -> dict[str, Any]:
        """Update an existing memory.

        Args:
            memory_id: ID of the memory to update
            **kwargs: Fields to update (content, tags, priority, etc.)

        Returns:
            Updated memory information

        Raises:
            ValueError: If memory ID is empty or no update fields provided
        """
        if not memory_id or not memory_id.strip():
            raise ValueError("Memory ID cannot be empty")
        if not kwargs:
            raise ValueError("At least one field must be provided for update")

        args = {"memory_id": memory_id.strip()}
        # Filter out None values and add update fields
        for key, value in kwargs.items():
            if value is not None:
                args[key] = value

        return await self.call_tool("memory_update", args)

    async def delete_memory(self, memory_id: str) -> dict[str, Any]:
        """Delete a memory.

        Args:
            memory_id: ID of the memory to delete

        Returns:
            Deletion confirmation

        Raises:
            ValueError: If memory ID is empty
        """
        if not memory_id or not memory_id.strip():
            raise ValueError("Memory ID cannot be empty")

        return await self.call_tool("memory_delete", {"memory_id": memory_id.strip()})

    async def __aenter__(self) -> "ClaudeCodeMCPClient":
        """Async context manager entry."""
        await self.start_server()
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop_server()


async def demo_claude_code_integration() -> None:
    """Demonstrate Claude Code integration with Mem0 AI.

    This comprehensive demo shows:
    - Server startup and initialization
    - Memory creation with metadata
    - Semantic search capabilities
    - Memory listing and filtering
    - Memory updates and deletion
    - Proper error handling
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger("demo")
    logger.info("Starting Claude Code MCP Client Demo")

    # Use context manager for automatic cleanup
    try:
        async with ClaudeCodeMCPClient() as client:
            logger.info("Connected to MCP server")

            # List available tools
            tools = await client.list_tools()
            logger.info("Available tools: %s", list(tools.get('tools', [])))

            # Create example memories with comprehensive metadata
            memories_to_create: list[dict[str, str | list[str]]] = [
                {
                    "content": "Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms and has extensive standard libraries.",
                    "tags": ["programming", "python", "language", "interpreted"],
                    "memory_type": "fact",
                    "priority": "medium",
                },
                {
                    "content": "FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python type hints. It provides automatic API documentation and data validation.",
                    "tags": ["python", "fastapi", "web", "api", "framework"],
                    "memory_type": "fact",
                    "priority": "high",
                },
                {
                    "content": "Remember to review the memory search algorithm performance next week. Focus on embedding generation speed and similarity calculation optimization.",
                    "tags": ["todo", "performance", "search", "optimization"],
                    "memory_type": "task",
                    "priority": "high",
                },
                {
                    "content": "AsyncIO provides infrastructure for writing single-threaded concurrent code using coroutines, multiplexing I/O access over sockets and other resources.",
                    "tags": ["python", "asyncio", "concurrency", "coroutines"],
                    "memory_type": "fact",
                    "priority": "medium",
                },
            ]

            created_memories: list[dict[str, Any]] = []
            logger.info("Creating %s example memories...", len(memories_to_create))

            for _i, memory_data in enumerate(memories_to_create, 1):
                try:
                    result = await client.create_memory(**memory_data)
                    if result.get("isError"):
                        logger.error("Failed to create memory {i}: %s", result.get('error'))
                    else:
                        memory = result.get("data", {}).get("memory", {})
                        created_memories.append(memory)
                        logger.info("Created memory {i}: %s", memory.get('id', 'unknown'))
                except Exception as e:
                    logger.error("Exception creating memory {i}: %s", e)

            logger.info("Successfully created %s memories", len(created_memories))

            # Demonstrate semantic search
            search_queries = [
                "Python programming language features",
                "web framework API development",
                "performance optimization tasks",
            ]

            for query in search_queries:
                try:
                    logger.info("Searching for: '%s'", query)
                    search_result = await client.search_memories(query, limit=3)

                    if search_result.get("isError"):
                        logger.error("Search failed: %s", search_result.get('error'))
                    else:
                        results = search_result.get("data", {}).get("results", [])
                        logger.info("Found %s results", len(results))
                        for result in results:
                            content_preview = result.get("content", "")[:60] + "..."
                            result.get("similarity_score", 0.0)
                            logger.info("  - Score: {score:.3f} | %s", content_preview)
                except Exception as e:
                    logger.error("Search exception for '{query}': %s", e)

            # List memories with filtering
            try:
                logger.info("Listing task-type memories...")
                list_result = await client.list_memories(memory_type="task", limit=10)

                if list_result.get("isError"):
                    logger.error("List failed: %s", list_result.get('error'))
                else:
                    memories = list_result.get("data", {}).get("memories", [])
                    logger.info("Found %s task memories", len(memories))
                    for memory in memories:
                        content_preview = memory.get("content", "")[:60] + "..."
                        tags = memory.get("tags", [])
                        logger.info("  - {content_preview} | Tags: %s", tags)
            except Exception as e:
                logger.error("List exception: %s", e)

            # Update a memory
            if created_memories:
                try:
                    memory_to_update = created_memories[0]
                    memory_id = memory_to_update.get("id")
                    if memory_id:
                        logger.info("Updating memory: %s", memory_id)

                        original_content = memory_to_update.get('content', '')
                        updated_content = f"{original_content} [UPDATED WITH DEMO TAG]"
                        original_tags = memory_to_update.get("tags", [])
                        updated_tags = [*original_tags, "demo", "updated"]

                        update_result = await client.update_memory(
                            memory_id,
                            content=updated_content,
                            tags=updated_tags,
                        )

                        if update_result.get("isError"):
                            logger.error("Update failed: %s", update_result.get('error'))
                        else:
                            updated_memory = update_result.get("data", {}).get("memory", {})
                            logger.info("Successfully updated memory: %s", updated_memory.get('id'))
                except Exception as e:
                    logger.error("Update exception: %s", e)

            # Demonstrate error handling with non-existent memory
            try:
                logger.info("Testing error handling with non-existent memory...")
                await client.get_memory("non-existent-id-12345")
            except Exception as e:
                logger.info("Expected error handled gracefully: %s", e)

            # Clean up: delete created memories (optional)
            logger.info("Demo completed successfully!")

    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error("Demo failed with exception: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Run the demo with proper error handling
    try:
        asyncio.run(demo_claude_code_integration())
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error("Demo failed", error=str(e))
        sys.exit(1)
