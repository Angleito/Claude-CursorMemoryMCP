#!/usr/bin/env python3
"""Cursor IDE MCP Client Example.

Demonstrates how to integrate Mem0 AI with Cursor IDE via MCP.
This example shows HTTP API integration, WebSocket real-time updates,
and IDE-specific features like code analysis and context management.

Requirements:
    uv add aiohttp websockets

Usage:
    python cursor_client.py

Features:
    - HTTP API client for memory operations
    - WebSocket integration for real-time updates
    - Server-Sent Events (SSE) support
    - Code context analysis
    - Automatic memory saving
    - Coding suggestions based on memory
"""

import asyncio
import json
import logging
import re
import sys
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

# Cursor client constants
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_NOT_FOUND = 404
HTTP_CLIENT_ERROR_THRESHOLD = 400
SIMILARITY_THRESHOLD = 0.9
MIN_KEYWORD_LENGTH = 2
MAX_KEYWORDS_EXTRACTED = 20

try:
    import aiohttp
except ImportError:
    sys.stderr.write("Error: aiohttp is required. Install with: uv add aiohttp\n")
    sys.exit(1)

try:
    import websockets
except ImportError:
    sys.stderr.write("Error: websockets is required. Install with: uv add websockets\n")
    sys.exit(1)


class CursorMCPClient:
    """Client for integrating Mem0 AI with Cursor IDE.

    This client provides HTTP API access to the Mem0 AI server with support for:
    - Authentication and session management
    - Memory CRUD operations
    - WebSocket connections for real-time updates
    - Server-Sent Events (SSE) streaming

    Args:
        base_url: Base URL of the Mem0 AI server (default: http://localhost:8000)
        timeout: Request timeout in seconds (default: 30)

    Example:
        >>> client = CursorMCPClient("http://localhost:8000")
        >>> await client.start_session()
        >>> success = await client.authenticate("username", "password")
        >>> if success:
        ...     memory = await client.create_memory("Hello world")
        >>> await client.close_session()
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30) -> None:
        """Initialize the Cursor MCP client."""
        # Validate URL format
        parsed_url = urlparse(base_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError(f"Invalid base URL: {base_url}")
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"URL must use http:// or https:// scheme: {base_url}")

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session: aiohttp.ClientSession | None = None
        self.access_token: str | None = None
        self.ws_connection: websockets.WebSocketClientProtocol | None = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self._authenticated = False

    async def start_session(self) -> None:
        """Start HTTP session with proper configuration.

        Raises:
            RuntimeError: If session is already started
        """
        if self.session and not self.session.closed:
            self.logger.warning("Session already started")
            return

        # Configure session with timeout and headers
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        headers = {
            "User-Agent": "Cursor-MCP-Client/1.0.0",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers=headers,
            raise_for_status=False  # Handle errors manually
        )
        self.logger.info("HTTP session started")

    async def close_session(self) -> None:
        """Close HTTP session and WebSocket connections."""
        # Close WebSocket first
        if self.ws_connection and not self.ws_connection.closed:
            try:
                await self.ws_connection.close()
                self.logger.info("WebSocket connection closed")
            except Exception as e:
                self.logger.warning("Error closing WebSocket: %s", e)
            finally:
                self.ws_connection = None

        # Close HTTP session
        if self.session and not self.session.closed:
            try:
                await self.session.close()
                self.logger.info("HTTP session closed")
            except Exception as e:
                self.logger.warning("Error closing session: %s", e)
            finally:
                self.session = None

        self._authenticated = False
        self.access_token = None

    async def authenticate(self, username: str, password: str) -> bool:
        """Authenticate with the server.

        Args:
            username: User login name
            password: User password

        Returns:
            True if authentication successful, False otherwise

        Raises:
            ValueError: If username or password is empty
            aiohttp.ClientError: For network-related errors
        """
        if not username or not username.strip():
            raise ValueError("Username cannot be empty")
        if not password:
            raise ValueError("Password cannot be empty")

        if not self.session:
            await self.start_session()

        auth_data = {"username": username.strip(), "password": password}

        try:
            async with self.session.post(
                f"{self.base_url}/auth/token", json=auth_data
            ) as response:
                if response.status == HTTP_OK:
                    try:
                        data = await response.json()
                        self.access_token = data.get("access_token")
                        if not self.access_token:
                            self.logger.error("No access token in response")
                            return False
                        self._authenticated = True
                        self.logger.info("Successfully authenticated user: %s", username)
                        return True
                    except (KeyError, json.JSONDecodeError) as e:
                        self.logger.error("Invalid authentication response: %s", e)
                        return False
                elif response.status == HTTP_UNAUTHORIZED:
                    self.logger.warning("Authentication failed for user: %s", username)
                    return False
                else:
                    error_text = await response.text()
                    self.logger.error("Authentication error %s: %s", response.status, error_text)
                    return False

        except aiohttp.ClientError as e:
            self.logger.error("Network error during authentication: %s", e)
            raise
        except Exception as e:
            self.logger.error("Unexpected error during authentication: %s", e)
            return False

    def _get_headers(self) -> dict[str, str]:
        """Get authentication headers.

        Returns:
            Dictionary with authorization header

        Raises:
            RuntimeError: If not authenticated
        """
        if not self._authenticated or not self.access_token:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        return {"Authorization": f"Bearer {self.access_token}"}

    async def _handle_response(self, response: aiohttp.ClientResponse) -> dict[str, Any] | None:
        """Handle HTTP response with proper error checking.

        Args:
            response: aiohttp response object

        Returns:
            JSON data if successful, None if error

        Raises:
            aiohttp.ClientError: For network errors
            json.JSONDecodeError: For invalid JSON
        """
        if response.status == HTTP_UNAUTHORIZED:
            self.logger.warning("Authentication expired")
            self._authenticated = False
            self.access_token = None
            raise RuntimeError("Authentication expired. Please re-authenticate.")
        elif response.status == HTTP_NOT_FOUND:
            self.logger.warning("Resource not found: %s", response.url)
            return None
        elif response.status >= HTTP_CLIENT_ERROR_THRESHOLD:
            error_text = await response.text()
            self.logger.error("HTTP %s error: %s", response.status, error_text)
            return None

        try:
            return await response.json()
        except json.JSONDecodeError as e:
            self.logger.error("Invalid JSON response: %s", e)
            raise

    async def create_memory(self, content: str, **kwargs) -> dict[str, Any] | None:
        """Create a new memory.

        Args:
            content: Memory content text
            **kwargs: Additional memory metadata (tags, memory_type, priority, etc.)

        Returns:
            Created memory data or None if failed

        Raises:
            ValueError: If content is empty
            RuntimeError: If not authenticated
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")

        memory_data = {"content": content.strip()}
        memory_data.update({k: v for k, v in kwargs.items() if v is not None})

        try:
            async with self.session.post(
                f"{self.base_url}/memories", json=memory_data, headers=self._get_headers()
            ) as response:
                result = await self._handle_response(response)
                if result:
                    self.logger.info("Created memory: %s", result.get('id', 'unknown'))
                return result
        except Exception as e:
            self.logger.error("Failed to create memory: %s", e)
            raise

    async def search_memories(self, query: str, **kwargs) -> list[dict[str, Any]]:
        """Search memories using semantic similarity.

        Args:
            query: Search query text
            **kwargs: Additional search parameters (limit, memory_types, tags, etc.)

        Returns:
            List of matching memories with similarity scores

        Raises:
            ValueError: If query is empty
            RuntimeError: If not authenticated
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        search_data = {"query": query.strip()}
        search_data.update({k: v for k, v in kwargs.items() if v is not None})

        try:
            async with self.session.post(
                f"{self.base_url}/memories/search",
                json=search_data,
                headers=self._get_headers(),
            ) as response:
                data = await self._handle_response(response)
                if data:
                    results = data.get("results", [])
                    self.logger.info("Search found {len(results)} results for: '%s'", query)
                    return results
                return []
        except Exception as e:
            self.logger.error("Failed to search memories: %s", e)
            return []

    async def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        """Get a specific memory by ID.

        Args:
            memory_id: Unique identifier of the memory

        Returns:
            Memory data or None if not found

        Raises:
            ValueError: If memory ID is empty
            RuntimeError: If not authenticated
        """
        if not memory_id or not memory_id.strip():
            raise ValueError("Memory ID cannot be empty")

        try:
            async with self.session.get(
                f"{self.base_url}/memories/{memory_id.strip()}", headers=self._get_headers()
            ) as response:
                return await self._handle_response(response)
        except Exception as e:
            self.logger.error("Failed to get memory {memory_id}: %s", e)
            return None

    async def update_memory(self, memory_id: str, **kwargs) -> dict[str, Any] | None:
        """Update an existing memory.

        Args:
            memory_id: ID of the memory to update
            **kwargs: Fields to update (content, tags, priority, etc.)

        Returns:
            Updated memory data or None if failed

        Raises:
            ValueError: If memory ID is empty or no update fields provided
            RuntimeError: If not authenticated
        """
        if not memory_id or not memory_id.strip():
            raise ValueError("Memory ID cannot be empty")
        if not kwargs:
            raise ValueError("At least one field must be provided for update")

        update_data = {k: v for k, v in kwargs.items() if v is not None}

        try:
            async with self.session.put(
                f"{self.base_url}/memories/{memory_id.strip()}",
                json=update_data,
                headers=self._get_headers(),
            ) as response:
                result = await self._handle_response(response)
                if result:
                    self.logger.info("Updated memory: %s", memory_id)
                return result
        except Exception as e:
            self.logger.error("Failed to update memory {memory_id}: %s", e)
            return None

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory.

        Args:
            memory_id: ID of the memory to delete

        Returns:
            True if successfully deleted, False otherwise

        Raises:
            ValueError: If memory ID is empty
            RuntimeError: If not authenticated
        """
        if not memory_id or not memory_id.strip():
            raise ValueError("Memory ID cannot be empty")

        try:
            async with self.session.delete(
                f"{self.base_url}/memories/{memory_id.strip()}", headers=self._get_headers()
            ) as response:
                success = response.status in (200, 204)
                if success:
                    self.logger.info("Deleted memory: %s", memory_id)
                else:
                    await self._handle_response(response)
                return success
        except Exception as e:
            self.logger.error("Failed to delete memory {memory_id}: %s", e)
            return False

    async def connect_websocket(self) -> bool:
        """Connect to WebSocket for real-time updates.

        Returns:
            True if connection successful, False otherwise

        Raises:
            RuntimeError: If not authenticated
        """
        if self.ws_connection and not self.ws_connection.closed:
            self.logger.warning("WebSocket already connected")
            return True

        try:
            ws_url = self.base_url.replace("http", "ws") + "/ws"
            headers = self._get_headers()

            self.ws_connection = await websockets.connect(
                ws_url,
                extra_headers=headers,
                ping_interval=20,
                ping_timeout=10
            )
            self.logger.info("WebSocket connected to %s", ws_url)
            return True
        except Exception as e:
            self.logger.error("Failed to connect WebSocket: %s", e)
            return False

    async def listen_for_updates(
        self, callback: Callable[[dict[str, Any]], None] | None = None
    ) -> None:
        """Listen for real-time memory updates via WebSocket.

        Args:
            callback: Optional callback function to handle updates
                     Can be sync or async function

        Raises:
            RuntimeError: If WebSocket connection fails
        """
        if (not self.ws_connection or self.ws_connection.closed) and not await self.connect_websocket():
                raise RuntimeError("Failed to establish WebSocket connection")

        try:
            self.logger.info("Starting to listen for WebSocket updates...")
            async for ws_message in self.ws_connection:
                try:
                    if isinstance(ws_message, bytes):
                        decoded_message = ws_message.decode('utf-8')
                    else:
                        decoded_message = ws_message
                    data = json.loads(decoded_message)

                    self.logger.debug("Received WebSocket message: %s", data)

                    if callback:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(data)
                            else:
                                callback(data)
                        except Exception as e:
                            self.logger.error("Error in callback: %s", e)
                    else:
                        # Default handling
                        data.get("type", "unknown")
                        self.logger.info("Received {msg_type} update: %s", data)

                except json.JSONDecodeError as e:
                    self.logger.error("Invalid JSON message: %s", e)
                except Exception as e:
                    self.logger.error("Error processing message: %s", e)

        except websockets.exceptions.ConnectionClosed:
            self.logger.info("WebSocket connection closed")
            self.ws_connection = None
        except Exception as e:
            self.logger.error("Error listening for messages: %s", e)
            self.ws_connection = None
            raise

    async def use_sse_stream(
        self, callback: Callable[[dict[str, Any]], None] | None = None
    ) -> None:
        """Use Server-Sent Events for real-time updates.

        Args:
            callback: Optional callback function to handle SSE events
                     Can be sync or async function

        Raises:
            RuntimeError: If not authenticated
            ConnectionError: If SSE connection fails
        """
        headers = self._get_headers()
        headers["Accept"] = "text/event-stream"
        headers["Cache-Control"] = "no-cache"

        try:
            self.logger.info("Starting SSE stream...")
            async with self.session.get(
                f"{self.base_url}/mcp/sse", headers=headers
            ) as response:
                if response.status != HTTP_OK:
                    error_text = await response.text()
                    raise ConnectionError(f"SSE connection failed ({response.status}): {error_text}")

                async for line in response.content:
                    try:
                        line_str = line.decode('utf-8').strip()
                        if not line_str:
                            continue

                        if line_str.startswith("data: "):
                            data_str = line_str[6:]
                            if data_str == "[DONE]":
                                self.logger.info("SSE stream completed")
                                break

                            data = json.loads(data_str)
                            self.logger.debug("Received SSE event: %s", data)

                            if callback:
                                try:
                                    if asyncio.iscoroutinefunction(callback):
                                        await callback(data)
                                    else:
                                        callback(data)
                                except Exception as e:
                                    self.logger.error("Error in SSE callback: %s", e)
                            else:
                                # Default handling
                                data.get("type", "unknown")
                                self.logger.info("SSE {event_type} event: %s", data)

                        elif line_str.startswith("event: "):
                            event_name = line_str[7:]
                            self.logger.debug("SSE event type: %s", event_name)

                    except json.JSONDecodeError as e:
                        self.logger.error("Invalid SSE JSON data: %s", e)
                    except UnicodeDecodeError as e:
                        self.logger.error("Invalid SSE encoding: %s", e)
                    except Exception as e:
                        self.logger.error("Error processing SSE event: %s", e)

        except aiohttp.ClientError as e:
            raise ConnectionError(f"SSE connection error: {e}") from e
        except Exception as e:
            self.logger.error("Unexpected SSE error: %s", e)
            raise

    async def __aenter__(self) -> "CursorMCPClient":
        """Async context manager entry."""
        await self.start_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close_session()


class CursorIntegration:
    """Integration layer for Cursor IDE specific features.

    This class provides IDE-specific functionality including:
    - Code context analysis and keyword extraction
    - Automatic memory saving for coding sessions
    - Code snippet storage and retrieval
    - Intelligent coding suggestions based on memory

    Args:
        client: Authenticated CursorMCPClient instance
        auto_save_enabled: Whether to automatically save context (default: True)
    """

    def __init__(self, client: CursorMCPClient, auto_save_enabled: bool = True):
        """Initialize Cursor IDE integration."""
        self.client = client
        self.context_memories: list[dict[str, Any]] = []
        self.auto_save_enabled = auto_save_enabled
        self.logger = logging.getLogger(self.__class__.__name__)

    async def analyze_code_context(
        self, code: str, filename: str | None = None
    ) -> list[dict[str, Any]]:
        """Analyze code and find relevant memories.

        Args:
            code: Source code to analyze
            filename: Optional filename for context

        Returns:
            List of relevant memories sorted by relevance
        """
        if not code or not code.strip():
            return []

        try:
            # Extract keywords and concepts from code
            keywords = self._extract_code_keywords(code)
            if not keywords:
                self.logger.warning("No keywords extracted from code")
                return []

            self.logger.info("Analyzing code with %s keywords", len(keywords))

            # Search for relevant memories using each keyword
            relevant_memories = []
            for keyword in keywords[:10]:  # Limit API calls
                try:
                    memories = await self.client.search_memories(
                        query=keyword,
                        limit=3,
                        memory_types=["fact", "skill", "preference"]
                    )
                    relevant_memories.extend(memories)

                    # Add short delay to avoid overwhelming the server
                    await asyncio.sleep(0.1)
                except Exception as e:
                    self.logger.warning("Search failed for keyword '{keyword}': %s", e)
                    continue

            # Remove duplicates and sort by relevance
            seen_ids = set()
            unique_memories = []
            for memory in relevant_memories:
                memory_id = memory.get("id")
                if memory_id and memory_id not in seen_ids:
                    unique_memories.append(memory)
                    seen_ids.add(memory_id)

            # Sort by similarity score (descending)
            sorted_memories = sorted(
                unique_memories,
                key=lambda m: m.get("similarity_score", 0),
                reverse=True
            )[:5]

            self.logger.info("Found %s relevant memories", len(sorted_memories))
            return sorted_memories

        except Exception as e:
            self.logger.error("Error analyzing code context: %s", e)
            return []

    def _extract_code_keywords(self, code: str) -> list[str]:
        """Extract keywords from code for memory search.

        This method uses regex patterns to identify:
        - Function and class definitions
        - Import statements
        - Variable assignments
        - Common programming constructs

        Args:
            code: Source code text to analyze

        Returns:
            List of extracted keywords and identifiers
        """
        if not code or not code.strip():
            return []

        keywords = set()  # Use set to avoid duplicates

        # Common programming patterns and keywords
        language_keywords = [
            "async", "await", "function", "const", "let", "var",
            "interface", "type", "class", "def", "import", "from",
            "export", "default", "return", "yield", "lambda"
        ]

        # Add language keywords found in code
        for keyword in language_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', code, re.IGNORECASE):
                keywords.add(keyword)

        # Extract function definitions (Python and JavaScript)
        func_patterns = [
            r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # Python functions
            r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # JS functions
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*function',  # JS function expressions
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=>',  # Arrow functions
        ]

        for pattern in func_patterns:
            matches = re.findall(pattern, code, re.MULTILINE)
            keywords.update(matches)

        # Extract class definitions
        class_patterns = [
            r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # Python/JS classes
            r'interface\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # TypeScript interfaces
            r'type\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # TypeScript types
        ]

        for pattern in class_patterns:
            matches = re.findall(pattern, code, re.MULTILINE)
            keywords.update(matches)

        # Extract import modules
        import_patterns = [
            r'import\s+(?:.*\s+from\s+)?[\'"]([^\'"]*)[\'"]',  # import statements
            r'from\s+[\'"]([^\'"]*)[\'"]',  # from imports
            r'require\([\'"]([^\'"]*)[\'"]\\)',  # Node.js requires
        ]

        for pattern in import_patterns:
            matches = re.findall(pattern, code, re.MULTILINE)
            # Clean up module names (remove file extensions, paths)
            for match in matches:
                module_name = match.split('/')[-1].split('.')[0]
                if module_name and module_name.isalnum():
                    keywords.add(module_name)

        # Extract variable assignments (simple patterns)
        var_patterns = [
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*',  # Variable assignments
            r'let\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # let declarations
            r'const\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # const declarations
            r'var\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # var declarations
        ]

        for pattern in var_patterns:
            matches = re.findall(pattern, code, re.MULTILINE)
            # Filter out very short or common names
            for match in matches:
                if len(match) > MIN_KEYWORD_LENGTH and match not in ['len', 'str', 'int', 'obj']:
                    keywords.add(match)

        # Convert set back to list and filter
        filtered_keywords = [
            kw for kw in keywords
            if len(kw) > 1 and kw.replace('_', '').isalnum()
        ]

        self.logger.debug("Extracted %s keywords from code", len(filtered_keywords))
        return filtered_keywords[:MAX_KEYWORDS_EXTRACTED]  # Limit to prevent too many searches

    async def save_code_snippet(
        self, code: str, description: str, tags: list[str] | None = None,
        language: str | None = None
    ) -> dict[str, Any] | None:
        """Save a code snippet as a memory.

        Args:
            code: Source code to save
            description: Human-readable description of the code
            tags: Optional list of tags
            language: Programming language (auto-detected if not provided)

        Returns:
            Created memory data or None if failed
        """
        if not code or not code.strip():
            raise ValueError("Code cannot be empty")
        if not description or not description.strip():
            raise ValueError("Description cannot be empty")

        # Auto-detect language if not provided
        if not language:
            language = self._detect_language(code)

        # Format code content with proper markdown
        formatted_content = f"Code snippet: {description.strip()}\n\n```{language}\n{code.strip()}\n```"

        # Build tags list
        snippet_tags = ["code", "snippet"]
        if language:
            snippet_tags.append(language)
        if tags:
            snippet_tags.extend([tag.strip() for tag in tags if tag.strip()])

        # Extract keywords from code for additional tags
        keywords = self._extract_code_keywords(code)
        snippet_tags.extend(keywords[:5])  # Add top 5 keywords as tags

        # Remove duplicates and clean tags
        unique_tags = list(dict.fromkeys(snippet_tags))  # Preserve order

        memory_data = {
            "content": formatted_content,
            "tags": unique_tags,
            "memory_type": "skill",
            "priority": "medium",
            "source": "cursor_ide",
            "metadata": {
                "language": language,
                "code_length": len(code),
                "description": description.strip()
            }
        }

        try:
            result = await self.client.create_memory(**memory_data)
            if result:
                self.logger.info("Saved code snippet: %s...", description[:50])
            return result
        except Exception as e:
            self.logger.error("Failed to save code snippet: %s", e)
            return None

    def _detect_language(self, code: str) -> str:
        """Simple language detection based on code patterns."""
        if not code:
            return "text"

        # Dictionary of language detectors in priority order
        language_detectors = [
            ("typescript", self._is_typescript_code),  # Check TypeScript before JavaScript
            ("javascript", self._is_javascript_code),
            ("python", self._is_python_code),
            ("c", self._is_c_code),
            ("java", self._is_java_code),
            ("rust", self._is_rust_code),
            ("go", self._is_go_code),
        ]

        # Check each language detector
        for language, detector in language_detectors:
            if detector(code):
                return language

        return "text"

    def _is_python_code(self, code: str) -> bool:
        """Check if code contains Python patterns."""
        return any(pattern in code for pattern in ['def ', 'import ', 'from ', '__init__'])

    def _is_typescript_code(self, code: str) -> bool:
        """Check if code contains TypeScript patterns."""
        js_patterns = ['function', 'const ', 'let ', '=>', 'var ']
        ts_patterns = ['interface ', ': string', ': number']
        return (any(pattern in code for pattern in js_patterns) and
                any(pattern in code for pattern in ts_patterns))

    def _is_javascript_code(self, code: str) -> bool:
        """Check if code contains JavaScript patterns."""
        js_patterns = ['function', 'const ', 'let ', '=>', 'var ']
        ts_patterns = ['interface ', ': string', ': number']
        return (any(pattern in code for pattern in js_patterns) and
                not any(pattern in code for pattern in ts_patterns))

    def _is_c_code(self, code: str) -> bool:
        """Check if code contains C patterns."""
        return any(pattern in code for pattern in ['#include', 'int main', 'printf'])

    def _is_java_code(self, code: str) -> bool:
        """Check if code contains Java patterns."""
        return any(pattern in code for pattern in ['public class', 'public static', 'System.out'])

    def _is_rust_code(self, code: str) -> bool:
        """Check if code contains Rust patterns."""
        return any(pattern in code for pattern in ['fn ', 'let mut', 'impl '])

    def _is_go_code(self, code: str) -> bool:
        """Check if code contains Go patterns."""
        return any(pattern in code for pattern in ['func ', 'package ', 'import "'])

    async def auto_save_context(self, file_path: str, code_context: str):
        """Automatically save coding context."""
        if not self.auto_save_enabled:
            return

        # Check if similar context already exists
        existing = await self.client.search_memories(
            query=f"file:{file_path}", limit=1, memory_types=["context"]
        )

        if existing and existing[0].get("similarity_score", 0) > SIMILARITY_THRESHOLD:
            # Update existing context
            await self.client.update_memory(
                existing[0]["id"],
                content=f"Working on {file_path}:\n{code_context}",
                metadata={"file_path": file_path, "last_updated": "now"},
            )
        else:
            # Create new context memory
            await self.client.create_memory(
                content=f"Working on {file_path}:\n{code_context}",
                tags=["context", "file", file_path.split("/")[-1]],
                memory_type="context",
                priority="low",
                source="cursor_auto_save",
                metadata={"file_path": file_path},
            )

    async def get_coding_suggestions(
        self, current_code: str, cursor_position: int | None = None
    ) -> list[str]:
        """Get coding suggestions based on memory."""
        # Analyze current code context
        relevant_memories = await self.analyze_code_context(current_code)

        suggestions = []
        for memory in relevant_memories:
            content = memory.get("content", "")
            if "Code snippet:" in content:
                # Extract code from memory
                if "```" in content:
                    code_part = (
                        content.split("```")[1] if len(content.split("```")) > 1 else ""
                    )
                    suggestions.append(f"Consider this pattern: {code_part[:100]}...")
            else:
                suggestions.append(f"Remember: {content[:100]}...")

        return suggestions[:3]  # Return top 3 suggestions


async def demo_cursor_integration() -> None:
    """Demonstrate Cursor IDE integration.

    This demo shows:
    - HTTP client setup and authentication
    - Code analysis and keyword extraction
    - Memory creation and search
    - Code snippet saving
    - Real-time updates via WebSocket or SSE
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger("demo")
    logger.info("Starting Cursor IDE integration demo")

    try:
        # Use context manager for automatic cleanup
        async with CursorMCPClient() as client:
            logger.info("Connected to Mem0 AI server")

            # Note: In real usage, you would authenticate with:
            # success = await client.authenticate("username", "password")  # noqa: ERA001

            # Create cursor integration
            cursor = CursorIntegration(client, auto_save_enabled=True)

            # Sample code for analysis
            sample_codes = [
                {
                    "code": '''
async def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts using embeddings"""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    # Load pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings
    embedding1 = model.encode([text1])[0]
    embedding2 = model.encode([text2])[0]

    # Calculate cosine similarity
    similarity = np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )
    return float(similarity)
''',
                    "description": "Function to calculate semantic similarity using sentence transformers",
                    "tags": ["similarity", "embedding", "nlp", "async"]
                },
                {
                    "code": '''
class MemoryManager:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.memories = []

    async def add_memory(self, content: str, tags: List[str] = None):
        """Add a new memory with automatic embedding generation"""
        embedding = self.model.encode([content])[0]
        memory = {
            "id": str(uuid.uuid4()),
            "content": content,
            "embedding": embedding.tolist(),
            "tags": tags or [],
            "timestamp": datetime.now().isoformat()
        }
        self.memories.append(memory)
        return memory["id"]
''',
                    "description": "Memory manager class with embedding generation",
                    "tags": ["memory", "class", "embedding", "manager"]
                }
            ]

            # Demonstrate keyword extraction
            for i, sample in enumerate(sample_codes, 1):
                logger.info("\n--- Analyzing Code Sample %d ---", i)

                # Extract keywords
                keywords = cursor._extract_code_keywords(sample["code"])
                logger.info("Extracted keywords: %s", keywords)

                # Detect language
                language = cursor._detect_language(sample["code"])
                logger.info("Detected language: %s", language)

                # With authentication, you could:
                # - Analyze code context for relevant memories
                # - Save code snippets to memory

            # Demonstrate WebSocket connection (would require authentication)
            logger.info("\n--- WebSocket Demo (Simulated) ---")
            logger.info("In real usage, you would:")
            logger.info("1. Authenticate with the server")
            logger.info("2. Connect to WebSocket for real-time updates")
            logger.info("3. Listen for memory creation/update events")
            logger.info("4. Automatically update IDE context")

            # Simulate callback for updates
            async def update_callback(data: dict[str, Any]) -> None:
                """Handle real-time memory updates."""
                update_type = data.get("type", "unknown")
                logger.info("Received %s update: %s", update_type, data)

                # In a real IDE integration, you would:
                # - Update syntax highlighting
                # - Refresh autocomplete suggestions
                # - Update context panel
                # - Notify user of relevant changes

            logger.info("\n--- Demo completed successfully! ---")
            logger.info("To use with authentication:")
            logger.info("1. Set up user credentials")
            logger.info("2. Call client.authenticate(username, password)")
            logger.info("3. Use cursor.analyze_code_context() for real analysis")
            logger.info("4. Use cursor.save_code_snippet() to save snippets")
            logger.info("5. Use client.listen_for_updates() for real-time sync")

    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error("Demo failed: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(demo_cursor_integration())
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception:
        sys.exit(1)
