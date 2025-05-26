#!/usr/bin/env python3
"""WebSocket Client Example for Real-time Memory Updates.

Demonstrates real-time synchronization between multiple clients using WebSockets.
This example shows advanced WebSocket handling, connection management, and
multi-client synchronization patterns.

Requirements:
    uv add websockets

Usage:
    python websocket_client.py

Features:
    - WebSocket client with automatic reconnection
    - Message handling and event dispatching
    - Multi-client synchronization manager
    - Health checking and monitoring
    - Proper error handling and logging
    - Context management for clean resource cleanup
"""

import asyncio
import json
import logging
import sys
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

# WebSocket client constants
CONTENT_PREVIEW_LENGTH = 50
LONG_CONTENT_PREVIEW_LENGTH = 30
SHORT_CONTENT_PREVIEW_LENGTH = 20

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
    from websockets.exceptions import WebSocketException
except ImportError:
    import logging
    logging.error("websockets is required. Install with: uv add websockets")
    sys.exit(1)


class MemoryWebSocketClient:
    """WebSocket client for real-time memory updates.

    This client provides robust WebSocket connectivity with:
    - Automatic reconnection with exponential backoff
    - Message handler registration and dispatch
    - Connection health monitoring
    - Proper error handling and logging

    Args:
        uri: WebSocket URI (default: ws://localhost:8000/ws)
        max_reconnect_attempts: Maximum reconnection attempts (default: 5)
        ping_interval: Ping interval in seconds (default: 20)

    Example:
        >>> client = MemoryWebSocketClient("ws://localhost:8000/ws")
        >>> await client.connect()
        >>> client.register_handler("memory_created", my_handler)
        >>> await client.subscribe_to_updates()
        >>> # Listen for messages...
        >>> await client.disconnect()
    """

    def __init__(
        self,
        uri: str = "ws://localhost:8000/ws",
        max_reconnect_attempts: int = 5,
        ping_interval: int = 20
    ) -> None:
        """Initialize WebSocket client."""
        # Validate URI format
        parsed_uri = urlparse(uri)
        if not parsed_uri.scheme or not parsed_uri.netloc:
            raise ValueError(f"Invalid WebSocket URI: {uri}")
        if parsed_uri.scheme not in ("ws", "wss"):
            raise ValueError(f"URI must use ws:// or wss:// scheme: {uri}")

        self.uri = uri
        self.websocket: websockets.WebSocketClientProtocol | None = None
        self.message_handlers: dict[str, Callable[[dict[str, Any]], None]] = {}
        self.is_connected: bool = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self._listen_task: asyncio.Task | None = None
        self._reconnect_attempts: int = 0
        self._max_reconnect_attempts: int = max_reconnect_attempts
        self._ping_interval: int = ping_interval
        self._client_id: str = str(uuid.uuid4())
        self._subscribed_events: list[str] = []

    async def connect(self, auth_token: str | None = None) -> None:
        """Connect to WebSocket server.

        Args:
            auth_token: Optional authentication token

        Raises:
            ValueError: If auth token is empty
            ConnectionError: If connection fails
        """
        if self.is_connected and self.websocket and not self.websocket.closed:
            self.logger.warning("Already connected to WebSocket server")
            return

        headers = {"X-Client-ID": self._client_id}
        if auth_token:
            if not auth_token.strip():
                raise ValueError("Auth token cannot be empty")
            headers["Authorization"] = f"Bearer {auth_token.strip()}"

        try:
            self.logger.info("Connecting to WebSocket server: %s", self.uri)

            self.websocket = await websockets.connect(
                self.uri,
                extra_headers=headers,
                ping_interval=self._ping_interval,
                ping_timeout=10,
                close_timeout=10,
                max_size=1024 * 1024,  # 1MB max message size
                compression=None  # Disable compression for better performance
            )

            self.is_connected = True
            self._reconnect_attempts = 0
            self.logger.info("Successfully connected to %s", self.uri)

            # Start listening for messages
            self._listen_task = asyncio.create_task(self._listen_for_messages())

        except websockets.InvalidURI as e:
            self.logger.error("Invalid WebSocket URI: %s", e)
            raise ValueError(f"Invalid WebSocket URI: {e}") from e
        except websockets.InvalidHandshake as e:
            self.logger.error("WebSocket handshake failed: %s", e)
            raise ConnectionError(f"WebSocket handshake failed: {e}") from e
        except OSError as e:
            self.logger.error("Network error during connection: %s", e)
            raise ConnectionError(f"Network error: {e}") from e
        except Exception as e:
            self.logger.error("Unexpected error during connection: %s", e)
            self.is_connected = False
            raise ConnectionError(f"Failed to connect to {self.uri}: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from WebSocket server with proper cleanup."""
        self.logger.info("Disconnecting from WebSocket server...")
        self.is_connected = False

        # Cancel listening task
        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
            try:
                await asyncio.wait_for(self._listen_task, timeout=5.0)
            except (TimeoutError, asyncio.CancelledError):
                pass
            except Exception as e:
                self.logger.warning("Error waiting for listen task: %s", e)

        # Close WebSocket connection
        if self.websocket and not self.websocket.closed:
            try:
                await asyncio.wait_for(self.websocket.close(), timeout=5.0)
                self.logger.info("WebSocket connection closed gracefully")
            except TimeoutError:
                self.logger.warning("WebSocket close timeout")
            except Exception as e:
                self.logger.error("Error closing WebSocket: %s", e)

        # Reset state
        self.websocket = None
        self._listen_task = None
        self._reconnect_attempts = 0

    async def send_message(self, message: dict[str, Any]) -> None:
        """Send a message to the server.

        Args:
            message: Message dictionary to send

        Raises:
            RuntimeError: If not connected
            TypeError: If message is not a dictionary
            ConnectionError: If WebSocket connection is lost
        """
        if not self.is_connected or not self.websocket or self.websocket.closed:
            raise RuntimeError("Not connected to WebSocket server")

        if not isinstance(message, dict):
            raise TypeError("Message must be a dictionary")

        # Add client metadata
        message_with_metadata = {
            **message,
            "client_id": self._client_id,
            "timestamp": datetime.utcnow().isoformat()
        }

        try:
            message_json = json.dumps(message_with_metadata, ensure_ascii=False)
            await self.websocket.send(message_json)
            self.logger.debug("Sent message: %s", message.get('type', 'unknown'))
        except json.JSONEncodeError as e:
            self.logger.error("Failed to serialize message: %s", e)
            raise ValueError(f"Failed to serialize message: {e}") from e
        except ConnectionClosed as e:
            self.is_connected = False
            self.logger.warning("WebSocket connection lost while sending")
            raise ConnectionError("WebSocket connection lost") from e
        except WebSocketException as e:
            self.logger.error("WebSocket error while sending: %s", e)
            raise ConnectionError(f"WebSocket error: {e}") from e
        except Exception as e:
            self.logger.error("Unexpected error while sending message: %s", e)
            raise

    async def _listen_for_messages(self) -> None:
        """Listen for incoming messages with robust error handling."""
        self.logger.info("Starting to listen for WebSocket messages...")

        try:
            async for message in self.websocket:
                try:
                    # Handle both text and binary messages
                    if isinstance(message, bytes):
                        message_text = message.decode("utf-8")
                    else:
                        message_text = message

                    if not message_text.strip():
                        continue

                    data = json.loads(message_text)

                    # Validate message structure
                    if not isinstance(data, dict):
                        self.logger.warning("Received non-dict message: %s", type(data))
                        continue

                    self.logger.debug("Received message: %s", data.get('type', 'unknown'))
                    await self._handle_message(data)

                except json.JSONDecodeError as e:
                    self.logger.error("Invalid JSON message: %s", e)
                    continue
                except UnicodeDecodeError as e:
                    self.logger.error("Invalid message encoding: %s", e)
                    continue
                except Exception as e:
                    self.logger.error("Error processing message: %s", e)
                    continue

        except ConnectionClosed:
            self.logger.info("WebSocket connection closed by server")
            self.is_connected = False

            # Attempt reconnection if not manually disconnected
            if self._reconnect_attempts < self._max_reconnect_attempts:
                await self._attempt_reconnect()

        except WebSocketException as e:
            self.logger.error("WebSocket error: %s", e)
            self.is_connected = False

        except Exception as e:
            self.logger.error("Unexpected error in message listener: %s", e)
            self.is_connected = False

    async def _handle_message(self, data: dict[str, Any]) -> None:
        """Handle incoming message by dispatching to registered handlers.

        Args:
            data: Parsed message data
        """
        message_type = data.get("type")
        if not message_type:
            self.logger.warning("Received message without type field")
            return

        # Handle built-in message types
        if message_type == "pong":
            self.logger.debug("Received pong response")
            return
        elif message_type == "error":
            error_msg = data.get("message", "Unknown error")
            self.logger.error("Server error: %s", error_msg)
            return
        elif message_type == "connection_ack":
            self.logger.info("Connection acknowledged by server")
            # Re-subscribe to events if we were previously subscribed
            if self._subscribed_events:
                await self.subscribe_to_updates(self._subscribed_events)
            return

        # Dispatch to registered handlers
        if message_type in self.message_handlers:
            try:
                handler = self.message_handlers[message_type]
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                self.logger.error("Error in message handler for '%s': %s", message_type, e)
        else:
            self.logger.debug("No handler registered for message type: %s", message_type)

    def register_handler(
        self, message_type: str, handler: Callable[[dict[str, Any]], None]
    ) -> None:
        """Register a message handler for a specific message type.

        Args:
            message_type: Type of message to handle
            handler: Handler function (can be sync or async)

        Raises:
            ValueError: If message type is empty
            TypeError: If handler is not callable
        """
        if not message_type or not message_type.strip():
            raise ValueError("Message type cannot be empty")
        if not callable(handler):
            raise TypeError("Handler must be callable")

        message_type = message_type.strip()
        self.message_handlers[message_type] = handler
        self.logger.debug("Registered handler for message type: %s", message_type)

    def unregister_handler(self, message_type: str) -> bool:
        """Unregister a message handler.

        Args:
            message_type: Type of message handler to remove

        Returns:
            True if handler was removed, False if not found
        """
        if message_type in self.message_handlers:
            del self.message_handlers[message_type]
            self.logger.debug("Unregistered handler for message type: %s", message_type)
            return True
        return False

    async def ping(self) -> None:
        """Send a ping message to test connectivity.

        Raises:
            RuntimeError: If not connected
        """
        await self.send_message({"type": "ping"})

    async def subscribe_to_updates(
        self, event_types: list[str] | None = None
    ) -> None:
        """Subscribe to memory update events.

        Args:
            event_types: List of event types to subscribe to
                        Default: ["memory_created", "memory_updated", "memory_deleted"]

        Raises:
            TypeError: If event_types is not a list
            ValueError: If no event types specified
            RuntimeError: If not connected
        """
        default_events = ["memory_created", "memory_updated", "memory_deleted"]
        events = event_types or default_events

        if not isinstance(events, list):
            raise TypeError("Event types must be a list")
        if not events:
            raise ValueError("At least one event type must be specified")

        # Clean and validate event types
        clean_events = [event.strip() for event in events if event and event.strip()]
        if not clean_events:
            raise ValueError("No valid event types provided")

        message = {"type": "subscribe", "events": clean_events}
        await self.send_message(message)

        self._subscribed_events = clean_events
        self.logger.info("Subscribed to events: %s", clean_events)

    async def unsubscribe_from_updates(
        self, event_types: list[str] | None = None
    ) -> None:
        """Unsubscribe from memory update events.

        Args:
            event_types: List of event types to unsubscribe from.
                        If None, unsubscribes from all.
        """
        if event_types is None:
            # Unsubscribe from all
            message = {"type": "unsubscribe", "events": self._subscribed_events}
            self._subscribed_events = []
        else:
            clean_events = [event.strip() for event in event_types if event and event.strip()]
            message = {"type": "unsubscribe", "events": clean_events}
            # Remove from subscribed list
            self._subscribed_events = [e for e in self._subscribed_events if e not in clean_events]

        await self.send_message(message)
        self.logger.info("Unsubscribed from events: %s", event_types or 'all')

    async def _attempt_reconnect(self) -> None:
        """Attempt to reconnect to the WebSocket server with exponential backoff."""
        self._reconnect_attempts += 1
        delay = min(2 ** self._reconnect_attempts, 60)  # Max 60 seconds

        self.logger.info(
            "Attempting reconnection %d/%d in %ds...",
            self._reconnect_attempts, self._max_reconnect_attempts, delay
        )

        await asyncio.sleep(delay)

        try:
            await self.connect()
            self.logger.info("Reconnection successful")
        except Exception as e:
            self.logger.error("Reconnection attempt %d failed: %s", self._reconnect_attempts, e)

            if self._reconnect_attempts < self._max_reconnect_attempts:
                await self._attempt_reconnect()
            else:
                self.logger.error("Max reconnection attempts reached. Giving up.")

    async def __aenter__(self) -> "MemoryWebSocketClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()


class MemorySyncManager:
    """Manages synchronization of memories across multiple clients.

    This manager coordinates multiple WebSocket clients and provides:
    - Client lifecycle management
    - Event broadcasting and synchronization
    - Local caching and consistency
    - Health monitoring and reporting

    Example:
        >>> manager = MemorySyncManager()
        >>> client1 = MemoryWebSocketClient("ws://localhost:8000/ws")
        >>> await manager.add_client("client1", client1)
        >>> await manager.broadcast_to_clients({"type": "test", "data": "hello"})
    """

    def __init__(self) -> None:
        """Initialize the sync manager."""
        self.clients: dict[str, MemoryWebSocketClient] = {}
        self.local_cache: dict[str, dict[str, Any]] = {}
        self.sync_callbacks: dict[str, Callable[[dict[str, Any]], None]] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._stats = {
            "messages_processed": 0,
            "errors_encountered": 0,
            "last_activity": None
        }

    async def add_client(self, client_id: str, client: MemoryWebSocketClient) -> None:
        """Add a client to sync management.

        Args:
            client_id: Unique identifier for the client
            client: WebSocket client instance

        Raises:
            ValueError: If client ID is empty
            TypeError: If client is not correct type
        """
        if not client_id or not client_id.strip():
            raise ValueError("Client ID cannot be empty")
        if not isinstance(client, MemoryWebSocketClient):
            raise TypeError("Client must be a MemoryWebSocketClient instance")

        client_id = client_id.strip()
        if client_id in self.clients:
            self.logger.warning("Client %s already exists, replacing", client_id)

        self.clients[client_id] = client

        # Register sync handlers
        client.register_handler("memory_created", self._handle_memory_created)
        client.register_handler("memory_updated", self._handle_memory_updated)
        client.register_handler("memory_deleted", self._handle_memory_deleted)
        client.register_handler("pong", self._handle_pong)
        client.register_handler("error", self._handle_error)

        self.logger.info("Added client %s to sync manager", client_id)

    async def remove_client(self, client_id: str) -> bool:
        """Remove a client from sync management.

        Args:
            client_id: Client identifier to remove

        Returns:
            True if client was removed, False if not found
        """
        if client_id in self.clients:
            client = self.clients[client_id]

            # Disconnect client if still connected
            if client.is_connected:
                try:
                    await client.disconnect()
                except Exception as e:
                    self.logger.warning("Error disconnecting client %s: %s", client_id, e)

            del self.clients[client_id]
            self.logger.info("Removed client %s from sync manager", client_id)
            return True
        return False

    async def _handle_memory_created(self, data: dict[str, Any]) -> None:
        """Handle memory creation event with proper error handling."""
        try:
            memory = data.get("data", {})
            memory_id = memory.get("id")

            if memory_id:
                self.local_cache[memory_id] = memory
                content_preview = memory.get("content", "")[:CONTENT_PREVIEW_LENGTH] + "..." if len(memory.get("content", "")) > CONTENT_PREVIEW_LENGTH else memory.get("content", "")
                self.logger.info("Memory created: %s - %s", memory_id, content_preview)

                # Update stats
                self._stats["messages_processed"] += 1
                self._stats["last_activity"] = datetime.utcnow().isoformat()

                # Notify callbacks
                if "memory_created" in self.sync_callbacks:
                    callback = self.sync_callbacks["memory_created"]
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(memory)
                        else:
                            callback(memory)
                    except Exception as e:
                        self.logger.error("Error in memory_created callback: %s", e)
                        self._stats["errors_encountered"] += 1
        except Exception as e:
            self.logger.error("Error handling memory creation: %s", e)
            self._stats["errors_encountered"] += 1

    async def _handle_memory_updated(self, data: dict[str, Any]) -> None:
        """Handle memory update event with proper error handling."""
        try:
            memory = data.get("data", {})
            memory_id = memory.get("id")

            if memory_id:
                old_memory = self.local_cache.get(memory_id, {})
                self.local_cache[memory_id] = memory
                self.logger.info("Memory updated: %s", memory_id)

                # Update stats
                self._stats["messages_processed"] += 1
                self._stats["last_activity"] = datetime.utcnow().isoformat()

                # Notify callbacks with old and new memory
                if "memory_updated" in self.sync_callbacks:
                    callback = self.sync_callbacks["memory_updated"]
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(memory, old_memory)
                        else:
                            callback(memory, old_memory)
                    except Exception as e:
                        self.logger.error("Error in memory_updated callback: %s", e)
                        self._stats["errors_encountered"] += 1
        except Exception as e:
            self.logger.error("Error handling memory update: %s", e)
            self._stats["errors_encountered"] += 1

    async def _handle_memory_deleted(self, data: dict[str, Any]) -> None:
        """Handle memory deletion event with proper error handling."""
        try:
            memory_data = data.get("data", {})
            memory_id = memory_data.get("memory_id") or memory_data.get("id")

            if memory_id and memory_id in self.local_cache:
                deleted_memory = self.local_cache.pop(memory_id)
                self.logger.info("Memory deleted: %s", memory_id)

                # Update stats
                self._stats["messages_processed"] += 1
                self._stats["last_activity"] = datetime.utcnow().isoformat()

                # Notify callbacks
                if "memory_deleted" in self.sync_callbacks:
                    callback = self.sync_callbacks["memory_deleted"]
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(deleted_memory)
                        else:
                            callback(deleted_memory)
                    except Exception as e:
                        self.logger.error("Error in memory_deleted callback: %s", e)
                        self._stats["errors_encountered"] += 1
        except Exception as e:
            self.logger.error("Error handling memory deletion: %s", e)
            self._stats["errors_encountered"] += 1

    async def _handle_pong(self, data: dict[str, Any]) -> None:
        """Handle pong response."""
        self.logger.debug("Received pong from client: %s", data.get('client_id', 'unknown'))

    async def _handle_error(self, data: dict[str, Any]) -> None:
        """Handle error messages from server."""
        error_msg = data.get("message", "Unknown error")
        client_id = data.get("client_id", "unknown")
        self.logger.error("Server error from %s: %s", client_id, error_msg)
        self._stats["errors_encountered"] += 1

    def register_sync_callback(
        self, event_type: str, callback: Callable[[dict[str, Any]], None]
    ) -> None:
        """Register a callback for sync events.

        Args:
            event_type: Type of event (memory_created, memory_updated, memory_deleted)
            callback: Callback function (can be sync or async)

        Raises:
            ValueError: If event type is empty
            TypeError: If callback is not callable
        """
        if not event_type or not event_type.strip():
            raise ValueError("Event type cannot be empty")
        if not callable(callback):
            raise TypeError("Callback must be callable")

        event_type = event_type.strip()
        self.sync_callbacks[event_type] = callback
        self.logger.debug("Registered sync callback for event: %s", event_type)

    def unregister_sync_callback(self, event_type: str) -> bool:
        """Unregister a sync callback.

        Args:
            event_type: Event type to unregister

        Returns:
            True if callback was removed, False if not found
        """
        if event_type in self.sync_callbacks:
            del self.sync_callbacks[event_type]
            self.logger.debug("Unregistered sync callback for event: %s", event_type)
            return True
        return False

    async def broadcast_to_clients(self, message: dict[str, Any]) -> dict[str, Any]:
        """Broadcast a message to all connected clients.

        Args:
            message: Message to broadcast

        Returns:
            Dictionary with broadcast results

        Raises:
            TypeError: If message is not a dictionary
        """
        if not isinstance(message, dict):
            raise TypeError("Message must be a dictionary")

        results = {
            "total_clients": len(self.clients),
            "successful_sends": 0,
            "failed_sends": 0,
            "errors": []
        }

        if not self.clients:
            self.logger.warning("No clients to broadcast to")
            return results

        # Broadcast to all connected clients
        for client_id, client in self.clients.items():
            if client.is_connected:
                try:
                    await client.send_message(message)
                    results["successful_sends"] += 1
                    self.logger.debug("Broadcast successful to %s", client_id)
                except Exception as e:
                    error_msg = f"Failed to send to {client_id}: {e}"
                    self.logger.warning(error_msg)
                    results["failed_sends"] += 1
                    results["errors"].append(error_msg)
            else:
                results["failed_sends"] += 1
                results["errors"].append(f"Client {client_id} not connected")

        self.logger.info(
            f"Broadcast complete: {results['successful_sends']} successful, "
            f"{results['failed_sends']} failed"
        )
        return results

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on all clients.

        Returns:
            Dictionary with health check results for each client
        """
        results = {}
        healthy_count = 0

        for client_id, client in self.clients.items():
            if client.is_connected:
                try:
                    await client.ping()
                    results[client_id] = {
                        "healthy": True,
                        "connected": True,
                        "last_check": datetime.utcnow().isoformat()
                    }
                    healthy_count += 1
                except Exception as e:
                    self.logger.warning("Health check failed for {client_id}: %s", e)
                    results[client_id] = {
                        "healthy": False,
                        "connected": True,
                        "error": str(e),
                        "last_check": datetime.utcnow().isoformat()
                    }
            else:
                results[client_id] = {
                    "healthy": False,
                    "connected": False,
                    "last_check": datetime.utcnow().isoformat()
                }

        total_count = len(results)
        self.logger.info("Health check complete: {healthy_count}/%s clients healthy", total_count)

        return {
            "summary": {
                "total_clients": total_count,
                "healthy_clients": healthy_count,
                "unhealthy_clients": total_count - healthy_count,
                "check_time": datetime.utcnow().isoformat()
            },
            "clients": results,
            "stats": self._stats.copy()
        }

    def get_stats(self) -> dict[str, Any]:
        """Get synchronization statistics.

        Returns:
            Dictionary with current statistics
        """
        return {
            **self._stats,
            "total_clients": len(self.clients),
            "connected_clients": sum(1 for c in self.clients.values() if c.is_connected),
            "cached_memories": len(self.local_cache),
            "registered_callbacks": len(self.sync_callbacks)
        }

    async def cleanup(self) -> None:
        """Clean up all clients and resources."""
        self.logger.info("Cleaning up sync manager...")

        # Disconnect all clients
        cleanup_tasks = []
        for _client_id, client in self.clients.items():
            if client.is_connected:
                cleanup_tasks.append(client.disconnect())

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        # Clear everything
        self.clients.clear()
        self.local_cache.clear()
        self.sync_callbacks.clear()

        self.logger.info("Sync manager cleanup complete")


async def demo_websocket_sync() -> None:
    """Demonstrate WebSocket synchronization with comprehensive examples.

    This demo shows:
    - Multi-client WebSocket connections
    - Message handling and event dispatch
    - Synchronization manager usage
    - Health monitoring and error handling
    - Proper resource cleanup
    """
    # Setup detailed logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger("demo")
    logger.info("Starting WebSocket synchronization demo")

    sync_manager = MemorySyncManager()

    # Create multiple clients with different configurations
    clients = {
        "client1": MemoryWebSocketClient(
            "ws://localhost:8000/ws",
            max_reconnect_attempts=3,
            ping_interval=15
        ),
        "client2": MemoryWebSocketClient(
            "ws://localhost:8000/ws",
            max_reconnect_attempts=5,
            ping_interval=20
        ),
        "client3": MemoryWebSocketClient(
            "ws://localhost:8000/ws",
            max_reconnect_attempts=2,
            ping_interval=25
        )
    }

    # Register comprehensive sync callbacks
    async def on_memory_created(memory: dict[str, Any]) -> None:
        """Handle memory creation events."""
        memory.get("content", "")[:LONG_CONTENT_PREVIEW_LENGTH] + "..." if len(memory.get("content", "")) > LONG_CONTENT_PREVIEW_LENGTH else memory.get("content", "")
        tags = memory.get("tags", [])
        logger.info("🆕 Memory created: {content_preview} | Tags: %s", tags)

    async def on_memory_updated(memory: dict[str, Any], old_memory: dict[str, Any]) -> None:
        """Handle memory update events."""
        memory_id = memory.get("id", "unknown")
        old_content = old_memory.get("content", "")[:SHORT_CONTENT_PREVIEW_LENGTH] + "..." if len(old_memory.get("content", "")) > SHORT_CONTENT_PREVIEW_LENGTH else old_memory.get("content", "")
        new_content = memory.get("content", "")[:SHORT_CONTENT_PREVIEW_LENGTH] + "..." if len(memory.get("content", "")) > SHORT_CONTENT_PREVIEW_LENGTH else memory.get("content", "")
        logger.info("📝 Memory updated: %s", memory_id)
        logger.info("   Before: %s", old_content)
        logger.info("   After:  %s", new_content)

    async def on_memory_deleted(memory: dict[str, Any]) -> None:
        """Handle memory deletion events."""
        memory.get("id", "unknown")
        content_preview = memory.get("content", "")[:LONG_CONTENT_PREVIEW_LENGTH] + "..." if len(memory.get("content", "")) > LONG_CONTENT_PREVIEW_LENGTH else memory.get("content", "")
        logger.info("🗑️  Memory deleted: {memory_id} - %s", content_preview)

    # Register callbacks with sync manager
    sync_manager.register_sync_callback("memory_created", on_memory_created)
    sync_manager.register_sync_callback("memory_updated", on_memory_updated)
    sync_manager.register_sync_callback("memory_deleted", on_memory_deleted)

    try:
        # Phase 1: Connection establishment
        logger.info("\n=== Phase 1: Establishing Connections ===")

        connection_tasks = []
        for client_id, client in clients.items():
            logger.info("Connecting %s...", client_id)
            connection_tasks.append(client.connect())

        # Connect all clients concurrently
        connection_results = await asyncio.gather(*connection_tasks, return_exceptions=True)

        # Check connection results and add successful clients to manager
        connected_clients = {}
        for i, (client_id, client) in enumerate(clients.items()):
            if isinstance(connection_results[i], Exception):
                logger.error("Failed to connect {client_id}: %s", connection_results[i])
            else:
                await sync_manager.add_client(client_id, client)
                connected_clients[client_id] = client
                logger.info("✓ %s connected and added to sync manager", client_id)

        if not connected_clients:
            logger.error("No clients connected successfully. Exiting demo.")
            return

        # Phase 2: Event subscription
        logger.info("\n=== Phase 2: Subscribing to Events ===")

        subscription_tasks = []
        for client in connected_clients.values():
            subscription_tasks.append(client.subscribe_to_updates())

        await asyncio.gather(*subscription_tasks, return_exceptions=True)
        logger.info("All connected clients subscribed to memory updates")

        # Phase 3: Health monitoring
        logger.info("\n=== Phase 3: Health Check ===")

        health_results = await sync_manager.health_check()
        logger.info("Health check summary: %s", health_results['summary'])

        for _client_id, health_info in health_results['clients'].items():
            status = "✓ Healthy" if health_info['healthy'] else "✗ Unhealthy"
            logger.info("  {client_id}: %s", status)

        # Phase 4: Message broadcasting
        logger.info("\n=== Phase 4: Testing Message Broadcasting ===")

        test_messages = [
            {"type": "test_broadcast", "data": "Hello from sync manager!", "sequence": 1},
            {"type": "heartbeat", "timestamp": datetime.utcnow().isoformat(), "sequence": 2},
            {"type": "status_update", "status": "demo_running", "sequence": 3}
        ]

        for message in test_messages:
            logger.info("Broadcasting: %s", message['type'])
            broadcast_result = await sync_manager.broadcast_to_clients(message)
            logger.info("  Result: {broadcast_result['successful_sends']}/%s successful", broadcast_result['total_clients'])

            if broadcast_result['errors']:
                for error in broadcast_result['errors']:
                    logger.warning("  Error: %s", error)

            await asyncio.sleep(1)  # Brief pause between broadcasts

        # Phase 5: Statistics and monitoring
        logger.info("\n=== Phase 5: Statistics ===")

        stats = sync_manager.get_stats()
        logger.info("Sync Manager Statistics:")
        for _key, value in stats.items():
            logger.info("  {key}: %s", value)

        # Phase 6: Simulated activity monitoring
        logger.info("\n=== Phase 6: Activity Monitoring ===")
        logger.info("Monitoring for real-time updates (simulated)...")
        logger.info("In a real environment, you would see:")
        logger.info("- Memory creation events")
        logger.info("- Memory update notifications")
        logger.info("- Memory deletion alerts")
        logger.info("- Client connection/disconnection events")

        # Simulate monitoring period
        await asyncio.sleep(3)

        # Phase 7: Testing individual client operations
        logger.info("\n=== Phase 7: Individual Client Operations ===")

        if connected_clients:
            test_client = next(iter(connected_clients.values()))

            # Test ping
            try:
                await test_client.ping()
                logger.info("✓ Ping test successful")
            except Exception as e:
                logger.error("✗ Ping test failed: %s", e)

            # Test custom message
            try:
                await test_client.send_message({
                    "type": "custom_test",
                    "data": "Direct client message",
                    "test_id": "demo_test_001"
                })
                logger.info("✓ Custom message sent successfully")
            except Exception as e:
                logger.error("✗ Custom message failed: %s", e)

        logger.info("\n=== Demo Completed Successfully! ===")
        logger.info("Demo Features Demonstrated:")
        logger.info("✓ Multi-client WebSocket connections")
        logger.info("✓ Event subscription and handling")
        logger.info("✓ Health monitoring and reporting")
        logger.info("✓ Message broadcasting")
        logger.info("✓ Error handling and resilience")
        logger.info("✓ Statistics collection")
        logger.info("✓ Proper resource cleanup")

    except KeyboardInterrupt:
        logger.info("\nDemo interrupted by user")
    except Exception as e:
        logger.error("Demo failed with unexpected error: %s", e)
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup phase
        logger.info("\n=== Cleanup Phase ===")

        try:
            await sync_manager.cleanup()
            logger.info("✓ Sync manager cleanup completed")
        except Exception as e:
            logger.error("Error during cleanup: %s", e)


if __name__ == "__main__":
    try:
        asyncio.run(demo_websocket_sync())
    except KeyboardInterrupt:
        logging.info("\nDemo interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.info("Demo failed: %s", e)
        sys.exit(1)
