"""WebSocket connection manager for real-time updates.

This module provides comprehensive WebSocket connection management including
connection lifecycle, health monitoring, message broadcasting, and automatic
cleanup of stale connections.
"""

from __future__ import annotations

# Standard library imports
import asyncio
import contextlib
import json
import time
import weakref
from enum import Enum
from typing import Any, Dict, List, Optional, Set

# Third-party imports
import structlog
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

logger = structlog.get_logger()


class ConnectionState(Enum):
    """WebSocket connection states."""

    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class WebSocketMessage(BaseModel):
    """WebSocket message model for validation."""

    type: str
    data: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None
    message_id: Optional[str] = None


class ConnectionInfo:
    """Information about a WebSocket connection.
    
    This class tracks connection state, health metrics, and provides
    ping/pong functionality for connection monitoring.
    """

    def __init__(self, websocket: WebSocket, user_id: Optional[str] = None) -> None:
        self.websocket = websocket
        self.user_id = user_id
        self.state = ConnectionState.CONNECTING
        self.connected_at = time.time()
        self.last_ping: Optional[float] = None
        self.last_pong: Optional[float] = None
        self.message_count = 0
        self.error_count = 0
        self.max_errors = 5
        self.ping_interval = 30  # seconds
        self.ping_timeout = 10  # seconds
        self.is_alive = True
        self._ping_task: Optional[asyncio.Task] = None

    def start_ping_task(self, manager: ConnectionManager) -> None:
        """Start periodic ping task.
        
        Args:
            manager: Connection manager instance for disconnect operations
        """
        if self._ping_task is None or self._ping_task.done():
            self._ping_task = asyncio.create_task(self._ping_loop(manager))

    def stop_ping_task(self) -> None:
        """Stop periodic ping task."""
        if self._ping_task and not self._ping_task.done():
            self._ping_task.cancel()

    async def _ping_loop(self, manager):
        """Periodic ping loop."""
        try:
            while self.is_alive and self.state == ConnectionState.CONNECTED:
                await asyncio.sleep(self.ping_interval)

                if not self.is_alive:
                    break

                # Send ping
                try:
                    await self.send_ping()

                    # Wait for pong with timeout
                    start_time = time.time()
                    while (time.time() - start_time) < self.ping_timeout:
                        if self.last_pong and self.last_pong > self.last_ping:
                            break
                        await asyncio.sleep(0.1)
                    else:
                        # Ping timeout
                        logger.warning("Ping timeout", user_id=self.user_id)
                        await manager.disconnect(self.websocket)
                        break

                except Exception as e:
                    logger.error(
                        "Error in ping loop", error=str(e), user_id=self.user_id
                    )
                    await manager.disconnect(self.websocket)
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(
                "Unexpected error in ping loop", error=str(e), user_id=self.user_id
            )

    async def send_ping(self) -> None:
        """Send ping message.
        
        Raises:
            Exception: If ping sending fails
        """
        try:
            ping_message = {"type": "ping", "timestamp": time.time()}
            await self.websocket.send_text(json.dumps(ping_message))
            self.last_ping = time.time()
        except Exception as e:
            logger.error("Failed to send ping", error=str(e), user_id=self.user_id)
            raise

    def handle_pong(self) -> None:
        """Handle pong message and update last pong timestamp."""
        self.last_pong = time.time()

    def increment_error(self) -> bool:
        """Increment error count and check if max reached."""
        self.error_count += 1
        return self.error_count >= self.max_errors


class ConnectionManager:
    """Manages WebSocket connections for real-time updates.
    
    This class provides comprehensive WebSocket connection management including:
    - Connection lifecycle management
    - Health monitoring with automatic ping/pong
    - Message broadcasting to users or all connections
    - Automatic cleanup of stale connections
    - Performance monitoring and statistics
    """

    def __init__(self) -> None:
        self.active_connections: Dict[WebSocket, ConnectionInfo] = {}
        self.user_connections: Dict[str, Set[WebSocket]] = {}
        self.connection_by_id: Dict[str, WebSocket] = {}
        self._closed = False
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_tasks: weakref.WeakSet = weakref.WeakSet()

        # Start background cleanup task
        self._start_cleanup_task()

    def _start_cleanup_task(self):
        """Start background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._background_cleanup())
            self._cleanup_tasks.add(self._cleanup_task)

    async def connect(
        self,
        websocket: WebSocket,
        user_id: Optional[str] = None,
        connection_id: Optional[str] = None,
    ) -> str:
        """Accept a new WebSocket connection.
        
        Args:
            websocket: WebSocket connection to accept
            user_id: Optional user ID for connection tracking
            connection_id: Optional custom connection identifier
            
        Returns:
            Generated or provided connection ID
            
        Raises:
            RuntimeError: If connection manager is closed
        """
        if self._closed:
            await websocket.close(code=1001, reason="Server shutting down")
            raise RuntimeError("Connection manager is closed")

        try:
            await websocket.accept()

            # Create connection info
            conn_info = ConnectionInfo(websocket, user_id)
            conn_info.state = ConnectionState.CONNECTED

            # Generate connection ID if not provided
            if connection_id is None:
                connection_id = (
                    f"conn_{int(time.time() * 1000)}_{len(self.active_connections)}"
                )

            # Store connection
            self.active_connections[websocket] = conn_info
            self.connection_by_id[connection_id] = websocket

            # Add to user connections
            if user_id:
                if user_id not in self.user_connections:
                    self.user_connections[user_id] = set()
                self.user_connections[user_id].add(websocket)

            # Start ping task
            conn_info.start_ping_task(self)

            # Send welcome message
            welcome_msg = {
                "type": "welcome",
                "connection_id": connection_id,
                "user_id": user_id,
                "timestamp": time.time(),
            }
            await self.send_personal_message(welcome_msg, websocket)

            logger.info(
                "WebSocket connected",
                user_id=user_id,
                connection_id=connection_id,
                total_connections=len(self.active_connections),
            )

            return connection_id

        except Exception as e:
            logger.error(
                "Failed to accept WebSocket connection", error=str(e), user_id=user_id
            )
            with contextlib.suppress(Exception):
                await websocket.close(code=1011, reason="Connection setup failed")
            raise

    async def disconnect(
        self, websocket: WebSocket, code: int = 1000, reason: str = "Normal closure"
    ) -> None:
        """Remove a WebSocket connection.
        
        Args:
            websocket: WebSocket connection to disconnect
            code: WebSocket close code
            reason: Human-readable close reason
        """
        conn_info = self.active_connections.get(websocket)
        if not conn_info:
            return  # Already disconnected

        user_id = conn_info.user_id

        # Update connection state
        conn_info.state = ConnectionState.DISCONNECTING
        conn_info.is_alive = False

        # Stop ping task
        conn_info.stop_ping_task()

        # Send goodbye message if possible
        try:
            goodbye_msg = {
                "type": "goodbye",
                "reason": reason,
                "timestamp": time.time(),
            }
            await websocket.send_text(json.dumps(goodbye_msg))
        except:
            pass  # Connection might already be closed

        # Close WebSocket
        try:
            if websocket.client_state.name != "DISCONNECTED":
                await websocket.close(code=code, reason=reason)
        except Exception as e:
            logger.debug("Error closing WebSocket", error=str(e))

        # Remove from active connections
        if websocket in self.active_connections:
            del self.active_connections[websocket]

        # Remove from user connections
        if user_id and user_id in self.user_connections:
            self.user_connections[user_id].discard(websocket)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]

        # Remove from connection_by_id
        to_remove = [
            conn_id for conn_id, ws in self.connection_by_id.items() if ws == websocket
        ]
        for conn_id in to_remove:
            del self.connection_by_id[conn_id]

        # Update final state
        conn_info.state = ConnectionState.DISCONNECTED

        logger.info(
            "WebSocket disconnected",
            user_id=user_id,
            code=code,
            reason=reason,
            total_connections=len(self.active_connections),
        )

    async def send_personal_message(
        self, message: Dict[str, Any], websocket: WebSocket
    ) -> bool:
        """Send a message to a specific WebSocket connection.
        
        Args:
            message: Message dictionary to send
            websocket: Target WebSocket connection
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        conn_info = self.active_connections.get(websocket)
        if not conn_info or conn_info.state != ConnectionState.CONNECTED:
            return False

        try:
            # Validate message structure
            try:
                WebSocketMessage(**message)
            except ValidationError as e:
                logger.error("Invalid message structure", error=str(e), message=message)
                return False

            # Add timestamp if not present
            if "timestamp" not in message:
                message["timestamp"] = time.time()

            # Handle pong messages
            if message.get("type") == "pong":
                conn_info.handle_pong()

            # Send message
            message_json = json.dumps(message)
            await websocket.send_text(message_json)

            # Update message count
            conn_info.message_count += 1

            return True

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected during send", user_id=conn_info.user_id)
            await self.disconnect(websocket, code=1001, reason="Client disconnected")
            return False
        except Exception as e:
            logger.error(
                "Failed to send personal message",
                error=str(e),
                user_id=conn_info.user_id,
            )

            # Increment error count
            if conn_info.increment_error():
                logger.warning(
                    "Max errors reached, disconnecting", user_id=conn_info.user_id
                )
                await self.disconnect(websocket, code=1011, reason="Too many errors")

            return False

    async def send_user_message(self, message: Dict[str, Any], user_id: str) -> int:
        """Send a message to all connections for a specific user.
        
        Args:
            message: Message dictionary to send
            user_id: Target user ID
            
        Returns:
            Number of successful sends
        """
        if user_id not in self.user_connections:
            return 0

        successful_sends = 0
        failed_connections = []

        # Create a copy of the connections set to avoid modification during iteration
        user_websockets = list(self.user_connections[user_id])

        for websocket in user_websockets:
            success = await self.send_personal_message(message, websocket)
            if success:
                successful_sends += 1
            else:
                failed_connections.append(websocket)

        # Clean up failed connections
        for websocket in failed_connections:
            if websocket in self.active_connections:
                await self.disconnect(websocket, code=1011, reason="Send failed")

        return successful_sends

    async def broadcast(self, message: Dict[str, Any]) -> int:
        """Broadcast a message to all connected clients.
        
        Args:
            message: Message dictionary to broadcast
            
        Returns:
            Number of successful sends
        """
        if self._closed:
            logger.warning("Cannot broadcast: connection manager is closed")
            return 0

        successful_sends = 0
        failed_connections = []

        # Create a copy of the connections to avoid modification during iteration
        active_websockets = list(self.active_connections.keys())

        for websocket in active_websockets:
            success = await self.send_personal_message(message, websocket)
            if success:
                successful_sends += 1
            else:
                failed_connections.append(websocket)

        # Clean up failed connections
        for websocket in failed_connections:
            if websocket in self.active_connections:
                await self.disconnect(websocket, code=1011, reason="Broadcast failed")

        logger.debug(
            "Broadcast completed",
            successful=successful_sends,
            failed=len(failed_connections),
            total=len(active_websockets),
        )

        return successful_sends

    async def broadcast_to_room(self, message: Dict[str, Any], room: str) -> int:
        """Broadcast a message to clients in a specific room."""
        # For future implementation of rooms/channels
        # For now, treat room as a user_id pattern
        successful_sends = 0

        for user_id in list(self.user_connections.keys()):
            if room in user_id or user_id.startswith(room):
                successful_sends += await self.send_user_message(message, user_id)

        return successful_sends

    def get_user_connections(self, user_id: str) -> List[WebSocket]:
        """Get all WebSocket connections for a user."""
        return list(self.user_connections.get(user_id, set()))

    def get_connection_info(self) -> Dict[str, Any]:
        """Get information about current connections."""
        connection_states = {}
        for websocket, conn_info in self.active_connections.items():
            connection_states[str(id(websocket))] = {
                "user_id": conn_info.user_id,
                "state": conn_info.state.value,
                "connected_at": conn_info.connected_at,
                "message_count": conn_info.message_count,
                "error_count": conn_info.error_count,
                "last_ping": conn_info.last_ping,
                "last_pong": conn_info.last_pong,
            }

        return {
            "total_connections": len(self.active_connections),
            "user_connections": {
                user_id: len(connections)
                for user_id, connections in self.user_connections.items()
            },
            "connection_states": connection_states,
            "is_closed": self._closed,
        }

    async def ping_all(self) -> int:
        """Send ping to all connections to check health."""
        ping_message = {"type": "ping", "timestamp": time.time()}
        return await self.broadcast(ping_message)

    async def cleanup_stale_connections(self):
        """Remove stale connections that haven't responded."""
        current_time = time.time()
        stale_connections = []

        for websocket, conn_info in list(self.active_connections.items()):
            # Check if connection is stale (no pong response for a while)
            if conn_info.last_ping and conn_info.last_pong:
                if (
                    conn_info.last_ping > conn_info.last_pong
                    and (current_time - conn_info.last_ping) > conn_info.ping_timeout
                ):
                    stale_connections.append(websocket)

            # Check if connection has been idle for too long (no activity for 10 minutes)
            elif (current_time - conn_info.connected_at) > 600:
                # Send a health check
                try:
                    health_check = {"type": "health_check", "timestamp": current_time}
                    success = await self.send_personal_message(health_check, websocket)
                    if not success:
                        stale_connections.append(websocket)
                except Exception:
                    stale_connections.append(websocket)

        # Disconnect stale connections
        for websocket in stale_connections:
            await self.disconnect(websocket, code=1001, reason="Stale connection")

        if stale_connections:
            logger.info("Cleaned up stale connections", count=len(stale_connections))

    async def _background_cleanup(self):
        """Background task to clean up stale connections."""
        while not self._closed:
            try:
                await asyncio.sleep(60)  # Run every minute
                await self.cleanup_stale_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in connection cleanup task", error=str(e))
                await asyncio.sleep(60)  # Wait before retrying

    async def close(self) -> None:
        """Close the connection manager and all connections.
        
        This method gracefully shuts down all connections and cleanup tasks.
        """
        if self._closed:
            return

        self._closed = True

        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()

        # Cancel all cleanup tasks
        for task in list(self._cleanup_tasks):
            if not task.done():
                task.cancel()

        # Disconnect all connections
        connections_to_close = list(self.active_connections.keys())
        for websocket in connections_to_close:
            await self.disconnect(websocket, code=1001, reason="Server shutdown")

        # Clear all data structures
        self.active_connections.clear()
        self.user_connections.clear()
        self.connection_by_id.clear()

        logger.info("Connection manager closed")

    def is_closed(self) -> bool:
        """Check if connection manager is closed."""
        return self._closed

    async def handle_message(self, websocket: WebSocket, message: str) -> bool:
        """Handle incoming WebSocket message."""
        conn_info = self.active_connections.get(websocket)
        if not conn_info:
            return False

        try:
            # Parse message
            try:
                data = json.loads(message)
            except json.JSONDecodeError as e:
                logger.error(
                    "Invalid JSON message", error=str(e), user_id=conn_info.user_id
                )
                return False

            # Validate message structure
            try:
                ws_message = WebSocketMessage(**data)
            except ValidationError as e:
                logger.error(
                    "Invalid message structure", error=str(e), user_id=conn_info.user_id
                )
                return False

            # Handle different message types
            if ws_message.type == "pong":
                conn_info.handle_pong()
                return True
            elif ws_message.type == "ping":
                # Respond with pong
                pong_msg = {"type": "pong", "timestamp": time.time()}
                return await self.send_personal_message(pong_msg, websocket)
            else:
                # Other message types can be handled by application logic
                logger.debug(
                    "Received message", type=ws_message.type, user_id=conn_info.user_id
                )
                return True

        except Exception as e:
            logger.error(
                "Error handling WebSocket message",
                error=str(e),
                user_id=conn_info.user_id,
            )
            if conn_info.increment_error():
                await self.disconnect(websocket, code=1011, reason="Too many errors")
            return False


# Background task to periodically clean up stale connections
async def connection_cleanup_task(connection_manager: ConnectionManager):
    """Background task to clean up stale connections (deprecated - use built-in cleanup)."""
    logger.warning(
        "connection_cleanup_task is deprecated - use ConnectionManager._background_cleanup instead"
    )
    while not connection_manager.is_closed():
        try:
            await asyncio.sleep(60)  # Run every minute
            await connection_manager.cleanup_stale_connections()
        except Exception as e:
            logger.error("Error in connection cleanup task", error=str(e))
