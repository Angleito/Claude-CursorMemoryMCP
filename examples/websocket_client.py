#!/usr/bin/env python3
"""
WebSocket Client Example for Real-time Memory Updates
Demonstrates real-time synchronization between multiple clients
"""

import asyncio
import json
import websockets
import logging
from typing import Dict, Any, Callable, Optional, List, Union
from urllib.parse import urlparse
import weakref


class MemoryWebSocketClient:
    """WebSocket client for real-time memory updates"""
    
    def __init__(self, uri: str = "ws://localhost:8000/ws") -> None:
        # Validate URI format
        parsed_uri = urlparse(uri)
        if not parsed_uri.scheme or not parsed_uri.netloc:
            raise ValueError(f"Invalid WebSocket URI: {uri}")
        if parsed_uri.scheme not in ('ws', 'wss'):
            raise ValueError(f"URI must use ws:// or wss:// scheme: {uri}")
            
        self.uri = uri
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.message_handlers: Dict[str, Callable[[Dict[str, Any]], None]] = {}
        self.is_connected: bool = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self._listen_task: Optional[asyncio.Task] = None
        self._reconnect_attempts: int = 0
        self._max_reconnect_attempts: int = 5
    
    async def connect(self, auth_token: Optional[str] = None) -> None:
        """Connect to WebSocket server"""
        headers = {}
        if auth_token:
            if not auth_token.strip():
                raise ValueError("Auth token cannot be empty")
            headers["Authorization"] = f"Bearer {auth_token.strip()}"
        
        try:
            self.websocket = await websockets.connect(
                self.uri, 
                extra_headers=headers,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            self.is_connected = True
            self._reconnect_attempts = 0
            self.logger.info(f"Connected to {self.uri}")
            
            # Start listening for messages
            self._listen_task = asyncio.create_task(self._listen_for_messages())
            
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            self.is_connected = False
            raise ConnectionError(f"Failed to connect to {self.uri}: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from WebSocket server"""
        self.is_connected = False
        
        # Cancel listening task
        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        
        # Close WebSocket connection
        if self.websocket and not self.websocket.closed:
            try:
                await self.websocket.close()
                self.logger.info("Disconnected from WebSocket server")
            except Exception as e:
                self.logger.error(f"Error closing WebSocket: {e}")
        
        self.websocket = None
        self._listen_task = None
    
    async def send_message(self, message: Dict[str, Any]) -> None:
        """Send a message to the server"""
        if not self.is_connected or not self.websocket or self.websocket.closed:
            raise RuntimeError("Not connected to WebSocket server")
        
        if not isinstance(message, dict):
            raise TypeError("Message must be a dictionary")
        
        try:
            message_json = json.dumps(message)
            await self.websocket.send(message_json)
            self.logger.debug(f"Sent message: {message}")
        except json.JSONEncodeError as e:
            raise ValueError(f"Failed to serialize message: {e}")
        except websockets.exceptions.ConnectionClosed:
            self.is_connected = False
            raise ConnectionError("WebSocket connection lost")
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            raise
    
    async def _listen_for_messages(self) -> None:
        """Listen for incoming messages"""
        try:
            async for message in self.websocket:
                try:
                    if isinstance(message, bytes):
                        message = message.decode('utf-8')
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON message: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
        except websockets.exceptions.ConnectionClosed:
            self.logger.info("WebSocket connection closed")
            self.is_connected = False
            # Attempt reconnection if not manually disconnected
            if self._reconnect_attempts < self._max_reconnect_attempts:
                await self._attempt_reconnect()
        except Exception as e:
            self.logger.error(f"Error listening for messages: {e}")
            self.is_connected = False
    
    async def _handle_message(self, data: Dict[str, Any]) -> None:
        """Handle incoming message"""
        if not isinstance(data, dict):
            self.logger.warning(f"Received non-dict message: {type(data)}")
            return
            
        message_type = data.get("type")
        
        if message_type and message_type in self.message_handlers:
            try:
                handler = self.message_handlers[message_type]
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                self.logger.error(f"Error in message handler for {message_type}: {e}")
        else:
            self.logger.debug(f"Received unhandled message type '{message_type}': {data}")
    
    def register_handler(self, message_type: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Register a message handler"""
        if not message_type or not message_type.strip():
            raise ValueError("Message type cannot be empty")
        if not callable(handler):
            raise TypeError("Handler must be callable")
            
        self.message_handlers[message_type.strip()] = handler
        self.logger.debug(f"Registered handler for message type: {message_type}")
    
    async def ping(self) -> None:
        """Send a ping message"""
        await self.send_message({"type": "ping"})
    
    async def subscribe_to_updates(self, event_types: Optional[List[str]] = None) -> None:
        """Subscribe to memory updates"""
        default_events = ["memory_created", "memory_updated", "memory_deleted"]
        events = event_types or default_events
        
        if not isinstance(events, list):
            raise TypeError("Event types must be a list")
        if not events:
            raise ValueError("At least one event type must be specified")
            
        message = {
            "type": "subscribe",
            "events": events
        }
        await self.send_message(message)
        self.logger.info(f"Subscribed to events: {events}")
        
    async def _attempt_reconnect(self) -> None:
        """Attempt to reconnect to the WebSocket server"""
        self._reconnect_attempts += 1
        delay = min(2 ** self._reconnect_attempts, 30)  # Exponential backoff, max 30s
        
        self.logger.info(f"Attempting reconnection {self._reconnect_attempts}/{self._max_reconnect_attempts} in {delay}s")
        await asyncio.sleep(delay)
        
        try:
            await self.connect()
        except Exception as e:
            self.logger.error(f"Reconnection attempt {self._reconnect_attempts} failed: {e}")
            if self._reconnect_attempts < self._max_reconnect_attempts:
                await self._attempt_reconnect()


class MemorySyncManager:
    """Manages synchronization of memories across multiple clients"""
    
    def __init__(self) -> None:
        self.clients: Dict[str, MemoryWebSocketClient] = {}
        self.local_cache: Dict[str, Dict[str, Any]] = {}
        self.sync_callbacks: Dict[str, Callable[[Dict[str, Any]], None]] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def add_client(self, client_id: str, client: MemoryWebSocketClient) -> None:
        """Add a client to sync management"""
        if not client_id or not client_id.strip():
            raise ValueError("Client ID cannot be empty")
        if not isinstance(client, MemoryWebSocketClient):
            raise TypeError("Client must be a MemoryWebSocketClient instance")
            
        client_id = client_id.strip()
        if client_id in self.clients:
            self.logger.warning(f"Client {client_id} already exists, replacing")
            
        self.clients[client_id] = client
        
        # Register sync handlers
        client.register_handler("memory_created", self._handle_memory_created)
        client.register_handler("memory_updated", self._handle_memory_updated)
        client.register_handler("memory_deleted", self._handle_memory_deleted)
        client.register_handler("pong", self._handle_pong)
        
        self.logger.info(f"Added client {client_id} to sync manager")
    
    async def _handle_memory_created(self, data: Dict[str, Any]) -> None:
        """Handle memory creation event"""
        try:
            memory = data.get("data", {})
            memory_id = memory.get("id")
            
            if memory_id:
                self.local_cache[memory_id] = memory
                content_preview = memory.get('content', '')[:50]
                self.logger.info(f"Memory created: {memory_id} - {content_preview}...")
                
                # Notify callbacks
                if "memory_created" in self.sync_callbacks:
                    callback = self.sync_callbacks["memory_created"]
                    if asyncio.iscoroutinefunction(callback):
                        await callback(memory)
                    else:
                        callback(memory)
        except Exception as e:
            self.logger.error(f"Error handling memory creation: {e}")
    
    async def _handle_memory_updated(self, data: Dict[str, Any]) -> None:
        """Handle memory update event"""
        try:
            memory = data.get("data", {})
            memory_id = memory.get("id")
            
            if memory_id:
                old_memory = self.local_cache.get(memory_id, {})
                self.local_cache[memory_id] = memory
                self.logger.info(f"Memory updated: {memory_id}")
                
                # Notify callbacks
                if "memory_updated" in self.sync_callbacks:
                    callback = self.sync_callbacks["memory_updated"]
                    if asyncio.iscoroutinefunction(callback):
                        await callback(memory, old_memory)
                    else:
                        callback(memory, old_memory)
        except Exception as e:
            self.logger.error(f"Error handling memory update: {e}")
    
    async def _handle_memory_deleted(self, data: Dict[str, Any]) -> None:
        """Handle memory deletion event"""
        try:
            memory_id = data.get("data", {}).get("memory_id")
            
            if memory_id and memory_id in self.local_cache:
                deleted_memory = self.local_cache.pop(memory_id)
                self.logger.info(f"Memory deleted: {memory_id}")
                
                # Notify callbacks
                if "memory_deleted" in self.sync_callbacks:
                    callback = self.sync_callbacks["memory_deleted"]
                    if asyncio.iscoroutinefunction(callback):
                        await callback(deleted_memory)
                    else:
                        callback(deleted_memory)
        except Exception as e:
            self.logger.error(f"Error handling memory deletion: {e}")
    
    async def _handle_pong(self, data: Dict[str, Any]) -> None:
        """Handle pong response"""
        self.logger.debug(f"Received pong: {data}")
    
    def register_sync_callback(self, event_type: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for sync events"""
        if not event_type or not event_type.strip():
            raise ValueError("Event type cannot be empty")
        if not callable(callback):
            raise TypeError("Callback must be callable")
            
        self.sync_callbacks[event_type.strip()] = callback
        self.logger.debug(f"Registered sync callback for event: {event_type}")
    
    async def broadcast_to_clients(self, message: Dict[str, Any]) -> None:
        """Broadcast a message to all connected clients"""
        if not isinstance(message, dict):
            raise TypeError("Message must be a dictionary")
            
        successful_sends = 0
        failed_sends = 0
        
        for client_id, client in self.clients.items():
            if client.is_connected:
                try:
                    await client.send_message(message)
                    successful_sends += 1
                except Exception as e:
                    self.logger.warning(f"Failed to send message to {client_id}: {e}")
                    failed_sends += 1
        
        self.logger.info(f"Broadcast complete: {successful_sends} successful, {failed_sends} failed")
    
    async def health_check(self) -> Dict[str, bool]:
        """Perform health check on all clients"""
        results: Dict[str, bool] = {}
        
        for client_id, client in self.clients.items():
            if client.is_connected:
                try:
                    await client.ping()
                    results[client_id] = True
                except Exception as e:
                    self.logger.warning(f"Health check failed for {client_id}: {e}")
                    results[client_id] = False
            else:
                results[client_id] = False
        
        healthy_count = sum(results.values())
        total_count = len(results)
        self.logger.info(f"Health check complete: {healthy_count}/{total_count} clients healthy")
        
        return results


async def demo_websocket_sync() -> None:
    """Demonstrate WebSocket synchronization"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    sync_manager = MemorySyncManager()
    
    # Create multiple clients
    client1 = MemoryWebSocketClient("ws://localhost:8000/ws")
    client2 = MemoryWebSocketClient("ws://localhost:8000/ws")
    
    # Register sync callbacks
    async def on_memory_created(memory: Dict[str, Any]) -> None:
        content_preview = memory.get('content', '')[:30]
        print(f"[SYNC] New memory synced: {content_preview}...")
    
    async def on_memory_updated(memory: Dict[str, Any], old_memory: Dict[str, Any]) -> None:
        print(f"[SYNC] Memory updated: {memory.get('id')}")
    
    async def on_memory_deleted(memory: Dict[str, Any]) -> None:
        print(f"[SYNC] Memory deleted: {memory.get('id')}")
    
    sync_manager.register_sync_callback("memory_created", on_memory_created)
    sync_manager.register_sync_callback("memory_updated", on_memory_updated)
    sync_manager.register_sync_callback("memory_deleted", on_memory_deleted)
    
    try:
        # Connect clients
        print("Connecting clients...")
        await client1.connect()
        await client2.connect()
        
        # Add to sync manager
        await sync_manager.add_client("client1", client1)
        await sync_manager.add_client("client2", client2)
        
        # Subscribe to updates
        await client1.subscribe_to_updates()
        await client2.subscribe_to_updates()
        
        print("WebSocket sync demo started. Listening for memory updates...")
        print("(In another terminal, create/update/delete memories to see sync in action)")
        
        # Simulate some activity
        await asyncio.sleep(2)
        
        # Send ping to test connectivity
        print("\nSending ping to test connectivity...")
        health_results = await sync_manager.health_check()
        print(f"Health check results: {health_results}")
        
        # Keep listening for updates
        print("\nListening for real-time updates... (Press Ctrl+C to stop)")
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client1.disconnect()
        await client2.disconnect()


if __name__ == "__main__":
    print("WebSocket Real-time Sync Demo")
    print("=" * 40)
    print("Note: Make sure the Mem0 AI server is running on localhost:8000")
    print()
    asyncio.run(demo_websocket_sync())