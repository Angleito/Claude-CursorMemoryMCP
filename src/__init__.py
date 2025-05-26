"""Mem0 AI MCP Server Package.

This package provides a Model Context Protocol (MCP) server implementation
for memory management with vector search capabilities, user authentication,
and real-time WebSocket communication.

Modules:
    config: Configuration management and settings
    models: Pydantic models for data validation
    database: Database connection and management
    memory: Memory management system with vector search
    auth: Authentication and authorization system
    mcp: Model Context Protocol server implementation
    websocket: WebSocket connection manager for real-time updates
    plugins: Plugin architecture for extensibility
    metrics: Metrics and monitoring system
"""

__version__ = "1.0.0"
__author__ = "Mem0 AI Team"

# Import main components for easier access
from .config import Settings
from .memory import MemoryManager
from .auth import AuthManager
from .mcp import MCPServer
from .websocket import ConnectionManager
from .plugins import PluginManager
from .metrics import MetricsCollector

__all__ = [
    "Settings",
    "MemoryManager", 
    "AuthManager",
    "MCPServer",
    "ConnectionManager",
    "PluginManager",
    "MetricsCollector",
]
