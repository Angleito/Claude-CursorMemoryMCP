#!/usr/bin/env python3
"""Simple MCP Server for testing Claude Code integration."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create MCP server instance
app = Server("mem0ai-memory")

# In-memory storage for testing
memories: List[Dict[str, Any]] = []

@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools."""
    return [
        Tool(
            name="store_memory",
            description="Store a new memory with content, context, and tags",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The memory content to store"},
                    "context": {"type": "object", "description": "Context metadata for the memory"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags for categorizing the memory"}
                },
                "required": ["content"]
            }
        ),
        Tool(
            name="search_memory",
            description="Search memories by query string",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Maximum number of results", "default": 10}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_recent_memories",
            description="Get recent memories",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Maximum number of results", "default": 10}
                }
            }
        ),
        Tool(
            name="delete_memory",
            description="Delete a memory by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "ID of the memory to delete"}
                },
                "required": ["memory_id"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    global memories
    logger.info(f"Tool called: {name} with arguments: {arguments}")
    
    try:
        if name == "store_memory":
            memory_id = f"mem_{len(memories) + 1}"
            memory = {
                "id": memory_id,
                "content": arguments["content"],
                "context": arguments.get("context", {}),
                "tags": arguments.get("tags", []),
                "timestamp": asyncio.get_event_loop().time()
            }
            memories.append(memory)
            
            return [TextContent(
                type="text",
                text=f"Memory stored successfully with ID: {memory_id}"
            )]
            
        elif name == "search_memory":
            query = arguments["query"].lower()
            limit = arguments.get("limit", 10)
            
            # Simple search - look for query in content
            results = []
            for memory in memories:
                if query in memory["content"].lower():
                    results.append(memory)
                    if len(results) >= limit:
                        break
            
            return [TextContent(
                type="text",
                text=f"Found {len(results)} memories:\n" + 
                     "\n".join([f"- {m['id']}: {m['content'][:100]}" for m in results])
            )]
            
        elif name == "get_recent_memories":
            limit = arguments.get("limit", 10)
            recent = memories[-limit:] if len(memories) > limit else memories
            
            return [TextContent(
                type="text",
                text=f"Recent {len(recent)} memories:\n" +
                     "\n".join([f"- {m['id']}: {m['content'][:100]}" for m in recent])
            )]
            
        elif name == "delete_memory":
            memory_id = arguments["memory_id"]
            memories = [m for m in memories if m["id"] != memory_id]
            
            return [TextContent(
                type="text",
                text=f"Memory {memory_id} deleted successfully"
            )]
            
        else:
            return [TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )]
            
    except Exception as e:
        logger.error(f"Error calling tool {name}: {e}")
        return [TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]

async def main():
    """Main function to run the MCP server."""
    logger.info("Starting Mem0AI MCP Server...")
    
    # Run the server using stdio transport
    async with stdio_server() as streams:
        await app.run(*streams)

if __name__ == "__main__":
    asyncio.run(main())