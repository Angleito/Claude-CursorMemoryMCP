#!/usr/bin/env python3
"""
Claude Code MCP Client Example
Demonstrates how to interact with Mem0 AI MCP Server from Claude Code
"""

import json
import asyncio
import subprocess
import sys
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path


class ClaudeCodeMCPClient:
    """Client for interacting with Mem0 AI MCP server via Claude Code"""
    
    def __init__(self, server_command: Optional[List[str]] = None) -> None:
        self.server_command = server_command or [
            "python", 
            str(Path(__file__).parent.parent / "main.py"), 
            "--stdio"
        ]
        self.process: Optional[asyncio.subprocess.Process] = None
        self.request_id: int = 0
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def start_server(self) -> None:
        """Start the MCP server process"""
        try:
            self.process = await asyncio.create_subprocess_exec(
                *self.server_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            self.logger.info("MCP server started")
        except Exception as e:
            self.logger.error(f"Failed to start MCP server: {e}")
            raise RuntimeError(f"Failed to start MCP server: {e}")
    
    async def stop_server(self) -> None:
        """Stop the MCP server process"""
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
                self.logger.info("MCP server stopped")
            except asyncio.TimeoutError:
                self.logger.warning("Server didn't stop gracefully, killing process")
                self.process.kill()
                await self.process.wait()
            except Exception as e:
                self.logger.error(f"Error stopping server: {e}")
            finally:
                self.process = None
    
    async def send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a request to the MCP server"""
        if not self.process or not self.process.stdin or not self.process.stdout:
            raise RuntimeError("Server not started or streams not available")
        
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params or {}
        }
        
        try:
            # Send request
            request_json = json.dumps(request) + "\n"
            self.process.stdin.write(request_json.encode())
            await self.process.stdin.drain()
            
            # Read response with timeout
            response_line = await asyncio.wait_for(
                self.process.stdout.readline(), 
                timeout=30.0
            )
            
            if not response_line:
                raise RuntimeError("Empty response from server")
                
            response = json.loads(response_line.decode().strip())
            
            if "error" in response:
                error_info = response['error']
                if isinstance(error_info, dict):
                    message = error_info.get('message', 'Unknown error')
                    code = error_info.get('code', 'Unknown')
                    raise Exception(f"MCP Error [{code}]: {message}")
                else:
                    raise Exception(f"MCP Error: {error_info}")
            
            return response.get("result", {})
            
        except asyncio.TimeoutError:
            raise TimeoutError("Request timed out")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}")
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            raise
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the MCP connection"""
        return await self.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "claude-code-client",
                "version": "1.0.0"
            }
        })
    
    async def list_tools(self) -> Dict[str, Any]:
        """List available MCP tools"""
        return await self.send_request("tools/list")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific MCP tool"""
        if not tool_name:
            raise ValueError("Tool name cannot be empty")
        if not isinstance(arguments, dict):
            raise ValueError("Arguments must be a dictionary")
            
        return await self.send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
    
    async def search_memories(self, query: str, limit: int = 10, **kwargs) -> Dict[str, Any]:
        """Search memories using semantic similarity"""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if limit <= 0 or limit > 1000:
            raise ValueError("Limit must be between 1 and 1000")
            
        args = {"query": query.strip(), "limit": limit}
        args.update(kwargs)
        return await self.call_tool("memory_search", args)
    
    async def create_memory(self, content: str, **kwargs) -> Dict[str, Any]:
        """Create a new memory"""
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")
        if len(content) > 100000:
            raise ValueError("Content too long (max 100,000 characters)")
            
        args = {"content": content.strip()}
        args.update(kwargs)
        return await self.call_tool("memory_create", args)
    
    async def get_memory(self, memory_id: str) -> Dict[str, Any]:
        """Get a specific memory"""
        if not memory_id or not memory_id.strip():
            raise ValueError("Memory ID cannot be empty")
            
        return await self.call_tool("memory_get", {"memory_id": memory_id.strip()})
    
    async def list_memories(self, **kwargs) -> Dict[str, Any]:
        """List memories with optional filtering"""
        return await self.call_tool("memory_list", kwargs)
    
    async def update_memory(self, memory_id: str, **kwargs) -> Dict[str, Any]:
        """Update an existing memory"""
        if not memory_id or not memory_id.strip():
            raise ValueError("Memory ID cannot be empty")
            
        args = {"memory_id": memory_id.strip()}
        args.update(kwargs)
        return await self.call_tool("memory_update", args)
    
    async def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """Delete a memory"""
        if not memory_id or not memory_id.strip():
            raise ValueError("Memory ID cannot be empty")
            
        return await self.call_tool("memory_delete", {"memory_id": memory_id.strip()})


async def demo_claude_code_integration() -> None:
    """Demonstrate Claude Code integration with Mem0 AI"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    client = ClaudeCodeMCPClient()
    
    try:
        # Start the server
        await client.start_server()
        
        # Initialize connection
        print("Initializing MCP connection...")
        init_result = await client.initialize()
        print(f"Initialization: {init_result}")
        
        # List available tools
        print("\nListing available tools...")
        tools = await client.list_tools()
        print(f"Available tools: {[tool['name'] for tool in tools.get('tools', [])]}")
        
        # Create some example memories
        print("\nCreating example memories...")
        
        memories_to_create: List[Dict[str, Union[str, List[str]]]] = [
            {
                "content": "Python is a high-level programming language known for its simplicity and readability.",
                "tags": ["programming", "python", "language"],
                "memory_type": "fact",
                "priority": "medium"
            },
            {
                "content": "FastAPI is a modern web framework for building APIs with Python based on type hints.",
                "tags": ["python", "fastapi", "web", "api"],
                "memory_type": "fact",
                "priority": "high"
            },
            {
                "content": "Remember to review the memory search algorithm performance next week.",
                "tags": ["todo", "performance", "search"],
                "memory_type": "task",
                "priority": "high"
            }
        ]
        
        created_memories: List[Dict[str, Any]] = []
        for memory_data in memories_to_create:
            try:
                result = await client.create_memory(**memory_data)
                if result.get("isError"):
                    print(f"Error creating memory: {result}")
                else:
                    memory = result.get("data", {}).get("memory", {})
                    created_memories.append(memory)
                    print(f"Created memory: {memory.get('id')} - {memory.get('content', '')[:50]}...")
            except Exception as e:
                print(f"Failed to create memory: {e}")
        
        # Search for memories
        print("\nSearching for Python-related memories...")
        search_result = await client.search_memories("Python programming language", limit=5)
        
        if search_result.get("isError"):
            print(f"Search error: {search_result}")
        else:
            results = search_result.get("data", {}).get("results", [])
            print(f"Found {len(results)} memories:")
            for memory in results:
                print(f"  - {memory.get('content')[:100]}... (score: {memory.get('similarity_score', 0):.3f})")
        
        # List memories with filtering
        print("\nListing task memories...")
        list_result = await client.list_memories(memory_type="task", limit=10)
        
        if list_result.get("isError"):
            print(f"List error: {list_result}")
        else:
            memories = list_result.get("data", {}).get("memories", [])
            print(f"Found {len(memories)} task memories:")
            for memory in memories:
                print(f"  - {memory.get('content')[:100]}...")
        
        # Update a memory
        if created_memories:
            memory_to_update = created_memories[0]
            memory_id = memory_to_update.get('id')
            if memory_id:
                print(f"\nUpdating memory {memory_id}...")
                
                try:
                    update_result = await client.update_memory(
                        memory_id,
                        content=f"{memory_to_update.get('content', '')} [UPDATED]",
                        tags=memory_to_update.get('tags', []) + ["updated"]
                    )
                    
                    if update_result.get("isError"):
                        print(f"Update error: {update_result}")
                    else:
                        updated_memory = update_result.get("data", {}).get("memory", {})
                        print(f"Updated memory: {updated_memory.get('content', '')[:100]}...")
                except Exception as e:
                    print(f"Failed to update memory: {e}")
        
        # Demonstrate error handling
        print("\nTesting error handling...")
        try:
            await client.get_memory("non-existent-id")
            print("WARNING: Expected error but got successful response")
        except Exception as e:
            print(f"Expected error for non-existent memory: {e}")
        
        print("\nDemo completed successfully!")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.stop_server()


if __name__ == "__main__":
    print("Claude Code MCP Client Demo")
    print("=" * 40)
    asyncio.run(demo_claude_code_integration())