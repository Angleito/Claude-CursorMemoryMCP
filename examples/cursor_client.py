#!/usr/bin/env python3
"""
Cursor IDE MCP Client Example
Demonstrates how to integrate Mem0 AI with Cursor IDE via MCP
"""

import json
import asyncio
import aiohttp
import websockets
import logging
from typing import Dict, Any, Optional, List, Union, Callable
from urllib.parse import urlparse


class CursorMCPClient:
    """Client for integrating Mem0 AI with Cursor IDE"""
    
    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        # Validate URL format
        parsed_url = urlparse(base_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError(f"Invalid base URL: {base_url}")
            
        self.base_url = base_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
        self.access_token: Optional[str] = None
        self.ws_connection: Optional[websockets.WebSocketServerProtocol] = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def start_session(self):
        """Start HTTP session"""
        self.session = aiohttp.ClientSession()
    
    async def close_session(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
        if self.ws_connection:
            await self.ws_connection.close()
    
    async def authenticate(self, username: str, password: str):
        """Authenticate with the server"""
        if not self.session:
            await self.start_session()
        
        auth_data = {
            "username": username,
            "password": password
        }
        
        async with self.session.post(
            f"{self.base_url}/auth/token",
            json=auth_data
        ) as response:
            if response.status == 200:
                data = await response.json()
                self.access_token = data["access_token"]
                print(f"Authenticated successfully")
                return True
            else:
                error = await response.text()
                print(f"Authentication failed: {error}")
                return False
    
    def _get_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        if not self.access_token:
            raise ValueError("Not authenticated")
        return {"Authorization": f"Bearer {self.access_token}"}
    
    async def create_memory(self, content: str, **kwargs) -> Dict[str, Any]:
        """Create a new memory"""
        memory_data = {"content": content}
        memory_data.update(kwargs)
        
        async with self.session.post(
            f"{self.base_url}/memories",
            json=memory_data,
            headers=self._get_headers()
        ) as response:
            return await response.json()
    
    async def search_memories(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search memories"""
        search_data = {"query": query}
        search_data.update(kwargs)
        
        async with self.session.post(
            f"{self.base_url}/memories/search",
            json=search_data,
            headers=self._get_headers()
        ) as response:
            data = await response.json()
            return data.get("results", [])
    
    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific memory"""
        async with self.session.get(
            f"{self.base_url}/memories/{memory_id}",
            headers=self._get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            return None
    
    async def update_memory(self, memory_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Update a memory"""
        async with self.session.put(
            f"{self.base_url}/memories/{memory_id}",
            json=kwargs,
            headers=self._get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            return None
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory"""
        async with self.session.delete(
            f"{self.base_url}/memories/{memory_id}",
            headers=self._get_headers()
        ) as response:
            return response.status == 200
    
    async def connect_websocket(self):
        """Connect to WebSocket for real-time updates"""
        ws_url = self.base_url.replace("http", "ws") + "/ws"
        headers = self._get_headers()
        
        self.ws_connection = await websockets.connect(
            ws_url,
            extra_headers=headers
        )
        self.logger.info("WebSocket connected")
    
    async def listen_for_updates(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        """Listen for real-time memory updates"""
        if not self.ws_connection or self.ws_connection.closed:
            await self.connect_websocket()
        
        try:
            async for message in self.ws_connection:
                try:
                    data = json.loads(message)
                    if callback:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(data)
                        else:
                            callback(data)
                    else:
                        self.logger.info(f"Received update: {data}")
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON message: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
        except websockets.exceptions.ConnectionClosed:
            self.logger.info("WebSocket connection closed")
        except Exception as e:
            self.logger.error(f"Error listening for messages: {e}")
            raise
    
    async def use_sse_stream(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        """Use Server-Sent Events for real-time updates"""
        headers = self._get_headers()
        headers["Accept"] = "text/event-stream"
        
        try:
            async with self.session.get(
                f"{self.base_url}/mcp/sse",
                headers=headers
            ) as response:
                if response.status != 200:
                    response.raise_for_status()
                    
                async for line in response.content:
                    try:
                        line_str = line.decode().strip()
                        if line_str.startswith("data: "):
                            data = json.loads(line_str[6:])
                            if callback:
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(data)
                                else:
                                    callback(data)
                            else:
                                self.logger.info(f"SSE Event: {data}")
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Invalid SSE data: {e}")
                    except Exception as e:
                        self.logger.error(f"Error processing SSE event: {e}")
        except aiohttp.ClientError as e:
            raise ConnectionError(f"SSE connection error: {e}")


class CursorIntegration:
    """Integration layer for Cursor IDE specific features"""
    
    def __init__(self, client: CursorMCPClient):
        self.client = client
        self.context_memories = []
        self.auto_save_enabled = True
    
    async def analyze_code_context(self, code: str, filename: str = None) -> List[Dict[str, Any]]:
        """Analyze code and find relevant memories"""
        # Extract keywords and concepts from code
        keywords = self._extract_code_keywords(code)
        
        # Search for relevant memories
        relevant_memories = []
        for keyword in keywords:
            memories = await self.client.search_memories(
                query=keyword,
                limit=3,
                memory_types=["fact", "skill", "preference"]
            )
            relevant_memories.extend(memories)
        
        # Remove duplicates and sort by relevance
        seen_ids = set()
        unique_memories = []
        for memory in relevant_memories:
            if memory["id"] not in seen_ids:
                unique_memories.append(memory)
                seen_ids.add(memory["id"])
        
        return sorted(unique_memories, key=lambda m: m.get("similarity_score", 0), reverse=True)[:5]
    
    def _extract_code_keywords(self, code: str) -> List[str]:
        """Extract keywords from code for memory search"""
        # Simple keyword extraction (can be enhanced with AST parsing)
        keywords = []
        
        # Common programming patterns
        patterns = [
            "def ", "class ", "import ", "from ", "async ", "await ",
            "function", "const", "let", "var", "interface", "type"
        ]
        
        for pattern in patterns:
            if pattern in code:
                keywords.append(pattern.strip())
        
        # Extract function/class names
        lines = code.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('def ') or line.startswith('class '):
                parts = line.split()
                if len(parts) > 1:
                    name = parts[1].split('(')[0].split(':')[0]
                    keywords.append(name)
        
        return keywords
    
    async def save_code_snippet(self, code: str, description: str, tags: List[str] = None):
        """Save a code snippet as a memory"""
        memory_data = {
            "content": f"Code snippet: {description}\n\n```\n{code}\n```",
            "tags": (tags or []) + ["code", "snippet"],
            "memory_type": "skill",
            "priority": "medium",
            "source": "cursor_ide"
        }
        
        return await self.client.create_memory(**memory_data)
    
    async def auto_save_context(self, file_path: str, code_context: str):
        """Automatically save coding context"""
        if not self.auto_save_enabled:
            return
        
        # Check if similar context already exists
        existing = await self.client.search_memories(
            query=f"file:{file_path}",
            limit=1,
            memory_types=["context"]
        )
        
        if existing and existing[0].get("similarity_score", 0) > 0.9:
            # Update existing context
            await self.client.update_memory(
                existing[0]["id"],
                content=f"Working on {file_path}:\n{code_context}",
                metadata={"file_path": file_path, "last_updated": "now"}
            )
        else:
            # Create new context memory
            await self.client.create_memory(
                content=f"Working on {file_path}:\n{code_context}",
                tags=["context", "file", file_path.split('/')[-1]],
                memory_type="context",
                priority="low",
                source="cursor_auto_save",
                metadata={"file_path": file_path}
            )
    
    async def get_coding_suggestions(self, current_code: str, cursor_position: int = None) -> List[str]:
        """Get coding suggestions based on memory"""
        # Analyze current code context
        relevant_memories = await self.analyze_code_context(current_code)
        
        suggestions = []
        for memory in relevant_memories:
            content = memory.get("content", "")
            if "Code snippet:" in content:
                # Extract code from memory
                if "```" in content:
                    code_part = content.split("```")[1] if len(content.split("```")) > 1 else ""
                    suggestions.append(f"Consider this pattern: {code_part[:100]}...")
            else:
                suggestions.append(f"Remember: {content[:100]}...")
        
        return suggestions[:3]  # Return top 3 suggestions


async def demo_cursor_integration() -> None:
    """Demonstrate Cursor IDE integration"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    client = CursorMCPClient()
    
    try:
        await client.start_session()
        
        # Note: In real usage, user would authenticate
        print("Note: Authentication required in production")
        print("Simulating authenticated session...")
        
        # Create cursor integration
        cursor = CursorIntegration(client)
        
        # Simulate code analysis
        sample_code = '''
def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts"""
    # Generate embeddings
    embedding1 = generate_embedding(text1)
    embedding2 = generate_embedding(text2)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity
'''
        
        print("Analyzing code context...")
        keywords = cursor._extract_code_keywords(sample_code)
        print(f"Extracted keywords: {keywords}")
        
        # Save code snippet
        print("\nSaving code snippet...")
        # Note: This would require authentication in real usage
        # await cursor.save_code_snippet(
        #     sample_code,
        #     "Function to calculate semantic similarity",
        #     ["similarity", "embedding", "nlp"]
        # )
        
        # Simulate getting suggestions
        print("\nGetting coding suggestions...")
        suggestions = await cursor.get_coding_suggestions(sample_code)
        print(f"Suggestions: {suggestions}")
        
        print("\nCursor integration demo completed!")
        
    except Exception as e:
        print(f"Demo error: {e}")
    finally:
        await client.close_session()


if __name__ == "__main__":
    print("Cursor IDE MCP Integration Demo")
    print("=" * 40)
    asyncio.run(demo_cursor_integration())