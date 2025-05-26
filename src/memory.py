"""Memory management system with vector search capabilities.

This module provides comprehensive memory management including vector embeddings,
semantic search, user authentication, and automatic cleanup. It integrates with
OpenAI for embeddings and Redis for caching.
"""

from __future__ import annotations

# Standard library imports
import asyncio
import contextlib
import hashlib
import json
import time
import uuid
from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING
from typing import Any

# Third-party imports
import numpy as np
import redis.asyncio as redis
import structlog
from openai import AsyncOpenAI
from passlib.context import CryptContext
from pydantic import ValidationError

# Constants
MAX_TEXT_LENGTH = 8000
DEFAULT_BATCH_SIZE = 1000
MIN_STRING_LENGTH = 3
MAX_TOKEN_LENGTH = 8

if TYPE_CHECKING:
    from .config import Settings

from .database import DatabaseManager
from .models import MemoryCreate
from .models import MemoryPriority
from .models import MemoryResponse
from .models import MemorySearch
from .models import MemoryType
from .models import MemoryUpdate
from .models import UserCreate
from .models import UserResponse

logger = structlog.get_logger()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class EmbeddingGenerator:
    """Generates embeddings using OpenAI API with proper validation and caching.

    This class handles embedding generation with caching, validation, and error handling.
    It integrates with OpenAI's embedding models and provides performance monitoring.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _validate_embedding(self, embedding: list[float]) -> bool:
        """Validate embedding dimensions and values."""
        if not isinstance(embedding, list):
            return False
        if len(embedding) != self.settings.vector_dimension:
            return False
        if not all(isinstance(x, int | float) for x in embedding):
            return False
        return all(np.isfinite(x) for x in embedding)

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for given text with validation and caching.

        Args:
            text: Input text to generate embedding for

        Returns:
            List of floating point numbers representing the embedding

        Raises:
            ValueError: If text is empty or embedding generation fails
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        if len(text) > MAX_TEXT_LENGTH:  # OpenAI's typical limit
            text = text[:8000]
            logger.warning("Text truncated for embedding", original_length=len(text))

        # Check cache first
        cache_key = self._get_cache_key(text)
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]

        try:
            response = await self.client.embeddings.create(
                model=self.model, input=text.strip()
            )

            embedding = response.data[0].embedding

            # Validate embedding
            if not self._validate_embedding(embedding):
                raise ValueError(
                    f"Invalid embedding: expected {self.settings.vector_dimension} dimensions"
                )

            # Cache result (limit cache size)
            if len(self._cache) < DEFAULT_BATCH_SIZE:
                self._cache[cache_key] = embedding

            self._cache_misses += 1
            return embedding

        except Exception as e:
            if "rate_limit_exceeded" in str(e).lower():
                logger.error("OpenAI rate limit exceeded", error=str(e))
                raise ValueError("Rate limit exceeded, please try again later") from e
            elif "invalid_request" in str(e).lower():
                logger.error(
                    "Invalid OpenAI request", error=str(e), text_length=len(text)
                )
                raise ValueError(f"Invalid request: {e!s}") from e
            else:
                logger.error(
                    "Failed to generate embedding", error=str(e), text_length=len(text)
                )
                raise RuntimeError(f"Failed to generate embedding: {e!s}") from e

    def get_cache_stats(self) -> dict[str, Any]:
        """Get embedding cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self._cache),
        }


class MemoryManager:
    """Manages memory storage, retrieval, and search operations with proper validation.

    This class provides comprehensive memory management including:
    - Memory creation with vector embeddings
    - Semantic search using vector similarity
    - User authentication and authorization
    - Automatic cleanup of expired memories
    - Redis caching for performance optimization
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.db = DatabaseManager(settings)
        self.embedding_generator = EmbeddingGenerator(settings)
        self.redis_client: redis.Redis | None = None
        self._cleanup_task: asyncio.Task | None = None
        self._stats: dict[str, int] = {
            "memories_created": 0,
            "memories_searched": 0,
            "memories_updated": 0,
            "memories_deleted": 0,
            "search_cache_hits": 0,
            "errors": 0,
        }

    async def initialize(self) -> None:
        """Initialize the memory manager."""
        try:
            await self.db.initialize()

            # Initialize Redis for caching
            self.redis_client = redis.from_url(
                self.settings.redis_url, decode_responses=True
            )
            await self.redis_client.ping()

            # Start background tasks
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_memories())

            logger.info("Memory manager initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize memory manager", error=str(e))
            raise

    def _validate_user_id(self, user_id: str) -> bool:
        """Validate user ID format."""
        try:
            uuid.UUID(user_id)
            return True
        except (ValueError, TypeError):
            return False

    def _validate_memory_id(self, memory_id: str) -> bool:
        """Validate memory ID format."""
        try:
            uuid.UUID(memory_id)
            return True
        except (ValueError, TypeError):
            return False

    def _validate_vector(self, vector: list[float]) -> bool:
        """Validate vector format and dimensions."""
        if not isinstance(vector, list):
            return False
        if len(vector) != self.settings.vector_dimension:
            return False
        return all(isinstance(x, int | float) and np.isfinite(x) for x in vector)

    async def create_user(self, user_data: UserCreate) -> UserResponse:
        """Create a new user with proper validation.

        Args:
            user_data: User creation data including username, email, password

        Returns:
            Created user information

        Raises:
            ValueError: If validation fails or user already exists
        """
        try:
            # Validate input data
            if not user_data.username or len(user_data.username) < MIN_STRING_LENGTH:
                raise ValueError("Username must be at least 3 characters")
            if not user_data.email or "@" not in user_data.email:
                raise ValueError("Valid email is required")
            if not user_data.password or len(user_data.password) < MAX_TOKEN_LENGTH:
                raise ValueError("Password must be at least 8 characters")

            password_hash = pwd_context.hash(user_data.password)

            # Use parameterized query to prevent SQL injection
            query = """
                INSERT INTO users (username, email, password_hash, full_name)
                VALUES ($1, $2, $3, $4)
                RETURNING id, username, email, full_name, is_active, created_at
            """

            result = await self.db.execute_one(
                query,
                user_data.username.strip().lower(),
                user_data.email.strip().lower(),
                password_hash,
                user_data.full_name.strip() if user_data.full_name else None,
            )

            if not result:
                raise ValueError("Failed to create user")

            logger.info("User created successfully", username=user_data.username)
            return UserResponse(**dict(result))

        except ValidationError as e:
            logger.error("User validation failed", error=str(e))
            raise ValueError(f"Validation error: {e!s}") from e
        except Exception as e:
            self._stats["errors"] += 1
            if "duplicate key" in str(e).lower():
                raise ValueError("Username or email already exists") from e
            logger.error("Failed to create user", error=str(e))
            raise RuntimeError(f"Failed to create user: {e!s}") from e

    async def authenticate_user(
        self, username: str, password: str
    ) -> UserResponse | None:
        """Authenticate user credentials with proper validation.

        Args:
            username: User's username
            password: User's password

        Returns:
            User information if authentication succeeds, None otherwise
        """
        try:
            if not username or not password:
                logger.warning("Authentication attempted with empty credentials")
                return None

            # Normalize username for lookup
            username = username.strip().lower()

            # Use parameterized query to prevent SQL injection
            query = """
                SELECT id, username, email, password_hash, full_name, is_active, created_at
                FROM users
                WHERE username = $1 AND is_active = true
            """

            result = await self.db.execute_one(query, username)
            if not result:
                logger.info("User not found or inactive", username=username)
                return None

            # Verify password with timing attack protection
            if not pwd_context.verify(password, result["password_hash"]):
                logger.info("Invalid password attempt", username=username)
                return None

            user_data = dict(result)
            del user_data["password_hash"]

            logger.info("User authenticated successfully", username=username)
            return UserResponse(**user_data)

        except Exception as e:
            logger.error("Authentication error", error=str(e))
            return None

    async def create_memory(
        self, memory_data: MemoryCreate, user_id: str
    ) -> MemoryResponse:
        """Create a new memory with embedding and proper validation.

        Args:
            memory_data: Memory creation data including content, tags, metadata
            user_id: ID of the user creating the memory

        Returns:
            Created memory with generated embedding

        Raises:
            ValueError: If validation fails or memory creation fails
        """
        try:
            # Validate inputs
            if not self._validate_user_id(user_id):
                raise ValueError("Invalid user ID format")

            if not memory_data.content or not memory_data.content.strip():
                raise ValueError("Memory content cannot be empty")

            if len(memory_data.content) > self.settings.max_memory_size:
                raise ValueError(
                    f"Memory content exceeds maximum size of {self.settings.max_memory_size}"
                )

            # Validate memory type and priority
            if memory_data.memory_type not in MemoryType:
                raise ValueError(f"Invalid memory type: {memory_data.memory_type}")

            if memory_data.priority not in MemoryPriority:
                raise ValueError(f"Invalid priority: {memory_data.priority}")

            # Generate embedding for the content
            content = memory_data.content.strip()
            embedding = await self.embedding_generator.generate_embedding(content)

            if not self._validate_vector(embedding):
                raise ValueError("Generated embedding is invalid")

            # Use transaction for atomicity
            async with self.db.get_transaction() as conn:
                # Insert memory into database with parameterized query
                query = """
                    INSERT INTO memories (
                        user_id, content, embedding, metadata, tags, memory_type,
                        priority, source, context, expires_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    RETURNING id, user_id, content, metadata, tags, memory_type,
                             priority, source, context, access_count, created_at,
                             updated_at, expires_at
                """

                result = await conn.fetchrow(
                    query,
                    user_id,
                    content,
                    embedding,
                    json.dumps(memory_data.metadata or {}),
                    memory_data.tags or [],
                    memory_data.memory_type.value,
                    memory_data.priority.value,
                    memory_data.source,
                    memory_data.context,
                    memory_data.expires_at,
                )

                if not result:
                    raise ValueError("Failed to create memory")

            # Cache the memory if Redis is available
            if self.redis_client:
                try:
                    cache_key = f"memory:{result['id']}"
                    await self.redis_client.setex(
                        cache_key,
                        3600,  # 1 hour cache
                        json.dumps(dict(result), default=str),
                    )
                except Exception as e:
                    logger.warning("Failed to cache memory", error=str(e))

            memory_response = MemoryResponse(**dict(result))
            memory_response.embedding = embedding

            self._stats["memories_created"] += 1
            logger.info("Memory created", memory_id=result["id"], user_id=user_id)
            return memory_response

        except ValidationError as e:
            logger.error("Memory validation failed", error=str(e))
            raise ValueError(f"Validation error: {e!s}") from e
        except Exception as e:
            self._stats["errors"] += 1
            logger.error("Failed to create memory", error=str(e), user_id=user_id)
            raise RuntimeError(f"Failed to create memory: {e!s}") from e

    async def get_memory(
        self, memory_id: str, user_id: str
    ) -> MemoryResponse | None:
        """Get a specific memory by ID with proper validation.

        Args:
            memory_id: Unique identifier of the memory
            user_id: ID of the user requesting the memory

        Returns:
            Memory information if found and authorized, None otherwise

        Raises:
            ValueError: If validation fails
        """
        try:
            # Validate inputs
            if not self._validate_memory_id(memory_id):
                raise ValueError("Invalid memory ID format")

            if not self._validate_user_id(user_id):
                raise ValueError("Invalid user ID format")

            # Try cache first
            cache_key = f"memory:{memory_id}"
            if self.redis_client:
                try:
                    cached = await self.redis_client.get(cache_key)
                    if cached:
                        data = json.loads(cached)
                        return MemoryResponse(**data)
                except Exception as e:
                    logger.warning("Cache retrieval failed", error=str(e))

            # Fetch from database with parameterized query
            query = """
                SELECT id, user_id, content, embedding, metadata, tags, memory_type,
                       priority, source, context, access_count, created_at,
                       updated_at, expires_at
                FROM memories
                WHERE id = $1 AND user_id = $2 AND (expires_at IS NULL OR expires_at > NOW())
            """

            result = await self.db.execute_one(query, memory_id, user_id)
            if not result:
                return None

            # Update access count in background
            try:
                await self.db.execute_command(
                    "UPDATE memories SET access_count = access_count + 1 WHERE id = $1",
                    memory_id,
                )
            except Exception as e:
                logger.warning("Failed to update access count", error=str(e))

            # Parse result and handle metadata
            memory_data = dict(result)
            if isinstance(memory_data.get("metadata"), str):
                memory_data["metadata"] = json.loads(memory_data["metadata"])

            # Cache the result
            if self.redis_client:
                try:
                    await self.redis_client.setex(
                        cache_key, 3600, json.dumps(memory_data, default=str)
                    )
                except Exception as e:
                    logger.warning("Failed to cache memory", error=str(e))

            return MemoryResponse(**memory_data)

        except ValidationError as e:
            logger.error("Memory validation failed", error=str(e))
            raise ValueError(f"Validation error: {e!s}") from e
        except Exception as e:
            self._stats["errors"] += 1
            logger.error("Failed to get memory", error=str(e), memory_id=memory_id)
            raise RuntimeError(f"Failed to get memory: {e!s}") from e

    async def _validate_search_inputs(
        self, search_data: MemorySearch, user_id: str
    ) -> None:
        """Validate search inputs."""
        if not self._validate_user_id(user_id):
            raise ValueError("Invalid user ID format")

        if not search_data.query or not search_data.query.strip():
            raise ValueError("Search query cannot be empty")

        if search_data.limit <= 0 or search_data.limit > DEFAULT_BATCH_SIZE:
            raise ValueError("Limit must be between 1 and 1000")

    async def _check_search_cache(
        self, user_id: str, search_data: MemorySearch
    ) -> tuple[str | None, list[MemoryResponse] | None]:
        """Check cache for search results."""
        if not self.redis_client:
            return None, None

        query_hash = hashlib.sha256(
            f"{user_id}:{search_data.query}:{search_data.limit}".encode(),
            usedforsecurity=False
        ).hexdigest()
        cache_key = f"search:{query_hash}"

        try:
            cached_result = await self.redis_client.get(cache_key)
            if cached_result:
                self._stats["search_cache_hits"] += 1
                cached_data = json.loads(cached_result)
                return cache_key, [MemoryResponse(**item) for item in cached_data]
        except Exception as e:
            logger.warning("Cache retrieval failed", error=str(e))

        return cache_key, None

    def _build_search_query(
        self, search_data: MemorySearch, user_id: str, query_embedding: list[float]
    ) -> tuple[str, list[Any]]:
        """Build optimized parameterized search query with filters."""
        # Pre-compute similarity threshold for optimization
        threshold = search_data.threshold or self.settings.similarity_threshold
        if threshold < 0 or threshold > 1:
            raise ValueError("Threshold must be between 0 and 1")

        # Use CTE for better query planning with large datasets
        base_query = """
            WITH filtered_memories AS (
                SELECT id, user_id, content, metadata, tags, memory_type,
                       priority, source, context, access_count, created_at,
                       updated_at, expires_at, embedding
                FROM memories
                WHERE user_id = $1 
                  AND embedding IS NOT NULL
                  AND (expires_at IS NULL OR expires_at > NOW())
        """

        params = [user_id]
        param_count = 1

        # Add selective filters in CTE for better performance
        if search_data.memory_types:
            param_count += 1
            type_values = [mt.value for mt in search_data.memory_types]
            base_query += f" AND memory_type = ANY(${param_count})"
            params.append(type_values)

        if search_data.tags:
            param_count += 1
            base_query += f" AND tags && ${param_count}"
            params.append(search_data.tags)

        if search_data.date_from:
            param_count += 1
            base_query += f" AND created_at >= ${param_count}"
            params.append(search_data.date_from)

        if search_data.date_to:
            param_count += 1
            base_query += f" AND created_at <= ${param_count}"
            params.append(search_data.date_to)

        # Close CTE and add similarity calculation
        param_count += 1
        base_query += f"""
            )
            SELECT fm.id, fm.user_id, fm.content, fm.metadata, fm.tags, fm.memory_type,
                   fm.priority, fm.source, fm.context, fm.access_count, fm.created_at,
                   fm.updated_at, fm.expires_at,
                   1 - (fm.embedding <=> ${param_count}) as similarity_score
            FROM filtered_memories fm
            WHERE 1 - (fm.embedding <=> ${param_count}) >= ${param_count + 1}
        """
        params.append(query_embedding)
        
        param_count += 1
        params.append(threshold)

        # Optimized ordering and limit
        param_count += 1
        base_query += f"""
            ORDER BY fm.embedding <=> ${param_count - 1}
            LIMIT ${param_count}
        """
        params.append(min(search_data.limit, 1000))

        return base_query, params

    async def _process_search_result(
        self, row: Any, search_data: MemorySearch
    ) -> MemoryResponse | None:
        """Process a single search result row."""
        try:
            memory_data = dict(row)
            memory_data["similarity_score"] = float(memory_data["similarity_score"])

            # Parse metadata if it's a string
            if isinstance(memory_data.get("metadata"), str):
                memory_data["metadata"] = json.loads(memory_data["metadata"])

            if search_data.include_embeddings:
                # Fetch embedding separately if needed
                embed_query = "SELECT embedding FROM memories WHERE id = $1"
                embed_result = await self.db.execute_one(embed_query, row["id"])
                if embed_result and embed_result["embedding"]:
                    memory_data["embedding"] = list(embed_result["embedding"])

            return MemoryResponse(**memory_data)

        except Exception as e:
            logger.warning(
                "Failed to parse memory result",
                error=str(e),
                memory_id=row.get("id"),
            )
            return None

    async def search_memories(
        self, search_data: MemorySearch, user_id: str
    ) -> list[MemoryResponse]:
        """Search memories using semantic similarity with proper validation.

        Args:
            search_data: Search parameters including query, filters, and limits
            user_id: ID of the user performing the search

        Returns:
            List of matching memories with similarity scores

        Raises:
            ValueError: If validation fails or search parameters are invalid
        """
        try:
            start_time = time.time()

            # Validate inputs
            await self._validate_search_inputs(search_data, user_id)

            # Check cache
            cache_key, cached_results = await self._check_search_cache(
                user_id, search_data
            )
            if cached_results:
                return cached_results

            # Generate query embedding
            query_embedding = await self.embedding_generator.generate_embedding(
                search_data.query.strip()
            )

            if not self._validate_vector(query_embedding):
                raise ValueError("Generated query embedding is invalid")

            # Build and execute query
            query, params = self._build_search_query(
                search_data, user_id, query_embedding
            )
            results = await self.db.execute_query(query, *params)

            # Process results
            memories = []
            for row in results:
                memory = await self._process_search_result(row, search_data)
                if memory:
                    memories.append(memory)

            # Cache results if enabled
            if cache_key and self.redis_client and memories:
                try:
                    cache_data = [memory.dict() for memory in memories]
                    await self.redis_client.setex(
                        cache_key,
                        300,  # 5 minute cache for search results
                        json.dumps(cache_data, default=str),
                    )
                except Exception as e:
                    logger.warning("Failed to cache search results", error=str(e))

            # Log search metrics
            execution_time = (time.time() - start_time) * 1000
            await self._log_search(
                user_id, search_data.query, len(memories), execution_time
            )

            self._stats["memories_searched"] += 1
            logger.info(
                "Memory search completed",
                user_id=user_id,
                query=search_data.query[:50],
                results_count=len(memories),
                execution_time_ms=execution_time,
            )

            return memories

        except ValidationError as e:
            logger.error("Search validation failed", error=str(e))
            raise ValueError(f"Validation error: {e!s}") from e
        except Exception as e:
            self._stats["errors"] += 1
            logger.error("Failed to search memories", error=str(e), user_id=user_id)
            raise RuntimeError(f"Failed to search memories: {e!s}") from e

    async def _validate_update_inputs(
        self, memory_id: str, user_id: str
    ) -> None:
        """Validate update inputs."""
        if not self._validate_memory_id(memory_id):
            raise ValueError("Invalid memory ID format")

        if not self._validate_user_id(user_id):
            raise ValueError("Invalid user ID format")

    async def _build_content_update(
        self,
        content: str,
        update_fields: list[str],
        params: list[Any],
        param_count: int
    ) -> int:
        """Build content update fields and generate embedding."""
        content = content.strip()
        if not content:
            raise ValueError("Memory content cannot be empty")
        if len(content) > self.settings.max_memory_size:
            raise ValueError(
                f"Memory content exceeds maximum size of {self.settings.max_memory_size}"
            )

        param_count += 1
        update_fields.append(f"content = ${param_count}")
        params.append(content)

        # Regenerate embedding if content changed
        embedding = await self.embedding_generator.generate_embedding(content)
        if not self._validate_vector(embedding):
            raise ValueError("Generated embedding is invalid")

        param_count += 1
        update_fields.append(f"embedding = ${param_count}")
        params.append(embedding)

        return param_count

    def _build_update_fields(
        self,
        update_data: MemoryUpdate,
        update_fields: list[str],
        params: list[Any],
        param_count: int
    ) -> int:
        """Build update fields from update data."""
        if update_data.metadata is not None:
            param_count += 1
            update_fields.append(f"metadata = ${param_count}")
            params.append(json.dumps(update_data.metadata))

        if update_data.tags is not None:
            param_count += 1
            update_fields.append(f"tags = ${param_count}")
            params.append(update_data.tags)

        if update_data.memory_type is not None:
            if update_data.memory_type not in MemoryType:
                raise ValueError(f"Invalid memory type: {update_data.memory_type}")
            param_count += 1
            update_fields.append(f"memory_type = ${param_count}")
            params.append(update_data.memory_type.value)

        if update_data.priority is not None:
            if update_data.priority not in MemoryPriority:
                raise ValueError(f"Invalid priority: {update_data.priority}")
            param_count += 1
            update_fields.append(f"priority = ${param_count}")
            params.append(update_data.priority.value)

        if update_data.context is not None:
            param_count += 1
            update_fields.append(f"context = ${param_count}")
            params.append(update_data.context)

        return param_count

    async def update_memory(
        self, memory_id: str, update_data: MemoryUpdate, user_id: str
    ) -> MemoryResponse | None:
        """Update an existing memory with proper validation.

        Args:
            memory_id: Unique identifier of the memory to update
            update_data: Updated memory data
            user_id: ID of the user updating the memory

        Returns:
            Updated memory information if successful, None if not found

        Raises:
            ValueError: If validation fails or update fails
        """
        try:
            # Validate inputs
            await self._validate_update_inputs(memory_id, user_id)

            # Build update query dynamically
            update_fields = []
            params = []
            param_count = 0

            # Handle content update
            if update_data.content is not None:
                param_count = await self._build_content_update(
                    update_data.content, update_fields, params, param_count
                )

            # Handle other fields
            param_count = self._build_update_fields(
                update_data, update_fields, params, param_count
            )

            if not update_fields:
                return await self.get_memory(memory_id, user_id)

            # Execute update
            result = await self._execute_update(
                memory_id, user_id, update_fields, params, param_count
            )

            if not result:
                return None

            # Post-update processing
            await self._post_update_processing(memory_id, user_id)

            return MemoryResponse(**dict(result))

        except ValidationError as e:
            logger.error("Memory update validation failed", error=str(e))
            raise ValueError(f"Validation error: {e!s}") from e
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(
                "Failed to update memory",
                error=str(e),
                memory_id=memory_id,
                user_id=user_id
            )
            raise RuntimeError(f"Failed to update memory: {e!s}") from e

    async def _execute_update(
        self,
        memory_id: str,
        user_id: str,
        update_fields: list[str],
        params: list[Any],
        param_count: int
    ) -> Any:
        """Execute memory update query."""
        # Add WHERE clause parameters
        param_count += 1
        params.append(memory_id)
        param_count += 1
        params.append(user_id)

        # Use transaction for consistency
        async with self.db.get_transaction() as conn:
            query = f"""
                UPDATE memories
                SET {', '.join(update_fields)}
                WHERE id = ${param_count - 1} AND user_id = ${param_count}
                RETURNING id, user_id, content, metadata, tags, memory_type,
                         priority, source, context, access_count, created_at,
                         updated_at, expires_at
            """

            return await conn.fetchrow(query, *params)

    async def _post_update_processing(
        self, memory_id: str, user_id: str
    ) -> None:
        """Post-update processing: invalidate cache and log."""
        # Invalidate cache
        if self.redis_client:
            try:
                cache_key = f"memory:{memory_id}"
                await self.redis_client.delete(cache_key)
            except Exception as e:
                logger.warning("Failed to invalidate cache", error=str(e))

        self._stats["memories_updated"] += 1
        logger.info("Memory updated", memory_id=memory_id, user_id=user_id)

    async def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """Delete a memory with proper validation.

        Args:
            memory_id: Unique identifier of the memory to delete
            user_id: ID of the user deleting the memory

        Returns:
            True if deletion was successful, False otherwise

        Raises:
            ValueError: If validation fails
        """
        try:
            # Validate inputs
            if not self._validate_memory_id(memory_id):
                raise ValueError("Invalid memory ID format")

            if not self._validate_user_id(user_id):
                raise ValueError("Invalid user ID format")

            # Use parameterized query to prevent SQL injection
            query = "DELETE FROM memories WHERE id = $1 AND user_id = $2"
            result = await self.db.execute_command(query, memory_id, user_id)

            # Remove from cache if deletion was successful
            success = "DELETE 1" in result
            if success:
                if self.redis_client:
                    try:
                        cache_key = f"memory:{memory_id}"
                        await self.redis_client.delete(cache_key)
                    except Exception as e:
                        logger.warning(
                            "Failed to remove memory from cache", error=str(e)
                        )

                self._stats["memories_deleted"] += 1
                logger.info("Memory deleted", memory_id=memory_id, user_id=user_id)
            else:
                logger.warning(
                    "Memory not found or not owned by user",
                    memory_id=memory_id,
                    user_id=user_id,
                )

            return success

        except Exception as e:
            self._stats["errors"] += 1
            logger.error("Failed to delete memory", error=str(e), memory_id=memory_id)
            raise RuntimeError(f"Failed to delete memory: {e!s}") from e

    async def get_memory_count(self, user_id: str | None = None) -> int:
        """Get total memory count with proper validation.

        Args:
            user_id: Optional user ID to filter by specific user

        Returns:
            Total number of memories (for user if specified, global otherwise)
        """
        try:
            if user_id and not self._validate_user_id(user_id):
                raise ValueError("Invalid user ID format")

            # Use parameterized queries to prevent SQL injection
            if user_id:
                query = "SELECT COUNT(*) as count FROM memories WHERE user_id = $1 AND (expires_at IS NULL OR expires_at > NOW())"
                result = await self.db.execute_one(query, user_id)
            else:
                query = "SELECT COUNT(*) as count FROM memories WHERE (expires_at IS NULL OR expires_at > NOW())"
                result = await self.db.execute_one(query)

            return result["count"] if result else 0

        except Exception as e:
            logger.error("Failed to get memory count", error=str(e))
            return 0

    async def get_stats(self) -> dict[str, Any]:
        """Get memory manager statistics."""
        try:
            db_stats = await self.db.get_stats()
            embedding_stats = self.embedding_generator.get_cache_stats()

            return {
                "memory_stats": self._stats,
                "database_stats": db_stats,
                "embedding_stats": embedding_stats,
            }
        except Exception as e:
            logger.error("Failed to get stats", error=str(e))
            return {"error": str(e)}

    async def _log_search(
        self, user_id: str, query: str, results_count: int, execution_time_ms: float
    ):
        """Log search metrics with proper validation."""
        try:
            if not self._validate_user_id(user_id):
                logger.warning("Invalid user ID for search logging", user_id=user_id)
                return

            # Truncate query for storage
            query_truncated = query[:500] if query else ""

            # Use parameterized query for safety
            search_query = """
                INSERT INTO search_history (user_id, query, results_count, execution_time_ms)
                VALUES ($1, $2, $3, $4)
            """
            await self.db.execute_command(
                search_query, user_id, query_truncated, results_count, execution_time_ms
            )
        except Exception as e:
            logger.error("Failed to log search", error=str(e))

    async def _cleanup_expired_memories(self):
        """Background task to cleanup expired memories safely."""
        while True:
            try:
                if self.settings.auto_cleanup_enabled:
                    # Use parameterized query for safety
                    query = "DELETE FROM memories WHERE expires_at < NOW()"
                    result = await self.db.execute_command(query)

                    if "DELETE" in result and result != "DELETE 0":
                        logger.info("Cleaned up expired memories", result=result)

                    # Also cleanup old search history
                    cleanup_date = datetime.now() - timedelta(days=30)
                    history_query = "DELETE FROM search_history WHERE created_at < $1"
                    await self.db.execute_command(history_query, cleanup_date)

                # Sleep for 1 hour
                await asyncio.sleep(3600)

            except asyncio.CancelledError:
                logger.info("Cleanup task cancelled")
                break
            except Exception as e:
                logger.error("Error in cleanup task", error=str(e))
                await asyncio.sleep(60)  # Retry after 1 minute

    async def close(self) -> None:
        """Close memory manager resources with proper cleanup.

        This method gracefully shuts down the memory manager, cancels background
        tasks, and closes all connections.
        """
        try:
            # Cancel cleanup task
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._cleanup_task

            # Close Redis connection
            if self.redis_client:
                try:
                    await self.redis_client.close()
                except Exception as e:
                    logger.error("Error closing Redis client", error=str(e))

            # Close database connection
            await self.db.close()

            logger.info("Memory manager closed", final_stats=self._stats)

        except Exception as e:
            logger.error("Error closing memory manager", error=str(e))
