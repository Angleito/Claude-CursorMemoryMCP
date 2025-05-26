#!/usr/bin/env python3
"""Advanced Similarity Search Optimization for mem0ai.

Production-grade search optimization with multiple algorithms and caching.
"""

import asyncio
import hashlib
import json
import logging
import os
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import asyncpg
import faiss
import numpy as np
import redis.asyncio as redis

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimilarityMetric(Enum):
    """Supported similarity metrics."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


class SearchAlgorithm(Enum):
    """Supported search algorithms."""

    EXACT = "exact"  # Direct database query
    FAISS_FLAT = "faiss_flat"  # FAISS exact search
    FAISS_IVF = "faiss_ivf"  # FAISS IVF index
    FAISS_HNSW = "faiss_hnsw"  # FAISS HNSW index
    HIERARCHICAL = "hierarchical"  # Custom hierarchical search
    HYBRID = "hybrid"  # Combine multiple approaches


@dataclass
class SearchConfig:
    """Configuration for similarity search."""

    algorithm: SearchAlgorithm = SearchAlgorithm.EXACT
    similarity_metric: SimilarityMetric = SimilarityMetric.COSINE
    top_k: int = 10
    similarity_threshold: float = 0.7
    max_results: int = 100
    use_cache: bool = True
    cache_ttl_seconds: int = 3600
    faiss_nprobe: int = 10  # For IVF
    faiss_ef_search: int = 64  # For HNSW
    enable_reranking: bool = True
    diversity_lambda: float = 0.1  # For diverse results
    temporal_decay_hours: float = 168.0  # 1 week
    importance_weight: float = 0.2


@dataclass
class SearchQuery:
    """Search query parameters."""

    user_id: str
    query_embedding: list[float] | np.ndarray
    query_text: str | None = None
    memory_types: list[str] | None = None
    metadata_filters: dict | None = None
    min_importance: float | None = None
    max_age_hours: float | None = None
    exclude_memory_ids: list[str] | None = None
    config: SearchConfig | None = None


@dataclass
class SearchResult:
    """Single search result."""

    memory_id: str
    memory_text: str
    similarity_score: float
    importance_score: float
    memory_type: str
    metadata: dict
    created_at: datetime
    last_accessed: datetime
    access_count: int
    final_score: float  # Combined score after reranking


@dataclass
class SearchResponse:
    """Complete search response."""

    results: list[SearchResult]
    total_found: int
    search_time_ms: float
    algorithm_used: SearchAlgorithm
    cache_hit: bool
    query_id: str


class SearchCache:
    """Redis-based search result caching."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None

    async def initialize(self):
        """Initialize Redis connection."""
        self.redis_client = redis.from_url(self.redis_url)

    def _get_cache_key(self, query: SearchQuery) -> str:
        """Generate cache key for search query."""
        # Create deterministic hash from query parameters
        key_data = {
            "user_id": query.user_id,
            "embedding": query.query_embedding,
            "memory_types": sorted(query.memory_types or []),
            "metadata_filters": json.dumps(
                query.metadata_filters or {}, sort_keys=True
            ),
            "min_importance": query.min_importance,
            "max_age_hours": query.max_age_hours,
            "exclude_ids": sorted(query.exclude_memory_ids or []),
            "config": asdict(query.config) if query.config else {},
        }

        key_str = json.dumps(key_data, sort_keys=True)
        hash_key = hashlib.sha256(key_str.encode()).hexdigest()
        return f"search_cache:{hash_key}"

    async def get(self, query: SearchQuery) -> SearchResponse | None:
        """Get cached search results."""
        if not self.redis_client or not query.config.use_cache:
            return None

        try:
            key = self._get_cache_key(query)
            cached_data = await self.redis_client.get(key)
            if cached_data:
                data = pickle.loads(cached_data)
                # Convert back to SearchResponse
                results = [SearchResult(**r) for r in data["results"]]
                return SearchResponse(
                    results=results,
                    total_found=data["total_found"],
                    search_time_ms=data["search_time_ms"],
                    algorithm_used=SearchAlgorithm(data["algorithm_used"]),
                    cache_hit=True,
                    query_id=data["query_id"],
                )
        except Exception as e:
            logger.warning("Cache get error: %s", e)
        return None

    async def set(self, query: SearchQuery, response: SearchResponse):
        """Cache search results."""
        if not self.redis_client or not query.config.use_cache:
            return

        try:
            key = self._get_cache_key(query)

            # Convert to serializable format
            data = {
                "results": [asdict(r) for r in response.results],
                "total_found": response.total_found,
                "search_time_ms": response.search_time_ms,
                "algorithm_used": response.algorithm_used.value,
                "query_id": response.query_id,
            }

            # Convert datetime objects to ISO strings
            for result in data["results"]:
                result["created_at"] = result["created_at"].isoformat()
                result["last_accessed"] = result["last_accessed"].isoformat()

            cached_data = pickle.dumps(data)
            await self.redis_client.setex(
                key, query.config.cache_ttl_seconds, cached_data
            )

        except Exception as e:
            logger.warning("Cache set error: %s", e)

    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()


class FAISSIndexManager:
    """Manages FAISS indexes for fast similarity search."""

    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions
        self.indexes = {}
        self.user_mappings = {}  # Maps user_id to index positions
        self.lock = threading.Lock()

    def create_index(
        self, user_id: str, algorithm: SearchAlgorithm, embeddings: np.ndarray
    ) -> None:
        """Create FAISS index for user."""
        with self.lock:
            if algorithm == SearchAlgorithm.FAISS_FLAT:
                index = faiss.IndexFlatIP(self.dimensions)  # Inner product for cosine
            elif algorithm == SearchAlgorithm.FAISS_IVF:
                # Choose number of clusters based on data size
                nlist = min(100, max(10, len(embeddings) // 10))
                quantizer = faiss.IndexFlatIP(self.dimensions)
                index = faiss.IndexIVFFlat(quantizer, self.dimensions, nlist)
                # Ensure embeddings are contiguous and float32
                training_embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
                index.train(training_embeddings)
            elif algorithm == SearchAlgorithm.FAISS_HNSW:
                index = faiss.IndexHNSWFlat(self.dimensions, 32)
                index.hnsw.efConstruction = 200
            else:
                raise ValueError(f"Unsupported FAISS algorithm: {algorithm}")

            # Normalize embeddings for cosine similarity
            normalized_embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
            faiss.normalize_L2(normalized_embeddings)

            # Add vectors to index
            index.add(normalized_embeddings)

            # Store index and mapping
            key = f"{user_id}_{algorithm.value}"
            self.indexes[key] = index
            self.user_mappings[key] = list(range(len(embeddings)))

            logger.info(
                "Created FAISS %s index for user %s with %d vectors",
                algorithm.value, user_id, len(embeddings)
            )

    def search(
        self,
        user_id: str,
        algorithm: SearchAlgorithm,
        query_embedding: np.ndarray | list[float],
        k: int,
        nprobe: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search FAISS index."""
        key = f"{user_id}_{algorithm.value}"

        if key not in self.indexes:
            raise ValueError(
                f"No index found for user {user_id} with algorithm {algorithm.value}"
            )

        index = self.indexes[key]

        # Set search parameters
        if algorithm == SearchAlgorithm.FAISS_IVF:
            index.nprobe = nprobe
        elif algorithm == SearchAlgorithm.FAISS_HNSW:
            index.hnsw.efSearch = max(k, 64)

        # Convert and normalize query embedding
        if isinstance(query_embedding, list):
            query_array = np.array(query_embedding, dtype=np.float32)
        else:
            query_array = query_embedding.astype(np.float32)

        query_normalized = np.ascontiguousarray(query_array.reshape(1, -1))
        faiss.normalize_L2(query_normalized)

        # Search
        similarities, indices = index.search(query_normalized, k)

        return similarities[0], indices[0]

    def update_index(
        self, user_id: str, algorithm: SearchAlgorithm, new_embeddings: np.ndarray
    ) -> None:
        """Update existing index with new embeddings."""
        key = f"{user_id}_{algorithm.value}"

        if key in self.indexes:
            # For simplicity, recreate the index
            # In production, you might want incremental updates
            self.remove_index(user_id, algorithm)

        # Get existing embeddings and combine with new ones
        # This is a simplified approach - in production you'd want to be more efficient
        self.create_index(user_id, algorithm, new_embeddings)

    def remove_index(self, user_id: str, algorithm: SearchAlgorithm) -> None:
        """Remove index for user."""
        key = f"{user_id}_{algorithm.value}"
        with self.lock:
            if key in self.indexes:
                del self.indexes[key]
                del self.user_mappings[key]


class SimilaritySearchOptimizer:
    """Main similarity search optimization engine."""

    def __init__(self, db_url: str, redis_url: str = "redis://localhost:6379"):
        self.db_url = db_url
        self.cache = SearchCache(redis_url)
        self.faiss_manager = FAISSIndexManager()
        self.db_pool = None

    async def initialize(self):
        """Initialize the search optimizer."""
        await self.cache.initialize()

        self.db_pool = await asyncpg.create_pool(
            self.db_url, min_size=5, max_size=20, command_timeout=60
        )

        logger.info("Similarity search optimizer initialized")

    async def cleanup(self):
        """Cleanup resources."""
        await self.cache.close()
        if self.db_pool:
            await self.db_pool.close()

    async def _load_user_embeddings(self, user_id: str) -> tuple[list[str], np.ndarray]:
        """Load all embeddings for a user from database."""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, embedding
                FROM mem0_vectors.memories
                WHERE user_id = $1 AND embedding IS NOT NULL
                ORDER BY created_at DESC
            """,
                user_id,
            )

            if not rows:
                return [], np.array([])

            memory_ids = [row["id"] for row in rows]
            embeddings = np.array([row["embedding"] for row in rows])

            return memory_ids, embeddings

    async def build_user_index(self, user_id: str, algorithm: SearchAlgorithm) -> None:
        """Build FAISS index for user."""
        memory_ids, embeddings = await self._load_user_embeddings(user_id)

        if len(embeddings) == 0:
            logger.warning("No embeddings found for user %s", user_id)
            return

        # Build index in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor,
                self.faiss_manager.create_index,
                user_id,
                algorithm,
                embeddings,
            )

    async def _exact_search(self, query: SearchQuery) -> list[SearchResult]:
        """Perform exact search using database."""
        conditions = ["m.user_id = $1"]
        params = [query.user_id]
        param_count = 1

        # Add filters
        if query.memory_types:
            param_count += 1
            conditions.append(f"m.memory_type = ANY(${param_count})")
            params.append(query.memory_types)

        if query.min_importance is not None:
            param_count += 1
            conditions.append(f"m.importance_score >= ${param_count}")
            params.append(query.min_importance)

        if query.max_age_hours is not None:
            param_count += 1
            conditions.append(
                f"m.created_at >= NOW() - INTERVAL '{query.max_age_hours} hours'"
            )

        if query.exclude_memory_ids:
            param_count += 1
            conditions.append(f"m.id != ALL(${param_count})")
            params.append(query.exclude_memory_ids)

        # Metadata filters
        if query.metadata_filters:
            for key, value in query.metadata_filters.items():
                param_count += 1
                conditions.append(f"m.metadata->>'{key}' = ${param_count}")
                params.append(str(value))

        where_clause = " AND ".join(conditions)

        # Choose similarity operator based on metric
        if query.config.similarity_metric == SimilarityMetric.COSINE:
            similarity_expr = f"1 - (m.embedding <=> ${param_count + 1})"
            order_expr = f"m.embedding <=> ${param_count + 1}"
        else:
            # Fallback to cosine for now
            similarity_expr = f"1 - (m.embedding <=> ${param_count + 1})"
            order_expr = f"m.embedding <=> ${param_count + 1}"

        # Convert query embedding to list if it's a numpy array
        if isinstance(query.query_embedding, np.ndarray):
            params.append(query.query_embedding.tolist())
        else:
            params.append(query.query_embedding)

        sql = f"""
            SELECT
                m.id,
                m.memory_text,
                {similarity_expr} as similarity_score,
                m.importance_score,
                m.memory_type,
                m.metadata,
                m.created_at,
                m.last_accessed,
                m.access_count
            FROM mem0_vectors.memories m
            WHERE {where_clause}
                AND {similarity_expr} >= {query.config.similarity_threshold}
            ORDER BY {order_expr}
            LIMIT {query.config.max_results}
        """

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

            results = []
            for row in rows:
                result = SearchResult(
                    memory_id=str(row["id"]),
                    memory_text=row["memory_text"],
                    similarity_score=float(row["similarity_score"]),
                    importance_score=float(row["importance_score"]),
                    memory_type=row["memory_type"],
                    metadata=row["metadata"] or {},
                    created_at=row["created_at"],
                    last_accessed=row["last_accessed"],
                    access_count=row["access_count"],
                    final_score=float(
                        row["similarity_score"]
                    ),  # Will be updated in reranking
                )
                results.append(result)

            return results

    async def _faiss_search(self, query: SearchQuery) -> list[SearchResult]:
        """Perform search using FAISS index."""
        algorithm = query.config.algorithm

        # Build index if it doesn't exist
        try:
            query_embedding_array = (
                query.query_embedding
                if isinstance(query.query_embedding, np.ndarray)
                else np.array(query.query_embedding, dtype=np.float32)
            )
            similarities, indices = self.faiss_manager.search(
                query.user_id,
                algorithm,
                query_embedding_array,
                query.config.max_results,
                query.config.faiss_nprobe,
            )
        except ValueError:
            # Index doesn't exist, build it
            await self.build_user_index(query.user_id, algorithm)
            similarities, indices = self.faiss_manager.search(
                query.user_id,
                algorithm,
                query_embedding_array,
                query.config.max_results,
                query.config.faiss_nprobe,
            )

        # Get memory details from database
        memory_ids, _ = await self._load_user_embeddings(query.user_id)

        # Filter results by similarity threshold
        valid_results = []
        for sim, idx in zip(similarities, indices, strict=False):
            if idx < len(memory_ids) and sim >= query.config.similarity_threshold:
                valid_results.append((memory_ids[idx], float(sim)))

        if not valid_results:
            return []

        # Get full memory details
        memory_id_list = [r[0] for r in valid_results]
        similarity_map = {r[0]: r[1] for r in valid_results}

        async with self.db_pool.acquire() as conn:
            placeholders = ",".join(f"${i+2}" for i in range(len(memory_id_list)))
            sql = f"""
                SELECT
                    m.id,
                    m.memory_text,
                    m.importance_score,
                    m.memory_type,
                    m.metadata,
                    m.created_at,
                    m.last_accessed,
                    m.access_count
                FROM mem0_vectors.memories m
                WHERE m.user_id = $1 AND m.id IN ({placeholders})
            """

            rows = await conn.fetch(sql, query.user_id, *memory_id_list)

            results = []
            for row in rows:
                memory_id = str(row["id"])
                similarity_score = similarity_map.get(memory_id, 0.0)

                result = SearchResult(
                    memory_id=memory_id,
                    memory_text=row["memory_text"],
                    similarity_score=similarity_score,
                    importance_score=float(row["importance_score"]),
                    memory_type=row["memory_type"],
                    metadata=row["metadata"] or {},
                    created_at=row["created_at"],
                    last_accessed=row["last_accessed"],
                    access_count=row["access_count"],
                    final_score=similarity_score,
                )
                results.append(result)

            # Sort by similarity score
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            return results

    def _rerank_results(
        self, results: list[SearchResult], config: SearchConfig
    ) -> list[SearchResult]:
        """Apply advanced reranking to search results."""
        if not config.enable_reranking or not results:
            return results

        current_time = datetime.now()

        for result in results:
            # Calculate temporal decay
            age_hours = (current_time - result.created_at).total_seconds() / 3600
            temporal_factor = np.exp(-age_hours / config.temporal_decay_hours)

            # Calculate access frequency boost
            access_factor = np.log1p(result.access_count) / 10.0

            # Calculate recency factor
            last_access_hours = (
                current_time - result.last_accessed
            ).total_seconds() / 3600
            recency_factor = np.exp(
                -last_access_hours / (config.temporal_decay_hours / 4)
            )

            # Combined score
            result.final_score = (
                result.similarity_score * (1 - config.importance_weight)
                + result.importance_score * config.importance_weight
                + temporal_factor * 0.1
                + access_factor * 0.05
                + recency_factor * 0.05
            )

        # Sort by final score
        results.sort(key=lambda x: x.final_score, reverse=True)

        # Apply diversity if needed
        if config.diversity_lambda > 0:
            results = self._apply_diversity(results, config.diversity_lambda)

        return results[: config.top_k]

    def _apply_diversity(
        self, results: list[SearchResult], diversity_lambda: float
    ) -> list[SearchResult]:
        """Apply maximal marginal relevance for diverse results."""
        if len(results) <= 1:
            return results

        selected = [results[0]]  # Start with highest scoring result
        remaining = results[1:]

        while remaining and len(selected) < len(results):
            best_idx = 0
            best_score = -float("inf")

            for i, candidate in enumerate(remaining):
                # Calculate similarity to already selected results
                max_sim_to_selected = 0
                for selected_result in selected:
                    # Simple text-based similarity (could use embeddings)
                    sim = len(
                        set(candidate.memory_text.lower().split())
                        & set(selected_result.memory_text.lower().split())
                    ) / len(
                        set(candidate.memory_text.lower().split())
                        | set(selected_result.memory_text.lower().split())
                    )
                    max_sim_to_selected = max(max_sim_to_selected, sim)

                # MMR score
                mmr_score = (
                    1 - diversity_lambda
                ) * candidate.final_score - diversity_lambda * max_sim_to_selected

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    async def search(self, query: SearchQuery) -> SearchResponse:
        """Main search function."""
        start_time = time.time()

        # Set default config
        if query.config is None:
            query.config = SearchConfig()

        # Generate query ID
        embedding_sample = (
            query.query_embedding[:5]
            if isinstance(query.query_embedding, list)
            else query.query_embedding.flatten()[:5].tolist()
        )
        query_id = hashlib.sha256(
            f"{query.user_id}_{time.time()}_{embedding_sample}".encode(), usedforsecurity=False
        ).hexdigest()

        # Check cache first
        cached_response = await self.cache.get(query)
        if cached_response:
            cached_response.query_id = query_id
            return cached_response

        # Perform search based on algorithm
        try:
            if query.config.algorithm == SearchAlgorithm.EXACT:
                results = await self._exact_search(query)
            elif query.config.algorithm in [
                SearchAlgorithm.FAISS_FLAT,
                SearchAlgorithm.FAISS_IVF,
                SearchAlgorithm.FAISS_HNSW,
            ]:
                results = await self._faiss_search(query)
            elif query.config.algorithm == SearchAlgorithm.HYBRID:
                # Combine exact and FAISS results
                exact_results = await self._exact_search(query)

                # Try FAISS search
                query.config.algorithm = SearchAlgorithm.FAISS_FLAT
                try:
                    faiss_results = await self._faiss_search(query)
                    # Merge and deduplicate results
                    seen_ids = set()
                    combined_results = []
                    for result in exact_results + faiss_results:
                        if result.memory_id not in seen_ids:
                            combined_results.append(result)
                            seen_ids.add(result.memory_id)
                    results = combined_results
                except Exception as err:
                    logger.warning("FAISS search failed, falling back to exact search: %s", err)
                    results = exact_results

                query.config.algorithm = SearchAlgorithm.HYBRID
            else:
                raise ValueError(
                    f"Unsupported search algorithm: {query.config.algorithm}"
                )

            # Apply reranking
            results = self._rerank_results(results, query.config)

            search_time_ms = (time.time() - start_time) * 1000

            response = SearchResponse(
                results=results,
                total_found=len(results),
                search_time_ms=search_time_ms,
                algorithm_used=query.config.algorithm,
                cache_hit=False,
                query_id=query_id,
            )

            # Cache the response
            await self.cache.set(query, response)

            return response

        except Exception as e:
            logger.error("Search error: %s", e)
            # Return empty response on error
            return SearchResponse(
                results=[],
                total_found=0,
                search_time_ms=(time.time() - start_time) * 1000,
                algorithm_used=query.config.algorithm,
                cache_hit=False,
                query_id=query_id,
            )

    async def update_memory_access(self, memory_id: str, user_id: str):
        """Update memory access statistics."""
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE mem0_vectors.memories
                SET access_count = access_count + 1,
                    last_accessed = NOW()
                WHERE id = $1 AND user_id = $2
            """,
                memory_id,
                user_id,
            )

    async def get_search_analytics(self, user_id: str, hours: int = 24) -> dict:
        """Get search analytics for user."""
        # This would typically be implemented with proper analytics tracking
        # For now, return basic stats
        async with self.db_pool.acquire() as conn:
            stats = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as total_memories,
                    AVG(importance_score) as avg_importance,
                    AVG(access_count) as avg_access_count,
                    MAX(last_accessed) as last_activity
                FROM mem0_vectors.memories
                WHERE user_id = $1
            """,
                user_id,
            )

            return {
                "total_memories": stats["total_memories"] or 0,
                "avg_importance": float(stats["avg_importance"] or 0),
                "avg_access_count": float(stats["avg_access_count"] or 0),
                "last_activity": stats["last_activity"],
            }


# Example usage
async def main():
    """Test the similarity search optimizer."""
    db_url = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/mem0ai")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

    optimizer = SimilaritySearchOptimizer(db_url, redis_url)

    try:
        await optimizer.initialize()

        # Example search query
        query_embedding = np.random.rand(1536).astype(np.float32)  # Random embedding for demo

        query = SearchQuery(
            user_id="test_user",
            query_embedding=query_embedding,
            query_text="test query",
            config=SearchConfig(
                algorithm=SearchAlgorithm.EXACT,
                top_k=10,
                similarity_threshold=0.7,
                enable_reranking=True,
            ),
        )

        # Perform search
        response = await optimizer.search(query)


        for i, result in enumerate(response.results[:5]):
            logger.info("Result %d: %s... (score: %.3f)", i, result.memory_text[:50], result.similarity_score)

        # Get analytics
        analytics = await optimizer.get_search_analytics("test_user")
        logger.info("Search analytics: %s", analytics)

    finally:
        await optimizer.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
