#!/usr/bin/env python3
"""Advanced Memory Deduplication System for mem0ai
Production-grade deduplication with multiple strategies and algorithms.
"""

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import asyncpg
import jellyfish
import Levenshtein
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DeduplicationStrategy(Enum):
    """Deduplication strategies."""

    EXACT_MATCH = "exact_match"
    FUZZY_TEXT = "fuzzy_text"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    HYBRID = "hybrid"
    CONTENT_HASH = "content_hash"
    TEMPORAL_CLUSTERING = "temporal_clustering"


class SimilarityMetric(Enum):
    """Text similarity metrics."""

    LEVENSHTEIN = "levenshtein"
    JARO_WINKLER = "jaro_winkler"
    COSINE_TFIDF = "cosine_tfidf"
    JACCARD = "jaccard"
    HAMMING = "hamming"
    SOUNDEX = "soundex"


@dataclass
class DeduplicationConfig:
    """Configuration for deduplication."""

    strategy: DeduplicationStrategy = DeduplicationStrategy.HYBRID
    similarity_threshold: float = 0.85
    vector_similarity_threshold: float = 0.95
    text_similarity_threshold: float = 0.80
    exact_match_threshold: float = 0.99
    batch_size: int = 1000
    max_workers: int = 4
    enable_stemming: bool = True
    remove_stopwords: bool = True
    case_sensitive: bool = False
    ignore_punctuation: bool = True
    temporal_window_hours: int = 24
    min_text_length: int = 10
    preserve_latest: bool = True  # Keep latest duplicate, remove older ones


@dataclass
class DuplicateGroup:
    """Group of duplicate memories."""

    canonical_id: str  # ID of the memory to keep
    duplicate_ids: List[str]  # IDs of memories to remove/merge
    similarity_scores: List[float]  # Similarity scores for each duplicate
    detection_method: str  # How duplicates were detected
    confidence: float  # Confidence in duplicate detection
    merge_strategy: str  # How to handle the duplicates


@dataclass
class DeduplicationResult:
    """Result from deduplication process."""

    user_id: str
    total_memories: int
    duplicate_groups: List[DuplicateGroup]
    duplicates_found: int
    duplicates_removed: int
    space_saved_percent: float
    processing_time_ms: float
    strategy_used: DeduplicationStrategy


class TextNormalizer:
    """Normalizes text for comparison."""

    def __init__(self, config: DeduplicationConfig):
        self.config = config
        self.stemmer = PorterStemmer() if config.enable_stemming else None
        self.stop_words = (
            set(stopwords.words("english")) if config.remove_stopwords else set()
        )

    def normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""

        # Case normalization
        if not self.config.case_sensitive:
            text = text.lower()

        # Remove punctuation if configured
        if self.config.ignore_punctuation:
            text = re.sub(r"[^\w\s]", " ", text)

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords
        if self.config.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]

        # Stem words
        if self.config.enable_stemming and self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]

        # Remove extra whitespace
        normalized = " ".join(tokens)
        normalized = re.sub(r"\s+", " ", normalized).strip()

        return normalized

    def get_content_hash(self, text: str) -> str:
        """Generate content hash for exact duplicate detection."""
        normalized = self.normalize(text)
        return hashlib.sha256(normalized.encode()).hexdigest()


class SimilarityCalculator:
    """Calculates various text similarity metrics."""

    def __init__(self):
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.text_to_index = {}

    def fit_tfidf(self, texts: List[str]):
        """Fit TF-IDF vectorizer on texts."""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000, stop_words="english", ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        self.text_to_index = {text: i for i, text in enumerate(texts)}

    def calculate_similarity(
        self, text1: str, text2: str, metric: SimilarityMetric
    ) -> float:
        """Calculate similarity between two texts using specified metric."""
        if metric == SimilarityMetric.LEVENSHTEIN:
            return 1 - (
                Levenshtein.distance(text1, text2) / max(len(text1), len(text2), 1)
            )

        elif metric == SimilarityMetric.JARO_WINKLER:
            return jellyfish.jaro_winkler_similarity(text1, text2)

        elif metric == SimilarityMetric.COSINE_TFIDF:
            if self.tfidf_vectorizer is None:
                return 0.0

            try:
                vec1 = self.tfidf_vectorizer.transform([text1])
                vec2 = self.tfidf_vectorizer.transform([text2])
                return cosine_similarity(vec1, vec2)[0][0]
            except:
                return 0.0

        elif metric == SimilarityMetric.JACCARD:
            set1 = set(text1.split())
            set2 = set(text2.split())
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union if union > 0 else 0.0

        elif metric == SimilarityMetric.SOUNDEX:
            try:
                soundex1 = jellyfish.soundex(text1[:50])  # Limit length for soundex
                soundex2 = jellyfish.soundex(text2[:50])
                return 1.0 if soundex1 == soundex2 else 0.0
            except:
                return 0.0

        else:
            # Fallback to basic string matching
            return 1.0 if text1 == text2 else 0.0


class MemoryDeduplicator:
    """Main memory deduplication engine."""

    def __init__(self, db_url: str, config: DeduplicationConfig = None):
        self.db_url = db_url
        self.config = config or DeduplicationConfig()
        self.pool = None
        self.text_normalizer = TextNormalizer(self.config)
        self.similarity_calculator = SimilarityCalculator()

    async def initialize(self):
        """Initialize the deduplicator."""
        self.pool = await asyncpg.create_pool(
            self.db_url, min_size=5, max_size=20, command_timeout=300
        )

        # Create deduplication tracking table
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mem0_vectors.deduplication_log (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id TEXT NOT NULL,
                    canonical_memory_id UUID NOT NULL,
                    duplicate_memory_ids UUID[] NOT NULL,
                    similarity_scores FLOAT[] NOT NULL,
                    detection_method TEXT NOT NULL,
                    confidence FLOAT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS dedup_log_user_id_idx
                ON mem0_vectors.deduplication_log (user_id);

                CREATE INDEX IF NOT EXISTS dedup_log_created_at_idx
                ON mem0_vectors.deduplication_log (created_at);
            """
            )

        logger.info("Memory deduplicator initialized")

    async def cleanup(self):
        """Cleanup resources."""
        if self.pool:
            await self.pool.close()

    async def find_duplicates(self, user_id: str) -> List[DuplicateGroup]:
        """Find duplicate memories for a user."""
        start_time = time.time()

        # Get all memories for the user
        async with self.pool.acquire() as conn:
            memories = await conn.fetch(
                """
                SELECT id, memory_text, embedding, created_at, importance_score, access_count
                FROM mem0_vectors.memories
                WHERE user_id = $1
                ORDER BY created_at DESC
            """,
                user_id,
            )

        if len(memories) < 2:
            return []

        logger.info(f"Analyzing {len(memories)} memories for user {user_id}")

        # Apply different strategies based on configuration
        if self.config.strategy == DeduplicationStrategy.EXACT_MATCH:
            duplicate_groups = await self._find_exact_duplicates(memories)
        elif self.config.strategy == DeduplicationStrategy.FUZZY_TEXT:
            duplicate_groups = await self._find_fuzzy_duplicates(memories)
        elif self.config.strategy == DeduplicationStrategy.SEMANTIC_SIMILARITY:
            duplicate_groups = await self._find_semantic_duplicates(memories)
        elif self.config.strategy == DeduplicationStrategy.CONTENT_HASH:
            duplicate_groups = await self._find_hash_duplicates(memories)
        elif self.config.strategy == DeduplicationStrategy.TEMPORAL_CLUSTERING:
            duplicate_groups = await self._find_temporal_duplicates(memories)
        else:  # HYBRID
            duplicate_groups = await self._find_hybrid_duplicates(memories)

        processing_time = (time.time() - start_time) * 1000
        logger.info(
            f"Found {len(duplicate_groups)} duplicate groups in {processing_time:.2f}ms"
        )

        return duplicate_groups

    async def _find_exact_duplicates(
        self, memories: List[Dict]
    ) -> List[DuplicateGroup]:
        """Find exact text duplicates."""
        text_to_memories = {}
        duplicate_groups = []

        for memory in memories:
            text = memory["memory_text"]
            normalized_text = self.text_normalizer.normalize(text)

            if normalized_text in text_to_memories:
                text_to_memories[normalized_text].append(memory)
            else:
                text_to_memories[normalized_text] = [memory]

        # Create duplicate groups
        for normalized_text, memory_group in text_to_memories.items():
            if len(memory_group) > 1:
                # Sort by creation time (newest first if preserve_latest is True)
                memory_group.sort(
                    key=lambda m: m["created_at"], reverse=self.config.preserve_latest
                )

                canonical_memory = memory_group[0]
                duplicates = memory_group[1:]

                duplicate_groups.append(
                    DuplicateGroup(
                        canonical_id=str(canonical_memory["id"]),
                        duplicate_ids=[str(m["id"]) for m in duplicates],
                        similarity_scores=[1.0] * len(duplicates),
                        detection_method="exact_match",
                        confidence=1.0,
                        merge_strategy="remove_duplicates",
                    )
                )

        return duplicate_groups

    async def _find_fuzzy_duplicates(
        self, memories: List[Dict]
    ) -> List[DuplicateGroup]:
        """Find fuzzy text duplicates using multiple similarity metrics."""
        duplicate_groups = []
        processed_pairs = set()

        # Normalize all texts for TF-IDF
        normalized_texts = []
        for memory in memories:
            normalized = self.text_normalizer.normalize(memory["memory_text"])
            normalized_texts.append(normalized)

        # Fit TF-IDF vectorizer
        if normalized_texts:
            self.similarity_calculator.fit_tfidf(normalized_texts)

        # Compare all pairs
        for i in range(len(memories)):
            memory1 = memories[i]
            text1 = self.text_normalizer.normalize(memory1["memory_text"])

            if len(text1) < self.config.min_text_length:
                continue

            similar_memories = []

            for j in range(i + 1, len(memories)):
                memory2 = memories[j]
                text2 = self.text_normalizer.normalize(memory2["memory_text"])

                if len(text2) < self.config.min_text_length:
                    continue

                pair_key = tuple(sorted([str(memory1["id"]), str(memory2["id"])]))
                if pair_key in processed_pairs:
                    continue

                # Calculate multiple similarity metrics
                similarities = {
                    SimilarityMetric.LEVENSHTEIN: self.similarity_calculator.calculate_similarity(
                        text1, text2, SimilarityMetric.LEVENSHTEIN
                    ),
                    SimilarityMetric.JARO_WINKLER: self.similarity_calculator.calculate_similarity(
                        text1, text2, SimilarityMetric.JARO_WINKLER
                    ),
                    SimilarityMetric.COSINE_TFIDF: self.similarity_calculator.calculate_similarity(
                        text1, text2, SimilarityMetric.COSINE_TFIDF
                    ),
                    SimilarityMetric.JACCARD: self.similarity_calculator.calculate_similarity(
                        text1, text2, SimilarityMetric.JACCARD
                    ),
                }

                # Calculate weighted average similarity with numpy for efficiency
                weights = np.array([0.25, 0.25, 0.35, 0.15])  # LEVENSHTEIN, JARO_WINKLER, COSINE_TFIDF, JACCARD
                similarity_values = np.array([
                    similarities[SimilarityMetric.LEVENSHTEIN],
                    similarities[SimilarityMetric.JARO_WINKLER], 
                    similarities[SimilarityMetric.COSINE_TFIDF],
                    similarities[SimilarityMetric.JACCARD]
                ])
                
                weighted_similarity = float(np.dot(similarity_values, weights))

                if weighted_similarity >= self.config.text_similarity_threshold:
                    similar_memories.append((memory2, weighted_similarity))
                    processed_pairs.add(pair_key)

            # Create duplicate group if similar memories found
            if similar_memories:
                all_memories = [memory1] + [mem for mem, _ in similar_memories]

                # Sort by importance and recency
                all_memories.sort(
                    key=lambda m: (m["importance_score"], m["created_at"]), reverse=True
                )

                canonical_memory = all_memories[0]
                duplicates = all_memories[1:]
                similarity_scores = [sim for _, sim in similar_memories]

                duplicate_groups.append(
                    DuplicateGroup(
                        canonical_id=str(canonical_memory["id"]),
                        duplicate_ids=[str(m["id"]) for m in duplicates],
                        similarity_scores=similarity_scores,
                        detection_method="fuzzy_text",
                        confidence=np.mean(similarity_scores),
                        merge_strategy="merge_metadata",
                    )
                )

        return duplicate_groups

    async def _find_semantic_duplicates(
        self, memories: List[Dict]
    ) -> List[DuplicateGroup]:
        """Find semantically similar memories using embeddings."""
        duplicate_groups = []

        # Filter memories with embeddings
        memories_with_embeddings = [m for m in memories if m["embedding"] is not None]

        if len(memories_with_embeddings) < 2:
            return duplicate_groups

        # Convert embeddings to numpy array with proper type
        embeddings = np.array(
            [m["embedding"] for m in memories_with_embeddings], 
            dtype=np.float32
        )

        # Calculate pairwise cosine similarities efficiently
        similarities = cosine_similarity(embeddings)

        processed_indices = set()

        for i in range(len(memories_with_embeddings)):
            if i in processed_indices:
                continue

            memory1 = memories_with_embeddings[i]
            similar_memories = []

            for j in range(i + 1, len(memories_with_embeddings)):
                if j in processed_indices:
                    continue

                similarity = similarities[i][j]

                if similarity >= self.config.vector_similarity_threshold:
                    memory2 = memories_with_embeddings[j]
                    similar_memories.append((memory2, similarity))
                    processed_indices.add(j)

            if similar_memories:
                processed_indices.add(i)

                all_memories = [memory1] + [mem for mem, _ in similar_memories]

                # Choose canonical memory based on importance and access count
                all_memories.sort(
                    key=lambda m: (
                        m["importance_score"],
                        m["access_count"],
                        m["created_at"],
                    ),
                    reverse=True,
                )

                canonical_memory = all_memories[0]
                duplicates = all_memories[1:]
                similarity_scores = [sim for _, sim in similar_memories]

                duplicate_groups.append(
                    DuplicateGroup(
                        canonical_id=str(canonical_memory["id"]),
                        duplicate_ids=[str(m["id"]) for m in duplicates],
                        similarity_scores=similarity_scores,
                        detection_method="semantic_similarity",
                        confidence=np.mean(similarity_scores),
                        merge_strategy="merge_with_highest_importance",
                    )
                )

        return duplicate_groups

    async def _find_hash_duplicates(self, memories: List[Dict]) -> List[DuplicateGroup]:
        """Find duplicates using content hashing."""
        hash_to_memories = {}

        for memory in memories:
            content_hash = self.text_normalizer.get_content_hash(memory["memory_text"])

            if content_hash in hash_to_memories:
                hash_to_memories[content_hash].append(memory)
            else:
                hash_to_memories[content_hash] = [memory]

        duplicate_groups = []

        for content_hash, memory_group in hash_to_memories.items():
            if len(memory_group) > 1:
                # Sort by creation time and importance
                memory_group.sort(
                    key=lambda m: (m["importance_score"], m["created_at"]), reverse=True
                )

                canonical_memory = memory_group[0]
                duplicates = memory_group[1:]

                duplicate_groups.append(
                    DuplicateGroup(
                        canonical_id=str(canonical_memory["id"]),
                        duplicate_ids=[str(m["id"]) for m in duplicates],
                        similarity_scores=[1.0] * len(duplicates),
                        detection_method="content_hash",
                        confidence=1.0,
                        merge_strategy="remove_duplicates",
                    )
                )

        return duplicate_groups

    async def _find_temporal_duplicates(
        self, memories: List[Dict]
    ) -> List[DuplicateGroup]:
        """Find duplicates within temporal windows."""
        duplicate_groups = []

        # Sort by creation time
        sorted_memories = sorted(memories, key=lambda m: m["created_at"])

        window_timedelta = timedelta(hours=self.config.temporal_window_hours)

        for i, memory1 in enumerate(sorted_memories):
            similar_memories = []

            # Look for similar memories within the temporal window
            for j in range(i + 1, len(sorted_memories)):
                memory2 = sorted_memories[j]

                # Check if within temporal window
                time_diff = memory2["created_at"] - memory1["created_at"]
                if time_diff > window_timedelta:
                    break

                # Check text similarity
                text1 = self.text_normalizer.normalize(memory1["memory_text"])
                text2 = self.text_normalizer.normalize(memory2["memory_text"])

                similarity = self.similarity_calculator.calculate_similarity(
                    text1, text2, SimilarityMetric.JARO_WINKLER
                )

                if similarity >= self.config.text_similarity_threshold:
                    similar_memories.append((memory2, similarity))

            if similar_memories:
                all_memories = [memory1] + [mem for mem, _ in similar_memories]

                # Choose canonical memory (latest with highest importance)
                canonical_memory = max(
                    all_memories, key=lambda m: (m["importance_score"], m["created_at"])
                )

                duplicates = [m for m in all_memories if m != canonical_memory]
                similarity_scores = [sim for _, sim in similar_memories]

                duplicate_groups.append(
                    DuplicateGroup(
                        canonical_id=str(canonical_memory["id"]),
                        duplicate_ids=[str(m["id"]) for m in duplicates],
                        similarity_scores=similarity_scores,
                        detection_method="temporal_clustering",
                        confidence=np.mean(similarity_scores),
                        merge_strategy="merge_recent",
                    )
                )

        return duplicate_groups

    async def _find_hybrid_duplicates(
        self, memories: List[Dict]
    ) -> List[DuplicateGroup]:
        """Find duplicates using hybrid approach combining multiple strategies."""
        all_groups = []

        # Apply each strategy
        strategies = [
            self._find_exact_duplicates,
            self._find_hash_duplicates,
            self._find_semantic_duplicates,
            self._find_fuzzy_duplicates,
        ]

        for strategy in strategies:
            try:
                groups = await strategy(memories)
                all_groups.extend(groups)
            except Exception as e:
                logger.warning(f"Strategy failed: {e}")

        # Merge overlapping groups and remove duplicates
        return self._merge_duplicate_groups(all_groups)

    def _merge_duplicate_groups(
        self, groups: List[DuplicateGroup]
    ) -> List[DuplicateGroup]:
        """Merge overlapping duplicate groups."""
        if not groups:
            return []

        # Create mapping of memory ID to group
        memory_to_group = {}
        for i, group in enumerate(groups):
            for memory_id in [group.canonical_id, *group.duplicate_ids]:
                if memory_id not in memory_to_group:
                    memory_to_group[memory_id] = []
                memory_to_group[memory_id].append(i)

        # Find groups that share memories
        group_clusters = []
        visited_groups = set()

        for group_idx in range(len(groups)):
            if group_idx in visited_groups:
                continue

            cluster = {group_idx}
            queue = [group_idx]

            while queue:
                current_group_idx = queue.pop(0)
                current_group = groups[current_group_idx]

                # Find all memories in this group
                all_memory_ids = [current_group.canonical_id, *current_group.duplicate_ids]

                # Find other groups that share any memory
                for memory_id in all_memory_ids:
                    for other_group_idx in memory_to_group.get(memory_id, []):
                        if other_group_idx not in cluster:
                            cluster.add(other_group_idx)
                            queue.append(other_group_idx)

            visited_groups.update(cluster)
            group_clusters.append(cluster)

        # Merge groups in each cluster
        merged_groups = []
        for cluster in group_clusters:
            if len(cluster) == 1:
                # No merging needed
                merged_groups.append(groups[next(iter(cluster))])
            else:
                # Merge multiple groups
                merged_group = self._merge_group_cluster([groups[i] for i in cluster])
                merged_groups.append(merged_group)

        return merged_groups

    def _merge_group_cluster(self, groups: List[DuplicateGroup]) -> DuplicateGroup:
        """Merge a cluster of overlapping groups."""
        all_memory_ids = set()
        all_similarities = []
        detection_methods = []
        confidences = []

        for group in groups:
            all_memory_ids.add(group.canonical_id)
            all_memory_ids.update(group.duplicate_ids)
            all_similarities.extend(group.similarity_scores)
            detection_methods.append(group.detection_method)
            confidences.append(group.confidence)

        # Choose canonical memory (could be improved with more sophisticated logic)
        canonical_id = groups[0].canonical_id
        duplicate_ids = list(all_memory_ids - {canonical_id})

        return DuplicateGroup(
            canonical_id=canonical_id,
            duplicate_ids=duplicate_ids,
            similarity_scores=all_similarities,
            detection_method="+".join(set(detection_methods)),
            confidence=np.mean(confidences),
            merge_strategy="hybrid_merge",
        )

    async def deduplicate_memories(
        self, user_id: str, dry_run: bool = True
    ) -> DeduplicationResult:
        """Perform deduplication for a user's memories."""
        start_time = time.time()

        # Get total count before deduplication
        async with self.pool.acquire() as conn:
            total_count = await conn.fetchval(
                "SELECT COUNT(*) FROM mem0_vectors.memories WHERE user_id = $1", user_id
            )

        # Find duplicates
        duplicate_groups = await self.find_duplicates(user_id)

        duplicates_found = sum(len(group.duplicate_ids) for group in duplicate_groups)
        duplicates_removed = 0

        if not dry_run and duplicate_groups:
            duplicates_removed = await self._remove_duplicates(
                user_id, duplicate_groups
            )

        processing_time = (time.time() - start_time) * 1000
        space_saved = (duplicates_removed / total_count * 100) if total_count > 0 else 0

        result = DeduplicationResult(
            user_id=user_id,
            total_memories=total_count,
            duplicate_groups=duplicate_groups,
            duplicates_found=duplicates_found,
            duplicates_removed=duplicates_removed,
            space_saved_percent=space_saved,
            processing_time_ms=processing_time,
            strategy_used=self.config.strategy,
        )

        logger.info(
            f"Deduplication completed for user {user_id}: "
            f"{duplicates_found} duplicates found, "
            f"{duplicates_removed} removed, "
            f"{space_saved:.1f}% space saved"
        )

        return result

    async def _remove_duplicates(
        self, user_id: str, duplicate_groups: List[DuplicateGroup]
    ) -> int:
        """Remove duplicate memories and log the operation."""
        removed_count = 0

        async with self.pool.acquire() as conn, conn.transaction():
            for group in duplicate_groups:
                try:
                    # Log the deduplication operation
                    await conn.execute(
                        """
                            INSERT INTO mem0_vectors.deduplication_log
                            (user_id, canonical_memory_id, duplicate_memory_ids,
                             similarity_scores, detection_method, confidence)
                            VALUES ($1, $2, $3, $4, $5, $6)
                        """,
                        user_id,
                        group.canonical_id,
                        group.duplicate_ids,
                        group.similarity_scores,
                        group.detection_method,
                        group.confidence,
                    )

                    # Remove duplicate memories
                    if group.duplicate_ids:
                        await conn.execute(
                            """
                                DELETE FROM mem0_vectors.memories
                                WHERE id = ANY($1) AND user_id = $2
                            """,
                            group.duplicate_ids,
                            user_id,
                        )

                        removed_count += len(group.duplicate_ids)

                except Exception as e:
                    logger.error(f"Failed to remove duplicate group: {e}")

        return removed_count

    async def get_deduplication_stats(self, user_id: str) -> Dict[str, Any]:
        """Get deduplication statistics for a user."""
        async with self.pool.acquire() as conn:
            stats = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as dedup_operations,
                    SUM(array_length(duplicate_memory_ids, 1)) as total_removed,
                    AVG(confidence) as avg_confidence,
                    MAX(created_at) as last_deduplication
                FROM mem0_vectors.deduplication_log
                WHERE user_id = $1
            """,
                user_id,
            )

            current_memory_count = await conn.fetchval(
                """
                SELECT COUNT(*) FROM mem0_vectors.memories WHERE user_id = $1
            """,
                user_id,
            )

            method_stats = await conn.fetch(
                """
                SELECT
                    detection_method,
                    COUNT(*) as count,
                    AVG(confidence) as avg_confidence
                FROM mem0_vectors.deduplication_log
                WHERE user_id = $1
                GROUP BY detection_method
                ORDER BY count DESC
            """,
                user_id,
            )

            return {
                "current_memory_count": current_memory_count,
                "deduplication_operations": stats["dedup_operations"] or 0,
                "total_memories_removed": stats["total_removed"] or 0,
                "average_confidence": float(stats["avg_confidence"] or 0),
                "last_deduplication": stats["last_deduplication"],
                "methods_used": [
                    {
                        "method": row["detection_method"],
                        "count": row["count"],
                        "avg_confidence": float(row["avg_confidence"]),
                    }
                    for row in method_stats
                ],
            }


# Example usage
async def main():
    """Test the memory deduplicator."""
    DB_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/mem0ai")

    config = DeduplicationConfig(
        strategy=DeduplicationStrategy.HYBRID,
        similarity_threshold=0.85,
        text_similarity_threshold=0.80,
        batch_size=1000,
    )

    deduplicator = MemoryDeduplicator(DB_URL, config)

    try:
        await deduplicator.initialize()

        # Test deduplication
        user_id = "test_user"

        result = await deduplicator.deduplicate_memories(user_id, dry_run=True)


        if result.duplicate_groups:
            for _i, _group in enumerate(result.duplicate_groups[:3]):
                pass

        # Get stats
        await deduplicator.get_deduplication_stats(user_id)

    finally:
        await deduplicator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
