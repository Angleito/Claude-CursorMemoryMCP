#!/usr/bin/env python3
"""Advanced Vector Compression and Storage Efficiency for mem0ai
Production-grade compression algorithms for optimal storage and performance.
"""

import asyncio
import gzip
import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import asyncpg
import blosc2
import faiss
import lz4.frame
import numpy as np
from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CompressionMethod(Enum):
    """Supported compression methods."""

    NONE = "none"
    PCA = "pca"  # Principal Component Analysis
    RANDOM_PROJECTION = "random_projection"  # Johnson-Lindenstrauss
    PRODUCT_QUANTIZATION = "product_quantization"  # Vector quantization
    SCALAR_QUANTIZATION = "scalar_quantization"  # Simple quantization
    BLOSC2 = "blosc2"  # General purpose compression
    LZ4 = "lz4"  # Fast compression
    GZIP = "gzip"  # Standard compression
    HYBRID = "hybrid"  # Combine multiple methods


class QuantizationBits(Enum):
    """Quantization bit depths."""

    FLOAT32 = 32
    FLOAT16 = 16
    INT8 = 8
    INT4 = 4
    BINARY = 1


@dataclass
class CompressionConfig:
    """Configuration for vector compression."""

    method: CompressionMethod = CompressionMethod.HYBRID
    target_dimensions: Optional[int] = None  # For dimensionality reduction
    quantization_bits: QuantizationBits = QuantizationBits.FLOAT16
    pq_subvectors: int = 8  # Product quantization subvectors
    pq_bits: int = 8  # Bits per subvector
    compression_level: int = 6  # For general compression algorithms
    preserve_accuracy: float = 0.95  # Minimum accuracy to preserve
    enable_delta_compression: bool = True  # For temporal sequences
    batch_size: int = 1000
    enable_caching: bool = True


@dataclass
class CompressionResult:
    """Result from vector compression."""

    original_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    compression_time_ms: float
    decompression_time_ms: float
    accuracy_loss: float
    method_used: CompressionMethod
    metadata: Dict[str, Any]


@dataclass
class CompressedVector:
    """Compressed vector representation."""

    data: bytes
    metadata: Dict[str, Any]
    method: CompressionMethod
    original_dimensions: int
    compressed_dimensions: Optional[int]
    quantization_info: Optional[Dict]
    checksum: str


class VectorQuantizer:
    """Advanced vector quantization methods."""

    def __init__(self, config: CompressionConfig):
        self.config = config
        self.pca_model = None
        self.projection_model = None
        self.pq_index = None
        self.quantization_params = {}
        self.lock = threading.Lock()

    def fit_pca(self, vectors: np.ndarray) -> None:
        """Fit PCA model for dimensionality reduction."""
        with self.lock:
            target_dim = self.config.target_dimensions or min(
                vectors.shape[1] // 2, 512
            )
            self.pca_model = PCA(n_components=target_dim)
            # Ensure vectors are float32 for memory efficiency
            vectors_f32 = vectors.astype(np.float32)
            self.pca_model.fit(vectors_f32)

            # Calculate variance explained
            explained_variance = np.sum(self.pca_model.explained_variance_ratio_)
            logger.info(
                f"PCA fitted: {vectors.shape[1]} -> {target_dim} dims, "
                f"variance explained: {explained_variance:.3f}"
            )

    def transform_pca(self, vectors: np.ndarray) -> np.ndarray:
        """Apply PCA transformation."""
        if self.pca_model is None:
            raise ValueError("PCA model not fitted")
        vectors_f32 = vectors.astype(np.float32)
        result = self.pca_model.transform(vectors_f32)
        return result.astype(np.float32)

    def inverse_transform_pca(self, compressed_vectors: np.ndarray) -> np.ndarray:
        """Inverse PCA transformation."""
        if self.pca_model is None:
            raise ValueError("PCA model not fitted")
        compressed_f32 = compressed_vectors.astype(np.float32)
        result = self.pca_model.inverse_transform(compressed_f32)
        return result.astype(np.float32)

    def fit_random_projection(self, vectors: np.ndarray) -> None:
        """Fit random projection for dimensionality reduction."""
        with self.lock:
            target_dim = self.config.target_dimensions or min(
                vectors.shape[1] // 2, 512
            )

            # Johnson-Lindenstrauss lemma for dimension calculation
            n_samples = vectors.shape[0]
            eps = 1 - self.config.preserve_accuracy
            min_dim = int(4 * np.log(n_samples) / (eps**2 / 2 - eps**3 / 3))
            target_dim = max(target_dim, min_dim)

            self.projection_model = SparseRandomProjection(
                n_components=target_dim, density="auto", random_state=42
            )
            self.projection_model.fit(vectors)

            logger.info(
                f"Random projection fitted: {vectors.shape[1]} -> {target_dim} dims"
            )

    def transform_random_projection(self, vectors: np.ndarray) -> np.ndarray:
        """Apply random projection."""
        if self.projection_model is None:
            raise ValueError("Random projection model not fitted")
        vectors_f32 = vectors.astype(np.float32)
        result = self.projection_model.transform(vectors_f32)
        return result.astype(np.float32)

    def fit_product_quantization(self, vectors: np.ndarray) -> None:
        """Fit product quantization index."""
        with self.lock:
            d = vectors.shape[1]

            # Ensure dimensions are divisible by number of subvectors
            if d % self.config.pq_subvectors != 0:
                # Pad vectors to make divisible
                pad_size = self.config.pq_subvectors - (d % self.config.pq_subvectors)
                vectors = np.pad(vectors, ((0, 0), (0, pad_size)), mode="constant")
                d = vectors.shape[1]

            # Create product quantization index
            self.pq_index = faiss.IndexPQ(
                d, self.config.pq_subvectors, self.config.pq_bits
            )
            # Ensure vectors are contiguous and float32
            training_vectors = np.ascontiguousarray(vectors, dtype=np.float32)
            self.pq_index.train(training_vectors)

            logger.info(
                f"Product quantization fitted: {d} dims, "
                f"{self.config.pq_subvectors} subvectors, {self.config.pq_bits} bits"
            )

    def transform_product_quantization(self, vectors: np.ndarray) -> bytes:
        """Apply product quantization."""
        if self.pq_index is None:
            raise ValueError("Product quantization not fitted")

        # Pad if necessary
        d_target = self.pq_index.d
        if vectors.shape[1] < d_target:
            pad_size = d_target - vectors.shape[1]
            vectors = np.pad(vectors, ((0, 0), (0, pad_size)), mode="constant")

        # Encode vectors with proper memory layout
        vectors_f32 = np.ascontiguousarray(vectors, dtype=np.float32)
        codes = self.pq_index.sa_encode(vectors_f32)
        return codes.tobytes()

    def inverse_transform_product_quantization(
        self, codes: bytes, n_vectors: int
    ) -> np.ndarray:
        """Decode product quantized vectors."""
        if self.pq_index is None:
            raise ValueError("Product quantization not fitted")

        # Convert bytes back to codes array
        code_size = self.config.pq_subvectors
        codes_array = np.frombuffer(codes, dtype=np.uint8).reshape(n_vectors, code_size)

        # Decode vectors
        decoded = self.pq_index.sa_decode(codes_array)
        return decoded

    def scalar_quantization(self, vectors: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        """Apply scalar quantization."""
        if self.config.quantization_bits == QuantizationBits.FLOAT16:
            quantized = vectors.astype(np.float16)
            quantization_info = {"dtype": "float16", "shape": vectors.shape}
        elif self.config.quantization_bits == QuantizationBits.INT8:
            # Normalize to [-1, 1] and quantize to int8
            v_min, v_max = vectors.min(), vectors.max()
            normalized = 2 * (vectors - v_min) / (v_max - v_min) - 1
            quantized = (normalized * 127).astype(np.int8)
            quantization_info = {
                "dtype": "int8",
                "shape": vectors.shape,
                "min": float(v_min),
                "max": float(v_max),
            }
        elif self.config.quantization_bits == QuantizationBits.BINARY:
            # Binary quantization (sign-based)
            quantized = (vectors > 0).astype(np.uint8)
            # Pack bits for better compression
            quantized = np.packbits(quantized, axis=1)
            quantization_info = {
                "dtype": "binary",
                "shape": vectors.shape,
                "packed_shape": quantized.shape,
            }
        else:
            quantized = vectors.astype(np.float32)
            quantization_info = {"dtype": "float32", "shape": vectors.shape}

        return quantized.tobytes(), quantization_info

    def inverse_scalar_quantization(
        self, data: bytes, quantization_info: Dict[str, Any]
    ) -> np.ndarray:
        """Inverse scalar quantization."""
        dtype = quantization_info["dtype"]
        shape = quantization_info["shape"]

        if dtype == "float16":
            return (
                np.frombuffer(data, dtype=np.float16).reshape(shape).astype(np.float32)
            )
        elif dtype == "int8":
            quantized = np.frombuffer(data, dtype=np.int8).reshape(shape)
            normalized = quantized.astype(np.float32) / 127.0
            v_min, v_max = quantization_info["min"], quantization_info["max"]
            return (normalized + 1) / 2 * (v_max - v_min) + v_min
        elif dtype == "binary":
            packed_shape = quantization_info["packed_shape"]
            packed = np.frombuffer(data, dtype=np.uint8).reshape(packed_shape)
            unpacked = np.unpackbits(packed, axis=1)
            # Trim to original shape
            unpacked = unpacked[:, : shape[1]]
            return unpacked.astype(np.float32) * 2 - 1  # Convert {0,1} to {-1,1}
        else:
            return np.frombuffer(data, dtype=np.float32).reshape(shape)


class CompressionEngine:
    """Main vector compression engine."""

    def __init__(self, config: CompressionConfig = None):
        self.config = config or CompressionConfig()
        self.quantizer = VectorQuantizer(self.config)
        self.compression_cache = {}
        self.stats = {
            "compressions": 0,
            "decompressions": 0,
            "total_compression_time": 0.0,
            "total_decompression_time": 0.0,
            "total_original_size": 0,
            "total_compressed_size": 0,
        }

    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate checksum for data integrity."""
        return hashlib.sha256(data).hexdigest()[:16]

    def _apply_general_compression(self, data: bytes, method: str) -> bytes:
        """Apply general purpose compression algorithms."""
        if method == "blosc2":
            return blosc2.compress(
                data, cname="zstd", clevel=self.config.compression_level
            )
        elif method == "lz4":
            return lz4.frame.compress(
                data, compression_level=self.config.compression_level
            )
        elif method == "gzip":
            return gzip.compress(data, compresslevel=self.config.compression_level)
        else:
            return data

    def _decompress_general(self, data: bytes, method: str) -> bytes:
        """Decompress general purpose algorithms."""
        if method == "blosc2":
            return blosc2.decompress(data)
        elif method == "lz4":
            return lz4.frame.decompress(data)
        elif method == "gzip":
            return gzip.decompress(data)
        else:
            return data

    async def compress_vectors(
        self, vectors: Union[np.ndarray, List[List[float]]], user_id: Optional[str] = None
    ) -> List[CompressedVector]:
        """Compress a batch of vectors."""
        # Convert to numpy array if needed
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)
        
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)
        
        # Ensure vectors are float32 for consistency
        vectors = vectors.astype(np.float32)

        start_time = time.time()
        original_size = vectors.nbytes

        compressed_vectors = []

        try:
            if self.config.method == CompressionMethod.NONE:
                # No compression
                for i, vector in enumerate(vectors):
                    data = vector.astype(np.float32).tobytes()
                    compressed_vectors.append(
                        CompressedVector(
                            data=data,
                            metadata={"index": i},
                            method=CompressionMethod.NONE,
                            original_dimensions=len(vector),
                            compressed_dimensions=len(vector),
                            quantization_info=None,
                            checksum=self._calculate_checksum(data),
                        )
                    )

            elif self.config.method == CompressionMethod.PCA:
                # PCA compression
                if self.quantizer.pca_model is None:
                    self.quantizer.fit_pca(vectors)

                compressed_data = self.quantizer.transform_pca(vectors)
                data_bytes, quant_info = self.quantizer.scalar_quantization(
                    compressed_data
                )
                data_bytes = self._apply_general_compression(data_bytes, "lz4")

                for i in range(len(vectors)):
                    compressed_vectors.append(
                        CompressedVector(
                            data=data_bytes,
                            metadata={"index": i, "method": "pca"},
                            method=CompressionMethod.PCA,
                            original_dimensions=vectors.shape[1],
                            compressed_dimensions=compressed_data.shape[1],
                            quantization_info=quant_info,
                            checksum=self._calculate_checksum(data_bytes),
                        )
                    )

            elif self.config.method == CompressionMethod.RANDOM_PROJECTION:
                # Random projection compression
                if self.quantizer.projection_model is None:
                    self.quantizer.fit_random_projection(vectors)

                compressed_data = self.quantizer.transform_random_projection(vectors)
                data_bytes, quant_info = self.quantizer.scalar_quantization(
                    compressed_data
                )
                data_bytes = self._apply_general_compression(data_bytes, "lz4")

                for i in range(len(vectors)):
                    compressed_vectors.append(
                        CompressedVector(
                            data=data_bytes,
                            metadata={"index": i, "method": "random_projection"},
                            method=CompressionMethod.RANDOM_PROJECTION,
                            original_dimensions=vectors.shape[1],
                            compressed_dimensions=compressed_data.shape[1],
                            quantization_info=quant_info,
                            checksum=self._calculate_checksum(data_bytes),
                        )
                    )

            elif self.config.method == CompressionMethod.PRODUCT_QUANTIZATION:
                # Product quantization
                if self.quantizer.pq_index is None:
                    self.quantizer.fit_product_quantization(vectors)

                data_bytes = self.quantizer.transform_product_quantization(vectors)

                for i in range(len(vectors)):
                    compressed_vectors.append(
                        CompressedVector(
                            data=data_bytes,
                            metadata={"index": i, "method": "product_quantization"},
                            method=CompressionMethod.PRODUCT_QUANTIZATION,
                            original_dimensions=vectors.shape[1],
                            compressed_dimensions=None,
                            quantization_info={"n_vectors": len(vectors)},
                            checksum=self._calculate_checksum(data_bytes),
                        )
                    )

            elif self.config.method == CompressionMethod.SCALAR_QUANTIZATION:
                # Scalar quantization only
                data_bytes, quant_info = self.quantizer.scalar_quantization(vectors)
                data_bytes = self._apply_general_compression(data_bytes, "blosc2")

                for i in range(len(vectors)):
                    compressed_vectors.append(
                        CompressedVector(
                            data=data_bytes,
                            metadata={"index": i, "method": "scalar_quantization"},
                            method=CompressionMethod.SCALAR_QUANTIZATION,
                            original_dimensions=vectors.shape[1],
                            compressed_dimensions=vectors.shape[1],
                            quantization_info=quant_info,
                            checksum=self._calculate_checksum(data_bytes),
                        )
                    )

            elif self.config.method == CompressionMethod.HYBRID:
                # Hybrid approach: dimensionality reduction + quantization + compression

                # Step 1: Dimensionality reduction
                if vectors.shape[1] > 512:  # Only if high dimensional
                    if self.quantizer.pca_model is None:
                        self.quantizer.fit_pca(vectors)
                    vectors = self.quantizer.transform_pca(vectors)

                # Step 2: Scalar quantization
                data_bytes, quant_info = self.quantizer.scalar_quantization(vectors)

                # Step 3: General compression
                data_bytes = self._apply_general_compression(data_bytes, "blosc2")

                for i in range(len(vectors)):
                    compressed_vectors.append(
                        CompressedVector(
                            data=data_bytes,
                            metadata={"index": i, "method": "hybrid"},
                            method=CompressionMethod.HYBRID,
                            original_dimensions=vectors.shape[1],
                            compressed_dimensions=vectors.shape[1],
                            quantization_info=quant_info,
                            checksum=self._calculate_checksum(data_bytes),
                        )
                    )

            # Calculate compression statistics
            total_compressed_size = sum(len(cv.data) for cv in compressed_vectors)
            compression_time = (time.time() - start_time) * 1000

            # Update stats
            self.stats["compressions"] += len(compressed_vectors)
            self.stats["total_compression_time"] += compression_time
            self.stats["total_original_size"] += original_size
            self.stats["total_compressed_size"] += total_compressed_size

            logger.info(
                f"Compressed {len(vectors)} vectors: "
                f"{original_size} -> {total_compressed_size} bytes "
                f"({total_compressed_size/original_size:.3f} ratio) in {compression_time:.2f}ms"
            )

            return compressed_vectors

        except Exception as e:
            logger.error(f"Vector compression failed: {e}")
            raise

    async def decompress_vectors(
        self, compressed_vectors: List[CompressedVector]
    ) -> np.ndarray:
        """Decompress vectors back to original format."""
        start_time = time.time()

        try:
            if not compressed_vectors:
                return np.array([])

            method = compressed_vectors[0].method

            if method == CompressionMethod.NONE:
                # No decompression needed
                vectors = []
                for cv in compressed_vectors:
                    vector = np.frombuffer(cv.data, dtype=np.float32)
                    vectors.append(vector)
                result = np.array(vectors)

            elif method == CompressionMethod.PCA:
                # PCA decompression
                data_bytes = self._decompress_general(compressed_vectors[0].data, "lz4")
                quantized_vectors = self.quantizer.inverse_scalar_quantization(
                    data_bytes, compressed_vectors[0].quantization_info
                )
                result = self.quantizer.inverse_transform_pca(quantized_vectors)

            elif method == CompressionMethod.RANDOM_PROJECTION:
                # Random projection - note: this is lossy and approximate
                data_bytes = self._decompress_general(compressed_vectors[0].data, "lz4")
                result = self.quantizer.inverse_scalar_quantization(
                    data_bytes, compressed_vectors[0].quantization_info
                )
                # Note: Random projection is not exactly invertible
                logger.warning("Random projection decompression is approximate")

            elif method == CompressionMethod.PRODUCT_QUANTIZATION:
                # Product quantization decompression
                cv = compressed_vectors[0]
                n_vectors = cv.quantization_info["n_vectors"]
                result = self.quantizer.inverse_transform_product_quantization(
                    cv.data, n_vectors
                )

            elif method == CompressionMethod.SCALAR_QUANTIZATION:
                # Scalar quantization decompression
                data_bytes = self._decompress_general(
                    compressed_vectors[0].data, "blosc2"
                )
                result = self.quantizer.inverse_scalar_quantization(
                    data_bytes, compressed_vectors[0].quantization_info
                )

            elif method == CompressionMethod.HYBRID:
                # Hybrid decompression
                cv = compressed_vectors[0]

                # Step 1: General decompression
                data_bytes = self._decompress_general(cv.data, "blosc2")

                # Step 2: Scalar dequantization
                vectors = self.quantizer.inverse_scalar_quantization(
                    data_bytes, cv.quantization_info
                )

                # Step 3: Dimensionality restoration (if PCA was used)
                if (
                    self.quantizer.pca_model is not None
                    and cv.original_dimensions != cv.compressed_dimensions
                ):
                    result = self.quantizer.inverse_transform_pca(vectors)
                else:
                    result = vectors

            else:
                raise ValueError(f"Unsupported compression method: {method}")

            decompression_time = (time.time() - start_time) * 1000

            # Update stats
            self.stats["decompressions"] += len(compressed_vectors)
            self.stats["total_decompression_time"] += decompression_time

            logger.info(
                f"Decompressed {len(compressed_vectors)} vectors in {decompression_time:.2f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Vector decompression failed: {e}")
            raise

    def get_compression_stats(self) -> Dict:
        """Get compression statistics."""
        stats = self.stats.copy()

        if stats["compressions"] > 0:
            stats["avg_compression_time_ms"] = (
                stats["total_compression_time"] / stats["compressions"]
            )
            stats["avg_compression_ratio"] = (
                stats["total_compressed_size"] / stats["total_original_size"]
            )
        else:
            stats["avg_compression_time_ms"] = 0
            stats["avg_compression_ratio"] = 0

        if stats["decompressions"] > 0:
            stats["avg_decompression_time_ms"] = (
                stats["total_decompression_time"] / stats["decompressions"]
            )
        else:
            stats["avg_decompression_time_ms"] = 0

        return stats

    def benchmark_compression_methods(
        self, test_vectors: np.ndarray
    ) -> Dict[str, CompressionResult]:
        """Benchmark different compression methods."""
        results = {}
        original_size = test_vectors.nbytes

        methods_to_test = [
            CompressionMethod.NONE,
            CompressionMethod.SCALAR_QUANTIZATION,
            CompressionMethod.PCA,
            CompressionMethod.PRODUCT_QUANTIZATION,
            CompressionMethod.HYBRID,
        ]

        for method in methods_to_test:
            try:
                # Create temporary config
                temp_config = CompressionConfig(method=method)
                temp_engine = CompressionEngine(temp_config)

                # Compression
                start_time = time.time()
                compressed = asyncio.run(temp_engine.compress_vectors(test_vectors))
                compression_time = (time.time() - start_time) * 1000

                compressed_size = sum(len(cv.data) for cv in compressed)

                # Decompression
                start_time = time.time()
                decompressed = asyncio.run(temp_engine.decompress_vectors(compressed))
                decompression_time = (time.time() - start_time) * 1000

                # Calculate accuracy loss
                if method != CompressionMethod.NONE:
                    mse = np.mean((test_vectors - decompressed) ** 2)
                    accuracy_loss = mse / np.mean(test_vectors**2)
                else:
                    accuracy_loss = 0.0

                results[method.value] = CompressionResult(
                    original_size_bytes=original_size,
                    compressed_size_bytes=compressed_size,
                    compression_ratio=compressed_size / original_size,
                    compression_time_ms=compression_time,
                    decompression_time_ms=decompression_time,
                    accuracy_loss=accuracy_loss,
                    method_used=method,
                    metadata={
                        "vectors_count": len(test_vectors),
                        "dimensions": test_vectors.shape[1],
                    },
                )

            except Exception as e:
                logger.error(f"Benchmark failed for method {method.value}: {e}")
                results[method.value] = CompressionResult(
                    original_size_bytes=original_size,
                    compressed_size_bytes=original_size,
                    compression_ratio=1.0,
                    compression_time_ms=float("inf"),
                    decompression_time_ms=float("inf"),
                    accuracy_loss=1.0,
                    method_used=method,
                    metadata={"error": str(e)},
                )

        return results


class DatabaseCompressionManager:
    """Manages compressed vectors in the database."""

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.pool = None
        self.compression_engine = CompressionEngine()

    async def initialize(self):
        """Initialize database connection."""
        self.pool = await asyncpg.create_pool(
            self.db_url, min_size=5, max_size=20, command_timeout=300
        )

        # Create compressed vectors table if it doesn't exist
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mem0_vectors.compressed_memories (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    original_memory_id UUID REFERENCES mem0_vectors.memories(id) ON DELETE CASCADE,
                    compressed_data BYTEA NOT NULL,
                    compression_method TEXT NOT NULL,
                    original_dimensions INTEGER NOT NULL,
                    compressed_dimensions INTEGER,
                    quantization_info JSONB,
                    checksum TEXT NOT NULL,
                    compression_ratio FLOAT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS compressed_memories_original_id_idx
                ON mem0_vectors.compressed_memories (original_memory_id);

                CREATE INDEX IF NOT EXISTS compressed_memories_method_idx
                ON mem0_vectors.compressed_memories (compression_method);
            """
            )

    async def cleanup(self):
        """Cleanup database connections."""
        if self.pool:
            await self.pool.close()

    async def compress_and_store(
        self, memory_ids: List[str], user_id: str
    ) -> Dict[str, bool]:
        """Compress vectors and store compressed versions."""
        results = {}

        async with self.pool.acquire() as conn:
            # Get original vectors
            rows = await conn.fetch(
                """
                SELECT id, embedding
                FROM mem0_vectors.memories
                WHERE id = ANY($1) AND user_id = $2 AND embedding IS NOT NULL
            """,
                memory_ids,
                user_id,
            )

            if not rows:
                return results

            # Extract vectors and IDs
            vectors = np.array([row["embedding"] for row in rows])
            db_memory_ids = [str(row["id"]) for row in rows]

            try:
                # Compress vectors
                compressed_vectors = await self.compression_engine.compress_vectors(
                    vectors
                )

                # Store compressed vectors
                for memory_id, compressed_vector in zip(
                    db_memory_ids, compressed_vectors
                ):
                    try:
                        await conn.execute(
                            """
                            INSERT INTO mem0_vectors.compressed_memories
                            (original_memory_id, compressed_data, compression_method,
                             original_dimensions, compressed_dimensions, quantization_info,
                             checksum, compression_ratio, metadata)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                            ON CONFLICT (original_memory_id) DO UPDATE SET
                                compressed_data = EXCLUDED.compressed_data,
                                compression_method = EXCLUDED.compression_method,
                                compressed_dimensions = EXCLUDED.compressed_dimensions,
                                quantization_info = EXCLUDED.quantization_info,
                                checksum = EXCLUDED.checksum,
                                compression_ratio = EXCLUDED.compression_ratio,
                                metadata = EXCLUDED.metadata,
                                created_at = NOW()
                        """,
                            memory_id,
                            compressed_vector.data,
                            compressed_vector.method.value,
                            compressed_vector.original_dimensions,
                            compressed_vector.compressed_dimensions,
                            compressed_vector.quantization_info,
                            compressed_vector.checksum,
                            len(compressed_vector.data)
                            / (compressed_vector.original_dimensions * 4),
                            compressed_vector.metadata,
                        )

                        results[memory_id] = True

                    except Exception as e:
                        logger.error(
                            f"Failed to store compressed vector for {memory_id}: {e}"
                        )
                        results[memory_id] = False

            except Exception as e:
                logger.error(f"Compression failed: {e}")
                for memory_id in db_memory_ids:
                    results[memory_id] = False

        return results

    async def get_compression_analytics(self, user_id: str) -> Dict:
        """Get compression analytics for user."""
        async with self.pool.acquire() as conn:
            stats = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as compressed_count,
                    AVG(compression_ratio) as avg_compression_ratio,
                    SUM(original_dimensions * 4) as total_original_size,
                    SUM(LENGTH(compressed_data)) as total_compressed_size,
                    COUNT(DISTINCT compression_method) as methods_used
                FROM mem0_vectors.compressed_memories cm
                JOIN mem0_vectors.memories m ON cm.original_memory_id = m.id
                WHERE m.user_id = $1
            """,
                user_id,
            )

            method_stats = await conn.fetch(
                """
                SELECT
                    compression_method,
                    COUNT(*) as count,
                    AVG(compression_ratio) as avg_ratio
                FROM mem0_vectors.compressed_memories cm
                JOIN mem0_vectors.memories m ON cm.original_memory_id = m.id
                WHERE m.user_id = $1
                GROUP BY compression_method
            """,
                user_id,
            )

            return {
                "compressed_count": stats["compressed_count"] or 0,
                "avg_compression_ratio": float(stats["avg_compression_ratio"] or 0),
                "total_original_size_bytes": stats["total_original_size"] or 0,
                "total_compressed_size_bytes": stats["total_compressed_size"] or 0,
                "methods_used": stats["methods_used"] or 0,
                "method_breakdown": [
                    {
                        "method": row["compression_method"],
                        "count": row["count"],
                        "avg_ratio": float(row["avg_ratio"]),
                    }
                    for row in method_stats
                ],
            }


# Example usage and testing
async def main():
    """Test the compression system."""
    import uuid

    DB_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/mem0ai")

    # Test compression engine

    # Generate test vectors
    test_vectors = np.random.randn(100, 1536).astype(np.float32)

    # Test different compression methods
    config = CompressionConfig(method=CompressionMethod.HYBRID)
    engine = CompressionEngine(config)

    # Benchmark all methods
    benchmark_results = engine.benchmark_compression_methods(test_vectors)

    for _method, result in benchmark_results.items():
        if "error" in result.metadata:
            pass

    # Test database compression manager

    manager = DatabaseCompressionManager(DB_URL)

    try:
        await manager.initialize()

        # Test compression and storage
        test_memory_ids = [str(uuid.uuid4()) for _ in range(5)]
        await manager.compress_and_store(test_memory_ids, "test_user")


        # Get analytics
        await manager.get_compression_analytics("test_user")

    except Exception as e:
        logger.error(f"Database testing failed: {e}")
    finally:
        await manager.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
