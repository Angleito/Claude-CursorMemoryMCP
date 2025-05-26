#!/usr/bin/env python3
"""High-Performance Batch Processing System for Large Memory Sets.

Production-grade batch processing with parallel execution and optimized I/O.
"""

import asyncio
import contextlib
import json
import logging
import multiprocessing as mp
import os
import queue
import signal
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import asyncpg
import numpy as np
import psutil

# Batch processing constants
SINGLE_WORKER_COUNT = 1
NO_FAILED_ITEMS = 0
MIN_PROCESSED_ITEMS = 0
MIN_PROCESSING_TIME = 0
CHECKPOINT_MODULO = 0

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BatchOperation(Enum):
    """Supported batch operations."""

    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    SEARCH = "search"
    EMBEDDING_GENERATION = "embedding_generation"
    INDEX_REBUILD = "index_rebuild"
    DEDUPLICATION = "deduplication"
    MIGRATION = "migration"


class BatchStatus(Enum):
    """Batch processing status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class BatchConfig:
    """Configuration for batch processing."""

    batch_size: int = 1000
    max_workers: int = None  # Auto-detect based on CPU count
    chunk_size: int = 100  # Size for parallel processing chunks
    max_memory_mb: int = 2048  # Maximum memory usage
    temp_dir: str = tempfile.gettempdir() + "/mem0ai_batch"
    enable_compression: bool = True
    enable_checkpointing: bool = True
    checkpoint_interval: int = 10000  # Items processed
    retry_attempts: int = 3
    retry_delay: float = 1.0
    progress_callback: Callable | None = None
    error_callback: Callable | None = None


@dataclass
class BatchItem:
    """Single item in a batch operation."""

    id: str
    operation: BatchOperation
    data: dict[str, Any]
    priority: int = 1
    retry_count: int = 0
    user_id: str | None = None
    metadata: dict | None = None


@dataclass
class BatchJob:
    """Batch job definition."""

    job_id: str
    operation: BatchOperation
    items: list[BatchItem]
    config: BatchConfig
    status: BatchStatus = BatchStatus.PENDING
    created_at: datetime = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress: float = 0.0
    processed_count: int = 0
    failed_count: int = 0
    error_log: list[str] = None
    result_data: dict | None = None

    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.error_log is None:
            self.error_log = []


@dataclass
class BatchResult:
    """Result from batch processing."""

    job_id: str
    status: BatchStatus
    total_items: int
    processed_items: int
    failed_items: int
    processing_time_seconds: float
    throughput_items_per_second: float
    memory_usage_mb: float
    error_summary: list[str]
    checkpoint_files: list[str]


class ProgressTracker:
    """Thread-safe progress tracking."""

    def __init__(self, total_items: int, callback: Callable | None = None):
        """Initialize progress tracker.

        Args:
            total_items: Total number of items to process.
            callback: Optional callback function for progress updates.
        """
        self.total_items = total_items
        self.processed_items = 0
        self.failed_items = 0
        self.callback = callback
        self.lock = threading.Lock()

    def update(self, processed: int = 0, failed: int = 0):
        """Update progress counters."""
        with self.lock:
            self.processed_items += processed
            self.failed_items += failed

            if self.callback:
                progress = (self.processed_items + self.failed_items) / self.total_items
                self.callback(progress, self.processed_items, self.failed_items)

    def get_progress(self) -> tuple[float, int, int]:
        """Get current progress."""
        with self.lock:
            progress = (self.processed_items + self.failed_items) / self.total_items
            return progress, self.processed_items, self.failed_items


class CheckpointManager:
    """Manages checkpointing for batch operations."""

    def __init__(self, job_id: str, temp_dir: str):
        """Initialize checkpoint manager.

        Args:
            job_id: Unique identifier for the batch job.
            temp_dir: Directory path for storing checkpoint files.
        """
        self.job_id = job_id
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_files = []

    def save_checkpoint(
        self, chunk_id: int, processed_items: list[dict], failed_items: list[dict]
    ) -> str:
        """Save checkpoint data."""
        checkpoint_data = {
            "chunk_id": chunk_id,
            "processed_items": processed_items,
            "failed_items": failed_items,
            "timestamp": datetime.now().isoformat(),
        }

        filename = f"checkpoint_{self.job_id}_{chunk_id}.json"
        filepath = self.temp_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, default=str)

        self.checkpoint_files.append(str(filepath))
        return str(filepath)

    def load_checkpoint(self, filepath: str) -> dict:
        """Load checkpoint data."""
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)

    def cleanup(self):
        """Remove all checkpoint files."""
        for filepath in self.checkpoint_files:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(filepath)

        # Remove temp directory if empty
        with contextlib.suppress(OSError):
            self.temp_dir.rmdir()


class MemoryMonitor:
    """Monitors memory usage during batch processing."""

    def __init__(self, max_memory_mb: int):
        """Initialize memory monitor.

        Args:
            max_memory_mb: Maximum memory usage limit in megabytes.
        """
        self.max_memory_mb = max_memory_mb
        self.process = psutil.Process()

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / (1024 * 1024)

    def check_memory_limit(self) -> bool:
        """Check if memory usage exceeds limit."""
        current_usage = self.get_memory_usage_mb()
        return current_usage > self.max_memory_mb

    def wait_for_memory(self, max_wait: float = 60.0):
        """Wait for memory usage to decrease."""
        start_time = time.time()
        while self.check_memory_limit() and (time.time() - start_time) < max_wait:
            time.sleep(1.0)

        if self.check_memory_limit():
            raise MemoryError(f"Memory usage still above limit after {max_wait}s")


class DatabaseBatchProcessor:
    """Optimized database batch operations."""

    def __init__(self, db_url: str):
        """Initialize database batch processor.

        Args:
            db_url: Database connection URL.
        """
        self.db_url = db_url
        self.pool = None

    async def initialize(self):
        """Initialize database connection pool."""
        self.pool = await asyncpg.create_pool(
            self.db_url, min_size=10, max_size=50, command_timeout=300
        )

    async def cleanup(self):
        """Cleanup database connections."""
        if self.pool:
            await self.pool.close()

    async def _process_individual_inserts(
        self, conn, query: str, chunk, chunk_items
    ) -> tuple[list[str], list[str]]:
        """Process items individually to identify failures."""
        success_ids = []
        failed_ids = []

        for _j, (data_row, item) in enumerate(zip(chunk, chunk_items, strict=False)):
            try:
                result = await conn.fetchrow(query, *data_row)
                if result:
                    success_ids.append(str(result["id"]))
                else:
                    failed_ids.append(item.id)
            except Exception as item_error:
                logger.error(
                    f"Individual insert failed for item {item.id}: {item_error}"
                )
                failed_ids.append(item.id)

        return success_ids, failed_ids

    async def batch_insert_memories(
        self, items: list[BatchItem]
    ) -> tuple[list[str], list[str]]:
        """Batch insert memory records."""
        success_ids: list[str] = []
        failed_ids: list[str] = []

        # Prepare batch data with proper type conversion
        batch_data = []
        for item in items:
            data = item.data
            embedding = data.get("embedding")
            # Convert numpy array to list if needed
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

            batch_data.append(
                (
                    data.get("user_id"),
                    data.get("memory_text"),
                    embedding,
                    data.get("metadata", {}),
                    data.get("memory_type", "general"),
                    data.get("importance_score", 0.5),
                )
            )

        async with self.pool.acquire() as conn:
            try:
                # Use COPY for maximum performance
                query = """
                    INSERT INTO mem0_vectors.memories
                    (user_id, memory_text, embedding, metadata, memory_type, importance_score)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING id
                """

                # Process in smaller chunks to avoid memory issues
                chunk_size = 100
                for i in range(0, len(batch_data), chunk_size):
                    chunk = batch_data[i : i + chunk_size]
                    chunk_items = items[i : i + chunk_size]

                    try:
                        async with conn.transaction():
                            results = await conn.fetch(query, *zip(*chunk, strict=False))
                            for _, result in enumerate(results):
                                success_ids.append(str(result["id"]))

                    except Exception as e:
                        logger.error("Batch insert chunk failed: %s", e)
                        # Process items individually to identify failures
                        chunk_success, chunk_failed = await self._process_individual_inserts(
                            conn, query, chunk, chunk_items
                        )
                        success_ids.extend(chunk_success)
                        failed_ids.extend(chunk_failed)

            except Exception as e:
                logger.error("Batch insert failed: %s", e)
                failed_ids.extend([item.id for item in items])

        return success_ids, failed_ids

    async def batch_update_embeddings(
        self, items: list[BatchItem]
    ) -> tuple[list[str], list[str]]:
        """Batch update embeddings."""
        success_ids: list[str] = []
        failed_ids: list[str] = []

        async with self.pool.acquire() as conn:
            try:
                # Prepare update data with proper type conversion
                update_data = []
                for item in items:
                    data = item.data
                    embedding = data.get("embedding")
                    # Convert numpy array to list if needed
                    if isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()
                    update_data.append((embedding, data.get("memory_id")))

                # Batch update using prepared statement
                stmt = await conn.prepare(
                    """
                    UPDATE mem0_vectors.memories
                    SET embedding = $1, updated_at = NOW()
                    WHERE id = $2
                """
                )

                # Execute batch update
                results = await stmt.executemany(update_data)

                # Check which updates succeeded
                for i, item in enumerate(items):
                    if i < len(results):
                        success_ids.append(item.id)
                    else:
                        failed_ids.append(item.id)

            except Exception as e:
                logger.error("Batch update failed: %s", e)
                failed_ids.extend([item.id for item in items])

        return success_ids, failed_ids

    async def batch_search_similar(self, items: list[BatchItem]) -> list[dict[str, Any]]:
        """Batch similarity search."""
        results: list[dict[str, Any]] = []

        async with self.pool.acquire() as conn:
            for item in items:
                try:
                    data = item.data
                    query_embedding = data.get("query_embedding")
                    # Convert numpy array to list if needed
                    if isinstance(query_embedding, np.ndarray):
                        query_embedding = query_embedding.tolist()

                    user_id = data.get("user_id")
                    top_k = data.get("top_k", 10)
                    threshold = data.get("similarity_threshold", 0.7)

                    # Perform similarity search
                    search_results = await conn.fetch(
                        """
                        SELECT
                            id,
                            memory_text,
                            1 - (embedding <=> $1) as similarity,
                            metadata
                        FROM mem0_vectors.memories
                        WHERE user_id = $2
                            AND 1 - (embedding <=> $1) >= $3
                        ORDER BY embedding <=> $1
                        LIMIT $4
                    """,
                        query_embedding,
                        user_id,
                        threshold,
                        top_k,
                    )

                    item_results: list[dict[str, Any]] = []
                    for row in search_results:
                        item_results.append(
                            {
                                "memory_id": str(row["id"]),
                                "memory_text": row["memory_text"],
                                "similarity": float(row["similarity"]),
                                "metadata": row["metadata"],
                            }
                        )

                    results.append(
                        {"item_id": item.id, "results": item_results, "success": True}
                    )

                except Exception as e:
                    logger.error("Search failed for item {item.id}: %s", e)
                    results.append(
                        {
                            "item_id": item.id,
                            "results": [],
                            "success": False,
                            "error": str(e),
                        }
                    )

        return results


class BatchProcessor:
    """Main batch processing engine."""

    def __init__(self, db_url: str):
        """Initialize batch processor.

        Args:
            db_url: Database connection URL.
        """
        self.db_url = db_url
        self.db_processor = DatabaseBatchProcessor(db_url)
        self.active_jobs = {}
        self.job_queue = queue.PriorityQueue()
        self.shutdown_event = threading.Event()

    async def initialize(self):
        """Initialize the batch processor."""
        await self.db_processor.initialize()

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("Batch processor initialized")

    async def cleanup(self):
        """Cleanup resources."""
        self.shutdown_event.set()
        await self.db_processor.cleanup()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Received signal %s, initiating graceful shutdown...", signum)
        self.shutdown_event.set()

    def submit_job(self, job: BatchJob) -> str:
        """Submit a batch job for processing."""
        # Validate job
        if not job.items:
            raise ValueError("Job must contain at least one item")

        if job.config.max_workers is None:
            job.config.max_workers = min(mp.cpu_count(), len(job.items))

        # Add to queue with priority (lower number = higher priority)
        priority = (
            1 if job.operation in [BatchOperation.INSERT, BatchOperation.UPDATE] else 2
        )
        self.job_queue.put((priority, time.time(), job.job_id, job))

        self.active_jobs[job.job_id] = job

        logger.info("Submitted batch job {job.job_id} with %s items", len(job.items))
        return job.job_id

    async def process_job(self, job: BatchJob) -> BatchResult:
        """Process a single batch job."""
        start_time = time.time()
        job.status = BatchStatus.RUNNING
        job.started_at = datetime.now()

        # Initialize components
        checkpoint_manager = CheckpointManager(job.job_id, job.config.temp_dir)
        memory_monitor = MemoryMonitor(job.config.max_memory_mb)
        progress_tracker = ProgressTracker(len(job.items), job.config.progress_callback)

        processed_items = 0
        failed_items = 0
        error_log = []

        try:
            # Split items into chunks for parallel processing
            chunks = self._create_chunks(job.items, job.config.chunk_size)

            # Process chunks
            if job.config.max_workers == SINGLE_WORKER_COUNT:
                processed_items, failed_items, error_log = await self._process_sequential(
                    chunks, job, progress_tracker, memory_monitor, checkpoint_manager
                )
            else:
                # Parallel processing
                processed_items, failed_items, error_log = await self._process_parallel(
                    chunks,
                    job.operation,
                    job.config,
                    progress_tracker,
                    memory_monitor,
                    checkpoint_manager,
                )

                # Determine final status
            job.status = self._determine_job_status(processed_items, failed_items)

        except Exception as e:
            logger.error("Job {job.job_id} failed: %s", e)
            job.status = BatchStatus.FAILED
            error_log.append(str(e))
            failed_items = len(job.items) - processed_items

        finally:
            job.completed_at = datetime.now()
            processing_time = time.time() - start_time

            # Create result
            result = BatchResult(
                job_id=job.job_id,
                status=job.status,
                total_items=len(job.items),
                processed_items=processed_items,
                failed_items=failed_items,
                processing_time_seconds=processing_time,
                throughput_items_per_second=(
                    processed_items / processing_time if processing_time > MIN_PROCESSING_TIME else MIN_PROCESSING_TIME
                ),
                memory_usage_mb=memory_monitor.get_memory_usage_mb(),
                error_summary=error_log[:100],  # Limit error log size
                checkpoint_files=checkpoint_manager.checkpoint_files.copy(),
            )

            # Cleanup if successful
            if job.status == BatchStatus.COMPLETED and not job.config.enable_checkpointing:
                checkpoint_manager.cleanup()

            # Update job
            job.progress = 1.0
            job.processed_count = processed_items
            job.failed_count = failed_items
            job.error_log = error_log
            job.result_data = asdict(result)

            logger.info(
                f"Job {job.job_id} completed: {processed_items} processed, "
                f"{failed_items} failed in {processing_time:.2f}s"
            )

        return result

    def _create_chunks(
        self, items: list[BatchItem], chunk_size: int
    ) -> list[list[BatchItem]]:
        """Split items into processing chunks."""
        chunks = []
        for i in range(0, len(items), chunk_size):
            chunks.append(items[i : i + chunk_size])
        return chunks

    async def _process_sequential(
        self,
        chunks: list[list[BatchItem]],
        job: BatchJob,
        progress_tracker: ProgressTracker,
        memory_monitor: MemoryMonitor,
        checkpoint_manager: CheckpointManager,
    ) -> tuple[int, int, list[str]]:
        """Process chunks sequentially."""
        processed_items = 0
        failed_items = 0
        error_log = []

        for chunk_id, chunk in enumerate(chunks):
            if self.shutdown_event.is_set():
                job.status = BatchStatus.CANCELLED
                break

            chunk_result = await self._process_chunk(
                chunk, job.operation, chunk_id, job.config
            )

            processed_items += chunk_result["processed"]
            failed_items += chunk_result["failed"]
            error_log.extend(chunk_result["errors"])

            progress_tracker.update(
                chunk_result["processed"], chunk_result["failed"]
            )

            # Checkpoint if enabled
            if (
                job.config.enable_checkpointing
                and processed_items % job.config.checkpoint_interval == CHECKPOINT_MODULO
            ):
                checkpoint_manager.save_checkpoint(
                    chunk_id,
                    chunk_result["processed_data"],
                    chunk_result["failed_data"],
                )

            # Check memory usage
            if memory_monitor.check_memory_limit():
                logger.warning("Memory limit exceeded, waiting...")
                memory_monitor.wait_for_memory()

        return processed_items, failed_items, error_log

    def _determine_job_status(self, processed_items: int, failed_items: int) -> BatchStatus:
        """Determine final job status based on results."""
        if self.shutdown_event.is_set():
            return BatchStatus.CANCELLED
        elif failed_items == NO_FAILED_ITEMS:
            return BatchStatus.COMPLETED
        elif processed_items > MIN_PROCESSED_ITEMS:
            return BatchStatus.COMPLETED  # Partial success
        else:
            return BatchStatus.FAILED

    async def _process_chunk(
        self,
        chunk: list[BatchItem],
        operation: BatchOperation,
        chunk_id: int,
        config: BatchConfig,
    ) -> dict:
        """Process a single chunk of items."""
        processed_data = []
        failed_data = []
        errors = []

        try:
            if operation == BatchOperation.INSERT:
                success_ids, failed_ids = await self.db_processor.batch_insert_memories(
                    chunk
                )

                for item in chunk:
                    if item.id in failed_ids:
                        failed_data.append(asdict(item))
                        errors.append(f"Insert failed for item {item.id}")
                    else:
                        processed_data.append(asdict(item))

            elif operation == BatchOperation.UPDATE:
                success_ids, failed_ids = (
                    await self.db_processor.batch_update_embeddings(chunk)
                )

                for item in chunk:
                    if item.id in failed_ids:
                        failed_data.append(asdict(item))
                        errors.append(f"Update failed for item {item.id}")
                    else:
                        processed_data.append(asdict(item))

            elif operation == BatchOperation.SEARCH:
                search_results = await self.db_processor.batch_search_similar(chunk)

                for result in search_results:
                    if result["success"]:
                        processed_data.append(result)
                    else:
                        failed_data.append(result)
                        errors.append(result.get("error", "Unknown search error"))

            else:
                raise ValueError(f"Unsupported operation: {operation}")

        except Exception as e:
            logger.error("Chunk {chunk_id} processing failed: %s", e)
            failed_data.extend([asdict(item) for item in chunk])
            errors.append(f"Chunk processing error: {e!s}")

        return {
            "processed": len(processed_data),
            "failed": len(failed_data),
            "errors": errors,
            "processed_data": processed_data,
            "failed_data": failed_data,
        }

    async def _process_parallel(
        self,
        chunks: list[list[BatchItem]],
        operation: BatchOperation,
        config: BatchConfig,
        progress_tracker: ProgressTracker,
        memory_monitor: MemoryMonitor,
        checkpoint_manager: CheckpointManager,
    ) -> tuple[int, int, list[str]]:
        """Process chunks in parallel."""
        total_processed = 0
        total_failed = 0
        all_errors = []

        # Use ThreadPoolExecutor for async operations
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            # Submit all chunks
            future_to_chunk = {}
            for chunk_id, chunk in enumerate(chunks):
                future = executor.submit(
                    asyncio.run, self._process_chunk(chunk, operation, chunk_id, config)
                )
                future_to_chunk[future] = (chunk_id, chunk)

            # Process completed chunks
            for future in as_completed(future_to_chunk):
                if self.shutdown_event.is_set():
                    # Cancel remaining futures
                    for f in future_to_chunk:
                        f.cancel()
                    break

                chunk_id, chunk = future_to_chunk[future]

                try:
                    result = future.result()
                    total_processed += result["processed"]
                    total_failed += result["failed"]
                    all_errors.extend(result["errors"])

                    progress_tracker.update(result["processed"], result["failed"])

                    # Checkpoint if enabled
                    if (
                        config.enable_checkpointing
                        and total_processed % config.checkpoint_interval == CHECKPOINT_MODULO
                    ):
                        checkpoint_manager.save_checkpoint(
                            chunk_id, result["processed_data"], result["failed_data"]
                        )

                    # Check memory usage
                    if memory_monitor.check_memory_limit():
                        logger.warning(
                            "Memory limit exceeded during parallel processing"
                        )
                        memory_monitor.wait_for_memory()

                except Exception as e:
                    logger.error("Parallel chunk processing failed: %s", e)
                    total_failed += len(chunk)
                    all_errors.append(f"Parallel processing error: {e!s}")

        return total_processed, total_failed, all_errors

    async def get_job_status(self, job_id: str) -> BatchJob | None:
        """Get status of a batch job."""
        return self.active_jobs.get(job_id)

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a batch job."""
        job = self.active_jobs.get(job_id)
        if job and job.status in [BatchStatus.PENDING, BatchStatus.RUNNING]:
            job.status = BatchStatus.CANCELLED
            return True
        return False

    async def list_jobs(
        self, status_filter: BatchStatus | None = None
    ) -> list[BatchJob]:
        """List all jobs, optionally filtered by status."""
        jobs = list(self.active_jobs.values())
        if status_filter:
            jobs = [job for job in jobs if job.status == status_filter]
        return jobs

    async def cleanup_completed_jobs(self, max_age_hours: int = 24):
        """Clean up old completed jobs."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        jobs_to_remove = []
        for job_id, job in self.active_jobs.items():
            if (
                job.status
                in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED]
                and job.completed_at
                and job.completed_at < cutoff_time
            ):
                jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            del self.active_jobs[job_id]

        logger.info("Cleaned up %s old jobs", len(jobs_to_remove))


# Example usage
async def main():
    """Test the batch processor."""
    import uuid

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL environment variable is required")
        return

    processor = BatchProcessor(db_url)

    try:
        await processor.initialize()

        # Create test batch job
        items = []
        for i in range(100):
            item = BatchItem(
                id=str(uuid.uuid4()),
                operation=BatchOperation.INSERT,
                data={
                    "user_id": "test_user",
                    "memory_text": f"Test memory {i}",
                    "embedding": np.random.rand(1536).tolist(),
                    "metadata": {"test": True, "index": i},
                    "memory_type": "test",
                    "importance_score": np.random.rand(),
                },
            )
            items.append(item)

        # Configure batch processing
        config = BatchConfig(
            batch_size=50,
            max_workers=4,
            chunk_size=25,
            enable_checkpointing=True,
            progress_callback=lambda p, proc, fail: logger.info(
                "Progress",
                percentage=f"{p:.1%}",
                processed=proc,
                failed=fail
            ),
        )

        # Create and submit job
        job = BatchJob(
            job_id=str(uuid.uuid4()),
            operation=BatchOperation.INSERT,
            items=items,
            config=config,
        )

        processor.submit_job(job)

        # Process the job
        result = await processor.process_job(job)
        logger.info("Batch job result: {result.status}, {result.processed_items} processed, %s failed", result.failed_items)
        logger.info("Job completed: {result.status}, processed: {result.processed_items}, failed: %s", result.failed_items)


        if result.error_summary:
            for error in result.error_summary[:5]:
                logger.error("Processing error: %s", error)

    finally:
        await processor.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
