#!/usr/bin/env python3
"""
High-Performance Batch Processing System for Large Memory Sets
Production-grade batch processing with parallel execution and optimized I/O
"""

import asyncio
import asyncpg
import numpy as np
import time
import logging
import json
from typing import List, Dict, Optional, Tuple, Any, Callable, Iterator
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
import queue
import psutil
import os
import pickle
import tempfile
import shutil
from pathlib import Path
import hashlib
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchOperation(Enum):
    """Supported batch operations"""
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    SEARCH = "search"
    EMBEDDING_GENERATION = "embedding_generation"
    INDEX_REBUILD = "index_rebuild"
    DEDUPLICATION = "deduplication"
    MIGRATION = "migration"

class BatchStatus(Enum):
    """Batch processing status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    batch_size: int = 1000
    max_workers: int = None  # Auto-detect based on CPU count
    chunk_size: int = 100  # Size for parallel processing chunks
    max_memory_mb: int = 2048  # Maximum memory usage
    temp_dir: str = "/tmp/mem0ai_batch"
    enable_compression: bool = True
    enable_checkpointing: bool = True
    checkpoint_interval: int = 10000  # Items processed
    retry_attempts: int = 3
    retry_delay: float = 1.0
    progress_callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None

@dataclass
class BatchItem:
    """Single item in a batch operation"""
    id: str
    operation: BatchOperation
    data: Dict[str, Any]
    priority: int = 1
    retry_count: int = 0
    user_id: Optional[str] = None
    metadata: Optional[Dict] = None

@dataclass
class BatchJob:
    """Batch job definition"""
    job_id: str
    operation: BatchOperation
    items: List[BatchItem]
    config: BatchConfig
    status: BatchStatus = BatchStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    processed_count: int = 0
    failed_count: int = 0
    error_log: List[str] = None
    result_data: Optional[Dict] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.error_log is None:
            self.error_log = []

@dataclass
class BatchResult:
    """Result from batch processing"""
    job_id: str
    status: BatchStatus
    total_items: int
    processed_items: int
    failed_items: int
    processing_time_seconds: float
    throughput_items_per_second: float
    memory_usage_mb: float
    error_summary: List[str]
    checkpoint_files: List[str]

class ProgressTracker:
    """Thread-safe progress tracking"""
    
    def __init__(self, total_items: int, callback: Optional[Callable] = None):
        self.total_items = total_items
        self.processed_items = 0
        self.failed_items = 0
        self.callback = callback
        self.lock = threading.Lock()
        
    def update(self, processed: int = 0, failed: int = 0):
        """Update progress counters"""
        with self.lock:
            self.processed_items += processed
            self.failed_items += failed
            
            if self.callback:
                progress = (self.processed_items + self.failed_items) / self.total_items
                self.callback(progress, self.processed_items, self.failed_items)
    
    def get_progress(self) -> Tuple[float, int, int]:
        """Get current progress"""
        with self.lock:
            progress = (self.processed_items + self.failed_items) / self.total_items
            return progress, self.processed_items, self.failed_items

class CheckpointManager:
    """Manages checkpointing for batch operations"""
    
    def __init__(self, job_id: str, temp_dir: str):
        self.job_id = job_id
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_files = []
        
    def save_checkpoint(self, chunk_id: int, processed_items: List[Dict], 
                       failed_items: List[Dict]) -> str:
        """Save checkpoint data"""
        checkpoint_data = {
            'chunk_id': chunk_id,
            'processed_items': processed_items,
            'failed_items': failed_items,
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f"checkpoint_{self.job_id}_{chunk_id}.pkl"
        filepath = self.temp_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint_data, f)
            
        self.checkpoint_files.append(str(filepath))
        return str(filepath)
    
    def load_checkpoint(self, filepath: str) -> Dict:
        """Load checkpoint data"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def cleanup(self):
        """Remove all checkpoint files"""
        for filepath in self.checkpoint_files:
            try:
                os.unlink(filepath)
            except FileNotFoundError:
                pass
        
        # Remove temp directory if empty
        try:
            self.temp_dir.rmdir()
        except OSError:
            pass

class MemoryMonitor:
    """Monitors memory usage during batch processing"""
    
    def __init__(self, max_memory_mb: int):
        self.max_memory_mb = max_memory_mb
        self.process = psutil.Process()
        
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / (1024 * 1024)
    
    def check_memory_limit(self) -> bool:
        """Check if memory usage exceeds limit"""
        current_usage = self.get_memory_usage_mb()
        return current_usage > self.max_memory_mb
    
    def wait_for_memory(self, max_wait: float = 60.0):
        """Wait for memory usage to decrease"""
        start_time = time.time()
        while self.check_memory_limit() and (time.time() - start_time) < max_wait:
            time.sleep(1.0)
            
        if self.check_memory_limit():
            raise MemoryError(f"Memory usage still above limit after {max_wait}s")

class DatabaseBatchProcessor:
    """Optimized database batch operations"""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.pool = None
        
    async def initialize(self):
        """Initialize database connection pool"""
        self.pool = await asyncpg.create_pool(
            self.db_url,
            min_size=10,
            max_size=50,
            command_timeout=300
        )
    
    async def cleanup(self):
        """Cleanup database connections"""
        if self.pool:
            await self.pool.close()
    
    async def batch_insert_memories(self, items: List[BatchItem]) -> Tuple[List[str], List[str]]:
        """Batch insert memory records"""
        success_ids = []
        failed_ids = []
        
        # Prepare batch data
        batch_data = []
        for item in items:
            data = item.data
            batch_data.append((
                data.get('user_id'),
                data.get('memory_text'),
                data.get('embedding'),
                data.get('metadata', {}),
                data.get('memory_type', 'general'),
                data.get('importance_score', 0.5)
            ))
        
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
                    chunk = batch_data[i:i + chunk_size]
                    chunk_items = items[i:i + chunk_size]
                    
                    try:
                        async with conn.transaction():
                            results = await conn.fetch(query, *zip(*chunk))
                            for j, result in enumerate(results):
                                success_ids.append(str(result['id']))
                                
                    except Exception as e:
                        logger.error(f"Batch insert chunk failed: {e}")
                        # Process items individually to identify failures
                        for j, (data_row, item) in enumerate(zip(chunk, chunk_items)):
                            try:
                                result = await conn.fetchrow(query, *data_row)
                                success_ids.append(str(result['id']))
                            except Exception as item_error:
                                logger.error(f"Individual insert failed for item {item.id}: {item_error}")
                                failed_ids.append(item.id)
                                
            except Exception as e:
                logger.error(f"Batch insert failed: {e}")
                failed_ids.extend([item.id for item in items])
        
        return success_ids, failed_ids
    
    async def batch_update_embeddings(self, items: List[BatchItem]) -> Tuple[List[str], List[str]]:
        """Batch update embeddings"""
        success_ids = []
        failed_ids = []
        
        async with self.pool.acquire() as conn:
            try:
                # Prepare update data
                update_data = []
                for item in items:
                    data = item.data
                    update_data.append((
                        data.get('embedding'),
                        data.get('memory_id')
                    ))
                
                # Batch update using prepared statement
                stmt = await conn.prepare("""
                    UPDATE mem0_vectors.memories 
                    SET embedding = $1, updated_at = NOW()
                    WHERE id = $2
                """)
                
                # Execute batch update
                results = await stmt.executemany(update_data)
                
                # Check which updates succeeded
                for i, item in enumerate(items):
                    if i < len(results):
                        success_ids.append(item.id)
                    else:
                        failed_ids.append(item.id)
                        
            except Exception as e:
                logger.error(f"Batch update failed: {e}")
                failed_ids.extend([item.id for item in items])
        
        return success_ids, failed_ids
    
    async def batch_search_similar(self, items: List[BatchItem]) -> List[Dict]:
        """Batch similarity search"""
        results = []
        
        async with self.pool.acquire() as conn:
            for item in items:
                try:
                    data = item.data
                    query_embedding = data.get('query_embedding')
                    user_id = data.get('user_id')
                    top_k = data.get('top_k', 10)
                    threshold = data.get('similarity_threshold', 0.7)
                    
                    # Perform similarity search
                    search_results = await conn.fetch("""
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
                    """, query_embedding, user_id, threshold, top_k)
                    
                    item_results = []
                    for row in search_results:
                        item_results.append({
                            'memory_id': str(row['id']),
                            'memory_text': row['memory_text'],
                            'similarity': float(row['similarity']),
                            'metadata': row['metadata']
                        })
                    
                    results.append({
                        'item_id': item.id,
                        'results': item_results,
                        'success': True
                    })
                    
                except Exception as e:
                    logger.error(f"Search failed for item {item.id}: {e}")
                    results.append({
                        'item_id': item.id,
                        'results': [],
                        'success': False,
                        'error': str(e)
                    })
        
        return results

class BatchProcessor:
    """Main batch processing engine"""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.db_processor = DatabaseBatchProcessor(db_url)
        self.active_jobs = {}
        self.job_queue = queue.PriorityQueue()
        self.shutdown_event = threading.Event()
        
    async def initialize(self):
        """Initialize the batch processor"""
        await self.db_processor.initialize()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Batch processor initialized")
    
    async def cleanup(self):
        """Cleanup resources"""
        self.shutdown_event.set()
        await self.db_processor.cleanup()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
    
    def submit_job(self, job: BatchJob) -> str:
        """Submit a batch job for processing"""
        # Validate job
        if not job.items:
            raise ValueError("Job must contain at least one item")
        
        if job.config.max_workers is None:
            job.config.max_workers = min(mp.cpu_count(), len(job.items))
        
        # Add to queue with priority (lower number = higher priority)
        priority = 1 if job.operation in [BatchOperation.INSERT, BatchOperation.UPDATE] else 2
        self.job_queue.put((priority, time.time(), job.job_id, job))
        
        self.active_jobs[job.job_id] = job
        
        logger.info(f"Submitted batch job {job.job_id} with {len(job.items)} items")
        return job.job_id
    
    async def process_job(self, job: BatchJob) -> BatchResult:
        """Process a single batch job"""
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
            if job.config.max_workers == 1:
                # Sequential processing
                for chunk_id, chunk in enumerate(chunks):
                    if self.shutdown_event.is_set():
                        job.status = BatchStatus.CANCELLED
                        break
                        
                    chunk_result = await self._process_chunk(
                        chunk, job.operation, chunk_id, job.config
                    )
                    
                    processed_items += chunk_result['processed']
                    failed_items += chunk_result['failed']
                    error_log.extend(chunk_result['errors'])
                    
                    progress_tracker.update(
                        chunk_result['processed'], 
                        chunk_result['failed']
                    )
                    
                    # Checkpoint if enabled
                    if job.config.enable_checkpointing and \
                       processed_items % job.config.checkpoint_interval == 0:
                        checkpoint_manager.save_checkpoint(
                            chunk_id, 
                            chunk_result['processed_data'],
                            chunk_result['failed_data']
                        )
                    
                    # Check memory usage
                    if memory_monitor.check_memory_limit():
                        logger.warning("Memory limit exceeded, waiting...")
                        memory_monitor.wait_for_memory()
            else:
                # Parallel processing
                processed_items, failed_items, error_log = await self._process_parallel(
                    chunks, job.operation, job.config, progress_tracker, 
                    memory_monitor, checkpoint_manager
                )
            
            # Determine final status
            if self.shutdown_event.is_set():
                job.status = BatchStatus.CANCELLED
            elif failed_items == 0:
                job.status = BatchStatus.COMPLETED
            elif processed_items > 0:
                job.status = BatchStatus.COMPLETED  # Partial success
            else:
                job.status = BatchStatus.FAILED
                
        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}")
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
                throughput_items_per_second=processed_items / processing_time if processing_time > 0 else 0,
                memory_usage_mb=memory_monitor.get_memory_usage_mb(),
                error_summary=error_log[:100],  # Limit error log size
                checkpoint_files=checkpoint_manager.checkpoint_files.copy()
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
            
            logger.info(f"Job {job.job_id} completed: {processed_items} processed, "
                       f"{failed_items} failed in {processing_time:.2f}s")
            
            return result
    
    def _create_chunks(self, items: List[BatchItem], chunk_size: int) -> List[List[BatchItem]]:
        """Split items into processing chunks"""
        chunks = []
        for i in range(0, len(items), chunk_size):
            chunks.append(items[i:i + chunk_size])
        return chunks
    
    async def _process_chunk(self, chunk: List[BatchItem], operation: BatchOperation,
                           chunk_id: int, config: BatchConfig) -> Dict:
        """Process a single chunk of items"""
        processed_data = []
        failed_data = []
        errors = []
        
        try:
            if operation == BatchOperation.INSERT:
                success_ids, failed_ids = await self.db_processor.batch_insert_memories(chunk)
                
                for item in chunk:
                    if item.id in failed_ids:
                        failed_data.append(asdict(item))
                        errors.append(f"Insert failed for item {item.id}")
                    else:
                        processed_data.append(asdict(item))
                        
            elif operation == BatchOperation.UPDATE:
                success_ids, failed_ids = await self.db_processor.batch_update_embeddings(chunk)
                
                for item in chunk:
                    if item.id in failed_ids:
                        failed_data.append(asdict(item))
                        errors.append(f"Update failed for item {item.id}")
                    else:
                        processed_data.append(asdict(item))
                        
            elif operation == BatchOperation.SEARCH:
                search_results = await self.db_processor.batch_search_similar(chunk)
                
                for result in search_results:
                    if result['success']:
                        processed_data.append(result)
                    else:
                        failed_data.append(result)
                        errors.append(result.get('error', 'Unknown search error'))
            
            else:
                raise ValueError(f"Unsupported operation: {operation}")
                
        except Exception as e:
            logger.error(f"Chunk {chunk_id} processing failed: {e}")
            failed_data.extend([asdict(item) for item in chunk])
            errors.append(f"Chunk processing error: {str(e)}")
        
        return {
            'processed': len(processed_data),
            'failed': len(failed_data),
            'errors': errors,
            'processed_data': processed_data,
            'failed_data': failed_data
        }
    
    async def _process_parallel(self, chunks: List[List[BatchItem]], operation: BatchOperation,
                              config: BatchConfig, progress_tracker: ProgressTracker,
                              memory_monitor: MemoryMonitor, 
                              checkpoint_manager: CheckpointManager) -> Tuple[int, int, List[str]]:
        """Process chunks in parallel"""
        total_processed = 0
        total_failed = 0
        all_errors = []
        
        # Use ThreadPoolExecutor for async operations
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            # Submit all chunks
            future_to_chunk = {}
            for chunk_id, chunk in enumerate(chunks):
                future = executor.submit(
                    asyncio.run,
                    self._process_chunk(chunk, operation, chunk_id, config)
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
                    total_processed += result['processed']
                    total_failed += result['failed']
                    all_errors.extend(result['errors'])
                    
                    progress_tracker.update(result['processed'], result['failed'])
                    
                    # Checkpoint if enabled
                    if config.enable_checkpointing and \
                       total_processed % config.checkpoint_interval == 0:
                        checkpoint_manager.save_checkpoint(
                            chunk_id,
                            result['processed_data'],
                            result['failed_data']
                        )
                    
                    # Check memory usage
                    if memory_monitor.check_memory_limit():
                        logger.warning("Memory limit exceeded during parallel processing")
                        memory_monitor.wait_for_memory()
                        
                except Exception as e:
                    logger.error(f"Parallel chunk processing failed: {e}")
                    total_failed += len(chunk)
                    all_errors.append(f"Parallel processing error: {str(e)}")
        
        return total_processed, total_failed, all_errors
    
    async def get_job_status(self, job_id: str) -> Optional[BatchJob]:
        """Get status of a batch job"""
        return self.active_jobs.get(job_id)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a batch job"""
        job = self.active_jobs.get(job_id)
        if job and job.status in [BatchStatus.PENDING, BatchStatus.RUNNING]:
            job.status = BatchStatus.CANCELLED
            return True
        return False
    
    async def list_jobs(self, status_filter: Optional[BatchStatus] = None) -> List[BatchJob]:
        """List all jobs, optionally filtered by status"""
        jobs = list(self.active_jobs.values())
        if status_filter:
            jobs = [job for job in jobs if job.status == status_filter]
        return jobs
    
    async def cleanup_completed_jobs(self, max_age_hours: int = 24):
        """Clean up old completed jobs"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        jobs_to_remove = []
        for job_id, job in self.active_jobs.items():
            if (job.status in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED] and
                job.completed_at and job.completed_at < cutoff_time):
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.active_jobs[job_id]
        
        logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")

# Example usage
async def main():
    """Test the batch processor"""
    import uuid
    
    DB_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost/mem0ai')
    
    processor = BatchProcessor(DB_URL)
    
    try:
        await processor.initialize()
        
        # Create test batch job
        items = []
        for i in range(100):
            item = BatchItem(
                id=str(uuid.uuid4()),
                operation=BatchOperation.INSERT,
                data={
                    'user_id': 'test_user',
                    'memory_text': f'Test memory {i}',
                    'embedding': np.random.rand(1536).tolist(),
                    'metadata': {'test': True, 'index': i},
                    'memory_type': 'test',
                    'importance_score': np.random.rand()
                }
            )
            items.append(item)
        
        # Configure batch processing
        config = BatchConfig(
            batch_size=50,
            max_workers=4,
            chunk_size=25,
            enable_checkpointing=True,
            progress_callback=lambda p, proc, fail: print(f"Progress: {p:.1%} ({proc} processed, {fail} failed)")
        )
        
        # Create and submit job
        job = BatchJob(
            job_id=str(uuid.uuid4()),
            operation=BatchOperation.INSERT,
            items=items,
            config=config
        )
        
        job_id = processor.submit_job(job)
        print(f"Submitted job: {job_id}")
        
        # Process the job
        result = await processor.process_job(job)
        
        print(f"\nBatch processing completed:")
        print(f"  Status: {result.status.value}")
        print(f"  Total items: {result.total_items}")
        print(f"  Processed: {result.processed_items}")
        print(f"  Failed: {result.failed_items}")
        print(f"  Time: {result.processing_time_seconds:.2f}s")
        print(f"  Throughput: {result.throughput_items_per_second:.1f} items/sec")
        print(f"  Memory usage: {result.memory_usage_mb:.1f} MB")
        
        if result.error_summary:
            print(f"  Errors: {len(result.error_summary)}")
            for error in result.error_summary[:5]:
                print(f"    - {error}")
                
    finally:
        await processor.cleanup()

if __name__ == "__main__":
    asyncio.run(main())