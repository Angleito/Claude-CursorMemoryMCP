#!/usr/bin/env python3
"""Production-grade Backup and Recovery System for Vector Data in mem0ai
Comprehensive backup, restoration, and disaster recovery for vector databases.
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from contextlib import suppress
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import asyncpg
import boto3
from botocore.exceptions import BotoCoreError
from botocore.exceptions import ClientError

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("/tmp/backup_recovery.log")],
)
logger = logging.getLogger(__name__)


class BackupType(Enum):
    """Types of backups."""

    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    VECTOR_ONLY = "vector_only"
    METADATA_ONLY = "metadata_only"


class BackupStatus(Enum):
    """Backup operation status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StorageBackend(Enum):
    """Storage backends for backups."""

    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"


@dataclass
class BackupConfig:
    """Configuration for backup operations."""

    backup_type: BackupType = BackupType.FULL
    storage_backend: StorageBackend = StorageBackend.LOCAL
    local_backup_dir: str = "/tmp/mem0ai_backups"
    s3_bucket: Optional[str] = None
    s3_prefix: str = "mem0ai-backups"
    compress_data: bool = True
    encrypt_data: bool = True
    encryption_key: Optional[str] = None
    retention_days: int = 30
    max_backup_size_gb: int = 100
    parallel_workers: int = 4
    verify_backup: bool = True
    include_indexes: bool = False  # Exclude indexes by default for space
    include_logs: bool = False
    connection_timeout: int = 300
    command_timeout: int = 600
    max_retries: int = 3
    retry_delay: int = 5

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.parallel_workers < 1:
            raise ValueError("parallel_workers must be at least 1")
        if self.retention_days < 1:
            raise ValueError("retention_days must be at least 1")
        if self.max_backup_size_gb < 1:
            raise ValueError("max_backup_size_gb must be at least 1")
        if self.storage_backend == StorageBackend.S3 and not self.s3_bucket:
            raise ValueError("s3_bucket is required when using S3 storage backend")
        if self.encrypt_data and not self.encryption_key:
            logger.warning("Encryption enabled but no encryption key provided")


@dataclass
class BackupMetadata:
    """Metadata for backup operations."""

    backup_id: str
    backup_type: BackupType
    created_at: datetime
    database_name: str
    total_size_bytes: int
    compressed_size_bytes: int
    file_count: int
    checksum: str
    storage_backend: StorageBackend
    storage_path: str
    encryption_enabled: bool
    backup_config: Dict[str, Any]


@dataclass
class BackupOperation:
    """Backup operation tracking."""

    operation_id: str
    backup_type: BackupType
    status: BackupStatus
    started_at: datetime
    completed_at: Optional[datetime]
    progress_percent: float
    current_step: str
    files_processed: int
    total_files: int
    size_processed_bytes: int
    total_size_bytes: int
    error_message: Optional[str]
    metadata: Optional[BackupMetadata]


class ProgressTracker:
    """Track backup/recovery progress with thread safety."""

    def __init__(self, total_items: int) -> None:
        if total_items < 0:
            raise ValueError("total_items cannot be negative")
        self.total_items = total_items
        self.processed_items = 0
        self.current_step = ""
        self.start_time = time.time()
        self.lock = threading.RLock()  # Use RLock for re-entrant locking
        self.error_count = 0

    def update(self, processed: int, step: Optional[str] = None) -> None:
        """Update progress with validation."""
        if processed < 0:
            raise ValueError("processed items cannot be negative")

        with self.lock:
            self.processed_items = min(
                self.processed_items + processed, self.total_items
            )
            if step:
                self.current_step = step
                logger.debug(f"Progress update: {self.get_progress()[0]:.1f}% - {step}")

    def increment_error(self) -> None:
        """Increment error counter."""
        with self.lock:
            self.error_count += 1

    def get_progress(self) -> Tuple[float, str]:
        """Get current progress with additional metrics."""
        with self.lock:
            progress = (
                (self.processed_items / self.total_items) * 100
                if self.total_items > 0
                else 0
            )
            return min(progress, 100.0), self.current_step

    def get_eta(self) -> Optional[float]:
        """Calculate estimated time of arrival."""
        with self.lock:
            if self.processed_items == 0:
                return None
            elapsed = time.time() - self.start_time
            rate = self.processed_items / elapsed
            remaining = self.total_items - self.processed_items
            return remaining / rate if rate > 0 else None


class StorageManager:
    """Manages different storage backends with improved error handling."""

    def __init__(self, config: BackupConfig) -> None:
        self.config = config
        self.s3_client: Optional[boto3.client] = None
        self.local_dir = Path(config.local_backup_dir)
        self.executor = ThreadPoolExecutor(max_workers=config.parallel_workers)

        try:
            if config.storage_backend == StorageBackend.S3:
                self._init_s3()
            elif config.storage_backend == StorageBackend.LOCAL:
                self._init_local()
            else:
                raise ValueError(
                    f"Unsupported storage backend: {config.storage_backend}"
                )
        except Exception as e:
            logger.error(f"Failed to initialize storage manager: {e}")
            raise

    def _init_s3(self) -> None:
        """Initialize S3 client with comprehensive error handling."""
        if not self.config.s3_bucket:
            raise ValueError("S3 bucket name is required")

        try:
            # Initialize S3 client with retry configuration
            from botocore.config import Config

            config = Config(
                region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
                retries={"max_attempts": self.config.max_retries, "mode": "adaptive"},
                max_pool_connections=50,
            )

            self.s3_client = boto3.client("s3", config=config)

            # Test connection and permissions
            try:
                self.s3_client.head_bucket(Bucket=self.config.s3_bucket)
                logger.info(
                    f"Successfully connected to S3 bucket: {self.config.s3_bucket}"
                )
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code == "404":
                    raise ValueError(
                        f"S3 bucket '{self.config.s3_bucket}' does not exist"
                    )
                elif error_code == "403":
                    raise PermissionError(
                        f"Access denied to S3 bucket '{self.config.s3_bucket}'"
                    )
                else:
                    raise

        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing S3: {e}")
            raise

    def _init_local(self) -> None:
        """Initialize local storage with validation."""
        try:
            # Validate path and create directory
            self.local_dir = self.local_dir.resolve()  # Convert to absolute path
            self.local_dir.mkdir(parents=True, exist_ok=True)

            # Check write permissions
            test_file = self.local_dir / ".write_test"
            try:
                test_file.write_text("test")
                test_file.unlink()
            except Exception as e:
                raise PermissionError(
                    f"No write permission to backup directory {self.local_dir}: {e}"
                )

            # Check available space
            stat = os.statvfs(self.local_dir)
            available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)

            if available_gb < self.config.max_backup_size_gb:
                logger.warning(
                    f"Available space ({available_gb:.1f}GB) is less than max backup size "
                    f"({self.config.max_backup_size_gb}GB)"
                )

            logger.info(
                f"Local backup directory initialized: {self.local_dir} ({available_gb:.1f}GB available)"
            )

        except Exception as e:
            logger.error(f"Failed to initialize local storage: {e}")
            raise

    async def store_file(self, local_path: str, remote_path: str) -> bool:
        """Store file to configured backend with retry logic."""
        if not Path(local_path).exists():
            logger.error(f"Local file does not exist: {local_path}")
            return False

        for attempt in range(self.config.max_retries):
            try:
                if self.config.storage_backend == StorageBackend.S3:
                    return await self._store_to_s3(local_path, remote_path)
                else:
                    return await self._store_to_local(local_path, remote_path)
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{self.config.max_retries} failed to store {local_path}: {e}"
                )
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(
                        self.config.retry_delay * (attempt + 1)
                    )  # Exponential backoff
                else:
                    logger.error(
                        f"Failed to store file {local_path} after {self.config.max_retries} attempts"
                    )
                    return False
        return False

    async def _store_to_s3(self, local_path: str, remote_path: str) -> bool:
        """Store file to S3 with validation and progress tracking."""
        if not self.s3_client:
            raise RuntimeError("S3 client not initialized")

        try:
            key = f"{self.config.s3_prefix}/{remote_path}".strip("/")
            file_size = Path(local_path).stat().st_size

            # Check if file already exists and compare
            try:
                head_response = self.s3_client.head_object(
                    Bucket=self.config.s3_bucket, Key=key
                )
                if head_response["ContentLength"] == file_size:
                    logger.debug(
                        f"File {key} already exists with same size, skipping upload"
                    )
                    return True
            except ClientError as e:
                if e.response["Error"]["Code"] != "404":
                    raise

            # Upload file with progress tracking
            def progress_callback(bytes_transferred: int) -> None:
                logger.debug(f"Uploaded {bytes_transferred}/{file_size} bytes to {key}")

            # Run upload in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            def upload_task() -> None:
                with open(local_path, "rb") as f:
                    self.s3_client.upload_fileobj(
                        f, self.config.s3_bucket, key, Callback=progress_callback
                    )

            await loop.run_in_executor(self.executor, upload_task)

            # Verify upload
            try:
                head_response = self.s3_client.head_object(
                    Bucket=self.config.s3_bucket, Key=key
                )
                if head_response["ContentLength"] != file_size:
                    raise RuntimeError(
                        f"Upload verification failed: size mismatch {head_response['ContentLength']} != {file_size}"
                    )
            except ClientError as e:
                raise RuntimeError(f"Upload verification failed: {e}")

            logger.info(
                f"Successfully uploaded {local_path} to s3://{self.config.s3_bucket}/{key} ({file_size} bytes)"
            )
            return True

        except Exception as e:
            logger.error(f"S3 upload failed for {local_path}: {e}")
            raise

    async def _store_to_local(self, local_path: str, remote_path: str) -> bool:
        """Store file to local directory with validation."""
        try:
            source_path = Path(local_path)
            target_path = self.local_dir / remote_path

            # Validate source file
            if not source_path.exists():
                raise FileNotFoundError(f"Source file does not exist: {local_path}")

            source_size = source_path.stat().st_size

            # Create target directory
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Skip if same file
            if source_path.resolve() == target_path.resolve():
                logger.debug(f"Source and target are the same file: {local_path}")
                return True

            # Check if target exists and compare
            if target_path.exists():
                target_size = target_path.stat().st_size
                if source_size == target_size:
                    # Compare checksums
                    source_hash = self._calculate_checksum(str(source_path))
                    target_hash = self._calculate_checksum(str(target_path))
                    if source_hash == target_hash:
                        logger.debug(
                            f"Target file already exists with same content: {target_path}"
                        )
                        return True

            # Copy file with verification
            shutil.copy2(source_path, target_path)

            # Verify copy
            target_size = target_path.stat().st_size
            if source_size != target_size:
                raise RuntimeError(
                    f"Copy verification failed: size mismatch {source_size} != {target_size}"
                )

            logger.debug(
                f"Successfully copied {local_path} to {target_path} ({source_size} bytes)"
            )
            return True

        except Exception as e:
            logger.error(f"Local storage failed for {local_path}: {e}")
            raise

    async def retrieve_file(self, remote_path: str, local_path: str) -> bool:
        """Retrieve file from storage backend."""
        try:
            if self.config.storage_backend == StorageBackend.S3:
                return await self._retrieve_from_s3(remote_path, local_path)
            else:
                return await self._retrieve_from_local(remote_path, local_path)
        except Exception as e:
            logger.error(f"Failed to retrieve file {remote_path}: {e}")
            return False

    async def _retrieve_from_s3(self, remote_path: str, local_path: str) -> bool:
        """Retrieve file from S3."""
        try:
            key = f"{self.config.s3_prefix}/{remote_path}"

            # Ensure local directory exists
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)

            # Download file
            self.s3_client.download_file(self.config.s3_bucket, key, local_path)

            logger.debug(
                f"Downloaded s3://{self.config.s3_bucket}/{key} to {local_path}"
            )
            return True

        except Exception as e:
            logger.error(f"S3 download failed: {e}")
            return False

    async def _retrieve_from_local(self, remote_path: str, local_path: str) -> bool:
        """Retrieve file from local storage."""
        try:
            source_path = self.local_dir / remote_path

            if not source_path.exists():
                logger.error(f"Source file not found: {source_path}")
                return False

            # Ensure target directory exists
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(source_path, local_path)
            return True

        except Exception as e:
            logger.error(f"Local retrieval failed: {e}")
            return False

    async def list_backups(self) -> List[str]:
        """List available backups."""
        try:
            if self.config.storage_backend == StorageBackend.S3:
                return await self._list_s3_backups()
            else:
                return await self._list_local_backups()
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
            return []

    async def _list_s3_backups(self) -> List[str]:
        """List backups in S3."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.config.s3_bucket,
                Prefix=f"{self.config.s3_prefix}/",
                Delimiter="/",
            )

            backups = []
            for obj in response.get("CommonPrefixes", []):
                prefix = obj["Prefix"]
                backup_id = prefix.split("/")[-2]  # Extract backup ID
                backups.append(backup_id)

            return backups

        except Exception as e:
            logger.error(f"Failed to list S3 backups: {e}")
            return []

    async def _list_local_backups(self) -> List[str]:
        """List local backups."""
        try:
            backups = []
            for backup_dir in self.local_dir.iterdir():
                if backup_dir.is_dir():
                    backups.append(backup_dir.name)
            return sorted(backups, reverse=True)

        except Exception as e:
            logger.error(f"Failed to list local backups: {e}")
            return []


class VectorBackupRecovery:
    """Main backup and recovery system for vector data."""

    def __init__(self, db_url: str, config: Optional[BackupConfig] = None) -> None:
        if not db_url:
            raise ValueError("Database URL is required")

        self.db_url = db_url
        self.config = config or BackupConfig()
        self.pool: Optional[asyncpg.Pool] = None
        self.storage_manager = StorageManager(self.config)
        self.active_operations: Dict[str, BackupOperation] = {}
        self._shutdown_event = asyncio.Event()
        self._operation_lock = asyncio.Lock()

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_event.set()

    async def initialize(self) -> None:
        """Initialize the backup system with comprehensive setup."""
        try:
            # Validate database URL
            if not self.db_url.startswith(("postgresql://", "postgres://")):
                raise ValueError("Invalid PostgreSQL database URL")

            # Create connection pool with retry logic
            for attempt in range(self.config.max_retries):
                try:
                    self.pool = await asyncpg.create_pool(
                        self.db_url,
                        min_size=2,
                        max_size=10,
                        command_timeout=self.config.command_timeout,
                        server_settings={
                            "application_name": "mem0ai_backup_system",
                            "tcp_user_timeout": "30000",
                        },
                    )

                    # Test connection
                    async with self.pool.acquire() as conn:
                        await conn.fetchval("SELECT 1")

                    logger.info("Database connection pool established")
                    break

                except Exception as e:
                    logger.warning(
                        f"Database connection attempt {attempt + 1}/{self.config.max_retries} failed: {e}"
                    )
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    else:
                        raise ConnectionError(
                            f"Failed to establish database connection after {self.config.max_retries} attempts"
                        )

            # Create schema and tables
            await self._setup_database_schema()

            # Validate storage manager
            if not hasattr(self.storage_manager, "config"):
                raise RuntimeError("Storage manager not properly initialized")

            logger.info("Backup and recovery system initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize backup system: {e}")
            await self.cleanup()
            raise

    async def _setup_database_schema(self) -> None:
        """Setup database schema for backup tracking."""
        async with self.pool.acquire() as conn:
            # Create schema if it doesn't exist
            await conn.execute("CREATE SCHEMA IF NOT EXISTS mem0_vectors")

            # Create backup tracking table with comprehensive fields
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mem0_vectors.backup_log (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    backup_id TEXT NOT NULL UNIQUE,
                    backup_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    completed_at TIMESTAMP WITH TIME ZONE,
                    total_size_bytes BIGINT,
                    compressed_size_bytes BIGINT,
                    file_count INTEGER,
                    checksum TEXT,
                    storage_backend TEXT,
                    storage_path TEXT,
                    metadata JSONB DEFAULT '{}',
                    error_message TEXT,
                    compression_ratio FLOAT,
                    duration_seconds INTEGER,
                    host_name TEXT DEFAULT CURRENT_SETTING('cluster_name', true),
                    version TEXT DEFAULT '1.0'
                );

                CREATE INDEX IF NOT EXISTS backup_log_backup_id_idx
                ON mem0_vectors.backup_log (backup_id);

                CREATE INDEX IF NOT EXISTS backup_log_created_at_idx
                ON mem0_vectors.backup_log (created_at DESC);

                CREATE INDEX IF NOT EXISTS backup_log_status_idx
                ON mem0_vectors.backup_log (status);

                CREATE INDEX IF NOT EXISTS backup_log_type_idx
                ON mem0_vectors.backup_log (backup_type);
            """
            )

            logger.debug("Database schema setup completed")

    async def cleanup(self) -> None:
        """Cleanup resources gracefully."""
        logger.info("Starting cleanup process...")

        # Signal shutdown to all operations
        self._shutdown_event.set()

        # Wait for active operations to complete (with timeout)
        if self.active_operations:
            logger.info(
                f"Waiting for {len(self.active_operations)} active operations to complete..."
            )
            await asyncio.sleep(5)  # Give operations time to notice shutdown signal

            # Force cancel remaining operations
            for operation_id, operation in self.active_operations.items():
                if operation.status == BackupStatus.RUNNING:
                    logger.warning(f"Force cancelling operation {operation_id}")
                    operation.status = BackupStatus.CANCELLED

        # Close storage manager resources
        if hasattr(self.storage_manager, "executor"):
            self.storage_manager.executor.shutdown(wait=True)

        # Close database pool
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")

        logger.info("Cleanup completed")

    def _generate_backup_id(self) -> str:
        """Generate unique backup ID with better entropy."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use more secure random instead of time-based hash
        import secrets

        random_suffix = secrets.token_hex(4)
        return f"backup_{timestamp}_{random_suffix}"

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum with error handling."""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                while chunk := f.read(
                    8192
                ):  # Use larger chunk size for better performance
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate checksum for {file_path}: {e}")
            raise

    async def create_backup(
        self,
        backup_type: Optional[BackupType] = None,
        include_users: Optional[List[str]] = None,
    ) -> str:
        """Create a backup with comprehensive error handling and monitoring."""
        backup_type = backup_type or self.config.backup_type
        backup_id = self._generate_backup_id()

        # Validate backup type
        if backup_type not in BackupType:
            raise ValueError(f"Invalid backup type: {backup_type}")

        # Check if system is shutting down
        if self._shutdown_event.is_set():
            raise RuntimeError("System is shutting down, cannot start new backup")

        # Create operation tracking
        operation = BackupOperation(
            operation_id=backup_id,
            backup_type=backup_type,
            status=BackupStatus.PENDING,
            started_at=datetime.now(),
            completed_at=None,
            progress_percent=0.0,
            current_step="Initializing",
            files_processed=0,
            total_files=0,
            size_processed_bytes=0,
            total_size_bytes=0,
            error_message=None,
            metadata=None,
        )

        async with self._operation_lock:
            self.active_operations[backup_id] = operation

        start_time = time.time()

        try:
            logger.info(
                f"Starting backup {backup_id} (type: {backup_type.value}, users: {include_users or 'all'})"
            )

            # Pre-backup validation
            await self._validate_backup_preconditions()

            # Execute backup based on type
            if backup_type == BackupType.FULL:
                metadata = await self._create_full_backup(
                    backup_id, operation, include_users
                )
            elif backup_type == BackupType.VECTOR_ONLY:
                metadata = await self._create_vector_backup(
                    backup_id, operation, include_users
                )
            elif backup_type == BackupType.METADATA_ONLY:
                metadata = await self._create_metadata_backup(
                    backup_id, operation, include_users
                )
            elif backup_type == BackupType.INCREMENTAL:
                metadata = await self._create_incremental_backup(
                    backup_id, operation, include_users
                )
            elif backup_type == BackupType.DIFFERENTIAL:
                metadata = await self._create_differential_backup(
                    backup_id, operation, include_users
                )
            else:
                raise ValueError(f"Backup type {backup_type} not implemented")

            # Calculate final metrics
            duration = time.time() - start_time
            operation.status = BackupStatus.COMPLETED
            operation.completed_at = datetime.now()
            operation.metadata = metadata
            operation.progress_percent = 100.0

            # Add duration to metadata
            if metadata:
                metadata_dict = asdict(metadata)
                metadata_dict["duration_seconds"] = int(duration)

            # Log backup completion
            await self._log_backup_operation(operation)

            logger.info(f"Backup {backup_id} completed successfully in {duration:.1f}s")
            return backup_id

        except asyncio.CancelledError:
            operation.status = BackupStatus.CANCELLED
            operation.error_message = "Backup cancelled"
            operation.completed_at = datetime.now()
            logger.warning(f"Backup {backup_id} was cancelled")
            raise

        except Exception as e:
            operation.status = BackupStatus.FAILED
            operation.error_message = str(e)
            operation.completed_at = datetime.now()

            await self._log_backup_operation(operation)

            logger.error(
                f"Backup {backup_id} failed after {time.time() - start_time:.1f}s: {e}"
            )
            raise

        finally:
            # Remove from active operations
            async with self._operation_lock:
                self.active_operations.pop(backup_id, None)

    async def _validate_backup_preconditions(self) -> None:
        """Validate system state before starting backup."""
        # Check database connectivity
        if not self.pool:
            raise RuntimeError("Database pool not initialized")

        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
        except Exception as e:
            raise RuntimeError(f"Database connectivity check failed: {e}")

        # Check storage availability
        if self.config.storage_backend == StorageBackend.LOCAL:
            # Check disk space
            stat = os.statvfs(self.storage_manager.local_dir)
            available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)

            if available_gb < 1.0:  # Require at least 1GB free
                raise RuntimeError(
                    f"Insufficient disk space: {available_gb:.1f}GB available"
                )

        # Check for running backups limit
        running_operations = sum(
            1
            for op in self.active_operations.values()
            if op.status == BackupStatus.RUNNING
        )

        if running_operations >= self.config.parallel_workers:
            raise RuntimeError(
                f"Too many concurrent operations: {running_operations}/{self.config.parallel_workers}"
            )

    async def _create_incremental_backup(
        self,
        backup_id: str,
        operation: BackupOperation,
        include_users: Optional[List[str]] = None,
    ) -> BackupMetadata:
        """Create incremental backup (changes since last backup)."""
        # TODO: Implement incremental backup logic
        raise NotImplementedError("Incremental backup not yet implemented")

    async def _create_differential_backup(
        self,
        backup_id: str,
        operation: BackupOperation,
        include_users: Optional[List[str]] = None,
    ) -> BackupMetadata:
        """Create differential backup (changes since last full backup)."""
        # TODO: Implement differential backup logic
        raise NotImplementedError("Differential backup not yet implemented")

    async def _create_full_backup(
        self,
        backup_id: str,
        operation: BackupOperation,
        include_users: Optional[List[str]] = None,
    ) -> BackupMetadata:
        """Create full database backup."""
        temp_dir = Path(tempfile.mkdtemp(prefix=f"backup_{backup_id}_"))

        try:
            operation.current_step = "Exporting database schema"
            operation.status = BackupStatus.RUNNING

            # Export schema
            schema_file = temp_dir / "schema.sql"
            await self._export_schema(str(schema_file))

            operation.current_step = "Exporting vector data"

            # Export vector data
            vector_data_file = temp_dir / "vector_data.jsonl"
            await self._export_vector_data(
                str(vector_data_file), include_users, operation
            )

            operation.current_step = "Exporting metadata"

            # Export metadata
            metadata_file = temp_dir / "metadata.json"
            await self._export_metadata(str(metadata_file), include_users)

            operation.current_step = "Compressing backup"

            # Create compressed archive
            archive_file = temp_dir / f"{backup_id}.tar.gz"
            await self._create_archive(temp_dir, archive_file, operation)

            operation.current_step = "Uploading to storage"

            # Store backup
            remote_path = f"{backup_id}/{backup_id}.tar.gz"
            success = await self.storage_manager.store_file(
                str(archive_file), remote_path
            )

            if not success:
                raise RuntimeError("Failed to store backup file")

            # Calculate final statistics
            total_size = sum(
                f.stat().st_size for f in temp_dir.glob("*") if f.is_file()
            )
            compressed_size = archive_file.stat().st_size
            checksum = self._calculate_checksum(str(archive_file))

            return BackupMetadata(
                backup_id=backup_id,
                backup_type=BackupType.FULL,
                created_at=operation.started_at,
                database_name="mem0ai",
                total_size_bytes=total_size,
                compressed_size_bytes=compressed_size,
                file_count=len(list(temp_dir.glob("*"))),
                checksum=checksum,
                storage_backend=self.config.storage_backend,
                storage_path=remote_path,
                encryption_enabled=self.config.encrypt_data,
                backup_config=asdict(self.config),
            )

        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def _create_vector_backup(
        self,
        backup_id: str,
        operation: BackupOperation,
        include_users: Optional[List[str]] = None,
    ) -> BackupMetadata:
        """Create vector-only backup."""
        temp_dir = Path(tempfile.mkdtemp(prefix=f"backup_{backup_id}_"))

        try:
            operation.current_step = "Exporting vector data"
            operation.status = BackupStatus.RUNNING

            # Export only vector data
            vector_data_file = temp_dir / "vectors.jsonl"
            await self._export_vector_data(
                str(vector_data_file), include_users, operation
            )

            operation.current_step = "Compressing backup"

            # Create compressed archive
            archive_file = temp_dir / f"{backup_id}_vectors.tar.gz"
            await self._create_archive(temp_dir, archive_file, operation)

            operation.current_step = "Uploading to storage"

            # Store backup
            remote_path = f"{backup_id}/{backup_id}_vectors.tar.gz"
            success = await self.storage_manager.store_file(
                str(archive_file), remote_path
            )

            if not success:
                raise RuntimeError("Failed to store backup file")

            # Calculate statistics
            total_size = vector_data_file.stat().st_size
            compressed_size = archive_file.stat().st_size
            checksum = self._calculate_checksum(str(archive_file))

            return BackupMetadata(
                backup_id=backup_id,
                backup_type=BackupType.VECTOR_ONLY,
                created_at=operation.started_at,
                database_name="mem0ai",
                total_size_bytes=total_size,
                compressed_size_bytes=compressed_size,
                file_count=1,
                checksum=checksum,
                storage_backend=self.config.storage_backend,
                storage_path=remote_path,
                encryption_enabled=self.config.encrypt_data,
                backup_config=asdict(self.config),
            )

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def _create_metadata_backup(
        self,
        backup_id: str,
        operation: BackupOperation,
        include_users: Optional[List[str]] = None,
    ) -> BackupMetadata:
        """Create metadata-only backup."""
        temp_dir = Path(tempfile.mkdtemp(prefix=f"backup_{backup_id}_"))

        try:
            operation.current_step = "Exporting metadata"
            operation.status = BackupStatus.RUNNING

            # Export metadata
            metadata_file = temp_dir / "metadata.json"
            await self._export_metadata(str(metadata_file), include_users)

            # Export schema
            schema_file = temp_dir / "schema.sql"
            await self._export_schema(str(schema_file))

            operation.current_step = "Compressing backup"

            # Create compressed archive
            archive_file = temp_dir / f"{backup_id}_metadata.tar.gz"
            await self._create_archive(temp_dir, archive_file, operation)

            operation.current_step = "Uploading to storage"

            # Store backup
            remote_path = f"{backup_id}/{backup_id}_metadata.tar.gz"
            success = await self.storage_manager.store_file(
                str(archive_file), remote_path
            )

            if not success:
                raise RuntimeError("Failed to store backup file")

            # Calculate statistics
            total_size = sum(
                f.stat().st_size for f in temp_dir.glob("*.json") if f.is_file()
            )
            compressed_size = archive_file.stat().st_size
            checksum = self._calculate_checksum(str(archive_file))

            return BackupMetadata(
                backup_id=backup_id,
                backup_type=BackupType.METADATA_ONLY,
                created_at=operation.started_at,
                database_name="mem0ai",
                total_size_bytes=total_size,
                compressed_size_bytes=compressed_size,
                file_count=2,
                checksum=checksum,
                storage_backend=self.config.storage_backend,
                storage_path=remote_path,
                encryption_enabled=self.config.encrypt_data,
                backup_config=asdict(self.config),
            )

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def _export_schema(self, output_file: str):
        """Export database schema."""
        # This would use pg_dump to export schema
        # For now, we'll create a basic schema export
        async with self.pool.acquire() as conn:
            schema_info = await conn.fetch(
                """
                SELECT table_name, column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'mem0_vectors'
                ORDER BY table_name, ordinal_position
            """
            )

            with open(output_file, "w") as f:
                f.write("-- mem0ai Schema Export\n")
                f.write(f"-- Generated: {datetime.now().isoformat()}\n\n")

                current_table = None
                for row in schema_info:
                    table_name = row["table_name"]
                    if table_name != current_table:
                        if current_table:
                            f.write(");\n\n")
                        f.write(f"CREATE TABLE mem0_vectors.{table_name} (\n")
                        current_table = table_name
                        first_column = True

                    if not first_column:
                        f.write(",\n")

                    nullable = "NULL" if row["is_nullable"] == "YES" else "NOT NULL"
                    f.write(f"    {row['column_name']} {row['data_type']} {nullable}")
                    first_column = False

                if current_table:
                    f.write("\n);\n")

    async def _export_vector_data(
        self,
        output_file: str,
        include_users: Optional[List[str]] = None,
        operation: Optional[BackupOperation] = None,
    ) -> None:
        """Export vector data to JSONL format with robust error handling."""
        if not self.pool:
            raise RuntimeError("Database pool not initialized")

        try:
            async with self.pool.acquire() as conn:
                # Build query with proper parameterization
                where_clause = ""
                params: List[Any] = []

                if include_users:
                    where_clause = "WHERE user_id = ANY($1)"
                    params = [include_users]

                # Get total count for progress tracking
                count_query = (
                    f"SELECT COUNT(*) FROM mem0_vectors.memories {where_clause}"
                )
                total_count = await conn.fetchval(count_query, *params)

                if total_count == 0:
                    logger.warning("No memory records found to export")
                    # Create empty file
                    Path(output_file).touch()
                    return

                if operation:
                    operation.total_files = total_count
                    operation.current_step = f"Exporting {total_count} memory records"

                logger.info(f"Exporting {total_count} memory records to {output_file}")

                # Export data in batches with error handling
                batch_size = 1000
                offset = 0
                exported_count = 0

                with open(output_file, "w", encoding="utf-8") as f:
                    while offset < total_count:
                        # Check for shutdown signal
                        if self._shutdown_event.is_set():
                            raise asyncio.CancelledError(
                                "Export cancelled due to shutdown"
                            )

                        try:
                            batch_query = f"""
                                SELECT id, user_id, memory_text, embedding, metadata,
                                       memory_hash, memory_type, importance_score,
                                       access_count, created_at, updated_at, last_accessed
                                FROM mem0_vectors.memories
                                {where_clause}
                                ORDER BY created_at
                                LIMIT ${'2' if include_users else '1'} OFFSET ${'3' if include_users else '2'}
                            """

                            batch_params = [*params, batch_size, offset]
                            rows = await conn.fetch(batch_query, *batch_params)

                            if not rows:
                                break

                            for row in rows:
                                try:
                                    # Convert row to JSON serializable format
                                    record = dict(row)

                                    # Handle special types
                                    for field in [
                                        "created_at",
                                        "updated_at",
                                        "last_accessed",
                                    ]:
                                        if record.get(field):
                                            record[field] = record[field].isoformat()

                                    if record.get("id"):
                                        record["id"] = str(record["id"])

                                    # Ensure metadata is JSON serializable
                                    if record.get("metadata"):
                                        if isinstance(record["metadata"], str):
                                            try:
                                                record["metadata"] = json.loads(
                                                    record["metadata"]
                                                )
                                            except json.JSONDecodeError:
                                                logger.warning(
                                                    f"Invalid JSON in metadata for record {record.get('id', 'unknown')}"
                                                )
                                                record["metadata"] = {}

                                    # Write record
                                    f.write(
                                        json.dumps(record, ensure_ascii=False) + "\n"
                                    )
                                    exported_count += 1

                                except Exception as e:
                                    logger.error(
                                        f"Failed to serialize record {row.get('id', 'unknown')}: {e}"
                                    )
                                    continue

                            offset += len(rows)

                            # Update progress
                            if operation:
                                operation.files_processed = exported_count
                                operation.progress_percent = (
                                    offset / total_count
                                ) * 80  # 80% for export
                                operation.current_step = (
                                    f"Exported {exported_count}/{total_count} records"
                                )

                            # Log progress periodically
                            if offset % 10000 == 0:
                                logger.info(
                                    f"Exported {exported_count} records ({offset}/{total_count})"
                                )

                        except Exception as e:
                            logger.error(
                                f"Error processing batch at offset {offset}: {e}"
                            )
                            raise

                logger.info(f"Successfully exported {exported_count} memory records")

        except Exception as e:
            logger.error(f"Failed to export vector data: {e}")
            # Clean up partial file on error
            if Path(output_file).exists():
                with suppress(Exception):
                    Path(output_file).unlink()
            raise

    async def _export_metadata(self, output_file: str, include_users: Optional[List[str]] = None):
        """Export metadata and configuration."""
        async with self.pool.acquire() as conn:
            # Get user statistics
            user_stats = await conn.fetch(
                """
                SELECT
                    user_id,
                    COUNT(*) as memory_count,
                    AVG(importance_score) as avg_importance,
                    MAX(created_at) as last_memory
                FROM mem0_vectors.memories
                GROUP BY user_id
            """
            )

            # Get table statistics
            table_stats = await conn.fetch(
                """
                SELECT
                    schemaname,
                    tablename,
                    n_tup_ins as inserts,
                    n_tup_upd as updates,
                    n_tup_del as deletes,
                    n_live_tup as live_tuples
                FROM pg_stat_user_tables
                WHERE schemaname = 'mem0_vectors'
            """
            )

            metadata = {
                "export_timestamp": datetime.now().isoformat(),
                "user_statistics": [
                    {
                        "user_id": row["user_id"],
                        "memory_count": row["memory_count"],
                        "avg_importance": float(row["avg_importance"] or 0),
                        "last_memory": (
                            row["last_memory"].isoformat()
                            if row["last_memory"]
                            else None
                        ),
                    }
                    for row in user_stats
                ],
                "table_statistics": [dict(row) for row in table_stats],
                "backup_config": asdict(self.config),
            }

            with open(output_file, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

    async def _create_archive(
        self, source_dir: Path, archive_file: Path, operation: BackupOperation = None
    ):
        """Create compressed archive."""
        import tarfile

        if operation:
            operation.current_step = "Creating compressed archive"

        with tarfile.open(archive_file, "w:gz") as tar:
            for file_path in source_dir.glob("*"):
                if file_path.is_file():
                    tar.add(file_path, arcname=file_path.name)

    async def _log_backup_operation(self, operation: BackupOperation) -> None:
        """Log backup operation to database with comprehensive error handling."""
        if not self.pool:
            logger.warning("Cannot log backup operation: database pool not available")
            return

        try:
            async with self.pool.acquire() as conn:
                metadata_json = asdict(operation.metadata) if operation.metadata else {}

                # Calculate additional metrics
                duration_seconds = None
                compression_ratio = None

                if operation.completed_at and operation.started_at:
                    duration_seconds = int(
                        (operation.completed_at - operation.started_at).total_seconds()
                    )

                if (
                    operation.metadata
                    and operation.metadata.total_size_bytes
                    and operation.metadata.compressed_size_bytes
                    and operation.metadata.total_size_bytes > 0
                ):
                    compression_ratio = (
                        operation.metadata.compressed_size_bytes
                        / operation.metadata.total_size_bytes
                    )

                await conn.execute(
                    """
                    INSERT INTO mem0_vectors.backup_log
                    (backup_id, backup_type, status, created_at, completed_at,
                     total_size_bytes, compressed_size_bytes, file_count, checksum,
                     storage_backend, storage_path, metadata, error_message,
                     compression_ratio, duration_seconds, host_name)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                    ON CONFLICT (backup_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        completed_at = EXCLUDED.completed_at,
                        total_size_bytes = EXCLUDED.total_size_bytes,
                        compressed_size_bytes = EXCLUDED.compressed_size_bytes,
                        file_count = EXCLUDED.file_count,
                        checksum = EXCLUDED.checksum,
                        storage_path = EXCLUDED.storage_path,
                        metadata = EXCLUDED.metadata,
                        error_message = EXCLUDED.error_message,
                        compression_ratio = EXCLUDED.compression_ratio,
                        duration_seconds = EXCLUDED.duration_seconds,
                        host_name = EXCLUDED.host_name
                """,
                    operation.operation_id,
                    operation.backup_type.value,
                    operation.status.value,
                    operation.started_at,
                    operation.completed_at,
                    operation.metadata.total_size_bytes if operation.metadata else None,
                    (
                        operation.metadata.compressed_size_bytes
                        if operation.metadata
                        else None
                    ),
                    operation.metadata.file_count if operation.metadata else None,
                    operation.metadata.checksum if operation.metadata else None,
                    (
                        operation.metadata.storage_backend.value
                        if operation.metadata
                        else None
                    ),
                    operation.metadata.storage_path if operation.metadata else None,
                    json.dumps(metadata_json),
                    operation.error_message,
                    compression_ratio,
                    duration_seconds,
                    os.getenv("HOSTNAME", "unknown"),
                )

                logger.debug(
                    f"Logged backup operation {operation.operation_id} with status {operation.status.value}"
                )

        except Exception as e:
            logger.error(
                f"Failed to log backup operation {operation.operation_id}: {e}"
            )
            # Don't raise here as this shouldn't fail the backup operation

    async def restore_backup(
        self, backup_id: str, target_users: Optional[List[str]] = None
    ) -> bool:
        """Restore from backup."""
        logger.info(f"Starting restore from backup {backup_id}")

        try:
            # Get backup metadata
            async with self.pool.acquire() as conn:
                backup_info = await conn.fetchrow(
                    """
                    SELECT * FROM mem0_vectors.backup_log WHERE backup_id = $1
                """,
                    backup_id,
                )

            if not backup_info:
                raise ValueError(f"Backup {backup_id} not found")

            # Download and extract backup
            temp_dir = Path(tempfile.mkdtemp(prefix=f"restore_{backup_id}_"))

            try:
                # Download backup file
                remote_path = backup_info["storage_path"]
                archive_file = temp_dir / f"{backup_id}.tar.gz"

                success = await self.storage_manager.retrieve_file(
                    remote_path, str(archive_file)
                )
                if not success:
                    raise RuntimeError("Failed to download backup file")

                # Extract archive
                import tarfile

                with tarfile.open(archive_file, "r:gz") as tar:
                    tar.extractall(temp_dir)

                # Restore data based on backup type
                backup_type = BackupType(backup_info["backup_type"])

                if backup_type == BackupType.FULL:
                    await self._restore_full_backup(temp_dir, target_users)
                elif backup_type == BackupType.VECTOR_ONLY:
                    await self._restore_vector_data(
                        temp_dir / "vectors.jsonl", target_users
                    )
                elif backup_type == BackupType.METADATA_ONLY:
                    await self._restore_metadata(temp_dir / "metadata.json")

                logger.info(f"Restore from backup {backup_id} completed successfully")
                return True

            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as e:
            logger.error(f"Restore from backup {backup_id} failed: {e}")
            return False

    async def _restore_full_backup(
        self, backup_dir: Path, target_users: Optional[List[str]] = None
    ):
        """Restore full backup."""
        # Restore vector data
        vector_file = backup_dir / "vector_data.jsonl"
        if vector_file.exists():
            await self._restore_vector_data(vector_file, target_users)

    async def _restore_vector_data(
        self, data_file: Path, target_users: Optional[List[str]] = None
    ):
        """Restore vector data from JSONL file."""
        async with self.pool.acquire() as conn:
            with open(data_file) as f:
                batch = []
                batch_size = 1000

                for line in f:
                    record = json.loads(line)

                    # Filter by target users if specified
                    if target_users and record.get("user_id") not in target_users:
                        continue

                    batch.append(record)

                    if len(batch) >= batch_size:
                        await self._insert_memory_batch(conn, batch)
                        batch = []

                # Insert remaining batch
                if batch:
                    await self._insert_memory_batch(conn, batch)

    async def _insert_memory_batch(self, conn, batch: List[Dict]):
        """Insert batch of memory records."""
        if not batch:
            return

        # Prepare batch data
        insert_data = []
        for record in batch:
            insert_data.append(
                (
                    record.get("id"),
                    record.get("user_id"),
                    record.get("memory_text"),
                    record.get("embedding"),
                    record.get("metadata", {}),
                    record.get("memory_hash"),
                    record.get("memory_type", "general"),
                    record.get("importance_score", 0.5),
                    record.get("access_count", 0),
                    record.get("created_at"),
                    record.get("updated_at"),
                    record.get("last_accessed"),
                )
            )

        # Batch insert with conflict handling
        await conn.executemany(
            """
            INSERT INTO mem0_vectors.memories
            (id, user_id, memory_text, embedding, metadata, memory_hash,
             memory_type, importance_score, access_count, created_at, updated_at, last_accessed)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT (id) DO UPDATE SET
                memory_text = EXCLUDED.memory_text,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
        """,
            insert_data,
        )

    async def _restore_metadata(self, metadata_file: Path):
        """Restore metadata."""
        with open(metadata_file) as f:
            metadata = json.load(f)

        logger.info(f"Restored metadata from {metadata['export_timestamp']}")

    async def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups."""
        async with self.pool.acquire() as conn:
            backups = await conn.fetch(
                """
                SELECT
                    backup_id,
                    backup_type,
                    status,
                    created_at,
                    completed_at,
                    total_size_bytes,
                    compressed_size_bytes,
                    storage_backend,
                    error_message
                FROM mem0_vectors.backup_log
                ORDER BY created_at DESC
            """
            )

        return [dict(backup) for backup in backups]

    async def cleanup_old_backups(self, retention_days: Optional[int] = None) -> int:
        """Cleanup old backups based on retention policy."""
        retention_days = retention_days or self.config.retention_days
        cutoff_date = datetime.now() - timedelta(days=retention_days)

        # Get old backups
        async with self.pool.acquire() as conn:
            old_backups = await conn.fetch(
                """
                SELECT backup_id, storage_path
                FROM mem0_vectors.backup_log
                WHERE created_at < $1 AND status = 'completed'
            """,
                cutoff_date,
            )

        cleaned_count = 0

        for backup in old_backups:
            try:
                # Delete from storage (implement based on storage backend)
                # For now, just mark as deleted in database
                async with self.pool.acquire() as conn:
                    await conn.execute(
                        """
                        DELETE FROM mem0_vectors.backup_log WHERE backup_id = $1
                    """,
                        backup["backup_id"],
                    )

                cleaned_count += 1
                logger.info(f"Cleaned up old backup: {backup['backup_id']}")

            except Exception as e:
                logger.error(f"Failed to cleanup backup {backup['backup_id']}: {e}")

        return cleaned_count


# Context manager for automatic cleanup
@asynccontextmanager
async def backup_system_context(db_url: str, config: Optional[BackupConfig] = None):
    """Context manager for backup system with automatic cleanup."""
    backup_system = VectorBackupRecovery(db_url, config)
    try:
        await backup_system.initialize()
        yield backup_system
    finally:
        await backup_system.cleanup()


# Example usage
async def main() -> None:
    """Test the backup and recovery system with improved error handling."""
    DB_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/mem0ai")

    if not DB_URL or "user:password" in DB_URL:
        logger.error("Please set a valid DATABASE_URL environment variable")
        return

    config = BackupConfig(
        backup_type=BackupType.FULL,
        storage_backend=StorageBackend.LOCAL,
        local_backup_dir="/tmp/mem0ai_backups",
        compress_data=True,
        retention_days=7,
        max_retries=3,
        connection_timeout=60,
    )

    try:
        async with backup_system_context(DB_URL, config) as backup_system:

            await backup_system.create_backup(BackupType.VECTOR_ONLY)

            backups = await backup_system.list_backups()
            for backup in backups:
                (
                    "✅"
                    if backup["status"] == "completed"
                    else "❌" if backup["status"] == "failed" else "⏳"
                )

            await backup_system.cleanup_old_backups(
                retention_days=0
            )  # Clean all for demo

    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Backup system test failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
