"""Enhanced metrics and monitoring system for Mem0 AI MCP Server.

This module provides comprehensive metrics collection and monitoring capabilities:
- High-performance Prometheus metrics integration
- Structured logging with context
- Automatic system resource monitoring
- Database health checks with timeout handling
- Request tracking and performance monitoring
- Error tracking with categorization
- Memory-efficient metric storage

Examples:
    >>> # Record a search operation
    >>> metrics_collector.record_search(0.150, 10, "user123")
    >>>
    >>> # Record an error with context
    >>> metrics_collector.record_error("ValidationError", "Invalid input", "api")
    >>>
    >>> # Get metrics summary
    >>> summary = metrics_collector.get_metrics_summary()
    >>> print(f"Total requests: {summary['total_requests']}")
    >>>
    >>> # Monitor database health
    >>> health_checker = DatabaseHealthChecker(memory_manager)
    >>> health = await health_checker.check_health()
    >>> if health["status"] == "healthy":
    ...     print("Database is operational")
"""

import asyncio

# Constants
ALERT_INTERVAL_SECONDS = 600  # 10 minutes
DEFAULT_BATCH_SIZE = 1000
import contextlib
import threading
import time
import weakref
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from typing import Protocol
from typing import TypeVar

import psutil
import structlog
from fastapi import Response
from prometheus_client import CONTENT_TYPE_LATEST
from prometheus_client import REGISTRY
from prometheus_client import CollectorRegistry
from prometheus_client import Counter
from prometheus_client import Gauge
from prometheus_client import Histogram
from prometheus_client import generate_latest

# Configure structured logging
logger = structlog.get_logger()

# Type aliases
MetricValue = int | float
Labels = dict[str, str]
T = TypeVar('T')

# HTTP status code constants
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_TOO_MANY_REQUESTS = 429
HTTP_INTERNAL_SERVER_ERROR = 500
HTTP_BAD_GATEWAY = 502
HTTP_SERVICE_UNAVAILABLE = 503

# Other constants
CONNECTION_CHANGE_THRESHOLD = 10
LARGE_QUEUE_THRESHOLD = 1000
MAX_CACHE_SIZE = 1000
CERTIFICATE_EXPIRY_WARNING_DAYS = 30


class MemoryManager(Protocol):
    """Protocol for memory manager interface."""

    async def get_memory_count(self) -> int:
        """Get total number of memories in the system."""
        ...

    @property
    def db(self) -> Any:
        """Database connection or client."""
        ...

@dataclass
class MetricConfig:
    """Configuration for metrics collection.

    Attributes:
        enable_detailed_metrics: Whether to collect detailed per-user metrics
        max_label_values: Maximum number of unique values per label
        retention_seconds: How long to keep metrics in memory
        batch_size: Size of batches for bulk operations
        enable_histogram_buckets: Whether to use custom histogram buckets
    """

    enable_detailed_metrics: bool = True
    max_label_values: int = 1000
    retention_seconds: int = 3600  # 1 hour
    batch_size: int = 100
    enable_histogram_buckets: bool = True
    custom_registry: CollectorRegistry | None = None


# Enhanced Prometheus metrics with optimized buckets
def create_prometheus_metrics(config: MetricConfig) -> dict[str, Any]:
    """Create Prometheus metrics with enhanced configuration.

    Args:
        config: Metric configuration

    Returns:
        Dictionary of Prometheus metric objects
    """
    registry = config.custom_registry or REGISTRY

    # Optimized histogram buckets for different metric types
    duration_buckets = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    size_buckets = [1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216]  # 1KB to 16MB
    search_buckets = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]  # Search-specific buckets

    metrics = {
        'REQUEST_COUNT': Counter(
            'mem0_requests_total',
            'Total HTTP requests processed',
            ['method', 'endpoint', 'status'],
            registry=registry
        ),
        'REQUEST_DURATION': Histogram(
            'mem0_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            buckets=duration_buckets if config.enable_histogram_buckets else None,
            registry=registry
        ),
        'MEMORY_COUNT': Gauge(
            'mem0_memories_total',
            'Total number of memories stored in database',
            registry=registry
        ),
        'USER_COUNT': Gauge(
            'mem0_users_total',
            'Total number of registered users',
            registry=registry
        ),
        'ACTIVE_CONNECTIONS': Gauge(
            'mem0_websocket_connections_active',
            'Number of active WebSocket connections',
            registry=registry
        ),
        'SEARCH_DURATION': Histogram(
            'mem0_search_duration_seconds',
            'Memory search operation duration',
            ['user_id', 'index_type'] if config.enable_detailed_metrics else ['index_type'],
            buckets=search_buckets if config.enable_histogram_buckets else None,
            registry=registry
        ),
        'EMBEDDING_DURATION': Histogram(
            'mem0_embedding_generation_duration_seconds',
            'Embedding generation duration',
            ['provider', 'model'],
            buckets=duration_buckets if config.enable_histogram_buckets else None,
            registry=registry
        ),
        'ERROR_COUNT': Counter(
            'mem0_errors_total',
            'Total number of errors by type and component',
            ['error_type', 'component'],
            registry=registry
        ),
        'DATABASE_OPERATIONS': Counter(
            'mem0_database_operations_total',
            'Total database operations by type and table',
            ['operation', 'table'],
            registry=registry
        ),
        'CACHE_HIT_RATE': Gauge(
            'mem0_cache_hit_rate',
            'Cache hit rate percentage by cache type',
            ['cache_type'],
            registry=registry
        ),
        'API_REQUEST_SIZE': Histogram(
            'mem0_api_request_size_bytes',
            'API request payload size in bytes',
            ['method', 'endpoint'],
            buckets=size_buckets if config.enable_histogram_buckets else None,
            registry=registry
        ),
        'API_RESPONSE_SIZE': Histogram(
            'mem0_api_response_size_bytes',
            'API response payload size in bytes',
            ['method', 'endpoint'],
            buckets=size_buckets if config.enable_histogram_buckets else None,
            registry=registry
        ),
        'SYSTEM_MEMORY_USAGE': Gauge(
            'mem0_system_memory_usage_bytes',
            'System memory usage in bytes',
            registry=registry
        ),
        'SYSTEM_CPU_USAGE': Gauge(
            'mem0_system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=registry
        ),
        'PROCESSING_QUEUE_SIZE': Gauge(
            'mem0_processing_queue_size',
            'Number of items in processing queue',
            ['queue_type'],
            registry=registry
        ),
        'BACKGROUND_TASK_DURATION': Histogram(
            'mem0_background_task_duration_seconds',
            'Duration of background tasks',
            ['task_type'],
            buckets=duration_buckets if config.enable_histogram_buckets else None,
            registry=registry
        ),
    }

    return metrics


# Initialize metrics with default configuration
default_config = MetricConfig()
prometheus_metrics = create_prometheus_metrics(default_config)


class MetricsCollector:
    """Enhanced metrics collector with advanced features.

    This collector provides:
    - Thread-safe metric recording
    - Label cardinality protection
    - Memory-efficient storage
    - Automatic cleanup of old metrics
    - Performance monitoring
    - Error tracking with context
    """

    def __init__(self, config: MetricConfig | None = None) -> None:
        """Initialize the metrics collector.

        Args:
            config: Optional configuration for metrics collection
        """
        self.config = config or MetricConfig()
        self.start_time = time.time()
        self.request_times: dict[str, float] = {}

        # Thread-safe storage
        self._lock = threading.RLock()

        # Label cardinality tracking to prevent metric explosion
        self._label_cardinality: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._label_value_counts: dict[str, int] = defaultdict(int)

        # Performance tracking
        self._operation_counts = defaultdict(int)
        self._operation_durations: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Error aggregation
        self._error_patterns: dict[str, int] = defaultdict(int)
        self._recent_errors: deque = deque(maxlen=100)

        # Weak references to prevent memory leaks
        self._metric_refs: weakref.WeakSet = weakref.WeakSet()

        logger.info("MetricsCollector initialized",
                   enable_detailed_metrics=self.config.enable_detailed_metrics,
                   max_label_values=self.config.max_label_values)

    def record_request_start(
        self,
        request_id: str,
        method: str,
        endpoint: str,
        request_size: int | None = None
    ) -> None:
        """Record the start of a request with enhanced tracking.

        Args:
            request_id: Unique identifier for the request
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            request_size: Size of request payload in bytes
        """
        try:
            with self._lock:
                self.request_times[request_id] = time.time()
                self._operation_counts['requests_started'] += 1

            # Record request size if provided
            if request_size is not None and request_size > 0:
                endpoint_normalized = self._normalize_endpoint(endpoint)
                prometheus_metrics['API_REQUEST_SIZE'].labels(
                    method=method,
                    endpoint=endpoint_normalized
                ).observe(request_size)

        except Exception as e:
            logger.error("Error recording request start",
                        request_id=request_id,
                        method=method,
                        endpoint=endpoint,
                        error=str(e))

    def record_request_end(
        self,
        request_id: str,
        method: str,
        endpoint: str,
        status_code: int,
        response_size: int | None = None,
        error_details: str | None = None
    ) -> None:
        """Record the end of a request with comprehensive metrics.

        Args:
            request_id: Unique identifier for the request
            method: HTTP method
            endpoint: API endpoint path
            status_code: HTTP status code
            response_size: Size of response payload in bytes
            error_details: Optional error details for failed requests
        """
        try:
            with self._lock:
                if request_id not in self.request_times:
                    logger.warning("Request end recorded without start", request_id=request_id)
                    return

                duration = time.time() - self.request_times[request_id]
                del self.request_times[request_id]
                self._operation_counts['requests_completed'] += 1

            # Normalize endpoint to prevent label explosion
            endpoint_normalized = self._normalize_endpoint(endpoint)

            # Record duration and count
            prometheus_metrics['REQUEST_DURATION'].labels(
                method=method,
                endpoint=endpoint_normalized
            ).observe(duration)

            prometheus_metrics['REQUEST_COUNT'].labels(
                method=method,
                endpoint=endpoint_normalized,
                status=str(status_code)
            ).inc()

            # Record response size if provided
            if response_size is not None and response_size > 0:
                prometheus_metrics['API_RESPONSE_SIZE'].labels(
                    method=method,
                    endpoint=endpoint_normalized
                ).observe(response_size)

            # Track request duration for performance analysis
            operation_key = f"{method}:{endpoint_normalized}"
            self._operation_durations[operation_key].append(duration)

            # Record errors for failed requests
            if status_code >= HTTP_BAD_REQUEST:
                error_type = self._categorize_http_error(status_code)
                self.record_error(
                    error_type=error_type,
                    error_message=error_details or f"HTTP {status_code}",
                    component="api",
                    context={
                        "method": method,
                        "endpoint": endpoint_normalized,
                        "status_code": status_code,
                        "duration": duration
                    }
                )

        except Exception as e:
            logger.error("Error recording request end",
                        request_id=request_id,
                        method=method,
                        endpoint=endpoint,
                        status_code=status_code,
                        error=str(e))

    def record_search(
        self,
        duration: float,
        results_count: int,
        user_id: str,
        index_type: str = "default",
        query_complexity: str | None = None,
        cache_hit: bool = False
    ) -> None:
        """Record search operation metrics with enhanced tracking.

        Args:
            duration: Search duration in seconds
            results_count: Number of results returned
            user_id: User identifier
            index_type: Type of index used
            query_complexity: Optional complexity indicator (simple, medium, complex)
            cache_hit: Whether the result was served from cache
        """
        try:
            # Validate inputs
            if duration < 0 or not isinstance(duration, int | float):
                logger.warning("Invalid search duration", duration=duration)
                return

            if results_count < 0:
                logger.warning("Invalid results count", results_count=results_count)
                return

            # Sanitize user_id to prevent label explosion
            sanitized_user_id = self._sanitize_label_value("user_id", user_id)

            # Record search duration
            labels = {'index_type': index_type}
            if self.config.enable_detailed_metrics and sanitized_user_id:
                labels['user_id'] = sanitized_user_id

            prometheus_metrics['SEARCH_DURATION'].labels(**labels).observe(duration)

            # Track search performance
            self._operation_counts['searches'] += 1
            self._operation_durations['search'].append(duration)

            # Record cache hit rate if applicable
            if cache_hit:
                prometheus_metrics['CACHE_HIT_RATE'].labels(cache_type='search').set(1.0)

            # Log with structured context
            log_context = {
                "operation": "search",
                "duration": duration,
                "results_count": results_count,
                "index_type": index_type,
                "cache_hit": cache_hit
            }

            if self.config.enable_detailed_metrics:
                log_context["user_id"] = user_id

            if query_complexity:
                log_context["query_complexity"] = query_complexity

            logger.info("Search metrics recorded", **log_context)

        except Exception as e:
            logger.error("Error recording search metrics",
                        duration=duration,
                        results_count=results_count,
                        user_id=user_id,
                        error=str(e))

    def record_embedding_generation(
        self,
        duration: float,
        text_length: int,
        provider: str = "default",
        model: str = "default",
        batch_size: int = 1,
        success: bool = True
    ) -> None:
        """Record embedding generation metrics with enhanced tracking.

        Args:
            duration: Generation duration in seconds
            text_length: Length of input text in characters
            provider: Embedding provider (openai, huggingface, etc.)
            model: Model name used
            batch_size: Number of texts processed in batch
            success: Whether generation was successful
        """
        try:
            # Validate inputs
            if duration < 0 or not isinstance(duration, int | float):
                logger.warning("Invalid embedding duration", duration=duration)
                return

            if text_length < 0:
                logger.warning("Invalid text length", text_length=text_length)
                return

            # Sanitize provider and model names
            sanitized_provider = self._sanitize_label_value("provider", provider)
            sanitized_model = self._sanitize_label_value("model", model)

            # Record duration
            prometheus_metrics['EMBEDDING_DURATION'].labels(
                provider=sanitized_provider,
                model=sanitized_model
            ).observe(duration)

            # Track performance
            self._operation_counts['embeddings'] += 1
            self._operation_durations['embedding'].append(duration)

            # Calculate throughput metrics
            tokens_per_second = text_length / max(duration, 0.001)  # Avoid division by zero

            logger.info("Embedding metrics recorded",
                       duration=duration,
                       text_length=text_length,
                       provider=sanitized_provider,
                       model=sanitized_model,
                       batch_size=batch_size,
                       tokens_per_second=tokens_per_second,
                       success=success)

            # Record error if generation failed
            if not success:
                self.record_error(
                    error_type="EmbeddingGenerationError",
                    error_message="Embedding generation failed",
                    component="embedding",
                    context={
                        "provider": sanitized_provider,
                        "model": sanitized_model,
                        "text_length": text_length,
                        "duration": duration
                    }
                )

        except Exception as e:
            logger.error("Error recording embedding metrics",
                        duration=duration,
                        text_length=text_length,
                        provider=provider,
                        model=model,
                        error=str(e))

    def record_error(
        self,
        error_type: str,
        error_message: str,
        component: str = "unknown",
        context: dict[str, Any] | None = None,
        severity: str = "error",
        recoverable: bool = True
    ) -> None:
        """Record error metrics with enhanced categorization and tracking.

        Args:
            error_type: Type/category of error
            error_message: Detailed error message
            component: Component where error occurred
            context: Additional context information
            severity: Error severity level (error, warning, critical)
            recoverable: Whether the error is recoverable
        """
        try:
            # Sanitize labels to prevent metric explosion
            sanitized_error_type = self._sanitize_label_value("error_type", error_type)
            sanitized_component = self._sanitize_label_value("component", component)

            # Record error count
            prometheus_metrics['ERROR_COUNT'].labels(
                error_type=sanitized_error_type,
                component=sanitized_component
            ).inc()

            # Track error patterns for analysis
            error_pattern = f"{sanitized_error_type}:{sanitized_component}"
            self._error_patterns[error_pattern] += 1

            # Store recent error for analysis
            error_record = {
                "timestamp": datetime.now(),
                "error_type": sanitized_error_type,
                "component": sanitized_component,
                "message": error_message[:500],  # Limit message length
                "severity": severity,
                "recoverable": recoverable,
                "context": context or {}
            }

            with self._lock:
                self._recent_errors.append(error_record)
                self._operation_counts['errors'] += 1

            # Enhanced logging with context
            log_context = {
                "event_type": "error_recorded",
                "error_type": sanitized_error_type,
                "error_message": error_message,
                "component": sanitized_component,
                "severity": severity,
                "recoverable": recoverable
            }

            if context:
                # Sanitize context to avoid logging sensitive data
                sanitized_context = self._sanitize_context(context)
                log_context["context"] = sanitized_context

            # Use appropriate log level based on severity
            if severity == "critical":
                logger.critical("Critical error recorded", **log_context)
            elif severity == "warning":
                logger.warning("Warning recorded", **log_context)
            else:
                logger.error("Error recorded", **log_context)

        except Exception as e:
            # Avoid recursive error logging
            logger.error("Failed to record error metric",
                        original_error_type=error_type,
                        recording_error=str(e))

    def update_memory_count(self, count: int) -> None:
        """Update memory count gauge with validation.

        Args:
            count: Total number of memories
        """
        try:
            if count < 0:
                logger.warning("Invalid memory count", count=count)
                return

            prometheus_metrics['MEMORY_COUNT'].set(count)

            # Track memory growth rate
            current_time = time.time()
            if hasattr(self, '_last_memory_count') and hasattr(self, '_last_memory_update'):
                time_diff = current_time - self._last_memory_update
                count_diff = count - self._last_memory_count

                if time_diff > 0:
                    growth_rate = count_diff / time_diff  # memories per second
                    logger.debug("Memory growth rate",
                               memories_per_second=growth_rate,
                               total_memories=count)

            self._last_memory_count = count
            self._last_memory_update = current_time

        except Exception as e:
            logger.error("Error updating memory count", count=count, error=str(e))

    def update_user_count(self, count: int) -> None:
        """Update user count gauge with validation.

        Args:
            count: Total number of users
        """
        try:
            if count < 0:
                logger.warning("Invalid user count", count=count)
                return

            prometheus_metrics['USER_COUNT'].set(count)

            logger.debug("User count updated", total_users=count)

        except Exception as e:
            logger.error("Error updating user count", count=count, error=str(e))

    def update_active_connections(self, count: int) -> None:
        """Update active connections gauge with validation.

        Args:
            count: Number of active connections
        """
        try:
            if count < 0:
                logger.warning("Invalid connection count", count=count)
                return

            prometheus_metrics['ACTIVE_CONNECTIONS'].set(count)

            # Log significant changes in connection count
            if hasattr(self, '_last_connection_count'):
                diff = abs(count - self._last_connection_count)
                if diff > CONNECTION_CHANGE_THRESHOLD:
                    logger.info("Significant connection count change",
                              previous=self._last_connection_count,
                              current=count,
                              change=count - self._last_connection_count)

            self._last_connection_count = count

        except Exception as e:
            logger.error("Error updating connection count", count=count, error=str(e))

    def get_uptime(self) -> float:
        """Get server uptime in seconds.

        Returns:
            Uptime in seconds since collector initialization
        """
        return time.time() - self.start_time

    def update_queue_size(self, queue_type: str, size: int) -> None:
        """Update processing queue size metric.

        Args:
            queue_type: Type of queue (e.g., 'embedding', 'indexing')
            size: Current queue size
        """
        try:
            if size < 0:
                logger.warning("Invalid queue size", queue_type=queue_type, size=size)
                return

            sanitized_queue_type = self._sanitize_label_value("queue_type", queue_type)
            prometheus_metrics['PROCESSING_QUEUE_SIZE'].labels(
                queue_type=sanitized_queue_type
            ).set(size)

            # Alert if queue is getting large
            if size > LARGE_QUEUE_THRESHOLD:
                logger.warning("Large processing queue detected",
                              queue_type=sanitized_queue_type,
                              size=size)

        except Exception as e:
            logger.error("Error updating queue size",
                        queue_type=queue_type,
                        size=size,
                        error=str(e))

    def record_background_task(
        self,
        task_type: str,
        duration: float,
        success: bool = True
    ) -> None:
        """Record background task execution metrics.

        Args:
            task_type: Type of background task
            duration: Task duration in seconds
            success: Whether task completed successfully
        """
        try:
            if duration < 0:
                logger.warning("Invalid task duration", task_type=task_type, duration=duration)
                return

            sanitized_task_type = self._sanitize_label_value("task_type", task_type)
            prometheus_metrics['BACKGROUND_TASK_DURATION'].labels(
                task_type=sanitized_task_type
            ).observe(duration)

            logger.debug("Background task completed",
                        task_type=sanitized_task_type,
                        duration=duration,
                        success=success)

            if not success:
                self.record_error(
                    error_type="BackgroundTaskError",
                    error_message=f"Background task {task_type} failed",
                    component="background_tasks",
                    context={"task_type": sanitized_task_type, "duration": duration}
                )

        except Exception as e:
            logger.error("Error recording background task",
                        task_type=task_type,
                        duration=duration,
                        error=str(e))

    def _sanitize_label_value(self, label_name: str, value: str) -> str:
        """Sanitize label values with cardinality protection.

        Args:
            label_name: Name of the label
            value: Label value to sanitize

        Returns:
            Sanitized label value
        """
        if not value:
            return "unknown"

        # Convert to string and limit length
        str_value = str(value)[:100]

        # Replace invalid characters for Prometheus
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9_.-]', '_', str_value)

        # Protect against label cardinality explosion
        with self._lock:
            self._label_cardinality[label_name][sanitized] += 1

            # If we have too many unique values for this label, use "other"
            if (len(self._label_cardinality[label_name]) > self.config.max_label_values and
                self._label_cardinality[label_name][sanitized] == 1):
                logger.warning("Label cardinality limit exceeded",
                              label_name=label_name,
                              unique_values=len(self._label_cardinality[label_name]),
                              limit=self.config.max_label_values)
                return "other"

        return sanitized if sanitized else "unknown"

    def _normalize_endpoint(self, endpoint: str) -> str:
        """Normalize API endpoint to prevent label explosion.

        Args:
            endpoint: Raw endpoint path

        Returns:
            Normalized endpoint path
        """
        if not endpoint:
            return "unknown"

        import re

        # Replace UUIDs with placeholder
        normalized = re.sub(
            r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '/{uuid}',
            endpoint
        )

        # Replace numeric IDs with placeholder
        normalized = re.sub(r'/\d+', '/{id}', normalized)

        # Replace hash-like strings with placeholder
        normalized = re.sub(r'/[a-f0-9]{16,}', '/{hash}', normalized)

        # Limit length
        return normalized[:200]

    def _categorize_http_error(self, status_code: int) -> str:
        """Categorize HTTP status codes into error types.

        Args:
            status_code: HTTP status code

        Returns:
            Error category string
        """
        # Use dictionary for cleaner lookup instead of consecutive if statements
        client_errors = {
            HTTP_BAD_REQUEST: "BadRequest",
            HTTP_UNAUTHORIZED: "Unauthorized",
            HTTP_FORBIDDEN: "Forbidden",
            HTTP_NOT_FOUND: "NotFound",
            HTTP_TOO_MANY_REQUESTS: "RateLimited",
        }

        server_errors = {
            HTTP_INTERNAL_SERVER_ERROR: "InternalServerError",
            HTTP_BAD_GATEWAY: "BadGateway",
            HTTP_SERVICE_UNAVAILABLE: "ServiceUnavailable",
            504: "GatewayTimeout",
        }

        if HTTP_BAD_REQUEST <= status_code < HTTP_INTERNAL_SERVER_ERROR:
            return client_errors.get(status_code, "ClientError")
        elif HTTP_INTERNAL_SERVER_ERROR <= status_code < ALERT_INTERVAL_SECONDS:
            return server_errors.get(status_code, "ServerError")
        else:
            return "UnknownError"

    def _sanitize_context(self, context: dict[str, Any]) -> dict[str, Any]:
        """Sanitize context dictionary to remove sensitive data.

        Args:
            context: Original context dictionary

        Returns:
            Sanitized context dictionary
        """
        if not context:
            return {}

        # List of sensitive keys to redact
        sensitive_keys = {
            'password', 'token', 'key', 'secret', 'auth', 'credential',
            'api_key', 'access_token', 'refresh_token', 'session_id'
        }

        sanitized = {}
        for key, value in context.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, str) and len(value) > DEFAULT_BATCH_SIZE:
                sanitized[key] = value[:1000] + "...[TRUNCATED]"
            else:
                sanitized[key] = value

        return sanitized

    def collect_system_metrics(self) -> None:
        """Collect system metrics including CPU and memory usage."""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            prometheus_metrics['SYSTEM_MEMORY_USAGE'].set(memory.used)

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            prometheus_metrics['SYSTEM_CPU_USAGE'].set(cpu_percent)

            logger.debug("System metrics collected",
                        memory_used_bytes=memory.used,
                        memory_percent=memory.percent,
                        cpu_percent=cpu_percent)

        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get a summary of current metrics."""
        try:
            # Safe access to metric values using prometheus_client API
            memory_count = prometheus_metrics['MEMORY_COUNT']._value._value if hasattr(prometheus_metrics['MEMORY_COUNT'], '_value') else 0
            user_count = prometheus_metrics['USER_COUNT']._value._value if hasattr(prometheus_metrics['USER_COUNT'], '_value') else 0
            active_connections = prometheus_metrics['ACTIVE_CONNECTIONS']._value._value if hasattr(prometheus_metrics['ACTIVE_CONNECTIONS'], '_value') else 0

            # Get counter totals safely
            total_requests = 0
            total_errors = 0

            if hasattr(prometheus_metrics['REQUEST_COUNT'], '_value'):
                total_requests = sum(sample.value for sample in prometheus_metrics['REQUEST_COUNT'].collect()[0].samples)

            if hasattr(prometheus_metrics['ERROR_COUNT'], '_value'):
                total_errors = sum(sample.value for sample in prometheus_metrics['ERROR_COUNT'].collect()[0].samples)

            return {
                "uptime_seconds": self.get_uptime(),
                "memory_count": memory_count,
                "user_count": user_count,
                "active_connections": active_connections,
                "total_requests": total_requests,
                "total_errors": total_errors
            }
        except Exception as e:
            logger.error("Error getting metrics summary", error=str(e))
            return {
                "uptime_seconds": self.get_uptime(),
                "memory_count": 0,
                "user_count": 0,
                "active_connections": 0,
                "total_requests": 0,
                "total_errors": 0
            }


# Global metrics collector
metrics_collector = MetricsCollector()


def setup_metrics(app):
    """Setup metrics endpoints."""

    # Schedule system metrics collection
    async def collect_system_metrics_periodically():
        while True:
            try:
                metrics_collector.collect_system_metrics()
                await asyncio.sleep(30)  # Collect every 30 seconds
            except Exception as e:
                logger.error("Error in system metrics collection", error=str(e))
                await asyncio.sleep(60)

    # Start background task for system metrics
    asyncio.create_task(collect_system_metrics_periodically())


def setup_metrics_old(app):
    """Setup metrics endpoints."""

    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/health/metrics")
    async def health_metrics():
        """Health check with metrics."""
        return metrics_collector.get_metrics_summary()


class PerformanceMonitor:
    """Monitor performance and trigger alerts."""

    def __init__(self, memory_manager=None):
        self.memory_manager = memory_manager
        self.thresholds = {
            "request_duration": 5.0,  # seconds
            "search_duration": 2.0,   # seconds
            "error_rate": 0.05,       # 5%
            "memory_usage": 0.8       # 80%
        }
        self.alerts_sent = set()
        self.monitoring_task = None
        self._running = False

    async def start_monitoring(self):
        """Start performance monitoring background task."""
        if self._running:
            logger.warning("Performance monitoring already running")
            return

        self._running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")

    async def stop_monitoring(self):
        """Stop performance monitoring."""
        self._running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.monitoring_task
        logger.info("Performance monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_performance()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(10)

    async def _check_performance(self):
        """Check performance metrics and trigger alerts."""
        metrics = metrics_collector.get_metrics_summary()

        # Check error rate
        total_requests = metrics.get("total_requests", 1)
        total_errors = metrics.get("total_errors", 0)
        error_rate = total_errors / total_requests if total_requests > 0 else 0

        if error_rate > self.thresholds["error_rate"]:
            alert_key = f"error_rate_{int(time.time() // 300)}"  # Group by 5-minute windows
            if alert_key not in self.alerts_sent:
                logger.warning("High error rate detected",
                             error_rate=error_rate,
                             threshold=self.thresholds["error_rate"])
                self.alerts_sent.add(alert_key)

        # Update database metrics safely
        if self.memory_manager and hasattr(self.memory_manager, 'get_memory_count'):
            try:
                memory_count = await self.memory_manager.get_memory_count()
                metrics_collector.update_memory_count(memory_count)
            except Exception as e:
                logger.error("Failed to update memory count", error=str(e))
                # Set to 0 as fallback
                metrics_collector.update_memory_count(0)

    def set_threshold(self, metric: str, value: float):
        """Set performance threshold."""
        if metric in self.thresholds:
            self.thresholds[metric] = value
            logger.info("Threshold updated", metric=metric, value=value)


# Global performance monitor
performance_monitor = PerformanceMonitor()


class RequestTracker:
    """Track individual request performance."""

    def __init__(self, request_id: str, method: str, endpoint: str):
        self.request_id = request_id
        self.method = method
        self.endpoint = endpoint
        self.start_time = time.time()

        metrics_collector.record_request_start(request_id, method, endpoint)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status_code = 500 if exc_type else 200
        metrics_collector.record_request_end(
            self.request_id, self.method, self.endpoint, status_code
        )

        if exc_type:
            metrics_collector.record_error(
                error_type=exc_type.__name__,
                error_message=str(exc_val),
                component="request_tracker",
                context={
                    "request_id": self.request_id,
                    "method": self.method,
                    "endpoint": self.endpoint
                }
            )


def track_request(request_id: str, method: str, endpoint: str):
    """Decorator for tracking request performance."""
    return RequestTracker(request_id, method, endpoint)


class DatabaseHealthChecker:
    """Monitor database health and performance."""

    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.last_check = None
        self.health_status = "unknown"

    async def check_health(self) -> dict[str, Any]:
        """Check database health."""
        try:
            start_time = time.time()

            # Test basic connectivity with timeout
            if hasattr(self.memory_manager, 'db') and self.memory_manager.db:
                await asyncio.wait_for(
                    self.memory_manager.db.execute_one("SELECT 1"),
                    timeout=5.0
                )
            else:
                raise Exception("Database connection not available")

            # Test memory count query with timeout
            if hasattr(self.memory_manager, 'get_memory_count'):
                memory_count = await asyncio.wait_for(
                    self.memory_manager.get_memory_count(),
                    timeout=10.0
                )
            else:
                memory_count = 0

            duration = time.time() - start_time

            self.health_status = "healthy"
            self.last_check = datetime.now()

            return {
                "status": "healthy",
                "response_time_ms": round(duration * 1000, 2),
                "memory_count": memory_count,
                "last_check": self.last_check.isoformat(),
                "database_connected": True
            }

        except TimeoutError:
            self.health_status = "unhealthy"
            self.last_check = datetime.now()

            logger.error("Database health check timed out")

            return {
                "status": "unhealthy",
                "error": "Database health check timed out",
                "last_check": self.last_check.isoformat(),
                "database_connected": False
            }

        except Exception as e:
            self.health_status = "unhealthy"
            self.last_check = datetime.now()

            logger.error("Database health check failed", error=str(e))

            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": self.last_check.isoformat(),
                "database_connected": False
            }

    async def start_health_monitoring(self):
        """Start database health monitoring."""
        logger.info("Starting database health monitoring")

        while True:
            try:
                health_result = await self.check_health()

                # Log health status changes
                if health_result["status"] != self.health_status:
                    logger.info("Database health status changed",
                              old_status=self.health_status,
                              new_status=health_result["status"])

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                logger.info("Database health monitoring cancelled")
                break
            except Exception as e:
                logger.error("Error in database health monitoring", error=str(e))
                await asyncio.sleep(10)
