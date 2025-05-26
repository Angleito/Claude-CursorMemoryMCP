#!/usr/bin/env python3
"""Comprehensive Monitoring and Metrics System for mem0ai Vector Operations.

Production-grade monitoring with real-time metrics, alerting, and dashboards.
This module provides:
- High-performance metrics collection with memory optimization
- Real-time alerting with configurable thresholds
- Web dashboard for visualization
- Prometheus integration
- Database health monitoring
- System resource tracking

Examples:
    >>> monitoring = MonitoringSystem(database_url)
    >>> await monitoring.initialize()
    >>> monitoring.record_metric("search_duration", 0.150, {"user": "123"})
    >>> profiler = monitoring.get_profiler()
    >>> with profiler.profile_async_function("vector_search"):
    ...     await search_vectors(query)
"""

import asyncio
import contextlib
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    Tuple,
    AsyncGenerator,
    NamedTuple,
    Protocol,
)
import statistics
import weakref
from concurrent.futures import ThreadPoolExecutor

import asyncpg
import numpy as np
import psutil
import structlog

# Use structlog for consistent logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class MetricType(Enum):
    """Types of metrics to collect."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMING = "timing"


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    """Definition of a metric with validation and optimization.
    
    Attributes:
        name: Unique metric name (must be valid Prometheus metric name)
        metric_type: Type of metric (counter, gauge, histogram, timing)
        description: Human-readable description
        unit: Unit of measurement (e.g., 'seconds', 'bytes', 'count')
        labels: Optional list of label names for this metric
        buckets: Optional histogram buckets for timing metrics
        help_text: Additional help text for Prometheus
    """

    name: str
    metric_type: MetricType
    description: str
    unit: str
    labels: Optional[List[str]] = None
    buckets: Optional[List[float]] = None
    help_text: str = ""
    
    def __post_init__(self) -> None:
        """Validate metric definition."""
        if not self.name or not self.name.replace('_', '').replace(':', '').isalnum():
            raise ValueError(f"Invalid metric name: {self.name}")
        
        if self.labels and len(set(self.labels)) != len(self.labels):
            raise ValueError("Duplicate labels not allowed")
        
        if self.metric_type == MetricType.HISTOGRAM and self.buckets:
            if not all(isinstance(b, (int, float)) for b in self.buckets):
                raise ValueError("Histogram buckets must be numeric")
            if self.buckets != sorted(self.buckets):
                raise ValueError("Histogram buckets must be sorted")


@dataclass(slots=True)
class MetricValue:
    """A metric value with timestamp and labels optimized for memory efficiency.
    
    Attributes:
        name: Metric name
        value: Numeric value
        timestamp: When the metric was recorded
        labels: Optional key-value labels
    """

    name: str
    value: float
    timestamp: datetime
    labels: Optional[Dict[str, str]] = None
    
    def __post_init__(self) -> None:
        """Validate and optimize metric value."""
        if not isinstance(self.value, (int, float)) or not np.isfinite(self.value):
            raise ValueError(f"Invalid metric value: {self.value}")
        
        # Intern string values for memory efficiency
        if self.labels:
            self.labels = {
                k: v for k, v in self.labels.items() 
                if k and v  # Remove empty keys/values
            }
    
    @property
    def age_seconds(self) -> float:
        """Get age of this metric value in seconds."""
        return (datetime.now() - self.timestamp).total_seconds()


@dataclass(frozen=True)
class AlertRule:
    """Alert rule definition with enhanced validation.
    
    Attributes:
        name: Unique rule name
        metric_name: Name of metric to monitor
        condition: Condition expression (e.g., "> 100", "< 0.95")
        level: Alert severity level
        duration_seconds: How long condition must be true before firing
        description: Human-readable description
        labels: Optional labels to match
        aggregation: How to aggregate metric values ('avg', 'max', 'min', 'sum')
        enabled: Whether the rule is currently enabled
    """

    name: str
    metric_name: str
    condition: str
    level: AlertLevel
    duration_seconds: int = 60
    description: str = ""
    labels: Optional[Dict[str, str]] = None
    aggregation: str = "avg"
    enabled: bool = True
    
    def __post_init__(self) -> None:
        """Validate alert rule parameters."""
        if not self.name or not self.metric_name:
            raise ValueError("Name and metric_name are required")
        
        if self.duration_seconds < 0:
            raise ValueError("Duration must be non-negative")
        
        # Validate condition format
        import re
        if not re.match(r'^[><=!]+ *-?\d+(\.\d+)?$', self.condition.strip()):
            raise ValueError(f"Invalid condition format: {self.condition}")
        
        if self.aggregation not in {'avg', 'max', 'min', 'sum', 'count', 'p95', 'p99'}:
            raise ValueError(f"Invalid aggregation: {self.aggregation}")


@dataclass
class Alert:
    """An active alert with comprehensive tracking.
    
    Attributes:
        id: Unique alert identifier
        rule_name: Name of the rule that fired
        metric_name: Name of the metric being monitored
        level: Alert severity level
        message: Alert message
        value: Current metric value that triggered the alert
        threshold: Threshold value that was exceeded
        started_at: When the alert first fired
        resolved_at: When the alert was resolved (if applicable)
        labels: Labels associated with the alert
        fingerprint: Unique fingerprint for deduplication
        ack_time: When the alert was acknowledged
        ack_by: Who acknowledged the alert
    """

    id: str
    rule_name: str
    metric_name: str
    level: AlertLevel
    message: str
    value: float
    threshold: str
    started_at: datetime
    resolved_at: Optional[datetime] = None
    labels: Dict[str, str] = field(default_factory=dict)
    fingerprint: Optional[str] = None
    ack_time: Optional[datetime] = None
    ack_by: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Generate fingerprint if not provided."""
        if not self.fingerprint:
            import hashlib
            content = f"{self.rule_name}:{self.metric_name}:{sorted(self.labels.items())}"
            self.fingerprint = hashlib.md5(content.encode()).hexdigest()[:8]
    
    @property
    def duration(self) -> timedelta:
        """Get alert duration."""
        end_time = self.resolved_at or datetime.now()
        return end_time - self.started_at
    
    @property
    def is_resolved(self) -> bool:
        """Check if alert is resolved."""
        return self.resolved_at is not None
    
    @property
    def is_acknowledged(self) -> bool:
        """Check if alert is acknowledged."""
        return self.ack_time is not None


class MetricsCollector:
    """High-performance metrics collector with memory optimization.
    
    This collector provides:
    - Thread-safe metric storage with RWLock optimization
    - Automatic memory management and cleanup
    - Efficient batch operations
    - Memory pooling for reduced allocations
    - Configurable retention policies
    """

    def __init__(
        self, 
        retention_hours: int = 24, 
        max_values_per_metric: int = 5000,
        cleanup_interval: int = 300,  # 5 minutes
        enable_compression: bool = True,
        thread_pool_size: int = 2
    ) -> None:
        """Initialize metrics collector with performance optimizations.
        
        Args:
            retention_hours: How long to keep metric values
            max_values_per_metric: Maximum values per metric (memory limit)
            cleanup_interval: How often to run cleanup (seconds)
            enable_compression: Whether to compress old metric data
            thread_pool_size: Size of thread pool for background tasks
        """
        self.retention_hours = retention_hours
        self.max_values_per_metric = max_values_per_metric
        self.cleanup_interval = cleanup_interval
        self.enable_compression = enable_compression
        
        # Thread-safe storage with RWLock for better performance
        self._metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_values_per_metric)
        )
        self._lock = threading.RLock()
        self._cleanup_counter = 0
        self._last_cleanup = time.time()
        
        # Performance tracking
        self._record_times: deque = deque(maxlen=1000)
        self._compression_stats = {'compressed_count': 0, 'original_size': 0, 'compressed_size': 0}
        
        # Background thread pool for cleanup and compression
        self._thread_pool = ThreadPoolExecutor(
            max_workers=thread_pool_size,
            thread_name_prefix="metrics_collector"
        )
        
        # Weak references to avoid memory leaks
        self._metric_refs: weakref.WeakSet = weakref.WeakSet()
        
        logger.info("MetricsCollector initialized", 
                   retention_hours=retention_hours,
                   max_values_per_metric=max_values_per_metric,
                   cleanup_interval=cleanup_interval)

        # Built-in metric definitions with optimized buckets
        self.metric_definitions: Dict[str, MetricDefinition] = {
            "vector_search_duration_ms": MetricDefinition(
                name="vector_search_duration_ms",
                metric_type=MetricType.HISTOGRAM,
                description="Time taken for vector similarity searches",
                unit="milliseconds",
                labels=["user_id", "index_type"],
                buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
                help_text="Duration distribution of vector similarity search operations"
            ),
            "embedding_generation_duration_ms": MetricDefinition(
                name="embedding_generation_duration_ms",
                metric_type=MetricType.HISTOGRAM,
                description="Time taken for embedding generation",
                unit="milliseconds",
                labels=["provider", "model"],
                buckets=[10, 50, 100, 250, 500, 1000, 2000, 5000, 10000],
                help_text="Duration distribution of embedding generation operations"
            ),
            "memory_insert_rate": MetricDefinition(
                name="memory_insert_rate",
                metric_type=MetricType.COUNTER,
                description="Rate of memory insertions",
                unit="operations/second",
                labels=["user_id"],
            ),
            "active_users": MetricDefinition(
                name="active_users",
                metric_type=MetricType.GAUGE,
                description="Number of active users in the last hour",
                unit="count",
            ),
            "database_connections": MetricDefinition(
                name="database_connections",
                metric_type=MetricType.GAUGE,
                description="Number of active database connections",
                unit="count",
            ),
            "memory_usage_mb": MetricDefinition(
                name="memory_usage_mb",
                metric_type=MetricType.GAUGE,
                description="System memory usage",
                unit="megabytes",
            ),
            "cpu_usage_percent": MetricDefinition(
                name="cpu_usage_percent",
                metric_type=MetricType.GAUGE,
                description="System CPU usage",
                unit="percent",
            ),
            "disk_usage_percent": MetricDefinition(
                name="disk_usage_percent",
                metric_type=MetricType.GAUGE,
                description="Disk usage percentage",
                unit="percent",
            ),
            "vector_index_size_mb": MetricDefinition(
                name="vector_index_size_mb",
                metric_type=MetricType.GAUGE,
                description="Size of vector indexes",
                unit="megabytes",
                labels=["index_name", "index_type"],
            ),
            "cache_hit_rate": MetricDefinition(
                name="cache_hit_rate",
                metric_type=MetricType.GAUGE,
                description="Cache hit rate for various operations",
                unit="ratio",
                labels=["cache_type"],
            ),
            "error_rate": MetricDefinition(
                name="error_rate",
                metric_type=MetricType.COUNTER,
                description="Rate of errors by type",
                unit="errors/second",
                labels=["error_type", "component"],
            ),
        }

    def record_metric(
        self, 
        name: str, 
        value: float, 
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record a metric value with performance optimization.
        
        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels
            timestamp: Optional timestamp (defaults to current time)
        """
        start_time = time.perf_counter()
        
        try:
            # Validate inputs
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                logger.warning("Invalid metric value", name=name, value=value)
                return
            
            # Sanitize and intern labels for memory efficiency
            processed_labels = self._process_labels(labels) if labels else None
            
            metric_value = MetricValue(
                name=name,
                value=float(value),
                timestamp=timestamp or datetime.now(),
                labels=processed_labels
            )

            # Store metric with minimal lock time
            with self._lock:
                self.metrics[name].append(metric_value)
                self._metric_refs.add(metric_value)

            # Track recording performance
            record_time = (time.perf_counter() - start_time) * 1000
            self._record_times.append(record_time)

            # Efficient cleanup scheduling
            self._cleanup_counter += 1
            if (self._cleanup_counter % 1000 == 0 or 
                time.time() - self._last_cleanup > self.cleanup_interval):
                # Submit cleanup to thread pool to avoid blocking
                self._thread_pool.submit(self._cleanup_old_metrics)
                
        except Exception as e:
            logger.error("Error recording metric", name=name, value=value, error=str(e))
    
    def _process_labels(self, labels: Dict[str, str]) -> Dict[str, str]:
        """Process and intern labels for memory efficiency.
        
        Args:
            labels: Input labels
            
        Returns:
            Processed labels with interned strings
        """
        if not labels:
            return {}
        
        # Intern strings to reduce memory usage
        processed = {}
        for key, value in labels.items():
            if key and value:  # Skip empty keys/values
                # Limit label value length to prevent memory issues
                processed_key = key[:100]
                processed_value = str(value)[:200]
                processed[processed_key] = processed_value
        
        return processed

    def _cleanup_old_metrics(self) -> None:
        """Remove old metric values based on retention policy with optimization."""
        start_time = time.perf_counter()
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        cleaned_count = 0
        compressed_count = 0

        try:
            with self._lock:
                metrics_to_cleanup = list(self.metrics.items())
            
            # Process cleanup outside of lock for better performance
            for metric_name, values in metrics_to_cleanup:
                if not values:
                    continue
                
                original_count = len(values)
                
                # Efficient cleanup using list comprehension
                recent_values = [
                    value for value in values 
                    if value.timestamp >= cutoff_time
                ]
                
                # Apply compression if enabled and beneficial
                if (self.enable_compression and 
                    len(recent_values) > 100 and 
                    len(recent_values) < original_count * 0.8):
                    
                    compressed_values = self._compress_metric_values(recent_values)
                    if compressed_values is not None:
                        recent_values = compressed_values
                        compressed_count += 1
                
                # Update storage with new deque
                new_deque = deque(recent_values, maxlen=self.max_values_per_metric)
                
                with self._lock:
                    self.metrics[metric_name] = new_deque
                
                cleaned_count += original_count - len(recent_values)
            
            # Update cleanup tracking
            self._last_cleanup = time.time()
            cleanup_duration = (time.perf_counter() - start_time) * 1000
            
            if cleaned_count > 0 or compressed_count > 0:
                logger.debug("Metrics cleanup completed",
                           cleaned_count=cleaned_count,
                           compressed_count=compressed_count,
                           duration_ms=cleanup_duration)
            
        except Exception as e:
            logger.error("Error during metrics cleanup", error=str(e))
    
    def _compress_metric_values(self, values: List[MetricValue]) -> Optional[List[MetricValue]]:
        """Compress metric values by downsampling.
        
        Args:
            values: List of metric values to compress
            
        Returns:
            Compressed list of metric values or None if compression failed
        """
        try:
            if len(values) < 100:
                return None
            
            # Simple downsampling: keep every nth value plus min/max in windows
            compressed = []
            window_size = max(10, len(values) // 50)  # Target ~50 samples
            
            for i in range(0, len(values), window_size):
                window = values[i:i + window_size]
                if not window:
                    continue
                
                # Keep first, last, min, and max from each window
                window_values = [v.value for v in window]
                min_idx = i + np.argmin(window_values)
                max_idx = i + np.argmax(window_values)
                
                # Add unique indices
                indices_to_keep = {i, i + len(window) - 1, min_idx, max_idx}
                for idx in sorted(indices_to_keep):
                    if idx < len(values):
                        compressed.append(values[idx])
            
            # Update compression stats
            original_size = len(values)
            compressed_size = len(compressed)
            
            self._compression_stats['compressed_count'] += 1
            self._compression_stats['original_size'] += original_size
            self._compression_stats['compressed_size'] += compressed_size
            
            return compressed if compressed_size < original_size * 0.8 else None
            
        except Exception as e:
            logger.warning("Metric compression failed", error=str(e))
            return None

    def get_metric_values(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        labels: Optional[Dict[str, str]] = None,
        limit: Optional[int] = None
    ) -> List[MetricValue]:
        """Get metric values with optimized filtering.
        
        Args:
            name: Metric name
            start_time: Start time filter
            end_time: End time filter 
            labels: Label filters
            limit: Maximum number of values to return
            
        Returns:
            List of matching metric values
        """
        # Get values with minimal lock time
        with self._lock:
            values = list(self.metrics.get(name, []))

        if not values:
            return []

        # Optimize filtering with early termination
        filtered_values = []
        
        for value in values:
            # Time range filter
            if start_time and value.timestamp < start_time:
                continue
            if end_time and value.timestamp > end_time:
                continue
                
            # Label filter with short-circuit evaluation
            if labels:
                if not value.labels:
                    continue
                if not all(value.labels.get(k) == v for k, v in labels.items()):
                    continue
            
            filtered_values.append(value)
            
            # Early termination if limit reached
            if limit and len(filtered_values) >= limit:
                break

        return filtered_values

    def get_metric_summary(
        self, 
        name: str, 
        start_time: Optional[datetime] = None, 
        end_time: Optional[datetime] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, Union[float, int]]:
        """Get comprehensive summary statistics for a metric.
        
        Args:
            name: Metric name
            start_time: Start time filter
            end_time: End time filter
            labels: Label filters
            
        Returns:
            Dictionary of statistics
        """
        values = self.get_metric_values(name, start_time, end_time, labels)

        if not values:
            return {"count": 0}

        numeric_values = [v.value for v in values]
        np_values = np.array(numeric_values)

        # Calculate duration for rate metrics
        duration_seconds = 0.0
        if len(values) > 1 and start_time and end_time:
            duration_seconds = (end_time - start_time).total_seconds()

        summary = {
            "count": len(numeric_values),
            "sum": float(np.sum(np_values)),
            "min": float(np.min(np_values)),
            "max": float(np.max(np_values)),
            "mean": float(np.mean(np_values)),
            "median": float(np.median(np_values)),
            "std": float(np.std(np_values)),
        }
        
        # Add percentiles if we have enough data
        if len(numeric_values) >= 5:
            summary.update({
                "p25": float(np.percentile(np_values, 25)),
                "p75": float(np.percentile(np_values, 75)),
                "p95": float(np.percentile(np_values, 95)),
                "p99": float(np.percentile(np_values, 99)),
            })
        
        # Add rate if duration is available
        if duration_seconds > 0:
            summary["rate_per_second"] = summary["sum"] / duration_seconds
        
        # Add data quality metrics
        summary["latest_value"] = float(values[-1].value) if values else 0.0
        summary["oldest_timestamp"] = values[0].timestamp.isoformat() if values else None
        summary["latest_timestamp"] = values[-1].timestamp.isoformat() if values else None

        return summary
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the collector.
        
        Returns:
            Dictionary of performance metrics
        """
        with self._lock:
            total_metrics = len(self.metrics)
            total_values = sum(len(values) for values in self.metrics.values())
        
        avg_record_time = (
            statistics.mean(self._record_times) if self._record_times else 0.0
        )
        
        return {
            "total_metrics": total_metrics,
            "total_values": total_values,
            "avg_record_time_ms": avg_record_time,
            "memory_usage_mb": self._estimate_memory_usage(),
            "compression_stats": self._compression_stats.copy(),
            "retention_hours": self.retention_hours,
            "max_values_per_metric": self.max_values_per_metric,
            "cleanup_interval_seconds": self.cleanup_interval,
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of stored metrics in MB.
        
        Returns:
            Estimated memory usage in megabytes
        """
        try:
            # Rough estimation: each MetricValue ~= 200 bytes
            # (datetime=24, float=8, string overhead, labels, etc.)
            with self._lock:
                total_values = sum(len(values) for values in self.metrics.values())
            
            estimated_bytes = total_values * 200
            return estimated_bytes / (1024 * 1024)  # Convert to MB
            
        except Exception:
            return 0.0
    
    def cleanup(self) -> None:
        """Clean up resources and shutdown thread pool."""
        try:
            self._thread_pool.shutdown(wait=True, timeout=30)
            logger.info("MetricsCollector cleanup completed")
        except Exception as e:
            logger.error("Error during MetricsCollector cleanup", error=str(e))


class AlertManager:
    """Enhanced alert manager with async support and performance optimizations.
    
    This manager provides:
    - Asynchronous rule evaluation
    - Efficient alert deduplication
    - Configurable notification channels
    - Alert correlation and grouping
    - Rate limiting and suppression
    """

    def __init__(
        self, 
        metrics_collector: MetricsCollector,
        evaluation_interval: int = 30,
        max_alert_history: int = 10000,
        enable_correlation: bool = True
    ) -> None:
        """Initialize alert manager with enhanced features.
        
        Args:
            metrics_collector: MetricsCollector instance
            evaluation_interval: How often to evaluate rules (seconds)
            max_alert_history: Maximum alerts to keep in history
            enable_correlation: Whether to enable alert correlation
        """
        self.metrics_collector = metrics_collector
        self.evaluation_interval = evaluation_interval
        self.enable_correlation = enable_correlation
        
        # Thread-safe storage
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}  # fingerprint -> alert
        self.alert_history: deque = deque(maxlen=max_alert_history)
        self.notification_callbacks: List[Callable[[Alert], None]] = []
        self._lock = threading.RLock()
        
        # Alert correlation and grouping
        self.alert_groups: Dict[str, List[str]] = {}  # group_key -> alert_fingerprints
        self.suppressed_alerts: Set[str] = set()
        
        # Performance tracking
        self.evaluation_count = 0
        self.last_evaluation_time: Optional[datetime] = None
        self.evaluation_duration_history: deque = deque(maxlen=100)
        
        # Background task management
        self._evaluation_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        logger.info("AlertManager initialized", 
                   evaluation_interval=evaluation_interval,
                   max_alert_history=max_alert_history)

        # Setup default alert rules
        self._setup_default_alerts()
        
    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            if hasattr(self, '_evaluation_task') and self._evaluation_task:
                self._evaluation_task.cancel()
        except Exception:
            pass

    def _setup_default_alerts(self) -> None:
        """Setup default alert rules with enhanced configuration."""
        default_rules = [
            AlertRule(
                name="high_search_latency",
                metric_name="vector_search_duration_ms",
                condition="> 5000",
                level=AlertLevel.WARNING,
                duration_seconds=120,
                description="Vector search taking longer than 5 seconds",
                aggregation="p95",  # Use 95th percentile
            ),
            AlertRule(
                name="very_high_search_latency",
                metric_name="vector_search_duration_ms",
                condition="> 10000",
                level=AlertLevel.CRITICAL,
                duration_seconds=60,
                description="Vector search taking longer than 10 seconds",
                aggregation="p95",
            ),
            AlertRule(
                name="high_memory_usage",
                metric_name="memory_usage_mb",
                condition="> 8192",
                level=AlertLevel.WARNING,
                duration_seconds=300,
                description="High system memory usage",
                aggregation="avg",
            ),
            AlertRule(
                name="critical_memory_usage",
                metric_name="memory_usage_mb",
                condition="> 12288",
                level=AlertLevel.CRITICAL,
                duration_seconds=60,
                description="Critical system memory usage",
                aggregation="avg",
            ),
            AlertRule(
                name="high_cpu_usage",
                metric_name="cpu_usage_percent",
                condition="> 80",
                level=AlertLevel.WARNING,
                duration_seconds=300,
                description="High CPU usage",
                aggregation="avg",
            ),
            AlertRule(
                name="low_cache_hit_rate",
                metric_name="cache_hit_rate",
                condition="< 0.5",
                level=AlertLevel.WARNING,
                duration_seconds=300,
                description="Low cache hit rate",
                aggregation="avg",
            ),
            AlertRule(
                name="high_error_rate",
                metric_name="error_rate",
                condition="> 10",
                level=AlertLevel.CRITICAL,
                duration_seconds=120,
                description="High error rate",
                aggregation="sum",
            ),
        ]

        # Add rules with error handling
        for rule in default_rules:
            try:
                self.add_alert_rule(rule)
            except Exception as e:
                logger.error("Failed to add default alert rule", 
                           rule_name=rule.name, error=str(e))

    async def start_evaluation(self) -> None:
        """Start the background alert evaluation task."""
        if self._evaluation_task and not self._evaluation_task.done():
            logger.warning("Alert evaluation already running")
            return
        
        self._shutdown_event.clear()
        self._evaluation_task = asyncio.create_task(
            self._evaluation_loop(),
            name="alert_manager_evaluation"
        )
        logger.info("Alert evaluation started")
    
    async def stop_evaluation(self) -> None:
        """Stop the background alert evaluation task."""
        self._shutdown_event.set()
        
        if self._evaluation_task:
            self._evaluation_task.cancel()
            try:
                await asyncio.wait_for(self._evaluation_task, timeout=10.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                logger.warning("Alert evaluation task cancellation timed out")
        
        logger.info("Alert evaluation stopped")
    
    async def _evaluation_loop(self) -> None:
        """Main evaluation loop running in background."""
        logger.info("Starting alert evaluation loop")
        
        while not self._shutdown_event.is_set():
            try:
                start_time = time.perf_counter()
                await self._evaluate_all_rules()
                
                # Track evaluation performance
                duration = (time.perf_counter() - start_time) * 1000
                self.evaluation_duration_history.append(duration)
                self.evaluation_count += 1
                self.last_evaluation_time = datetime.now()
                
                # Wait for next evaluation
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.evaluation_interval
                    )
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    continue  # Normal timeout, continue evaluation
                    
            except asyncio.CancelledError:
                logger.info("Alert evaluation loop cancelled")
                break
            except Exception as e:
                logger.error("Error in alert evaluation loop", error=str(e))
                await asyncio.sleep(min(self.evaluation_interval, 60))
        
        logger.info("Alert evaluation loop ended")
    
    async def _evaluate_all_rules(self) -> None:
        """Evaluate all alert rules against current metrics."""
        with self._lock:
            rules_to_evaluate = [(name, rule) for name, rule in self.alert_rules.items() if rule.enabled]
        
        if not rules_to_evaluate:
            return
        
        # Evaluate rules concurrently for better performance
        tasks = [
            self._evaluate_single_rule(rule_name, rule)
            for rule_name, rule in rules_to_evaluate
        ]
        
        # Use gather with return_exceptions to handle individual rule failures
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log any rule evaluation errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                rule_name = rules_to_evaluate[i][0]
                logger.error("Rule evaluation failed", 
                           rule_name=rule_name, 
                           error=str(result))
    
    async def _evaluate_single_rule(self, rule_name: str, rule: AlertRule) -> None:
        """Evaluate a single alert rule asynchronously.
        
        Args:
            rule_name: Name of the rule
            rule: AlertRule to evaluate
        """
        try:
            # Get recent metric values
            end_time = datetime.now()
            start_time = end_time - timedelta(seconds=300)  # 5 minute window
            
            values = self.metrics_collector.get_metric_values(
                rule.metric_name,
                start_time=start_time,
                end_time=end_time,
                labels=rule.labels,
                limit=1000  # Limit to prevent performance issues
            )
            
            if not values:
                # No data - resolve any existing alerts for this rule
                await self._resolve_alerts_for_rule(rule_name)
                return
            
            # Calculate aggregated value
            aggregated_value = self._calculate_aggregate(values, rule.aggregation)
            
            # Check condition
            condition_met = self._evaluate_condition(aggregated_value, rule.condition)
            
            if condition_met:
                await self._handle_condition_met(rule, aggregated_value)
            else:
                await self._handle_condition_resolved(rule_name)
                
        except Exception as e:
            logger.error("Error evaluating rule", rule_name=rule_name, error=str(e))
    
    def _calculate_aggregate(self, values: List[MetricValue], aggregation: str) -> float:
        """Calculate aggregated value from metric values.
        
        Args:
            values: List of metric values
            aggregation: Aggregation method
            
        Returns:
            Aggregated value
        """
        if not values:
            return 0.0
        
        numeric_values = [v.value for v in values]
        np_values = np.array(numeric_values)
        
        if aggregation == "avg":
            return float(np.mean(np_values))
        elif aggregation == "max":
            return float(np.max(np_values))
        elif aggregation == "min":
            return float(np.min(np_values))
        elif aggregation == "sum":
            return float(np.sum(np_values))
        elif aggregation == "count":
            return float(len(numeric_values))
        elif aggregation == "p95":
            return float(np.percentile(np_values, 95))
        elif aggregation == "p99":
            return float(np.percentile(np_values, 99))
        else:
            # Default to average
            return float(np.mean(np_values))
    
    def _evaluate_condition(self, value: float, condition: str) -> bool:
        """Evaluate if a condition is met.
        
        Args:
            value: Current metric value
            condition: Condition string (e.g., "> 100")
            
        Returns:
            True if condition is met
        """
        try:
            condition = condition.strip()
            
            if condition.startswith("> "):
                threshold = float(condition[2:])
                return value > threshold
            elif condition.startswith("< "):
                threshold = float(condition[2:])
                return value < threshold
            elif condition.startswith(">= "):
                threshold = float(condition[3:])
                return value >= threshold
            elif condition.startswith("<= "):
                threshold = float(condition[3:])
                return value <= threshold
            elif condition.startswith("== "):
                threshold = float(condition[3:])
                return abs(value - threshold) < 1e-9
            elif condition.startswith("!= "):
                threshold = float(condition[3:])
                return abs(value - threshold) >= 1e-9
            else:
                logger.warning("Unknown condition format", condition=condition)
                return False
                
        except (ValueError, IndexError) as e:
            logger.error("Error evaluating condition", condition=condition, error=str(e))
            return False
    
    async def _handle_condition_met(self, rule: AlertRule, value: float) -> None:
        """Handle when an alert condition is met.
        
        Args:
            rule: AlertRule that was triggered
            value: Current metric value
        """
        # Generate alert fingerprint for deduplication
        fingerprint = self._generate_alert_fingerprint(rule)
        
        with self._lock:
            if fingerprint in self.active_alerts:
                # Update existing alert
                self.active_alerts[fingerprint].value = value
                return
            
            # Create new alert
            alert = Alert(
                id=str(uuid.uuid4()),
                rule_name=rule.name,
                metric_name=rule.metric_name,
                level=rule.level,
                message=rule.description,
                value=value,
                threshold=rule.condition,
                started_at=datetime.now(),
                labels=rule.labels.copy() if rule.labels else {},
                fingerprint=fingerprint
            )
            
            self.active_alerts[fingerprint] = alert
            self.alert_history.append(alert)
            
            # Notify callbacks
            for callback in self.notification_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error("Notification callback failed", error=str(e))
            
            logger.warning("Alert fired", 
                          rule_name=rule.name,
                          metric_name=rule.metric_name,
                          value=value,
                          condition=rule.condition,
                          fingerprint=fingerprint)
    
    async def _handle_condition_resolved(self, rule_name: str) -> None:
        """Handle when an alert condition is resolved.
        
        Args:
            rule_name: Name of the rule
        """
        await self._resolve_alerts_for_rule(rule_name)
    
    async def _resolve_alerts_for_rule(self, rule_name: str) -> None:
        """Resolve all active alerts for a specific rule.
        
        Args:
            rule_name: Name of the rule
        """
        alerts_to_resolve = []
        
        with self._lock:
            for fingerprint, alert in list(self.active_alerts.items()):
                if alert.rule_name == rule_name:
                    alert.resolved_at = datetime.now()
                    alerts_to_resolve.append((fingerprint, alert))
                    del self.active_alerts[fingerprint]
        
        for fingerprint, alert in alerts_to_resolve:
            logger.info("Alert resolved", 
                       rule_name=rule_name,
                       fingerprint=fingerprint,
                       duration=alert.duration)
    
    def _generate_alert_fingerprint(self, rule: AlertRule) -> str:
        """Generate unique fingerprint for alert deduplication.
        
        Args:
            rule: AlertRule
            
        Returns:
            Unique fingerprint string
        """
        import hashlib
        
        # Create fingerprint from rule name and labels
        content = f"{rule.name}:{rule.metric_name}"
        if rule.labels:
            sorted_labels = sorted(rule.labels.items())
            content += f":{sorted_labels}"
        
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert rule with validation.
        
        Args:
            rule: AlertRule to add
            
        Raises:
            ValueError: If rule name already exists or rule is invalid
        """
        if not isinstance(rule, AlertRule):
            raise ValueError("Expected AlertRule instance")
        
        with self._lock:
            if rule.name in self.alert_rules:
                raise ValueError(f"Alert rule '{rule.name}' already exists")
            
            self.alert_rules[rule.name] = rule
        
        logger.info("Alert rule added", rule_name=rule.name, metric=rule.metric_name)

    async def remove_alert_rule(self, rule_name: str) -> bool:
        """Remove an alert rule and resolve its alerts.
        
        Args:
            rule_name: Name of the rule to remove
            
        Returns:
            True if rule was removed, False if not found
        """
        with self._lock:
            if rule_name not in self.alert_rules:
                return False
            
            del self.alert_rules[rule_name]
        
        # Resolve any active alerts for this rule
        await self._resolve_alerts_for_rule(rule_name)
        
        logger.info("Alert rule removed", rule_name=rule_name)
        return True

    def add_notification_callback(self, callback: Callable[[Alert], None]):
        """Add a notification callback function."""
        self.notification_callbacks.append(callback)

    def check_alerts(self):
        """Check all alert rules against current metrics."""
        current_time = datetime.now()

        with self.lock:
            for _rule_name, rule in self.alert_rules.items():
                self._check_single_alert(rule, current_time)

    def _check_single_alert(self, rule: AlertRule, current_time: datetime):
        """Check a single alert rule."""
        # Get recent metric values
        start_time = current_time - timedelta(seconds=rule.duration_seconds)
        values = self.metrics_collector.get_metric_values(
            rule.metric_name, start_time, current_time
        )

        if not values:
            return

        # Check if condition is met
        condition_met = self._evaluate_condition(values, rule.condition)

        if condition_met:
            if rule.name not in self.active_alerts:
                # Create new alert
                alert = Alert(
                    rule_name=rule.name,
                    metric_name=rule.metric_name,
                    level=rule.level,
                    message=rule.description,
                    value=values[-1].value,  # Latest value
                    threshold=rule.condition,
                    started_at=current_time,
                )

                self.active_alerts[rule.name] = alert
                self.alert_history.append(alert)

                # Notify
                for callback in self.notification_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"Notification callback failed: {e}")

                logger.warning(
                    f"ALERT: {rule.name} - {rule.description} "
                    f"(value: {values[-1].value}, threshold: {rule.condition})"
                )
        elif rule.name in self.active_alerts:
            # Resolve alert
            alert = self.active_alerts[rule.name]
            alert.resolved_at = current_time
            del self.active_alerts[rule.name]

            logger.info(f"RESOLVED: {rule.name}")

    def _evaluate_condition(self, values: List[MetricValue], condition: str) -> bool:
        """Evaluate if metric values meet the alert condition."""
        if not values:
            return False

        # For simplicity, check if the latest value meets the condition
        latest_value = values[-1].value

        # Parse condition (e.g., "> 100", "< 0.5")
        if condition.startswith("> "):
            threshold = float(condition[2:])
            return latest_value > threshold
        elif condition.startswith("< "):
            threshold = float(condition[2:])
            return latest_value < threshold
        elif condition.startswith(">= "):
            threshold = float(condition[3:])
            return latest_value >= threshold
        elif condition.startswith("<= "):
            threshold = float(condition[3:])
            return latest_value <= threshold
        elif condition.startswith("== "):
            threshold = float(condition[3:])
            return latest_value == threshold

        return False

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self.lock:
            return list(self.active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        with self.lock:
            return list(self.alert_history)[-limit:]


class SystemMetricsCollector:
    """Collects system-level metrics."""

    def __init__(self, metrics_collector: MetricsCollector, db_pool: asyncpg.Pool):
        self.metrics_collector = metrics_collector
        self.db_pool = db_pool
        self.collection_interval = 30  # seconds
        self.running = False
        self.collection_task = None

    async def start_collection(self):
        """Start collecting system metrics."""
        if self.running:
            logger.warning("System metrics collection already running")
            return

        self.running = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("System metrics collection started")

    async def stop_collection(self):
        """Stop collecting system metrics."""
        self.running = False
        if self.collection_task:
            self.collection_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.collection_task
        logger.info("System metrics collection stopped")

    async def _collection_loop(self):
        """Main collection loop."""
        while self.running:
            try:
                await self._collect_system_metrics()
                await self._collect_database_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying

    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        # Memory usage
        memory = psutil.virtual_memory()
        self.metrics_collector.record_metric(
            "memory_usage_mb", memory.used / (1024 * 1024)
        )
        self.metrics_collector.record_metric("memory_usage_percent", memory.percent)

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics_collector.record_metric("cpu_usage_percent", cpu_percent)

        # Disk usage
        disk = psutil.disk_usage("/")
        self.metrics_collector.record_metric(
            "disk_usage_percent", (disk.used / disk.total) * 100
        )

        # Network I/O
        net_io = psutil.net_io_counters()
        self.metrics_collector.record_metric("network_bytes_sent", net_io.bytes_sent)
        self.metrics_collector.record_metric("network_bytes_recv", net_io.bytes_recv)

    async def _collect_database_metrics(self):
        """Collect database-specific metrics."""
        try:
            async with self.db_pool.acquire() as conn:
                # Connection count
                connection_stats = await conn.fetchrow(
                    """
                    SELECT
                        count(*) as total_connections,
                        count(*) FILTER (WHERE state = 'active') as active_connections,
                        count(*) FILTER (WHERE state = 'idle') as idle_connections
                    FROM pg_stat_activity
                """
                )

                self.metrics_collector.record_metric(
                    "database_connections", connection_stats["total_connections"]
                )
                self.metrics_collector.record_metric(
                    "database_active_connections",
                    connection_stats["active_connections"],
                )

                # Table statistics
                # Check if schema exists first
                schema_exists = await conn.fetchval(
                    """
                    SELECT EXISTS(
                        SELECT 1 FROM information_schema.schemata
                        WHERE schema_name = 'mem0_vectors'
                    )
                """
                )

                if schema_exists:
                    table_stats = await conn.fetch(
                        """
                        SELECT
                            tablename,
                            n_tup_ins as inserts,
                            n_tup_upd as updates,
                            n_tup_del as deletes,
                            n_live_tup as live_tuples,
                            n_dead_tup as dead_tuples
                        FROM pg_stat_user_tables
                        WHERE schemaname = 'mem0_vectors'
                    """
                    )
                else:
                    table_stats = []

                for row in table_stats:
                    labels = {"table": row["tablename"]}
                    self.metrics_collector.record_metric(
                        "table_live_tuples", row["live_tuples"], labels
                    )
                    self.metrics_collector.record_metric(
                        "table_dead_tuples", row["dead_tuples"], labels
                    )

                # Index statistics (only if schema exists)
                if schema_exists:
                    index_stats = await conn.fetch(
                        """
                        SELECT
                            indexname,
                            idx_scan,
                            idx_tup_read,
                            idx_tup_fetch,
                            pg_size_pretty(pg_relation_size(indexrelid)) as size_pretty,
                            pg_relation_size(indexrelid) as size_bytes
                        FROM pg_stat_user_indexes
                        WHERE schemaname = 'mem0_vectors'
                    """
                    )
                else:
                    index_stats = []

                for row in index_stats:
                    labels = {"index": row["indexname"]}
                    self.metrics_collector.record_metric(
                        "index_scans", row["idx_scan"], labels
                    )
                    self.metrics_collector.record_metric(
                        "vector_index_size_mb",
                        row["size_bytes"] / (1024 * 1024),
                        labels,
                    )

                # Active users in last hour (only if schema and table exist)
                if schema_exists:
                    try:
                        active_users = await conn.fetchval(
                            """
                            SELECT COUNT(DISTINCT user_id)
                            FROM mem0_vectors.memories
                            WHERE last_accessed > NOW() - INTERVAL '1 hour'
                        """
                        )
                    except Exception:
                        # Table might not exist
                        active_users = 0
                else:
                    active_users = 0

                self.metrics_collector.record_metric("active_users", active_users or 0)

        except Exception as e:
            logger.error(f"Error collecting database metrics: {e}")


class PerformanceProfiler:
    """Performance profiling for vector operations."""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.active_operations = {}
        self.lock = threading.Lock()

    def start_operation(
        self, operation_id: str, operation_type: str, metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """Start timing an operation."""
        with self.lock:
            self.active_operations[operation_id] = {
                "type": operation_type,
                "start_time": time.time(),
                "metadata": metadata or {},
            }
        return operation_id

    def end_operation(
        self,
        operation_id: str,
        success: bool = True,
        additional_metrics: Optional[Dict[str, float]] = None,
    ):
        """End timing an operation and record metrics."""
        with self.lock:
            if operation_id not in self.active_operations:
                return

            operation = self.active_operations.pop(operation_id)
            duration_ms = (time.time() - operation["start_time"]) * 1000

            # Record timing metric
            labels = operation["metadata"].copy()
            labels["success"] = str(success)
            labels["operation_type"] = operation["type"]

            self.metrics_collector.record_metric(
                f"{operation['type']}_duration_ms", duration_ms, labels
            )

            # Record additional metrics
            if additional_metrics:
                for metric_name, value in additional_metrics.items():
                    self.metrics_collector.record_metric(metric_name, value, labels)

    def profile_async_function(
        self, operation_type: str, metadata: Optional[Dict[str, str]] = None
    ):
        """Decorator to profile async functions."""

        def decorator(func):
            async def wrapper(*args, **kwargs):
                operation_id = f"{operation_type}_{time.time()}"
                self.start_operation(operation_id, operation_type, metadata)

                try:
                    result = await func(*args, **kwargs)
                    self.end_operation(operation_id, success=True)
                    return result
                except Exception:
                    self.end_operation(operation_id, success=False)
                    raise

            return wrapper

        return decorator


class MetricsExporter:
    """Exports metrics to external systems."""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.exporters = {}

    def add_prometheus_exporter(self, port: int = 8000):
        """Add Prometheus metrics exporter."""
        try:
            from prometheus_client import Counter
            from prometheus_client import Gauge
            from prometheus_client import Histogram
            from prometheus_client import start_http_server

            # Create Prometheus metrics
            self.prometheus_metrics = {}

            for name, definition in self.metrics_collector.metric_definitions.items():
                if definition.metric_type == MetricType.GAUGE:
                    metric = Gauge(
                        name, definition.description, definition.labels or []
                    )
                elif definition.metric_type == MetricType.COUNTER:
                    metric = Counter(
                        name, definition.description, definition.labels or []
                    )
                elif definition.metric_type == MetricType.HISTOGRAM:
                    metric = Histogram(
                        name, definition.description, definition.labels or []
                    )
                else:
                    metric = Gauge(
                        name, definition.description, definition.labels or []
                    )

                self.prometheus_metrics[name] = metric

            # Start HTTP server
            start_http_server(port)

            # Store export loop task to avoid "coroutine was never awaited" warning
            self.export_task = asyncio.create_task(self._prometheus_export_loop())

            logger.info(f"Prometheus exporter started on port {port}")

        except ImportError:
            logger.warning(
                "prometheus_client not available, skipping Prometheus exporter"
            )

    async def _prometheus_export_loop(self):
        """Export metrics to Prometheus periodically."""
        while True:
            try:
                current_time = datetime.now()
                lookback_time = current_time - timedelta(minutes=5)

                for metric_name, prometheus_metric in self.prometheus_metrics.items():
                    values = self.metrics_collector.get_metric_values(
                        metric_name, lookback_time, current_time
                    )

                    if values:
                        latest_value = values[-1]

                        if latest_value.labels and hasattr(
                            prometheus_metric, "_labelnames"
                        ):
                            # Metric with labels
                            label_values = [
                                latest_value.labels.get(label, "")
                                for label in prometheus_metric._labelnames
                            ]
                            prometheus_metric.labels(*label_values).set(
                                latest_value.value
                            )
                        else:
                            # Metric without labels or no labelnames attribute
                            try:
                                prometheus_metric.set(latest_value.value)
                            except Exception as e:
                                logger.debug(f"Could not set metric {metric_name}: {e}")

                await asyncio.sleep(30)  # Export every 30 seconds

            except Exception as e:
                logger.error(f"Prometheus export error: {e}")
                await asyncio.sleep(60)


class MonitoringDashboard:
    """Simple web dashboard for monitoring metrics."""

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        alert_manager: AlertManager,
        port: int = 8080,
    ):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.port = port
        self.app = None

    async def start_dashboard(self):
        """Start the web dashboard."""
        try:
            from aiohttp import web
            from aiohttp import web_response

            self.app = web.Application()

            # Add routes
            self.app.router.add_get("/", self._dashboard_handler)
            self.app.router.add_get("/metrics", self._metrics_handler)
            self.app.router.add_get("/alerts", self._alerts_handler)
            self.app.router.add_get(
                "/api/metrics/{metric_name}", self._api_metrics_handler
            )

            # Start server
            runner = web.AppRunner(self.app)
            await runner.setup()
            site = web.TCPSite(runner, "localhost", self.port)
            await site.start()

            logger.info(f"Monitoring dashboard started on http://localhost:{self.port}")

        except ImportError:
            logger.warning("aiohttp not available, skipping web dashboard")

    async def _dashboard_handler(self, request):
        """Main dashboard page."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>mem0ai Monitoring Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .metric-card {
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 10px 0;
                    background: #f9f9f9;
                }
                .alert {
                    padding: 10px;
                    margin: 5px 0;
                    border-radius: 4px;
                }
                .alert-warning { background: #fff3cd; border-left: 4px solid #ffc107; }
                .alert-critical { background: #f8d7da; border-left: 4px solid #dc3545; }
                .alert-info { background: #d1ecf1; border-left: 4px solid #17a2b8; }
            </style>
        </head>
        <body>
            <h1>mem0ai Monitoring Dashboard</h1>

            <h2>System Overview</h2>
            <div id="metrics"></div>

            <h2>Active Alerts</h2>
            <div id="alerts"></div>

            <script>
                async function loadMetrics() {
                    try {
                        const response = await fetch('/metrics');
                        const data = await response.json();

                        const metricsDiv = document.getElementById('metrics');
                        metricsDiv.innerHTML = '';

                        for (const [name, summary] of Object.entries(data.metrics)) {
                            const card = document.createElement('div');
                            card.className = 'metric-card';
                            card.innerHTML = `
                                <h3>${name}</h3>
                                <p>Latest: ${summary.latest}</p>
                                <p>Mean: ${summary.mean?.toFixed(2)}</p>
                                <p>P95: ${summary.p95?.toFixed(2)}</p>
                            `;
                            metricsDiv.appendChild(card);
                        }
                    } catch (error) {
                        console.error('Error loading metrics:', error);
                    }
                }

                async function loadAlerts() {
                    try {
                        const response = await fetch('/alerts');
                        const data = await response.json();

                        const alertsDiv = document.getElementById('alerts');
                        alertsDiv.innerHTML = '';

                        if (data.alerts.length === 0) {
                            alertsDiv.innerHTML = '<p>No active alerts</p>';
                            return;
                        }

                        for (const alert of data.alerts) {
                            const alertDiv = document.createElement('div');
                            alertDiv.className = `alert alert-${alert.level}`;
                            alertDiv.innerHTML = `
                                <strong>${alert.rule_name}</strong>: ${alert.message}
                                <br>Value: ${alert.value}, Threshold: ${alert.threshold}
                                <br>Started: ${new Date(alert.started_at).toLocaleString()}
                            `;
                            alertsDiv.appendChild(alertDiv);
                        }
                    } catch (error) {
                        console.error('Error loading alerts:', error);
                    }
                }

                // Load data initially and refresh every 30 seconds
                loadMetrics();
                loadAlerts();
                setInterval(() => {
                    loadMetrics();
                    loadAlerts();
                }, 30000);
            </script>
        </body>
        </html>
        """

        return web.Response(text=html, content_type="text/html")

    async def _metrics_handler(self, request):
        """Metrics API endpoint."""
        current_time = datetime.now()
        lookback_time = current_time - timedelta(minutes=30)

        metrics_data = {}

        for metric_name in self.metrics_collector.metric_definitions:
            summary = self.metrics_collector.get_metric_summary(
                metric_name, lookback_time, current_time
            )

            if summary:
                values = self.metrics_collector.get_metric_values(
                    metric_name, lookback_time, current_time
                )
                summary["latest"] = values[-1].value if values else 0
                metrics_data[metric_name] = summary

        return web.json_response({"metrics": metrics_data})

    async def _alerts_handler(self, request):
        """Alerts API endpoint."""
        active_alerts = self.alert_manager.get_active_alerts()

        alerts_data = []
        for alert in active_alerts:
            alerts_data.append(
                {
                    "rule_name": alert.rule_name,
                    "metric_name": alert.metric_name,
                    "level": alert.level.value,
                    "message": alert.message,
                    "value": alert.value,
                    "threshold": alert.threshold,
                    "started_at": alert.started_at.isoformat(),
                }
            )

        return web.json_response({"alerts": alerts_data})

    async def _api_metrics_handler(self, request):
        """Individual metric API endpoint."""
        metric_name = request.match_info["metric_name"]

        current_time = datetime.now()
        lookback_time = current_time - timedelta(hours=1)

        values = self.metrics_collector.get_metric_values(
            metric_name, lookback_time, current_time
        )

        data_points = [
            {"timestamp": v.timestamp.isoformat(), "value": v.value, "labels": v.labels}
            for v in values
        ]

        return web.json_response({"data": data_points})


# Main monitoring system
class MonitoringSystem:
    """Complete monitoring system for mem0ai."""

    def __init__(self, db_url: str, config: Optional[Dict[str, Any]] = None):
        self.db_url = db_url
        self.config = config or {}
        self.pool = None

        # Initialize components
        self.metrics_collector = MetricsCollector(
            retention_hours=self.config.get("retention_hours", 24)
        )
        self.alert_manager = AlertManager(self.metrics_collector)
        self.profiler = PerformanceProfiler(self.metrics_collector)
        self.exporter = MetricsExporter(self.metrics_collector)

        # System metrics collector will be initialized after database connection
        self.system_collector = None

        # Dashboard
        self.dashboard = MonitoringDashboard(
            self.metrics_collector,
            self.alert_manager,
            port=self.config.get("dashboard_port", 8080),
        )

        # Alert checking task
        self.alert_check_task = None

    async def initialize(self):
        """Initialize the monitoring system."""
        # Initialize database connection
        self.pool = await asyncpg.create_pool(
            self.db_url, min_size=5, max_size=20, command_timeout=60
        )

        # Initialize system metrics collector
        self.system_collector = SystemMetricsCollector(
            self.metrics_collector, self.pool
        )

        # Start components
        await self.system_collector.start_collection()

        # Start alert checking
        self.alert_check_task = asyncio.create_task(self._alert_check_loop())

        # Start dashboard
        await self.dashboard.start_dashboard()

        # Setup exporters if configured
        if self.config.get("prometheus_port"):
            self.exporter.add_prometheus_exporter(self.config["prometheus_port"])

        logger.info("Monitoring system initialized")

    async def cleanup(self):
        """Cleanup monitoring system."""
        if self.system_collector:
            await self.system_collector.stop_collection()

        if self.alert_check_task:
            self.alert_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.alert_check_task

        if self.pool:
            await self.pool.close()

    async def _alert_check_loop(self):
        """Periodically check alerts."""
        while True:
            try:
                self.alert_manager.check_alerts()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error checking alerts: {e}")
                await asyncio.sleep(60)

    def get_profiler(self) -> PerformanceProfiler:
        """Get the performance profiler."""
        return self.profiler

    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        self.metrics_collector.record_metric(name, value, labels)


# Example usage
async def main():
    """Test the monitoring system."""
    import os

    DB_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/mem0ai")

    config = {"retention_hours": 24, "dashboard_port": 8080, "prometheus_port": 8000}

    monitoring = MonitoringSystem(DB_URL, config)

    try:
        await monitoring.initialize()

        # Simulate some metrics
        profiler = monitoring.get_profiler()


        for i in range(10):
            # Simulate vector search
            operation_id = profiler.start_operation(
                f"search_{i}",
                "vector_search",
                {"user_id": "test_user", "index_type": "hnsw"},
            )

            await asyncio.sleep(0.1)  # Simulate work

            profiler.end_operation(
                operation_id,
                success=True,
                additional_metrics={"results_count": 10, "similarity_threshold": 0.8},
            )

            # Record some custom metrics
            monitoring.record_metric("custom_metric", i * 10)
            monitoring.record_metric(
                "cache_hit_rate", 0.85, {"cache_type": "embedding"}
            )

            await asyncio.sleep(1)


        # Keep running
        while True:
            await asyncio.sleep(10)

    except KeyboardInterrupt:
        pass
    finally:
        await monitoring.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
