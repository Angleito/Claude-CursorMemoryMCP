"""Enhanced metrics collection system for mem0ai.

This module provides a comprehensive metrics collection system with:
- High-performance metrics collection
- Type-safe metric definitions
- Automatic aggregation and bucketing
- Memory-efficient storage
- Prometheus integration
- Real-time metric queries

Examples:
    >>> collector = MetricsCollector()
    >>> await collector.initialize()
    >>> collector.record_counter("api_requests", 1, {"endpoint": "/search"})
    >>> collector.record_histogram("request_duration", 0.150, {"method": "POST"})
    >>> metrics = await collector.get_metrics_summary()
"""

import asyncio
import statistics
import threading
import time
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from enum import Enum
from typing import Any

import structlog
from prometheus_client import REGISTRY
from prometheus_client import CollectorRegistry
from prometheus_client import Counter as PrometheusCounter
from prometheus_client import Gauge as PrometheusGauge
from prometheus_client import Histogram as PrometheusHistogram
from prometheus_client import Summary as PrometheusSummary
from prometheus_client import generate_latest

logger = structlog.get_logger()


class MetricType(str, Enum):
    """Types of metrics that can be collected."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMING = "timing"


class AggregationType(str, Enum):
    """Types of aggregation for metrics."""

    SUM = "sum"
    COUNT = "count"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    P50 = "p50"
    P95 = "p95"
    P99 = "p99"
    STDDEV = "stddev"


@dataclass
class MetricDefinition:
    """Definition of a metric with metadata and configuration.

    Attributes:
        name: Unique metric name
        metric_type: Type of metric (counter, gauge, etc.)
        description: Human-readable description
        unit: Unit of measurement
        labels: List of label names for this metric
        help_text: Additional help text for documentation
        buckets: Optional histogram buckets
        ttl_seconds: Time-to-live for metric values (default: 1 hour)
    """

    name: str
    metric_type: MetricType
    description: str
    unit: str
    labels: list[str] = field(default_factory=list)
    help_text: str = ""
    buckets: list[float] | None = None
    ttl_seconds: int = 3600  # 1 hour default TTL

    def __post_init__(self) -> None:
        """Validate metric definition after initialization."""
        if not self.name or not self.name.strip():
            raise ValueError("Metric name cannot be empty")

        if not self.description:
            self.description = f"Metric: {self.name}"

        if not self.unit:
            self.unit = "count"

        # Sanitize metric name for Prometheus
        self.name = self._sanitize_name(self.name)

    def _sanitize_name(self, name: str) -> str:
        """Sanitize metric name for Prometheus compatibility."""
        import re
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_:]', '_', name)
        # Ensure it starts with a letter or underscore
        if sanitized and not sanitized[0].isalpha() and sanitized[0] != '_':
            sanitized = f"metric_{sanitized}"
        return sanitized


@dataclass
class MetricValue:
    """A single metric value with timestamp and labels.

    Attributes:
        value: The numeric value
        timestamp: When the metric was recorded
        labels: Key-value pairs for metric labels
    """

    value: float
    timestamp: datetime
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure timestamp is set if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.utcnow()


@dataclass
class MetricSummary:
    """Summary statistics for a metric over a time period.

    Attributes:
        metric_name: Name of the metric
        start_time: Start of the summary period
        end_time: End of the summary period
        count: Number of data points
        sum_value: Sum of all values
        min_value: Minimum value
        max_value: Maximum value
        avg_value: Average value
        p50_value: 50th percentile
        p95_value: 95th percentile
        p99_value: 99th percentile
        stddev_value: Standard deviation
        rate_per_second: Rate per second (for counters)
    """

    metric_name: str
    start_time: datetime
    end_time: datetime
    count: int
    sum_value: float
    min_value: float
    max_value: float
    avg_value: float
    p50_value: float
    p95_value: float
    p99_value: float
    stddev_value: float
    rate_per_second: float = 0.0


class MetricsCollector:
    """High-performance metrics collector with type safety and memory efficiency.

    This collector provides:
    - Thread-safe metric recording
    - Automatic cleanup of old metrics
    - Prometheus integration
    - Efficient memory usage
    - Flexible querying
    """

    def __init__(
        self,
        retention_hours: int = 24,
        max_values_per_metric: int = 10000,
        cleanup_interval: int = 300,  # 5 minutes
        prometheus_registry: CollectorRegistry | None = None,
    ) -> None:
        """Initialize the metrics collector.

        Args:
            retention_hours: How long to keep metric values
            max_values_per_metric: Maximum values to store per metric
            cleanup_interval: How often to clean up old metrics (seconds)
            prometheus_registry: Optional Prometheus registry
        """
        self.retention_hours = retention_hours
        self.max_values_per_metric = max_values_per_metric
        self.cleanup_interval = cleanup_interval
        self.prometheus_registry = prometheus_registry or REGISTRY

        # Thread-safe storage
        self._lock = threading.RLock()
        self._metrics: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_values_per_metric)
        )
        self._metric_definitions: dict[str, MetricDefinition] = {}
        self._prometheus_metrics: dict[str, PrometheusCounter | PrometheusGauge | PrometheusHistogram] = {}

        # Performance tracking
        self._last_cleanup = time.time()
        self._recording_times: deque = deque(maxlen=1000)
        self._cleanup_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()

        # Built-in metric definitions
        self._setup_builtin_metrics()

        logger.info("MetricsCollector initialized",
                   retention_hours=retention_hours,
                   max_values_per_metric=max_values_per_metric)

    def _setup_builtin_metrics(self) -> None:
        """Setup built-in metric definitions."""
        builtin_metrics = [
            MetricDefinition(
                name="mem0_requests_total",
                metric_type=MetricType.COUNTER,
                description="Total number of requests",
                unit="requests",
                labels=["method", "endpoint", "status_code"],
                help_text="Total number of HTTP requests processed"
            ),
            MetricDefinition(
                name="mem0_request_duration_seconds",
                metric_type=MetricType.HISTOGRAM,
                description="Request duration in seconds",
                unit="seconds",
                labels=["method", "endpoint"],
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
                help_text="HTTP request duration distribution"
            ),
            MetricDefinition(
                name="mem0_memory_usage_bytes",
                metric_type=MetricType.GAUGE,
                description="Memory usage in bytes",
                unit="bytes",
                help_text="Current memory usage of the application"
            ),
            MetricDefinition(
                name="mem0_active_connections",
                metric_type=MetricType.GAUGE,
                description="Number of active connections",
                unit="connections",
                help_text="Current number of active database/WebSocket connections"
            ),
            MetricDefinition(
                name="mem0_cache_hit_rate",
                metric_type=MetricType.GAUGE,
                description="Cache hit rate",
                unit="ratio",
                labels=["cache_type"],
                help_text="Cache hit rate for various cache types"
            ),
            MetricDefinition(
                name="mem0_errors_total",
                metric_type=MetricType.COUNTER,
                description="Total number of errors",
                unit="errors",
                labels=["error_type", "component"],
                help_text="Total number of errors by type and component"
            ),
            MetricDefinition(
                name="mem0_vector_search_duration_seconds",
                metric_type=MetricType.HISTOGRAM,
                description="Vector search duration",
                unit="seconds",
                labels=["user_id", "index_type"],
                buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
                help_text="Duration of vector similarity searches"
            ),
            MetricDefinition(
                name="mem0_embedding_generation_duration_seconds",
                metric_type=MetricType.HISTOGRAM,
                description="Embedding generation duration",
                unit="seconds",
                labels=["provider", "model"],
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
                help_text="Duration of embedding generation operations"
            ),
        ]

        for metric_def in builtin_metrics:
            self.register_metric(metric_def)

    async def initialize(self) -> None:
        """Initialize the metrics collector and start background tasks."""
        try:
            # Create Prometheus metrics
            self._create_prometheus_metrics()

            # Start cleanup task
            self._cleanup_task = asyncio.create_task(
                self._periodic_cleanup(),
                name="metrics_collector_cleanup"
            )

            logger.info("MetricsCollector initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize MetricsCollector", error=str(e))
            raise RuntimeError(f"MetricsCollector initialization failed: {e}") from e

    async def close(self) -> None:
        """Close the metrics collector and cleanup resources."""
        logger.info("Shutting down MetricsCollector")

        # Signal shutdown
        self._shutdown_event.set()

        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await asyncio.wait_for(self._cleanup_task, timeout=5.0)
            except (TimeoutError, asyncio.CancelledError):
                logger.warning("Cleanup task shutdown timed out")

        logger.info("MetricsCollector shutdown complete")

    def register_metric(self, metric_def: MetricDefinition) -> None:
        """Register a new metric definition.

        Args:
            metric_def: The metric definition to register

        Raises:
            ValueError: If metric name already exists with different definition
        """
        with self._lock:
            if metric_def.name in self._metric_definitions:
                existing = self._metric_definitions[metric_def.name]
                if existing.metric_type != metric_def.metric_type:
                    raise ValueError(
                        f"Metric {metric_def.name} already exists with different type: "
                        f"{existing.metric_type} vs {metric_def.metric_type}"
                    )

            self._metric_definitions[metric_def.name] = metric_def

            # Create Prometheus metric if not exists
            if metric_def.name not in self._prometheus_metrics:
                self._create_prometheus_metric(metric_def)

            logger.debug("Metric registered",
                        name=metric_def.name,
                        type=metric_def.metric_type.value)

    def _create_prometheus_metrics(self) -> None:
        """Create Prometheus metrics for all registered definitions."""
        with self._lock:
            for metric_def in self._metric_definitions.values():
                self._create_prometheus_metric(metric_def)

    def _create_prometheus_metric(self, metric_def: MetricDefinition) -> None:
        """Create a single Prometheus metric.

        Args:
            metric_def: The metric definition
        """
        try:
            if metric_def.metric_type == MetricType.COUNTER:
                prometheus_metric = PrometheusCounter(
                    metric_def.name,
                    metric_def.description,
                    labelnames=metric_def.labels,
                    registry=self.prometheus_registry
                )
            elif metric_def.metric_type == MetricType.GAUGE:
                prometheus_metric = PrometheusGauge(
                    metric_def.name,
                    metric_def.description,
                    labelnames=metric_def.labels,
                    registry=self.prometheus_registry
                )
            elif metric_def.metric_type == MetricType.HISTOGRAM:
                prometheus_metric = PrometheusHistogram(
                    metric_def.name,
                    metric_def.description,
                    labelnames=metric_def.labels,
                    buckets=metric_def.buckets,
                    registry=self.prometheus_registry
                )
            elif metric_def.metric_type == MetricType.SUMMARY:
                prometheus_metric = PrometheusSummary(
                    metric_def.name,
                    metric_def.description,
                    labelnames=metric_def.labels,
                    registry=self.prometheus_registry
                )
            else:
                # Default to gauge for unknown types
                prometheus_metric = PrometheusGauge(
                    metric_def.name,
                    metric_def.description,
                    labelnames=metric_def.labels,
                    registry=self.prometheus_registry
                )

            self._prometheus_metrics[metric_def.name] = prometheus_metric

        except Exception as e:
            logger.warning("Failed to create Prometheus metric",
                          name=metric_def.name, error=str(e))

    def record_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
        timestamp: datetime | None = None
    ) -> None:
        """Record a counter metric.

        Args:
            name: Metric name
            value: Counter increment value (default: 1.0)
            labels: Optional labels
            timestamp: Optional timestamp (default: current time)
        """
        self._record_metric(name, value, labels, timestamp, MetricType.COUNTER)

    def record_gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
        timestamp: datetime | None = None
    ) -> None:
        """Record a gauge metric.

        Args:
            name: Metric name
            value: Gauge value
            labels: Optional labels
            timestamp: Optional timestamp (default: current time)
        """
        self._record_metric(name, value, labels, timestamp, MetricType.GAUGE)

    def record_histogram(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
        timestamp: datetime | None = None
    ) -> None:
        """Record a histogram metric.

        Args:
            name: Metric name
            value: Observed value
            labels: Optional labels
            timestamp: Optional timestamp (default: current time)
        """
        self._record_metric(name, value, labels, timestamp, MetricType.HISTOGRAM)

    def record_timing(
        self,
        name: str,
        duration_seconds: float,
        labels: dict[str, str] | None = None,
        timestamp: datetime | None = None
    ) -> None:
        """Record a timing metric (histogram of durations).

        Args:
            name: Metric name
            duration_seconds: Duration in seconds
            labels: Optional labels
            timestamp: Optional timestamp (default: current time)
        """
        self._record_metric(name, duration_seconds, labels, timestamp, MetricType.HISTOGRAM)

    def _record_metric(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None,
        timestamp: datetime | None,
        expected_type: MetricType | None = None
    ) -> None:
        """Internal method to record a metric value.

        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels
            timestamp: Optional timestamp
            expected_type: Expected metric type for validation
        """
        start_time = time.time()

        try:
            # Sanitize inputs
            if not isinstance(value, int | float):
                logger.warning("Invalid metric value type", name=name, value_type=type(value))
                return

            labels = labels or {}
            timestamp = timestamp or datetime.utcnow()

            # Get or create metric definition
            metric_def = self._get_or_create_metric_def(name, expected_type)

            # Validate labels
            labels = self._validate_labels(metric_def, labels)

            # Create metric value
            metric_value = MetricValue(
                value=float(value),
                timestamp=timestamp,
                labels=labels
            )

            # Store in internal storage
            with self._lock:
                self._metrics[name].append(metric_value)

            # Update Prometheus metric
            self._update_prometheus_metric(name, value, labels)

            # Track recording performance
            recording_time = (time.time() - start_time) * 1000  # ms
            self._recording_times.append(recording_time)

            logger.debug("Metric recorded",
                        name=name,
                        value=value,
                        labels=labels,
                        recording_time_ms=recording_time)

        except Exception as e:
            logger.error("Failed to record metric",
                        name=name,
                        value=value,
                        error=str(e))

    def _get_or_create_metric_def(
        self,
        name: str,
        expected_type: MetricType | None = None
    ) -> MetricDefinition:
        """Get existing metric definition or create a new one.

        Args:
            name: Metric name
            expected_type: Expected metric type

        Returns:
            MetricDefinition object
        """
        with self._lock:
            if name in self._metric_definitions:
                return self._metric_definitions[name]

            # Create new metric definition
            metric_type = expected_type or MetricType.GAUGE
            metric_def = MetricDefinition(
                name=name,
                metric_type=metric_type,
                description=f"Auto-generated metric: {name}",
                unit="count"
            )

            self._metric_definitions[name] = metric_def
            self._create_prometheus_metric(metric_def)

            return metric_def

    def _validate_labels(
        self,
        metric_def: MetricDefinition,
        labels: dict[str, str]
    ) -> dict[str, str]:
        """Validate and sanitize labels.

        Args:
            metric_def: Metric definition
            labels: Input labels

        Returns:
            Validated labels
        """
        if not metric_def.labels:
            return {}

        validated = {}
        for label_name in metric_def.labels:
            if label_name in labels:
                # Sanitize label value
                value = str(labels[label_name])[:100]  # Limit length
                validated[label_name] = value
            else:
                validated[label_name] = "unknown"

        return validated

    def _update_prometheus_metric(
        self,
        name: str,
        value: float,
        labels: dict[str, str]
    ) -> None:
        """Update the corresponding Prometheus metric.

        Args:
            name: Metric name
            value: Metric value
            labels: Metric labels
        """
        try:
            prometheus_metric = self._prometheus_metrics.get(name)
            if not prometheus_metric:
                return

            metric_def = self._metric_definitions[name]

            if labels and hasattr(prometheus_metric, 'labels'):
                # Metric with labels
                label_values = [labels.get(label, "unknown") for label in metric_def.labels]
                labeled_metric = prometheus_metric.labels(*label_values)

                if metric_def.metric_type == MetricType.COUNTER:
                    labeled_metric.inc(value)
                elif metric_def.metric_type == MetricType.GAUGE:
                    labeled_metric.set(value)
                elif metric_def.metric_type in (MetricType.HISTOGRAM, MetricType.SUMMARY):
                    labeled_metric.observe(value)
            # Metric without labels
            elif metric_def.metric_type == MetricType.COUNTER:
                prometheus_metric.inc(value)
            elif metric_def.metric_type == MetricType.GAUGE:
                prometheus_metric.set(value)
            elif metric_def.metric_type in (MetricType.HISTOGRAM, MetricType.SUMMARY):
                prometheus_metric.observe(value)

        except Exception as e:
            logger.debug("Failed to update Prometheus metric",
                        name=name, error=str(e))

    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup of old metric values."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.cleanup_interval)

                if self._shutdown_event.is_set():
                    break

                await self._cleanup_old_metrics()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in periodic cleanup", error=str(e))
                await asyncio.sleep(60)  # Wait before retrying

    async def _cleanup_old_metrics(self) -> None:
        """Remove old metric values based on TTL."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)
        cleaned_count = 0

        with self._lock:
            for metric_name, values in list(self._metrics.items()):
                original_count = len(values)

                # Create new deque with only recent values
                recent_values = deque(maxlen=values.maxlen)
                for value in values:
                    if value.timestamp >= cutoff_time:
                        recent_values.append(value)

                self._metrics[metric_name] = recent_values
                cleaned_count += original_count - len(recent_values)

        if cleaned_count > 0:
            logger.debug("Cleaned up old metrics",
                        cleaned_count=cleaned_count,
                        cutoff_time=cutoff_time.isoformat())

    def get_metric_values(
        self,
        name: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        labels: dict[str, str] | None = None,
        limit: int | None = None
    ) -> list[MetricValue]:
        """Get metric values within a time range.

        Args:
            name: Metric name
            start_time: Start time filter
            end_time: End time filter
            labels: Label filters
            limit: Maximum number of values to return

        Returns:
            List of matching metric values
        """
        with self._lock:
            values = list(self._metrics.get(name, []))

        # Apply time filters
        if start_time:
            values = [v for v in values if v.timestamp >= start_time]
        if end_time:
            values = [v for v in values if v.timestamp <= end_time]

        # Apply label filters
        if labels:
            filtered_values = []
            for value in values:
                if all(value.labels.get(k) == v for k, v in labels.items()):
                    filtered_values.append(value)
            values = filtered_values

        # Apply limit
        if limit and len(values) > limit:
            values = values[-limit:]  # Take most recent

        return values

    async def get_metric_summary(
        self,
        name: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        labels: dict[str, str] | None = None
    ) -> MetricSummary | None:
        """Get summary statistics for a metric.

        Args:
            name: Metric name
            start_time: Start time (default: 1 hour ago)
            end_time: End time (default: now)
            labels: Label filters

        Returns:
            MetricSummary or None if no data
        """
        if not end_time:
            end_time = datetime.utcnow()
        if not start_time:
            start_time = end_time - timedelta(hours=1)

        values = self.get_metric_values(name, start_time, end_time, labels)

        if not values:
            return None

        numeric_values = [v.value for v in values]

        # Calculate statistics
        count = len(numeric_values)
        sum_value = sum(numeric_values)
        min_value = min(numeric_values)
        max_value = max(numeric_values)
        avg_value = sum_value / count

        # Calculate percentiles
        sorted_values = sorted(numeric_values)
        p50_value = statistics.median(sorted_values)
        p95_value = sorted_values[int(0.95 * len(sorted_values))] if len(sorted_values) > 1 else sorted_values[0]
        p99_value = sorted_values[int(0.99 * len(sorted_values))] if len(sorted_values) > 1 else sorted_values[0]

        # Calculate standard deviation
        stddev_value = statistics.stdev(numeric_values) if count > 1 else 0.0

        # Calculate rate (for counters)
        duration_seconds = (end_time - start_time).total_seconds()
        rate_per_second = sum_value / max(duration_seconds, 1)

        return MetricSummary(
            metric_name=name,
            start_time=start_time,
            end_time=end_time,
            count=count,
            sum_value=sum_value,
            min_value=min_value,
            max_value=max_value,
            avg_value=avg_value,
            p50_value=p50_value,
            p95_value=p95_value,
            p99_value=p99_value,
            stddev_value=stddev_value,
            rate_per_second=rate_per_second
        )

    async def get_all_metrics_summary(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None
    ) -> dict[str, MetricSummary]:
        """Get summary for all metrics.

        Args:
            start_time: Start time
            end_time: End time

        Returns:
            Dictionary of metric summaries
        """
        summaries = {}

        with self._lock:
            metric_names = list(self._metrics.keys())

        for name in metric_names:
            summary = await self.get_metric_summary(name, start_time, end_time)
            if summary:
                summaries[name] = summary

        return summaries

    def get_prometheus_metrics(self) -> str:
        """Get Prometheus-formatted metrics.

        Returns:
            Prometheus metrics text
        """
        try:
            return generate_latest(self.prometheus_registry).decode('utf-8')
        except Exception as e:
            logger.error("Failed to generate Prometheus metrics", error=str(e))
            return ""

    def get_metric_definitions(self) -> dict[str, MetricDefinition]:
        """Get all registered metric definitions.

        Returns:
            Dictionary of metric definitions
        """
        with self._lock:
            return self._metric_definitions.copy()

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics for the collector.

        Returns:
            Performance statistics
        """
        with self._lock:
            metric_count = sum(len(values) for values in self._metrics.values())
            avg_recording_time = (
                statistics.mean(self._recording_times)
                if self._recording_times else 0.0
            )

        return {
            "total_metrics": len(self._metric_definitions),
            "total_values": metric_count,
            "avg_recording_time_ms": avg_recording_time,
            "retention_hours": self.retention_hours,
            "max_values_per_metric": self.max_values_per_metric,
            "cleanup_interval_seconds": self.cleanup_interval,
        }


# Context manager for timing operations
class MetricTimer:
    """Context manager for measuring operation duration.

    Examples:
        >>> collector = MetricsCollector()
        >>> with MetricTimer(collector, "operation_duration", {"type": "search"}):
        ...     # Perform operation
        ...     pass
    """

    def __init__(
        self,
        collector: MetricsCollector,
        metric_name: str,
        labels: dict[str, str] | None = None,
        record_as: MetricType = MetricType.HISTOGRAM
    ) -> None:
        """Initialize timer.

        Args:
            collector: MetricsCollector instance
            metric_name: Name of the timing metric
            labels: Optional labels
            record_as: How to record the timing (histogram or gauge)
        """
        self.collector = collector
        self.metric_name = metric_name
        self.labels = labels or {}
        self.record_as = record_as
        self.start_time: float | None = None

    def __enter__(self) -> 'MetricTimer':
        """Start timing."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timing and record metric."""
        if self.start_time is not None:
            duration = time.time() - self.start_time

            if self.record_as == MetricType.HISTOGRAM:
                self.collector.record_histogram(
                    self.metric_name,
                    duration,
                    self.labels
                )
            else:
                self.collector.record_gauge(
                    self.metric_name,
                    duration,
                    self.labels
                )


# Global metrics collector instance
metrics_collector = MetricsCollector()


async def initialize_metrics_collector() -> MetricsCollector:
    """Initialize the global metrics collector.

    Returns:
        Initialized MetricsCollector instance
    """
    await metrics_collector.initialize()
    return metrics_collector


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance.

    Returns:
        MetricsCollector instance
    """
    return metrics_collector
