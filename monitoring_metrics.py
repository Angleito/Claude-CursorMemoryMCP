#!/usr/bin/env python3
"""
Comprehensive Monitoring and Metrics System for mem0ai Vector Operations
Production-grade monitoring with real-time metrics, alerting, and dashboards
"""

import asyncio
import asyncpg
import time
import logging
import json
import threading
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import psutil
import queue
from collections import defaultdict, deque
import aiohttp
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics to collect"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMING = "timing"

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class MetricDefinition:
    """Definition of a metric"""
    name: str
    metric_type: MetricType
    description: str
    unit: str
    labels: List[str] = None
    
@dataclass
class MetricValue:
    """A metric value with timestamp and labels"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = None
    
@dataclass
class AlertRule:
    """Alert rule definition"""
    name: str
    metric_name: str
    condition: str  # e.g., "> 100", "< 0.95"
    level: AlertLevel
    duration_seconds: int = 60  # How long condition must be true
    description: str = ""
    
@dataclass
class Alert:
    """An active alert"""
    rule_name: str
    metric_name: str
    level: AlertLevel
    message: str
    value: float
    threshold: str
    started_at: datetime
    resolved_at: Optional[datetime] = None

class MetricsCollector:
    """Collects and stores metrics"""
    
    def __init__(self, retention_hours: int = 24, max_values_per_metric: int = 5000):
        # Reduced max values for better memory efficiency
        self.metrics = defaultdict(lambda: deque(maxlen=max_values_per_metric))
        self.retention_hours = retention_hours
        self.max_values_per_metric = max_values_per_metric
        self.lock = threading.Lock()
        self._cleanup_counter = 0
        
        # Built-in metric definitions
        self.metric_definitions = {
            'vector_search_duration_ms': MetricDefinition(
                name='vector_search_duration_ms',
                metric_type=MetricType.HISTOGRAM,
                description='Time taken for vector similarity searches',
                unit='milliseconds',
                labels=['user_id', 'index_type']
            ),
            'embedding_generation_duration_ms': MetricDefinition(
                name='embedding_generation_duration_ms',
                metric_type=MetricType.HISTOGRAM,
                description='Time taken for embedding generation',
                unit='milliseconds',
                labels=['provider', 'model']
            ),
            'memory_insert_rate': MetricDefinition(
                name='memory_insert_rate',
                metric_type=MetricType.COUNTER,
                description='Rate of memory insertions',
                unit='operations/second',
                labels=['user_id']
            ),
            'active_users': MetricDefinition(
                name='active_users',
                metric_type=MetricType.GAUGE,
                description='Number of active users in the last hour',
                unit='count'
            ),
            'database_connections': MetricDefinition(
                name='database_connections',
                metric_type=MetricType.GAUGE,
                description='Number of active database connections',
                unit='count'
            ),
            'memory_usage_mb': MetricDefinition(
                name='memory_usage_mb',
                metric_type=MetricType.GAUGE,
                description='System memory usage',
                unit='megabytes'
            ),
            'cpu_usage_percent': MetricDefinition(
                name='cpu_usage_percent',
                metric_type=MetricType.GAUGE,
                description='System CPU usage',
                unit='percent'
            ),
            'disk_usage_percent': MetricDefinition(
                name='disk_usage_percent',
                metric_type=MetricType.GAUGE,
                description='Disk usage percentage',
                unit='percent'
            ),
            'vector_index_size_mb': MetricDefinition(
                name='vector_index_size_mb',
                metric_type=MetricType.GAUGE,
                description='Size of vector indexes',
                unit='megabytes',
                labels=['index_name', 'index_type']
            ),
            'cache_hit_rate': MetricDefinition(
                name='cache_hit_rate',
                metric_type=MetricType.GAUGE,
                description='Cache hit rate for various operations',
                unit='ratio',
                labels=['cache_type']
            ),
            'error_rate': MetricDefinition(
                name='error_rate',
                metric_type=MetricType.COUNTER,
                description='Rate of errors by type',
                unit='errors/second',
                labels=['error_type', 'component']
            )
        }
    
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a metric value"""
        metric_value = MetricValue(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {}
        )
        
        with self.lock:
            self.metrics[name].append(metric_value)
            
        # Periodic cleanup to manage memory
        self._cleanup_counter += 1
        if self._cleanup_counter % 100 == 0:  # Cleanup every 100 metrics
            self._cleanup_old_metrics()
    
    def _cleanup_old_metrics(self):
        """Remove old metric values based on retention policy"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        with self.lock:
            for metric_name, values in self.metrics.items():
                # Create new deque with only recent values
                recent_values = deque(maxlen=values.maxlen)
                for value in values:
                    if value.timestamp >= cutoff_time:
                        recent_values.append(value)
                self.metrics[metric_name] = recent_values
    
    def get_metric_values(self, name: str, start_time: datetime = None, 
                         end_time: datetime = None, labels: Dict[str, str] = None) -> List[MetricValue]:
        """Get metric values within time range and matching labels"""
        with self.lock:
            values = list(self.metrics.get(name, []))
        
        # Filter by time range
        if start_time:
            values = [v for v in values if v.timestamp >= start_time]
        if end_time:
            values = [v for v in values if v.timestamp <= end_time]
        
        # Filter by labels
        if labels:
            filtered_values = []
            for value in values:
                if all(value.labels.get(k) == v for k, v in labels.items()):
                    filtered_values.append(value)
            values = filtered_values
        
        return values
    
    def get_metric_summary(self, name: str, start_time: datetime = None, 
                          end_time: datetime = None) -> Dict[str, float]:
        """Get summary statistics for a metric"""
        values = self.get_metric_values(name, start_time, end_time)
        
        if not values:
            return {}
        
        numeric_values = [v.value for v in values]
        
        return {
            'count': len(numeric_values),
            'min': min(numeric_values),
            'max': max(numeric_values),
            'mean': np.mean(numeric_values),
            'median': np.median(numeric_values),
            'p95': np.percentile(numeric_values, 95),
            'p99': np.percentile(numeric_values, 99),
            'std': np.std(numeric_values)
        }

class AlertManager:
    """Manages alerting rules and notifications"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.notification_callbacks = []
        self.lock = threading.Lock()
        
        # Default alert rules
        self._setup_default_alerts()
    
    def _setup_default_alerts(self):
        """Setup default alert rules"""
        default_rules = [
            AlertRule(
                name="high_search_latency",
                metric_name="vector_search_duration_ms",
                condition="> 5000",
                level=AlertLevel.WARNING,
                duration_seconds=120,
                description="Vector search taking longer than 5 seconds"
            ),
            AlertRule(
                name="very_high_search_latency",
                metric_name="vector_search_duration_ms",
                condition="> 10000",
                level=AlertLevel.CRITICAL,
                duration_seconds=60,
                description="Vector search taking longer than 10 seconds"
            ),
            AlertRule(
                name="high_memory_usage",
                metric_name="memory_usage_mb",
                condition="> 8192",
                level=AlertLevel.WARNING,
                duration_seconds=300,
                description="High system memory usage"
            ),
            AlertRule(
                name="critical_memory_usage",
                metric_name="memory_usage_mb",
                condition="> 12288",
                level=AlertLevel.CRITICAL,
                duration_seconds=60,
                description="Critical system memory usage"
            ),
            AlertRule(
                name="high_cpu_usage",
                metric_name="cpu_usage_percent",
                condition="> 80",
                level=AlertLevel.WARNING,
                duration_seconds=300,
                description="High CPU usage"
            ),
            AlertRule(
                name="low_cache_hit_rate",
                metric_name="cache_hit_rate",
                condition="< 0.5",
                level=AlertLevel.WARNING,
                duration_seconds=300,
                description="Low cache hit rate"
            ),
            AlertRule(
                name="high_error_rate",
                metric_name="error_rate",
                condition="> 10",
                level=AlertLevel.CRITICAL,
                duration_seconds=120,
                description="High error rate"
            )
        ]
        
        for rule in default_rules:
            self.add_alert_rule(rule)
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule"""
        with self.lock:
            self.alert_rules[rule.name] = rule
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule"""
        with self.lock:
            if rule_name in self.alert_rules:
                del self.alert_rules[rule_name]
            if rule_name in self.active_alerts:
                del self.active_alerts[rule_name]
    
    def add_notification_callback(self, callback: Callable[[Alert], None]):
        """Add a notification callback function"""
        self.notification_callbacks.append(callback)
    
    def check_alerts(self):
        """Check all alert rules against current metrics"""
        current_time = datetime.now()
        
        with self.lock:
            for rule_name, rule in self.alert_rules.items():
                self._check_single_alert(rule, current_time)
    
    def _check_single_alert(self, rule: AlertRule, current_time: datetime):
        """Check a single alert rule"""
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
                    started_at=current_time
                )
                
                self.active_alerts[rule.name] = alert
                self.alert_history.append(alert)
                
                # Notify
                for callback in self.notification_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"Notification callback failed: {e}")
                
                logger.warning(f"ALERT: {rule.name} - {rule.description} "
                             f"(value: {values[-1].value}, threshold: {rule.condition})")
        else:
            if rule.name in self.active_alerts:
                # Resolve alert
                alert = self.active_alerts[rule.name]
                alert.resolved_at = current_time
                del self.active_alerts[rule.name]
                
                logger.info(f"RESOLVED: {rule.name}")
    
    def _evaluate_condition(self, values: List[MetricValue], condition: str) -> bool:
        """Evaluate if metric values meet the alert condition"""
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
        """Get all active alerts"""
        with self.lock:
            return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history"""
        with self.lock:
            return list(self.alert_history)[-limit:]

class SystemMetricsCollector:
    """Collects system-level metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector, db_pool: asyncpg.Pool):
        self.metrics_collector = metrics_collector
        self.db_pool = db_pool
        self.collection_interval = 30  # seconds
        self.running = False
        self.collection_task = None
    
    async def start_collection(self):
        """Start collecting system metrics"""
        if self.running:
            logger.warning("System metrics collection already running")
            return
            
        self.running = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("System metrics collection started")
    
    async def stop_collection(self):
        """Stop collecting system metrics"""
        self.running = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        logger.info("System metrics collection stopped")
    
    async def _collection_loop(self):
        """Main collection loop"""
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
        """Collect system-level metrics"""
        # Memory usage
        memory = psutil.virtual_memory()
        self.metrics_collector.record_metric(
            'memory_usage_mb', 
            memory.used / (1024 * 1024)
        )
        self.metrics_collector.record_metric(
            'memory_usage_percent', 
            memory.percent
        )
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics_collector.record_metric('cpu_usage_percent', cpu_percent)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        self.metrics_collector.record_metric(
            'disk_usage_percent', 
            (disk.used / disk.total) * 100
        )
        
        # Network I/O
        net_io = psutil.net_io_counters()
        self.metrics_collector.record_metric('network_bytes_sent', net_io.bytes_sent)
        self.metrics_collector.record_metric('network_bytes_recv', net_io.bytes_recv)
    
    async def _collect_database_metrics(self):
        """Collect database-specific metrics"""
        try:
            async with self.db_pool.acquire() as conn:
                # Connection count
                connection_stats = await conn.fetchrow("""
                    SELECT 
                        count(*) as total_connections,
                        count(*) FILTER (WHERE state = 'active') as active_connections,
                        count(*) FILTER (WHERE state = 'idle') as idle_connections
                    FROM pg_stat_activity
                """)
                
                self.metrics_collector.record_metric(
                    'database_connections', 
                    connection_stats['total_connections']
                )
                self.metrics_collector.record_metric(
                    'database_active_connections', 
                    connection_stats['active_connections']
                )
                
                # Table statistics
                # Check if schema exists first
                schema_exists = await conn.fetchval("""
                    SELECT EXISTS(
                        SELECT 1 FROM information_schema.schemata 
                        WHERE schema_name = 'mem0_vectors'
                    )
                """)
                
                if schema_exists:
                    table_stats = await conn.fetch("""
                        SELECT 
                            tablename,
                            n_tup_ins as inserts,
                            n_tup_upd as updates,
                            n_tup_del as deletes,
                            n_live_tup as live_tuples,
                            n_dead_tup as dead_tuples
                        FROM pg_stat_user_tables
                        WHERE schemaname = 'mem0_vectors'
                    """)
                else:
                    table_stats = []
                
                for row in table_stats:
                    labels = {'table': row['tablename']}
                    self.metrics_collector.record_metric(
                        'table_live_tuples', row['live_tuples'], labels
                    )
                    self.metrics_collector.record_metric(
                        'table_dead_tuples', row['dead_tuples'], labels
                    )
                
                # Index statistics (only if schema exists)
                if schema_exists:
                    index_stats = await conn.fetch("""
                        SELECT 
                            indexname,
                            idx_scan,
                            idx_tup_read,
                            idx_tup_fetch,
                            pg_size_pretty(pg_relation_size(indexrelid)) as size_pretty,
                            pg_relation_size(indexrelid) as size_bytes
                        FROM pg_stat_user_indexes
                        WHERE schemaname = 'mem0_vectors'
                    """)
                else:
                    index_stats = []
                
                for row in index_stats:
                    labels = {'index': row['indexname']}
                    self.metrics_collector.record_metric(
                        'index_scans', row['idx_scan'], labels
                    )
                    self.metrics_collector.record_metric(
                        'vector_index_size_mb', 
                        row['size_bytes'] / (1024 * 1024), 
                        labels
                    )
                
                # Active users in last hour (only if schema and table exist)
                if schema_exists:
                    try:
                        active_users = await conn.fetchval("""
                            SELECT COUNT(DISTINCT user_id)
                            FROM mem0_vectors.memories
                            WHERE last_accessed > NOW() - INTERVAL '1 hour'
                        """)
                    except Exception:
                        # Table might not exist
                        active_users = 0
                else:
                    active_users = 0
                
                self.metrics_collector.record_metric('active_users', active_users or 0)
                
        except Exception as e:
            logger.error(f"Error collecting database metrics: {e}")

class PerformanceProfiler:
    """Performance profiling for vector operations"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.active_operations = {}
        self.lock = threading.Lock()
    
    def start_operation(self, operation_id: str, operation_type: str, 
                       metadata: Dict[str, str] = None) -> str:
        """Start timing an operation"""
        with self.lock:
            self.active_operations[operation_id] = {
                'type': operation_type,
                'start_time': time.time(),
                'metadata': metadata or {}
            }
        return operation_id
    
    def end_operation(self, operation_id: str, success: bool = True, 
                     additional_metrics: Dict[str, float] = None):
        """End timing an operation and record metrics"""
        with self.lock:
            if operation_id not in self.active_operations:
                return
            
            operation = self.active_operations.pop(operation_id)
            duration_ms = (time.time() - operation['start_time']) * 1000
            
            # Record timing metric
            labels = operation['metadata'].copy()
            labels['success'] = str(success)
            labels['operation_type'] = operation['type']
            
            self.metrics_collector.record_metric(
                f"{operation['type']}_duration_ms",
                duration_ms,
                labels
            )
            
            # Record additional metrics
            if additional_metrics:
                for metric_name, value in additional_metrics.items():
                    self.metrics_collector.record_metric(metric_name, value, labels)
    
    def profile_async_function(self, operation_type: str, metadata: Dict[str, str] = None):
        """Decorator to profile async functions"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                operation_id = f"{operation_type}_{time.time()}"
                self.start_operation(operation_id, operation_type, metadata)
                
                try:
                    result = await func(*args, **kwargs)
                    self.end_operation(operation_id, success=True)
                    return result
                except Exception as e:
                    self.end_operation(operation_id, success=False)
                    raise
            
            return wrapper
        return decorator

class MetricsExporter:
    """Exports metrics to external systems"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.exporters = {}
    
    def add_prometheus_exporter(self, port: int = 8000):
        """Add Prometheus metrics exporter"""
        try:
            from prometheus_client import start_http_server, Gauge, Counter, Histogram
            
            # Create Prometheus metrics
            self.prometheus_metrics = {}
            
            for name, definition in self.metrics_collector.metric_definitions.items():
                if definition.metric_type == MetricType.GAUGE:
                    metric = Gauge(name, definition.description, definition.labels or [])
                elif definition.metric_type == MetricType.COUNTER:
                    metric = Counter(name, definition.description, definition.labels or [])
                elif definition.metric_type == MetricType.HISTOGRAM:
                    metric = Histogram(name, definition.description, definition.labels or [])
                else:
                    metric = Gauge(name, definition.description, definition.labels or [])
                
                self.prometheus_metrics[name] = metric
            
            # Start HTTP server
            start_http_server(port)
            
            # Store export loop task to avoid "coroutine was never awaited" warning
            self.export_task = asyncio.create_task(self._prometheus_export_loop())
            
            logger.info(f"Prometheus exporter started on port {port}")
            
        except ImportError:
            logger.warning("prometheus_client not available, skipping Prometheus exporter")
    
    async def _prometheus_export_loop(self):
        """Export metrics to Prometheus periodically"""
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
                        
                        if latest_value.labels and hasattr(prometheus_metric, '_labelnames'):
                            # Metric with labels
                            label_values = [latest_value.labels.get(label, '') 
                                          for label in prometheus_metric._labelnames]
                            prometheus_metric.labels(*label_values).set(latest_value.value)
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
    """Simple web dashboard for monitoring metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector, alert_manager: AlertManager,
                 port: int = 8080):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.port = port
        self.app = None
    
    async def start_dashboard(self):
        """Start the web dashboard"""
        try:
            from aiohttp import web, web_response
            
            self.app = web.Application()
            
            # Add routes
            self.app.router.add_get('/', self._dashboard_handler)
            self.app.router.add_get('/metrics', self._metrics_handler)
            self.app.router.add_get('/alerts', self._alerts_handler)
            self.app.router.add_get('/api/metrics/{metric_name}', self._api_metrics_handler)
            
            # Start server
            runner = web.AppRunner(self.app)
            await runner.setup()
            site = web.TCPSite(runner, 'localhost', self.port)
            await site.start()
            
            logger.info(f"Monitoring dashboard started on http://localhost:{self.port}")
            
        except ImportError:
            logger.warning("aiohttp not available, skipping web dashboard")
    
    async def _dashboard_handler(self, request):
        """Main dashboard page"""
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
        
        return web.Response(text=html, content_type='text/html')
    
    async def _metrics_handler(self, request):
        """Metrics API endpoint"""
        current_time = datetime.now()
        lookback_time = current_time - timedelta(minutes=30)
        
        metrics_data = {}
        
        for metric_name in self.metrics_collector.metric_definitions.keys():
            summary = self.metrics_collector.get_metric_summary(
                metric_name, lookback_time, current_time
            )
            
            if summary:
                values = self.metrics_collector.get_metric_values(
                    metric_name, lookback_time, current_time
                )
                summary['latest'] = values[-1].value if values else 0
                metrics_data[metric_name] = summary
        
        return web.json_response({'metrics': metrics_data})
    
    async def _alerts_handler(self, request):
        """Alerts API endpoint"""
        active_alerts = self.alert_manager.get_active_alerts()
        
        alerts_data = []
        for alert in active_alerts:
            alerts_data.append({
                'rule_name': alert.rule_name,
                'metric_name': alert.metric_name,
                'level': alert.level.value,
                'message': alert.message,
                'value': alert.value,
                'threshold': alert.threshold,
                'started_at': alert.started_at.isoformat()
            })
        
        return web.json_response({'alerts': alerts_data})
    
    async def _api_metrics_handler(self, request):
        """Individual metric API endpoint"""
        metric_name = request.match_info['metric_name']
        
        current_time = datetime.now()
        lookback_time = current_time - timedelta(hours=1)
        
        values = self.metrics_collector.get_metric_values(
            metric_name, lookback_time, current_time
        )
        
        data_points = [
            {
                'timestamp': v.timestamp.isoformat(),
                'value': v.value,
                'labels': v.labels
            }
            for v in values
        ]
        
        return web.json_response({'data': data_points})

# Main monitoring system
class MonitoringSystem:
    """Complete monitoring system for mem0ai"""
    
    def __init__(self, db_url: str, config: Dict[str, Any] = None):
        self.db_url = db_url
        self.config = config or {}
        self.pool = None
        
        # Initialize components
        self.metrics_collector = MetricsCollector(
            retention_hours=self.config.get('retention_hours', 24)
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
            port=self.config.get('dashboard_port', 8080)
        )
        
        # Alert checking task
        self.alert_check_task = None
    
    async def initialize(self):
        """Initialize the monitoring system"""
        # Initialize database connection
        self.pool = await asyncpg.create_pool(
            self.db_url,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        
        # Initialize system metrics collector
        self.system_collector = SystemMetricsCollector(self.metrics_collector, self.pool)
        
        # Start components
        await self.system_collector.start_collection()
        
        # Start alert checking
        self.alert_check_task = asyncio.create_task(self._alert_check_loop())
        
        # Start dashboard
        await self.dashboard.start_dashboard()
        
        # Setup exporters if configured
        if self.config.get('prometheus_port'):
            self.exporter.add_prometheus_exporter(self.config['prometheus_port'])
        
        logger.info("Monitoring system initialized")
    
    async def cleanup(self):
        """Cleanup monitoring system"""
        if self.system_collector:
            await self.system_collector.stop_collection()
        
        if self.alert_check_task:
            self.alert_check_task.cancel()
            try:
                await self.alert_check_task
            except asyncio.CancelledError:
                pass
        
        if self.pool:
            await self.pool.close()
    
    async def _alert_check_loop(self):
        """Periodically check alerts"""
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
        """Get the performance profiler"""
        return self.profiler
    
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a metric value"""
        self.metrics_collector.record_metric(name, value, labels)

# Example usage
async def main():
    """Test the monitoring system"""
    import os
    
    DB_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost/mem0ai')
    
    config = {
        'retention_hours': 24,
        'dashboard_port': 8080,
        'prometheus_port': 8000
    }
    
    monitoring = MonitoringSystem(DB_URL, config)
    
    try:
        await monitoring.initialize()
        
        # Simulate some metrics
        profiler = monitoring.get_profiler()
        
        print("Monitoring system started. Simulating some metrics...")
        
        for i in range(10):
            # Simulate vector search
            operation_id = profiler.start_operation(
                f"search_{i}", 
                "vector_search",
                {'user_id': 'test_user', 'index_type': 'hnsw'}
            )
            
            await asyncio.sleep(0.1)  # Simulate work
            
            profiler.end_operation(operation_id, success=True, additional_metrics={
                'results_count': 10,
                'similarity_threshold': 0.8
            })
            
            # Record some custom metrics
            monitoring.record_metric('custom_metric', i * 10)
            monitoring.record_metric('cache_hit_rate', 0.85, {'cache_type': 'embedding'})
            
            await asyncio.sleep(1)
        
        print("Metrics recorded. Dashboard available at http://localhost:8080")
        print("Prometheus metrics at http://localhost:8000")
        print("Press Ctrl+C to stop...")
        
        # Keep running
        while True:
            await asyncio.sleep(10)
            
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await monitoring.cleanup()

if __name__ == "__main__":
    asyncio.run(main())