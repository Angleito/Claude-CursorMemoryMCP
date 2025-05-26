"""Metrics and monitoring for Mem0 AI MCP Server"""

import time
import asyncio
import psutil
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import structlog

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('mem0_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('mem0_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
MEMORY_COUNT = Gauge('mem0_memories_total', 'Total memories in database')
USER_COUNT = Gauge('mem0_users_total', 'Total users in database')
ACTIVE_CONNECTIONS = Gauge('mem0_websocket_connections_active', 'Active WebSocket connections')
SEARCH_DURATION = Histogram('mem0_search_duration_seconds', 'Memory search duration', ['user_id', 'index_type'])
EMBEDDING_DURATION = Histogram('mem0_embedding_generation_duration_seconds', 'Embedding generation duration', ['provider', 'model'])
ERROR_COUNT = Counter('mem0_errors_total', 'Total errors', ['error_type', 'component'])
DATABASE_OPERATIONS = Counter('mem0_database_operations_total', 'Total database operations', ['operation', 'table'])
CACHE_HIT_RATE = Gauge('mem0_cache_hit_rate', 'Cache hit rate', ['cache_type'])
API_REQUEST_SIZE = Histogram('mem0_api_request_size_bytes', 'API request size in bytes', ['method', 'endpoint'])
API_RESPONSE_SIZE = Histogram('mem0_api_response_size_bytes', 'API response size in bytes', ['method', 'endpoint'])
SYSTEM_MEMORY_USAGE = Gauge('mem0_system_memory_usage_bytes', 'System memory usage in bytes')
SYSTEM_CPU_USAGE = Gauge('mem0_system_cpu_usage_percent', 'System CPU usage percentage')


class MetricsCollector:
    """Collects and manages application metrics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_times: Dict[str, float] = {}
        
    def record_request_start(self, request_id: str, method: str, endpoint: str):
        """Record the start of a request"""
        self.request_times[request_id] = time.time()
    
    def record_request_end(self, request_id: str, method: str, endpoint: str, status_code: int):
        """Record the end of a request"""
        if request_id in self.request_times:
            duration = time.time() - self.request_times[request_id]
            REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status_code).inc()
            del self.request_times[request_id]
    
    def record_search(self, duration: float, results_count: int, user_id: str, index_type: str = "default"):
        """Record search metrics"""
        SEARCH_DURATION.labels(user_id=user_id, index_type=index_type).observe(duration)
        logger.info("Search metrics recorded", 
                   duration=duration, 
                   results_count=results_count, 
                   user_id=user_id,
                   index_type=index_type)
    
    def record_embedding_generation(self, duration: float, text_length: int, provider: str = "default", model: str = "default"):
        """Record embedding generation metrics"""
        EMBEDDING_DURATION.labels(provider=provider, model=model).observe(duration)
        logger.info("Embedding metrics recorded", 
                   duration=duration, 
                   text_length=text_length,
                   provider=provider,
                   model=model)
    
    def record_error(self, error_type: str, error_message: str, component: str = "unknown", context: Dict[str, Any] = None):
        """Record error metrics"""
        # Sanitize labels to prevent metric explosion
        error_type = self._sanitize_label(error_type)
        component = self._sanitize_label(component)
        
        ERROR_COUNT.labels(error_type=error_type, component=component).inc()
        logger.error("Error recorded", 
                    error_type=error_type, 
                    error_message=error_message, 
                    component=component,
                    context=context)
    
    def update_memory_count(self, count: int):
        """Update memory count gauge"""
        MEMORY_COUNT.set(count)
    
    def update_user_count(self, count: int):
        """Update user count gauge"""
        USER_COUNT.set(count)
    
    def update_active_connections(self, count: int):
        """Update active connections gauge"""
        ACTIVE_CONNECTIONS.set(count)
    
    def get_uptime(self) -> float:
        """Get server uptime in seconds"""
        return time.time() - self.start_time
    
    def _sanitize_label(self, label: str) -> str:
        """Sanitize label values to prevent metric explosion"""
        if not label:
            return "unknown"
        # Keep only alphanumeric, underscore, and hyphen
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', str(label))
        # Limit length to prevent issues
        return sanitized[:50] if sanitized else "unknown"
    
    def collect_system_metrics(self):
        \"\"\"Collect system metrics\"\"\"
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            SYSTEM_MEMORY_USAGE.set(memory.used)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            SYSTEM_CPU_USAGE.set(cpu_percent)
            
            logger.debug(\"System metrics collected\",
                        memory_used_bytes=memory.used,
                        memory_percent=memory.percent,
                        cpu_percent=cpu_percent)
                        
        except Exception as e:
            logger.error(\"Failed to collect system metrics\", error=str(e))
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics"""
        try:
            # Safe access to metric values using prometheus_client API
            memory_count = MEMORY_COUNT._value._value if hasattr(MEMORY_COUNT, '_value') else 0
            user_count = USER_COUNT._value._value if hasattr(USER_COUNT, '_value') else 0
            active_connections = ACTIVE_CONNECTIONS._value._value if hasattr(ACTIVE_CONNECTIONS, '_value') else 0
            
            # Get counter totals safely
            total_requests = 0
            total_errors = 0
            
            if hasattr(REQUEST_COUNT, '_value'):
                total_requests = sum(sample.value for sample in REQUEST_COUNT.collect()[0].samples)
            
            if hasattr(ERROR_COUNT, '_value'):
                total_errors = sum(sample.value for sample in ERROR_COUNT.collect()[0].samples)
            
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
    \"\"\"Setup metrics endpoints\"\"\"
    
    # Schedule system metrics collection
    async def collect_system_metrics_periodically():
        while True:
            try:
                metrics_collector.collect_system_metrics()
                await asyncio.sleep(30)  # Collect every 30 seconds
            except Exception as e:
                logger.error(\"Error in system metrics collection\", error=str(e))
                await asyncio.sleep(60)
    
    # Start background task for system metrics
    asyncio.create_task(collect_system_metrics_periodically())


def setup_metrics_old(app):
    """Setup metrics endpoints"""
    
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint"""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    
    @app.get("/health/metrics")
    async def health_metrics():
        """Health check with metrics"""
        return metrics_collector.get_metrics_summary()


class PerformanceMonitor:
    """Monitor performance and trigger alerts"""
    
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
        """Start performance monitoring background task"""
        if self._running:
            logger.warning("Performance monitoring already running")
            return
            
        self._running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")
        
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self._running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
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
        """Check performance metrics and trigger alerts"""
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
        """Set performance threshold"""
        if metric in self.thresholds:
            self.thresholds[metric] = value
            logger.info("Threshold updated", metric=metric, value=value)


# Global performance monitor
performance_monitor = PerformanceMonitor()


class RequestTracker:
    """Track individual request performance"""
    
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
    """Decorator for tracking request performance"""
    return RequestTracker(request_id, method, endpoint)


class DatabaseHealthChecker:
    """Monitor database health and performance"""
    
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.last_check = None
        self.health_status = "unknown"
    
    async def check_health(self) -> Dict[str, Any]:
        """Check database health"""
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
            
        except asyncio.TimeoutError:
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
        """Start database health monitoring"""
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