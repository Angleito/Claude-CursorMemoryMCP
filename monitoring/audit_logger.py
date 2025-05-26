"""Comprehensive audit logging and monitoring system.

This module provides a complete audit logging solution with:
- Structured logging with context
- Prometheus metrics integration
- Background event processing
- Security event tracking
- Performance monitoring

Examples:
    >>> audit_logger = AuditLogger()
    >>> await audit_logger.initialize()
    >>> await audit_logger.log_event("user_login", "authentication", user_id=user.id)
    >>> await audit_logger.log_security_event("failed_login", ip_address="192.168.1.1")
"""

import asyncio
import json
import uuid
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import asyncpg
import structlog
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    start_http_server,
    CollectorRegistry,
    REGISTRY,
)

from config.settings import get_settings

settings = get_settings()

# Prometheus metrics
AUDIT_EVENTS_TOTAL = Counter(
    "audit_events_total", "Total audit events", ["action", "resource", "user_role"]
)
SECURITY_EVENTS_TOTAL = Counter(
    "security_events_total", "Total security events", ["event_type", "severity"]
)
API_REQUESTS_TOTAL = Counter(
    "api_requests_total", "Total API requests", ["method", "endpoint", "status_code"]
)
API_REQUEST_DURATION = Histogram(
    "api_request_duration_seconds", "API request duration", ["method", "endpoint"]
)
FAILED_LOGINS_TOTAL = Counter(
    "failed_logins_total", "Total failed login attempts", ["reason"]
)
ACTIVE_SESSIONS = Gauge("active_sessions_total", "Number of active user sessions")
DATABASE_OPERATIONS_TOTAL = Counter(
    "database_operations_total", "Total database operations", ["operation", "table"]
)

# Configure structured logging
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


class AuditEventType(str, Enum):
    """Types of audit events."""

    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_REGISTRATION = "user_registration"
    PASSWORD_CHANGE = "password_change"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    MEMORY_CREATED = "memory_created"
    MEMORY_UPDATED = "memory_updated"
    MEMORY_DELETED = "memory_deleted"
    MEMORY_ACCESSED = "memory_accessed"
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"
    SYSTEM_CONFIG_CHANGE = "system_config_change"
    SECURITY_POLICY_CHANGE = "security_policy_change"


class SecurityEventType(str, Enum):
    """Types of security events."""

    FAILED_LOGIN = "failed_login"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    BRUTE_FORCE_ATTEMPT = "brute_force_attempt"
    MALICIOUS_REQUEST = "malicious_request"
    DATA_BREACH_ATTEMPT = "data_breach_attempt"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"


class SecuritySeverity(str, Enum):
    """Security event severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditLogger:
    """Main audit logging class."""

    def __init__(self, max_queue_size: int = 10000, metrics_registry: Optional[CollectorRegistry] = None) -> None:
        """Initialize the audit logger.
        
        Args:
            max_queue_size: Maximum size of the event queue to prevent memory overflow
            metrics_registry: Optional Prometheus registry for metrics
        """
        self.settings = settings
        self.db_pool: Optional[asyncpg.Pool] = None
        self.event_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=max_queue_size)
        self.background_task: Optional[asyncio.Task[None]] = None
        self._shutdown_event = asyncio.Event()
        self._metrics_registry = metrics_registry or REGISTRY

        # Security thresholds - configurable
        self.failed_login_threshold = int(getattr(settings.security, 'failed_login_threshold', 5))
        self.suspicious_activity_threshold = int(getattr(settings.security, 'suspicious_activity_threshold', 10))
        self.rate_limit_threshold = int(getattr(settings.security, 'rate_limit_threshold', 100))
        
        # Performance tracking
        self._event_processing_times: List[float] = []
        self._max_processing_time_samples = 1000

    async def initialize(self) -> None:
        """Initialize audit logger with proper error handling and logging.
        
        Raises:
            ConnectionError: If database connection fails
            RuntimeError: If initialization fails
        """
        try:
            logger.info("Initializing audit logger", 
                       failed_login_threshold=self.failed_login_threshold,
                       queue_size=self.event_queue.maxsize)
            
            # Create database connection pool with retry logic
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    self.db_pool = await asyncpg.create_pool(
                        self.settings.database.database_url, 
                        min_size=2, 
                        max_size=10,
                        command_timeout=30,
                        server_settings={
                            'application_name': 'audit_logger',
                            'jit': 'off'  # Disable JIT for consistent performance
                        }
                    )
                    logger.info("Database connection pool created successfully")
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.error("Failed to create database pool after retries", 
                                   error=str(e), retry_count=retry_count)
                        raise ConnectionError(f"Database connection failed: {e}") from e
                    
                    logger.warning("Database connection attempt failed, retrying", 
                                 error=str(e), retry=retry_count, max_retries=max_retries)
                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff

            # Initialize database tables
            await self._ensure_audit_tables()
            
            # Start background processing task
            self.background_task = asyncio.create_task(
                self._process_events(), 
                name="audit_logger_background_processor"
            )
            logger.info("Background event processing task started")

            # Start Prometheus metrics server if configured
            if hasattr(self.settings, 'monitoring') and getattr(self.settings.monitoring, 'prometheus_port', None):
                try:
                    start_http_server(self.settings.monitoring.prometheus_port, registry=self._metrics_registry)
                    logger.info("Prometheus metrics server started", 
                               port=self.settings.monitoring.prometheus_port)
                except Exception as e:
                    logger.warning("Failed to start Prometheus metrics server", error=str(e))
            
            logger.info("Audit logger initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize audit logger", error=str(e))
            await self.close()
            raise RuntimeError(f"Audit logger initialization failed: {e}") from e

    async def close(self) -> None:
        """Close audit logger gracefully with proper cleanup."""
        logger.info("Shutting down audit logger")
        
        # Signal shutdown to background task
        self._shutdown_event.set()
        
        # Cancel and wait for background task
        if self.background_task and not self.background_task.done():
            self.background_task.cancel()
            with suppress(asyncio.CancelledError):
                try:
                    await asyncio.wait_for(self.background_task, timeout=10.0)
                    logger.info("Background task shut down gracefully")
                except asyncio.TimeoutError:
                    logger.warning("Background task shutdown timed out")
        
        # Process remaining events in queue with timeout
        remaining_events = 0
        start_time = asyncio.get_event_loop().time()
        timeout = 30.0  # 30 seconds to process remaining events
        
        while not self.event_queue.empty() and (asyncio.get_event_loop().time() - start_time) < timeout:
            try:
                event = self.event_queue.get_nowait()
                await self._process_single_event(event)
                remaining_events += 1
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                logger.error("Error processing remaining event during shutdown", error=str(e))
        
        if remaining_events > 0:
            logger.info("Processed remaining events during shutdown", count=remaining_events)
        
        # Close database pool
        if self.db_pool:
            try:
                await self.db_pool.close()
                logger.info("Database connection pool closed")
            except Exception as e:
                logger.error("Error closing database pool", error=str(e))
        
        logger.info("Audit logger shutdown complete")

    async def _ensure_audit_tables(self) -> None:
        """Ensure audit tables exist with proper error handling.
        
        Raises:
            DatabaseError: If table creation fails
        """
        if not self.db_pool:
            raise RuntimeError("Database pool not initialized")
            
        try:
            async with self.db_pool.acquire() as conn:
                logger.debug("Creating audit tables if they don't exist")
                
                # Audit logs table with improved schema
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        user_id UUID,
                        api_key_id UUID,
                        action VARCHAR(100) NOT NULL,
                        resource VARCHAR(100) NOT NULL,
                        resource_id VARCHAR(100),
                        ip_address INET,
                        user_agent TEXT,
                        details JSONB DEFAULT '{}',
                        timestamp TIMESTAMPTZ DEFAULT NOW(),
                        session_id VARCHAR(100),
                        success BOOLEAN DEFAULT TRUE,
                        duration_ms INTEGER,
                        CONSTRAINT audit_logs_action_check CHECK (action != ''),
                        CONSTRAINT audit_logs_resource_check CHECK (resource != '')
                    )
                """
                )

                # Create indexes for better query performance
                index_queries = [
                    "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs (timestamp DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_audit_user_id ON audit_logs (user_id) WHERE user_id IS NOT NULL",
                    "CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs (action)",
                    "CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_logs (resource)",
                    "CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_logs (session_id) WHERE session_id IS NOT NULL",
                    "CREATE INDEX IF NOT EXISTS idx_audit_ip_address ON audit_logs (ip_address) WHERE ip_address IS NOT NULL",
                    "CREATE INDEX IF NOT EXISTS idx_audit_success ON audit_logs (success, timestamp DESC)"
                ]
                
                for query in index_queries:
                    try:
                        await conn.execute(query)
                    except Exception as e:
                        logger.warning("Failed to create index", query=query, error=str(e))

                # Security events table with enhanced schema
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS security_events (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        event_type VARCHAR(50) NOT NULL,
                        user_id UUID,
                        ip_address INET,
                        details JSONB DEFAULT '{}',
                        severity VARCHAR(20) NOT NULL,
                        timestamp TIMESTAMPTZ DEFAULT NOW(),
                        resolved BOOLEAN DEFAULT FALSE,
                        resolved_at TIMESTAMPTZ,
                        resolved_by UUID,
                        alert_sent BOOLEAN DEFAULT FALSE,
                        alert_sent_at TIMESTAMPTZ,
                        risk_score INTEGER DEFAULT 0,
                        session_id VARCHAR(100),
                        user_agent TEXT,
                        CONSTRAINT security_events_severity_check CHECK (severity IN ('low', 'medium', 'high', 'critical')),
                        CONSTRAINT security_events_risk_score_check CHECK (risk_score >= 0 AND risk_score <= 100)
                    )
                """
                )

                # Create indexes for security events
                security_index_queries = [
                    "CREATE INDEX IF NOT EXISTS idx_security_timestamp ON security_events (timestamp DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_security_event_type ON security_events (event_type)",
                    "CREATE INDEX IF NOT EXISTS idx_security_severity ON security_events (severity, timestamp DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_security_resolved ON security_events (resolved, timestamp DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_security_ip_address ON security_events (ip_address) WHERE ip_address IS NOT NULL",
                    "CREATE INDEX IF NOT EXISTS idx_security_user_id ON security_events (user_id) WHERE user_id IS NOT NULL",
                    "CREATE INDEX IF NOT EXISTS idx_security_risk_score ON security_events (risk_score DESC) WHERE risk_score > 0",
                    "CREATE INDEX IF NOT EXISTS idx_security_unresolved ON security_events (timestamp DESC) WHERE resolved = FALSE"
                ]
                
                for query in security_index_queries:
                    try:
                        await conn.execute(query)
                    except Exception as e:
                        logger.warning("Failed to create security index", query=query, error=str(e))

                # API call logs table with enhanced tracking
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS api_call_logs (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        user_id UUID,
                        api_key_id UUID,
                        method VARCHAR(10) NOT NULL,
                        path VARCHAR(500) NOT NULL,
                        status_code INTEGER NOT NULL,
                        duration_ms INTEGER,
                        ip_address INET,
                        user_agent TEXT,
                        request_size INTEGER,
                        response_size INTEGER,
                        timestamp TIMESTAMPTZ DEFAULT NOW(),
                        session_id VARCHAR(100),
                        endpoint_normalized VARCHAR(200),
                        rate_limit_remaining INTEGER,
                        cache_hit BOOLEAN DEFAULT FALSE,
                        error_message TEXT,
                        trace_id VARCHAR(100),
                        CONSTRAINT api_call_logs_method_check CHECK (method IN ('GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS')),
                        CONSTRAINT api_call_logs_status_check CHECK (status_code >= 100 AND status_code < 600),
                        CONSTRAINT api_call_logs_duration_check CHECK (duration_ms >= 0)
                    )
                """
                )

                # Create indexes for API call logs
                api_index_queries = [
                    "CREATE INDEX IF NOT EXISTS idx_api_timestamp ON api_call_logs (timestamp DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_api_user_id ON api_call_logs (user_id) WHERE user_id IS NOT NULL",
                    "CREATE INDEX IF NOT EXISTS idx_api_status_code ON api_call_logs (status_code, timestamp DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_api_endpoint ON api_call_logs (endpoint_normalized, timestamp DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_api_method ON api_call_logs (method, timestamp DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_api_errors ON api_call_logs (timestamp DESC) WHERE status_code >= 400",
                    "CREATE INDEX IF NOT EXISTS idx_api_slow ON api_call_logs (duration_ms DESC, timestamp DESC) WHERE duration_ms > 1000",
                    "CREATE INDEX IF NOT EXISTS idx_api_session ON api_call_logs (session_id) WHERE session_id IS NOT NULL"
                ]
                
                for query in api_index_queries:
                    try:
                        await conn.execute(query)
                    except Exception as e:
                        logger.warning("Failed to create API index", query=query, error=str(e))
                
                logger.info("Audit tables and indexes created successfully")
                
        except Exception as e:
            logger.error("Failed to ensure audit tables", error=str(e))
            raise RuntimeError(f"Database table creation failed: {e}") from e

    # Public logging methods
    async def log_event(
        self,
        action: str,
        resource: str,
        user_id: Optional[uuid.UUID] = None,
        api_key_id: Optional[uuid.UUID] = None,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        success: bool = True,
        duration_ms: Optional[int] = None,
    ) -> None:
        """Log an audit event with comprehensive context.
        
        Args:
            action: The action being performed (e.g., 'user_login', 'memory_created')
            resource: The resource being acted upon (e.g., 'authentication', 'memory')
            user_id: Optional user ID performing the action
            api_key_id: Optional API key used for the action
            resource_id: Optional ID of the specific resource
            ip_address: Optional IP address of the requester
            user_agent: Optional user agent string
            details: Optional additional context data
            session_id: Optional session identifier
            success: Whether the action was successful (default: True)
            duration_ms: Optional duration of the action in milliseconds
            
        Raises:
            ValueError: If required parameters are invalid
            QueueFullError: If the event queue is full
        """
        # Input validation
        if not action or not action.strip():
            raise ValueError("Action cannot be empty")
        if not resource or not resource.strip():
            raise ValueError("Resource cannot be empty")
        
        # Sanitize inputs
        action = action.strip()[:100]  # Limit length
        resource = resource.strip()[:100]
        
        event_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        event = {
            "type": "audit",
            "id": event_id,
            "user_id": str(user_id) if user_id else None,
            "api_key_id": str(api_key_id) if api_key_id else None,
            "action": action,
            "resource": resource,
            "resource_id": resource_id[:100] if resource_id else None,  # Limit length
            "ip_address": ip_address,
            "user_agent": user_agent[:500] if user_agent else None,  # Limit length
            "details": self._sanitize_details(details or {}),
            "timestamp": timestamp.isoformat(),
            "session_id": session_id[:100] if session_id else None,
            "success": success,
            "duration_ms": duration_ms,
        }

        # Add to queue for background processing with error handling
        try:
            self.event_queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.error("Audit event queue is full, dropping event", 
                        action=action, resource=resource, event_id=event_id)
            # Try to process some events immediately to make space
            await self._emergency_queue_drain()
            raise RuntimeError("Audit event queue is full") from None

        # Update metrics with proper error handling
        try:
            # Use cached user role or default to avoid blocking
            user_role = await self._get_user_role_cached(user_id) if user_id else "anonymous"
            AUDIT_EVENTS_TOTAL.labels(
                action=action, resource=resource, user_role=user_role
            ).inc()
        except Exception as e:
            logger.warning("Failed to update audit metrics", 
                         error=str(e), action=action, resource=resource)
            # Use fallback metrics without user role lookup
            try:
                AUDIT_EVENTS_TOTAL.labels(
                    action=action, resource=resource, user_role="unknown"
                ).inc()
            except Exception as metrics_error:
                logger.error("Critical: Failed to update any audit metrics", 
                           error=str(metrics_error), original_error=str(e))

        # Structured logging with enhanced context
        log_context = {
            "event_type": "audit_event",
            "event_id": event_id,
            "action": action,
            "resource": resource,
            "user_id": str(user_id) if user_id else None,
            "resource_id": resource_id,
            "session_id": session_id,
            "success": success,
            "ip_address": ip_address,
        }
        
        # Add duration if provided
        if duration_ms is not None:
            log_context["duration_ms"] = duration_ms
        
        # Add sanitized details (avoid logging sensitive data)
        if details:
            log_context["details_keys"] = list(details.keys())
            log_context["details_count"] = len(details)
        
        logger.info("Audit event logged", **log_context)

    async def log_security_event(
        self,
        event_type: str,
        user_id: Optional[uuid.UUID] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: str = SecuritySeverity.MEDIUM,
    ):
        """Log a security event."""
        event = {
            "type": "security",
            "id": str(uuid.uuid4()),
            "event_type": event_type,
            "user_id": str(user_id) if user_id else None,
            "ip_address": ip_address,
            "details": details or {},
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat(),
            "resolved": False,
        }

        # Add to queue for background processing
        await self.event_queue.put(event)

        # Update metrics
        SECURITY_EVENTS_TOTAL.labels(event_type=event_type, severity=severity).inc()

        # Structured logging
        logger.warning(
            "security_event",
            event_type=event_type,
            user_id=str(user_id) if user_id else None,
            ip_address=ip_address,
            severity=severity,
            details=details,
        )

        # Alert on high severity events
        if severity in [SecuritySeverity.HIGH, SecuritySeverity.CRITICAL]:
            await self._send_security_alert(event)

    async def log_api_call(
        self,
        method: str,
        path: str,
        status_code: int,
        user_id: Optional[uuid.UUID] = None,
        api_key_id: Optional[uuid.UUID] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        duration: float = 0.0,
        request_size: int = 0,
        response_size: int = 0,
    ):
        """Log an API call."""
        event = {
            "type": "api_call",
            "id": str(uuid.uuid4()),
            "user_id": str(user_id) if user_id else None,
            "api_key_id": str(api_key_id) if api_key_id else None,
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": int(duration * 1000),
            "ip_address": ip_address,
            "user_agent": user_agent,
            "request_size": request_size,
            "response_size": response_size,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Add to queue for background processing
        await self.event_queue.put(event)

        # Update metrics
        endpoint = self._normalize_endpoint(path)
        API_REQUESTS_TOTAL.labels(
            method=method, endpoint=endpoint, status_code=status_code
        ).inc()
        API_REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)

        # Log failed requests
        if status_code >= 400:
            logger.warning(
                "api_error",
                method=method,
                path=path,
                status_code=status_code,
                user_id=str(user_id) if user_id else None,
                ip_address=ip_address,
            )

    async def log_failed_login(
        self, email: str, ip_address: str, reason: str, user_agent: Optional[str] = None
    ):
        """Log failed login attempt."""
        details = {"email": email, "reason": reason, "user_agent": user_agent}

        await self.log_security_event(
            SecurityEventType.FAILED_LOGIN,
            ip_address=ip_address,
            details=details,
            severity=SecuritySeverity.MEDIUM,
        )

        # Update metrics
        FAILED_LOGINS_TOTAL.labels(reason=reason).inc()

        # Check for brute force patterns
        await self._check_brute_force_pattern(ip_address, email)

    async def log_database_operation(
        self,
        operation: str,
        table: str,
        user_id: Optional[uuid.UUID] = None,
        affected_rows: int = 0,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log database operation."""
        await self.log_event(
            action=f"db_{operation}",
            resource=f"database.{table}",
            user_id=user_id,
            details={"affected_rows": affected_rows, **(details or {})},
        )

        # Update metrics
        DATABASE_OPERATIONS_TOTAL.labels(operation=operation, table=table).inc()

    # Helper methods
    def _sanitize_details(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize details to prevent logging sensitive data.
        
        Args:
            details: Original details dictionary
            
        Returns:
            Sanitized details dictionary
        """
        if not details:
            return {}
        
        # List of sensitive keys to exclude or mask
        sensitive_keys = {
            'password', 'passwd', 'secret', 'token', 'key', 'api_key',
            'authorization', 'auth', 'credential', 'private', 'confidential'
        }
        
        sanitized = {}
        for key, value in details.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, str) and len(value) > 1000:
                sanitized[key] = value[:1000] + "...[TRUNCATED]"
            elif isinstance(value, (dict, list)) and len(str(value)) > 2000:
                sanitized[key] = "[LARGE_OBJECT_REDACTED]"
            else:
                sanitized[key] = value
        
        return sanitized
    
    async def _emergency_queue_drain(self) -> None:
        """Emergency queue draining when queue is full."""
        drained_count = 0
        max_drain = 100  # Drain up to 100 events
        
        while not self.event_queue.empty() and drained_count < max_drain:
            try:
                event = self.event_queue.get_nowait()
                await self._process_single_event(event)
                drained_count += 1
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                logger.error("Error during emergency queue drain", error=str(e))
                break
        
        if drained_count > 0:
            logger.warning("Emergency queue drain completed", drained_count=drained_count)
    
    async def _get_user_role_cached(self, user_id: uuid.UUID) -> str:
        """Get user role with caching to avoid blocking operations.
        
        Args:
            user_id: User ID to look up
            
        Returns:
            User role string or 'unknown' if lookup fails
        """
        # Use a simple in-memory cache with TTL
        if not hasattr(self, '_user_role_cache'):
            self._user_role_cache = {}
            self._user_role_cache_ttl = {}
        
        now = datetime.utcnow().timestamp()
        cache_key = str(user_id)
        
        # Check cache first
        if (cache_key in self._user_role_cache and 
            cache_key in self._user_role_cache_ttl and
            now - self._user_role_cache_ttl[cache_key] < 300):  # 5 minute TTL
            return self._user_role_cache[cache_key]
        
        # Fallback to database lookup with timeout
        try:
            role = await asyncio.wait_for(self._get_user_role(user_id), timeout=1.0)
            self._user_role_cache[cache_key] = role
            self._user_role_cache_ttl[cache_key] = now
            return role
        except (asyncio.TimeoutError, Exception):
            # Clean up cache periodically
            if len(self._user_role_cache) > 1000:
                self._user_role_cache.clear()
                self._user_role_cache_ttl.clear()
            return "unknown"
    
    # Background processing
    async def _process_events(self) -> None:
        """Background task to process audit events with improved error handling."""
        logger.info("Starting audit event processing loop")
        batch_size = 10
        batch_timeout = 5.0
        
        while not self._shutdown_event.is_set():
            try:
                # Process events in batches for better performance
                events_batch = []
                batch_start_time = asyncio.get_event_loop().time()
                
                # Collect a batch of events
                while (len(events_batch) < batch_size and 
                       (asyncio.get_event_loop().time() - batch_start_time) < batch_timeout):
                    try:
                        remaining_timeout = batch_timeout - (asyncio.get_event_loop().time() - batch_start_time)
                        if remaining_timeout <= 0:
                            break
                        
                        event = await asyncio.wait_for(
                            self.event_queue.get(), 
                            timeout=min(remaining_timeout, 1.0)
                        )
                        events_batch.append(event)
                        
                    except asyncio.TimeoutError:
                        break  # Process whatever we have
                
                # Process the batch
                if events_batch:
                    await self._process_events_batch(events_batch)
                
                # Brief pause if no events to process
                if not events_batch:
                    await asyncio.sleep(0.1)
                    
            except asyncio.CancelledError:
                logger.info("Audit event processing cancelled")
                break
            except Exception as e:
                logger.error("Error in audit event processing loop", error=str(e))
                await asyncio.sleep(1)  # Pause before retrying
        
        logger.info("Audit event processing loop ended")
    
    async def _process_events_batch(self, events: List[Dict[str, Any]]) -> None:
        """Process a batch of events efficiently.
        
        Args:
            events: List of events to process
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Group events by type for batch processing
            audit_events = [e for e in events if e.get("type") == "audit"]
            security_events = [e for e in events if e.get("type") == "security"]
            api_events = [e for e in events if e.get("type") == "api_call"]
            
            # Process each type in parallel
            tasks = []
            if audit_events:
                tasks.append(self._store_audit_events_batch(audit_events))
            if security_events:
                tasks.append(self._store_security_events_batch(security_events))
            if api_events:
                tasks.append(self._store_api_events_batch(api_events))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Track processing performance
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            self._event_processing_times.append(processing_time)
            
            # Keep only recent samples
            if len(self._event_processing_times) > self._max_processing_time_samples:
                self._event_processing_times = self._event_processing_times[-self._max_processing_time_samples//2:]
            
            logger.debug("Processed events batch", 
                        batch_size=len(events), 
                        processing_time_ms=processing_time)
                        
        except Exception as e:
            logger.error("Error processing events batch", 
                        error=str(e), batch_size=len(events))
    
    async def _process_single_event(self, event: Dict[str, Any]) -> None:
        """Process a single event (fallback method).
        
        Args:
            event: Event dictionary to process
        """
        try:
            event_type = event.get("type")
            
            if event_type == "audit":
                await self._store_audit_event(event)
            elif event_type == "security":
                await self._store_security_event(event)
            elif event_type == "api_call":
                await self._store_api_call(event)
            else:
                logger.warning("Unknown event type", event_type=event_type, event_id=event.get("id"))
                
        except Exception as e:
            logger.error("Error processing single event", 
                        error=str(e), event_id=event.get("id"), event_type=event.get("type"))

    async def _store_audit_events_batch(self, events: List[Dict[str, Any]]) -> None:
        """Store multiple audit events in a single transaction.
        
        Args:
            events: List of audit events to store
        """
        if not events or not self.db_pool:
            return
            
        try:
            async with self.db_pool.acquire() as conn:
                # Prepare batch insert
                values = []
                for event in events:
                    values.append((
                        uuid.UUID(event["id"]),
                        uuid.UUID(event["user_id"]) if event["user_id"] else None,
                        uuid.UUID(event["api_key_id"]) if event["api_key_id"] else None,
                        event["action"],
                        event["resource"],
                        event["resource_id"],
                        event["ip_address"],
                        event["user_agent"],
                        json.dumps(event["details"]),
                        datetime.fromisoformat(event["timestamp"]),
                        event.get("session_id"),
                        event.get("success", True),
                        event.get("duration_ms"),
                    ))
                
                await conn.executemany(
                    """
                    INSERT INTO audit_logs (
                        id, user_id, api_key_id, action, resource, resource_id,
                        ip_address, user_agent, details, timestamp, session_id, success, duration_ms
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    """,
                    values
                )
                
                logger.debug("Stored audit events batch", count=len(events))
                
        except Exception as e:
            logger.error("Failed to store audit events batch", 
                        error=str(e), batch_size=len(events))
            # Fall back to individual inserts
            for event in events:
                try:
                    await self._store_audit_event(event)
                except Exception as individual_error:
                    logger.error("Failed to store individual audit event", 
                               error=str(individual_error), event_id=event.get("id"))

    async def _store_audit_event(self, event: Dict[str, Any]) -> None:
        """Store single audit event in database.
        
        Args:
            event: Audit event to store
        """
        if not self.db_pool:
            logger.error("Database pool not available")
            return
            
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO audit_logs (
                        id, user_id, api_key_id, action, resource, resource_id,
                        ip_address, user_agent, details, timestamp, session_id, success, duration_ms
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    """,
                    uuid.UUID(event["id"]),
                    uuid.UUID(event["user_id"]) if event["user_id"] else None,
                    uuid.UUID(event["api_key_id"]) if event["api_key_id"] else None,
                    event["action"],
                    event["resource"],
                    event["resource_id"],
                    event["ip_address"],
                    event["user_agent"],
                    json.dumps(event["details"]),
                    datetime.fromisoformat(event["timestamp"]),
                    event.get("session_id"),
                    event.get("success", True),
                    event.get("duration_ms"),
                )
        except Exception as e:
            logger.error("Failed to store audit event", 
                        error=str(e), event_id=event.get("id"))

    async def _store_security_events_batch(self, events: List[Dict[str, Any]]) -> None:
        """Store multiple security events in a single transaction.
        
        Args:
            events: List of security events to store
        """
        if not events or not self.db_pool:
            return
            
        try:
            async with self.db_pool.acquire() as conn:
                values = []
                for event in events:
                    values.append((
                        uuid.UUID(event["id"]),
                        event["event_type"],
                        uuid.UUID(event["user_id"]) if event["user_id"] else None,
                        event["ip_address"],
                        json.dumps(event["details"]),
                        event["severity"],
                        datetime.fromisoformat(event["timestamp"]),
                        event["resolved"],
                        event.get("resolved_at"),
                        event.get("resolved_by"),
                        event.get("alert_sent", False),
                        event.get("alert_sent_at"),
                        event.get("risk_score", 0),
                        event.get("session_id"),
                        event.get("user_agent"),
                    ))
                
                await conn.executemany(
                    """
                    INSERT INTO security_events (
                        id, event_type, user_id, ip_address, details, severity, timestamp, resolved,
                        resolved_at, resolved_by, alert_sent, alert_sent_at, risk_score, session_id, user_agent
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                    """,
                    values
                )
                
                logger.debug("Stored security events batch", count=len(events))
                
        except Exception as e:
            logger.error("Failed to store security events batch", 
                        error=str(e), batch_size=len(events))
            # Fall back to individual inserts
            for event in events:
                try:
                    await self._store_security_event(event)
                except Exception as individual_error:
                    logger.error("Failed to store individual security event", 
                               error=str(individual_error), event_id=event.get("id"))

    async def _store_security_event(self, event: Dict[str, Any]) -> None:
        """Store single security event in database.
        
        Args:
            event: Security event to store
        """
        if not self.db_pool:
            logger.error("Database pool not available")
            return
            
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO security_events (
                        id, event_type, user_id, ip_address, details, severity, timestamp, resolved,
                        resolved_at, resolved_by, alert_sent, alert_sent_at, risk_score, session_id, user_agent
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                    """,
                    uuid.UUID(event["id"]),
                    event["event_type"],
                    uuid.UUID(event["user_id"]) if event["user_id"] else None,
                    event["ip_address"],
                    json.dumps(event["details"]),
                    event["severity"],
                    datetime.fromisoformat(event["timestamp"]),
                    event["resolved"],
                    event.get("resolved_at"),
                    event.get("resolved_by"),
                    event.get("alert_sent", False),
                    event.get("alert_sent_at"),
                    event.get("risk_score", 0),
                    event.get("session_id"),
                    event.get("user_agent"),
                )
        except Exception as e:
            logger.error("Failed to store security event", 
                        error=str(e), event_id=event.get("id"))

    async def _store_api_events_batch(self, events: List[Dict[str, Any]]) -> None:
        """Store multiple API call events in a single transaction.
        
        Args:
            events: List of API call events to store
        """
        if not events or not self.db_pool:
            return
            
        try:
            async with self.db_pool.acquire() as conn:
                values = []
                for event in events:
                    values.append((
                        uuid.UUID(event["id"]),
                        uuid.UUID(event["user_id"]) if event["user_id"] else None,
                        uuid.UUID(event["api_key_id"]) if event["api_key_id"] else None,
                        event["method"],
                        event["path"],
                        event["status_code"],
                        event["duration_ms"],
                        event["ip_address"],
                        event["user_agent"],
                        event["request_size"],
                        event["response_size"],
                        datetime.fromisoformat(event["timestamp"]),
                        event.get("session_id"),
                        event.get("endpoint_normalized"),
                        event.get("rate_limit_remaining"),
                        event.get("cache_hit", False),
                        event.get("error_message"),
                        event.get("trace_id"),
                    ))
                
                await conn.executemany(
                    """
                    INSERT INTO api_call_logs (
                        id, user_id, api_key_id, method, path, status_code, duration_ms,
                        ip_address, user_agent, request_size, response_size, timestamp,
                        session_id, endpoint_normalized, rate_limit_remaining, cache_hit, error_message, trace_id
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
                    """,
                    values
                )
                
                logger.debug("Stored API events batch", count=len(events))
                
        except Exception as e:
            logger.error("Failed to store API events batch", 
                        error=str(e), batch_size=len(events))
            # Fall back to individual inserts
            for event in events:
                try:
                    await self._store_api_call(event)
                except Exception as individual_error:
                    logger.error("Failed to store individual API event", 
                               error=str(individual_error), event_id=event.get("id"))

    async def _store_api_call(self, event: Dict[str, Any]) -> None:
        """Store single API call log in database.
        
        Args:
            event: API call event to store
        """
        if not self.db_pool:
            logger.error("Database pool not available")
            return
            
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO api_call_logs (
                        id, user_id, api_key_id, method, path, status_code, duration_ms,
                        ip_address, user_agent, request_size, response_size, timestamp,
                        session_id, endpoint_normalized, rate_limit_remaining, cache_hit, error_message, trace_id
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
                    """,
                    uuid.UUID(event["id"]),
                    uuid.UUID(event["user_id"]) if event["user_id"] else None,
                    uuid.UUID(event["api_key_id"]) if event["api_key_id"] else None,
                    event["method"],
                    event["path"],
                    event["status_code"],
                    event["duration_ms"],
                    event["ip_address"],
                    event["user_agent"],
                    event["request_size"],
                    event["response_size"],
                    datetime.fromisoformat(event["timestamp"]),
                    event.get("session_id"),
                    event.get("endpoint_normalized"),
                    event.get("rate_limit_remaining"),
                    event.get("cache_hit", False),
                    event.get("error_message"),
                    event.get("trace_id"),
                )
        except Exception as e:
            logger.error("Failed to store API call event", 
                        error=str(e), event_id=event.get("id"))

    # Security analysis methods
    async def _check_brute_force_pattern(self, ip_address: str, email: str):
        """Check for brute force attack patterns."""
        # Check recent failed attempts from this IP
        async with self.db_pool.acquire() as conn:
            # Count failed attempts in last 15 minutes
            recent_attempts = await conn.fetchval(
                """
                SELECT COUNT(*) FROM security_events
                WHERE event_type = 'failed_login'
                AND ip_address = $1
                AND timestamp > NOW() - INTERVAL '15 minutes'
            """,
                ip_address,
            )

            if recent_attempts >= self.failed_login_threshold:
                await self.log_security_event(
                    SecurityEventType.BRUTE_FORCE_ATTEMPT,
                    ip_address=ip_address,
                    details={
                        "failed_attempts": recent_attempts,
                        "target_email": email,
                        "detection_method": "ip_based",
                    },
                    severity=SecuritySeverity.HIGH,
                )

    async def _send_security_alert(self, event: Dict[str, Any]):
        """Send security alert for high severity events."""
        # This would integrate with your alerting system
        # For now, just log the alert
        logger.critical(
            "security_alert",
            event_type=event["event_type"],
            severity=event["severity"],
            details=event["details"],
        )

        # TODO: Implement email/Slack/PagerDuty integration

    async def _get_user_role(self, user_id: uuid.UUID) -> str:
        """Get user role for metrics."""
        try:
            async with self.db_pool.acquire() as conn:
                role = await conn.fetchval(
                    "SELECT role FROM users WHERE id = $1", user_id
                )
                return role or "unknown"
        except Exception:
            return "unknown"

    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint path for metrics."""
        # Replace UUIDs and IDs with placeholders
        import re

        path = re.sub(r"/[0-9a-f-]{36}", "/{id}", path)  # UUIDs
        path = re.sub(r"/\d+", "/{id}", path)  # Numeric IDs
        return path

    # Query methods
    async def get_audit_logs(
        self,
        user_id: Optional[uuid.UUID] = None,
        action: Optional[str] = None,
        resource: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get audit logs with filtering."""
        conditions = []
        params = []
        param_count = 1

        if user_id:
            conditions.append(f"user_id = ${param_count}")
            params.append(user_id)
            param_count += 1

        if action:
            conditions.append(f"action = ${param_count}")
            params.append(action)
            param_count += 1

        if resource:
            conditions.append(f"resource = ${param_count}")
            params.append(resource)
            param_count += 1

        if start_time:
            conditions.append(f"timestamp >= ${param_count}")
            params.append(start_time)
            param_count += 1

        if end_time:
            conditions.append(f"timestamp <= ${param_count}")
            params.append(end_time)
            param_count += 1

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT * FROM audit_logs
                {where_clause}
                ORDER BY timestamp DESC
                LIMIT ${param_count}
            """,
                *params,
                limit,
            )

            return [dict(row) for row in rows]

    async def get_security_events(
        self,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        resolved: Optional[bool] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get security events with filtering."""
        conditions = []
        params = []
        param_count = 1

        if event_type:
            conditions.append(f"event_type = ${param_count}")
            params.append(event_type)
            param_count += 1

        if severity:
            conditions.append(f"severity = ${param_count}")
            params.append(severity)
            param_count += 1

        if resolved is not None:
            conditions.append(f"resolved = ${param_count}")
            params.append(resolved)
            param_count += 1

        if start_time:
            conditions.append(f"timestamp >= ${param_count}")
            params.append(start_time)
            param_count += 1

        if end_time:
            conditions.append(f"timestamp <= ${param_count}")
            params.append(end_time)
            param_count += 1

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT * FROM security_events
                {where_clause}
                ORDER BY timestamp DESC
                LIMIT ${param_count}
            """,
                *params,
                limit,
            )

            return [dict(row) for row in rows]

    async def get_audit_statistics(
        self, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """Get audit statistics for a time period."""
        async with self.db_pool.acquire() as conn:
            # Total events
            total_events = await conn.fetchval(
                """
                SELECT COUNT(*) FROM audit_logs
                WHERE timestamp BETWEEN $1 AND $2
            """,
                start_time,
                end_time,
            )

            # Events by action
            action_stats = await conn.fetch(
                """
                SELECT action, COUNT(*) as count
                FROM audit_logs
                WHERE timestamp BETWEEN $1 AND $2
                GROUP BY action
                ORDER BY count DESC
            """,
                start_time,
                end_time,
            )

            # Security events
            security_stats = await conn.fetch(
                """
                SELECT event_type, severity, COUNT(*) as count
                FROM security_events
                WHERE timestamp BETWEEN $1 AND $2
                GROUP BY event_type, severity
                ORDER BY count DESC
            """,
                start_time,
                end_time,
            )

            return {
                "period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                },
                "total_events": total_events,
                "action_breakdown": [dict(row) for row in action_stats],
                "security_events": [dict(row) for row in security_stats],
            }


# Global audit logger instance
audit_logger = AuditLogger()


# Context manager for request tracing
@asynccontextmanager
async def audit_context(
    action: str, resource: str, user_id: Optional[uuid.UUID] = None
):
    """Context manager for audit logging."""
    start_time = datetime.utcnow()

    try:
        yield
        # Log successful operation
        await audit_logger.log_event(
            action=action,
            resource=resource,
            user_id=user_id,
            details={
                "duration_ms": int(
                    (datetime.utcnow() - start_time).total_seconds() * 1000
                )
            },
        )
    except Exception as e:
        # Log failed operation
        await audit_logger.log_event(
            action=f"{action}_failed",
            resource=resource,
            user_id=user_id,
            details={
                "error": str(e),
                "duration_ms": int(
                    (datetime.utcnow() - start_time).total_seconds() * 1000
                ),
            },
        )
        raise


# Initialization function
async def initialize_audit_logger():
    """Initialize the audit logger."""
    await audit_logger.initialize()
    return audit_logger
