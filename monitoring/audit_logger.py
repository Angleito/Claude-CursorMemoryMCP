"""
Comprehensive audit logging and monitoring system
"""
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum
import json
import asyncio
import uuid
import structlog
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import asyncpg
from contextlib import asynccontextmanager

from config.settings import get_settings
from auth.models import AuditLog, SecurityEvent, User


settings = get_settings()

# Prometheus metrics
AUDIT_EVENTS_TOTAL = Counter('audit_events_total', 'Total audit events', ['action', 'resource', 'user_role'])
SECURITY_EVENTS_TOTAL = Counter('security_events_total', 'Total security events', ['event_type', 'severity'])
API_REQUESTS_TOTAL = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status_code'])
API_REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration', ['method', 'endpoint'])
FAILED_LOGINS_TOTAL = Counter('failed_logins_total', 'Total failed login attempts', ['reason'])
ACTIVE_SESSIONS = Gauge('active_sessions_total', 'Number of active user sessions')
DATABASE_OPERATIONS_TOTAL = Counter('database_operations_total', 'Total database operations', ['operation', 'table'])

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
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class AuditEventType(str, Enum):
    """Types of audit events"""
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
    """Types of security events"""
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
    """Security event severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditLogger:
    """Main audit logging class"""
    
    def __init__(self):
        self.settings = settings
        self.db_pool: Optional[asyncpg.Pool] = None
        self.event_queue = asyncio.Queue()
        self.background_task: Optional[asyncio.Task] = None
        
        # Security thresholds
        self.failed_login_threshold = 5
        self.suspicious_activity_threshold = 10
        self.rate_limit_threshold = 100
        
    async def initialize(self):
        """Initialize audit logger"""
        # Create database connection pool
        self.db_pool = await asyncpg.create_pool(
            self.settings.database.database_url,
            min_size=2,
            max_size=10
        )
        
        # Start background processing task
        self.background_task = asyncio.create_task(self._process_events())
        
        # Start Prometheus metrics server
        if self.settings.monitoring.prometheus_port:
            start_http_server(self.settings.monitoring.prometheus_port)
        
        # Initialize database tables
        await self._ensure_audit_tables()
    
    async def close(self):
        """Close audit logger"""
        if self.background_task:
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
        
        if self.db_pool:
            await self.db_pool.close()
    
    async def _ensure_audit_tables(self):
        """Ensure audit tables exist"""
        async with self.db_pool.acquire() as conn:
            # Audit logs table
            await conn.execute("""
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
                    timestamp TIMESTAMPTZ DEFAULT NOW()
                )
            """);
            
            # Create indexes separately
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs (timestamp)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_user_id ON audit_logs (user_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs (action)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_logs (resource)")
            
            # Security events table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS security_events (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    event_type VARCHAR(50) NOT NULL,
                    user_id UUID,
                    ip_address INET,
                    details JSONB DEFAULT '{}',
                    severity VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    resolved BOOLEAN DEFAULT FALSE
                )
            """);
            
            # Create indexes separately
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_security_timestamp ON security_events (timestamp)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_security_event_type ON security_events (event_type)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_security_severity ON security_events (severity)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_security_resolved ON security_events (resolved)")
            
            # API call logs table
            await conn.execute("""
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
                    timestamp TIMESTAMPTZ DEFAULT NOW()
                )
            """);
            
            # Create indexes separately
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_api_timestamp ON api_call_logs (timestamp)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_api_user_id ON api_call_logs (user_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_api_status_code ON api_call_logs (status_code)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_api_path ON api_call_logs (path)")
    
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
        details: Dict[str, Any] = None
    ):
        """Log an audit event"""
        event = {
            'type': 'audit',
            'id': str(uuid.uuid4()),
            'user_id': str(user_id) if user_id else None,
            'api_key_id': str(api_key_id) if api_key_id else None,
            'action': action,
            'resource': resource,
            'resource_id': resource_id,
            'ip_address': ip_address,
            'user_agent': user_agent,
            'details': details or {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Add to queue for background processing
        await self.event_queue.put(event)
        
        # Update metrics (make async call safe)
        try:
            user_role = await self._get_user_role(user_id) if user_id else 'anonymous'
            AUDIT_EVENTS_TOTAL.labels(action=action, resource=resource, user_role=user_role).inc()
        except Exception as e:
            logger.warning("Failed to update audit metrics", error=str(e))
            # Use fallback without user role lookup
            AUDIT_EVENTS_TOTAL.labels(action=action, resource=resource, user_role='unknown').inc()
        
        # Structured logging
        logger.info(
            "audit_event",
            action=action,
            resource=resource,
            user_id=str(user_id) if user_id else None,
            resource_id=resource_id,
            details=details
        )
    
    async def log_security_event(
        self,
        event_type: str,
        user_id: Optional[uuid.UUID] = None,
        ip_address: Optional[str] = None,
        details: Dict[str, Any] = None,
        severity: str = SecuritySeverity.MEDIUM
    ):
        """Log a security event"""
        event = {
            'type': 'security',
            'id': str(uuid.uuid4()),
            'event_type': event_type,
            'user_id': str(user_id) if user_id else None,
            'ip_address': ip_address,
            'details': details or {},
            'severity': severity,
            'timestamp': datetime.utcnow().isoformat(),
            'resolved': False
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
            details=details
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
        response_size: int = 0
    ):
        """Log an API call"""
        event = {
            'type': 'api_call',
            'id': str(uuid.uuid4()),
            'user_id': str(user_id) if user_id else None,
            'api_key_id': str(api_key_id) if api_key_id else None,
            'method': method,
            'path': path,
            'status_code': status_code,
            'duration_ms': int(duration * 1000),
            'ip_address': ip_address,
            'user_agent': user_agent,
            'request_size': request_size,
            'response_size': response_size,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Add to queue for background processing
        await self.event_queue.put(event)
        
        # Update metrics
        endpoint = self._normalize_endpoint(path)
        API_REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        API_REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
        
        # Log failed requests
        if status_code >= 400:
            logger.warning(
                "api_error",
                method=method,
                path=path,
                status_code=status_code,
                user_id=str(user_id) if user_id else None,
                ip_address=ip_address
            )
    
    async def log_failed_login(
        self,
        email: str,
        ip_address: str,
        reason: str,
        user_agent: str = None
    ):
        """Log failed login attempt"""
        details = {
            'email': email,
            'reason': reason,
            'user_agent': user_agent
        }
        
        await self.log_security_event(
            SecurityEventType.FAILED_LOGIN,
            ip_address=ip_address,
            details=details,
            severity=SecuritySeverity.MEDIUM
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
        details: Dict[str, Any] = None
    ):
        """Log database operation"""
        await self.log_event(
            action=f"db_{operation}",
            resource=f"database.{table}",
            user_id=user_id,
            details={
                'affected_rows': affected_rows,
                **(details or {})
            }
        )
        
        # Update metrics
        DATABASE_OPERATIONS_TOTAL.labels(operation=operation, table=table).inc()
    
    # Background processing
    async def _process_events(self):
        """Background task to process audit events"""
        while True:
            try:
                # Wait for events with timeout
                try:
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue
                
                # Process event based on type
                if event['type'] == 'audit':
                    await self._store_audit_event(event)
                elif event['type'] == 'security':
                    await self._store_security_event(event)
                elif event['type'] == 'api_call':
                    await self._store_api_call(event)
                
                # Mark task as done
                self.event_queue.task_done()
                
            except Exception as e:
                logger.error("Error processing audit event", error=str(e))
                await asyncio.sleep(1)
    
    async def _store_audit_event(self, event: Dict[str, Any]):
        """Store audit event in database"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO audit_logs (
                    id, user_id, api_key_id, action, resource, resource_id,
                    ip_address, user_agent, details, timestamp
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, 
                uuid.UUID(event['id']),
                uuid.UUID(event['user_id']) if event['user_id'] else None,
                uuid.UUID(event['api_key_id']) if event['api_key_id'] else None,
                event['action'],
                event['resource'],
                event['resource_id'],
                event['ip_address'],
                event['user_agent'],
                json.dumps(event['details']),
                datetime.fromisoformat(event['timestamp'])
            )
    
    async def _store_security_event(self, event: Dict[str, Any]):
        """Store security event in database"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO security_events (
                    id, event_type, user_id, ip_address, details, severity, timestamp, resolved
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
                uuid.UUID(event['id']),
                event['event_type'],
                uuid.UUID(event['user_id']) if event['user_id'] else None,
                event['ip_address'],
                json.dumps(event['details']),
                event['severity'],
                datetime.fromisoformat(event['timestamp']),
                event['resolved']
            )
    
    async def _store_api_call(self, event: Dict[str, Any]):
        """Store API call log in database"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO api_call_logs (
                    id, user_id, api_key_id, method, path, status_code, duration_ms,
                    ip_address, user_agent, request_size, response_size, timestamp
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """,
                uuid.UUID(event['id']),
                uuid.UUID(event['user_id']) if event['user_id'] else None,
                uuid.UUID(event['api_key_id']) if event['api_key_id'] else None,
                event['method'],
                event['path'],
                event['status_code'],
                event['duration_ms'],
                event['ip_address'],
                event['user_agent'],
                event['request_size'],
                event['response_size'],
                datetime.fromisoformat(event['timestamp'])
            )
    
    # Security analysis methods
    async def _check_brute_force_pattern(self, ip_address: str, email: str):
        """Check for brute force attack patterns"""
        # Check recent failed attempts from this IP
        async with self.db_pool.acquire() as conn:
            # Count failed attempts in last 15 minutes
            recent_attempts = await conn.fetchval("""
                SELECT COUNT(*) FROM security_events 
                WHERE event_type = 'failed_login' 
                AND ip_address = $1 
                AND timestamp > NOW() - INTERVAL '15 minutes'
            """, ip_address)
            
            if recent_attempts >= self.failed_login_threshold:
                await self.log_security_event(
                    SecurityEventType.BRUTE_FORCE_ATTEMPT,
                    ip_address=ip_address,
                    details={
                        'failed_attempts': recent_attempts,
                        'target_email': email,
                        'detection_method': 'ip_based'
                    },
                    severity=SecuritySeverity.HIGH
                )
    
    async def _send_security_alert(self, event: Dict[str, Any]):
        """Send security alert for high severity events"""
        # This would integrate with your alerting system
        # For now, just log the alert
        logger.critical(
            "security_alert",
            event_type=event['event_type'],
            severity=event['severity'],
            details=event['details']
        )
        
        # TODO: Implement email/Slack/PagerDuty integration
    
    async def _get_user_role(self, user_id: uuid.UUID) -> str:
        """Get user role for metrics"""
        try:
            async with self.db_pool.acquire() as conn:
                role = await conn.fetchval(
                    "SELECT role FROM users WHERE id = $1", user_id
                )
                return role or 'unknown'
        except Exception:
            return 'unknown'
    
    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint path for metrics"""
        # Replace UUIDs and IDs with placeholders
        import re
        path = re.sub(r'/[0-9a-f-]{36}', '/{id}', path)  # UUIDs
        path = re.sub(r'/\d+', '/{id}', path)  # Numeric IDs
        return path
    
    # Query methods
    async def get_audit_logs(
        self,
        user_id: Optional[uuid.UUID] = None,
        action: Optional[str] = None,
        resource: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get audit logs with filtering"""
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
            rows = await conn.fetch(f"""
                SELECT * FROM audit_logs
                {where_clause}
                ORDER BY timestamp DESC
                LIMIT ${param_count}
            """, *params, limit)
            
            return [dict(row) for row in rows]
    
    async def get_security_events(
        self,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        resolved: Optional[bool] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get security events with filtering"""
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
            rows = await conn.fetch(f"""
                SELECT * FROM security_events
                {where_clause}
                ORDER BY timestamp DESC
                LIMIT ${param_count}
            """, *params, limit)
            
            return [dict(row) for row in rows]
    
    async def get_audit_statistics(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> Dict[str, Any]:
        """Get audit statistics for a time period"""
        async with self.db_pool.acquire() as conn:
            # Total events
            total_events = await conn.fetchval("""
                SELECT COUNT(*) FROM audit_logs 
                WHERE timestamp BETWEEN $1 AND $2
            """, start_time, end_time)
            
            # Events by action
            action_stats = await conn.fetch("""
                SELECT action, COUNT(*) as count 
                FROM audit_logs 
                WHERE timestamp BETWEEN $1 AND $2
                GROUP BY action 
                ORDER BY count DESC
            """, start_time, end_time)
            
            # Security events
            security_stats = await conn.fetch("""
                SELECT event_type, severity, COUNT(*) as count 
                FROM security_events 
                WHERE timestamp BETWEEN $1 AND $2
                GROUP BY event_type, severity 
                ORDER BY count DESC
            """, start_time, end_time)
            
            return {
                'period': {'start': start_time.isoformat(), 'end': end_time.isoformat()},
                'total_events': total_events,
                'action_breakdown': [dict(row) for row in action_stats],
                'security_events': [dict(row) for row in security_stats]
            }


# Global audit logger instance
audit_logger = AuditLogger()


# Context manager for request tracing
@asynccontextmanager
async def audit_context(action: str, resource: str, user_id: Optional[uuid.UUID] = None):
    """Context manager for audit logging"""
    start_time = datetime.utcnow()
    
    try:
        yield
        # Log successful operation
        await audit_logger.log_event(
            action=action,
            resource=resource,
            user_id=user_id,
            details={'duration_ms': int((datetime.utcnow() - start_time).total_seconds() * 1000)}
        )
    except Exception as e:
        # Log failed operation
        await audit_logger.log_event(
            action=f"{action}_failed",
            resource=resource,
            user_id=user_id,
            details={
                'error': str(e),
                'duration_ms': int((datetime.utcnow() - start_time).total_seconds() * 1000)
            }
        )
        raise


# Initialization function
async def initialize_audit_logger():
    """Initialize the audit logger"""
    await audit_logger.initialize()
    return audit_logger