"""Security monitoring, alerting, and incident response system."""

import asyncio
import json
import logging
import smtplib
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

import aiohttp
import psutil
from pydantic import BaseModel, Field

from config.settings import get_settings
from monitoring.audit_logger import audit_logger

settings = get_settings()
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(Enum):
    """Incident status levels."""
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    CLOSED = "closed"


@dataclass
class SecurityAlert:
    """Represents a security alert."""
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    source: str
    event_data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved: bool = False
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class SecurityMetric:
    """Represents a security metric."""
    metric_name: str
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    threshold_breached: bool = False


@dataclass
class Incident:
    """Represents a security incident."""
    incident_id: str
    title: str
    description: str
    severity: AlertSeverity
    status: IncidentStatus
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    assigned_to: Optional[str] = None
    alerts: List[str] = field(default_factory=list)  # Alert IDs
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    affected_systems: List[str] = field(default_factory=list)
    root_cause: Optional[str] = None
    resolution: Optional[str] = None


class AlertRule:
    """Defines conditions for triggering alerts."""
    
    def __init__(self, rule_id: str, name: str, condition: Callable[[Dict[str, Any]], bool],
                 severity: AlertSeverity, description: str = "", enabled: bool = True):
        self.rule_id = rule_id
        self.name = name
        self.condition = condition
        self.severity = severity
        self.description = description
        self.enabled = enabled
        self.created_at = datetime.utcnow()
        self.triggered_count = 0
        self.last_triggered = None


class SecurityMetricsCollector:
    """Collects security-related metrics from various sources."""
    
    def __init__(self):
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.thresholds = self._load_default_thresholds()
    
    def _load_default_thresholds(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """Load default metric thresholds."""
        return {
            "failed_login_attempts": {"warning": 10, "critical": 50},
            "blocked_ips": {"warning": 100, "critical": 500},
            "suspicious_requests": {"warning": 20, "critical": 100},
            "cpu_usage": {"warning": 80, "critical": 95},
            "memory_usage": {"warning": 85, "critical": 95},
            "disk_usage": {"warning": 80, "critical": 90},
            "active_connections": {"warning": 1000, "critical": 5000},
            "error_rate": {"warning": 5, "critical": 15},
        }
    
    async def collect_system_metrics(self) -> List[SecurityMetric]:
        """Collect system-level security metrics."""
        metrics = []
        timestamp = datetime.utcnow()
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics.append(SecurityMetric(
            metric_name="cpu_usage",
            value=cpu_percent,
            timestamp=timestamp,
            threshold_breached=cpu_percent > self.thresholds["cpu_usage"]["critical"]
        ))
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        metrics.append(SecurityMetric(
            metric_name="memory_usage",
            value=memory_percent,
            timestamp=timestamp,
            threshold_breached=memory_percent > self.thresholds["memory_usage"]["critical"]
        ))
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        metrics.append(SecurityMetric(
            metric_name="disk_usage",
            value=disk_percent,
            timestamp=timestamp,
            threshold_breached=disk_percent > self.thresholds["disk_usage"]["critical"]
        ))
        
        # Network connections
        connections = len(psutil.net_connections())
        metrics.append(SecurityMetric(
            metric_name="active_connections",
            value=connections,
            timestamp=timestamp,
            threshold_breached=connections > self.thresholds["active_connections"]["critical"]
        ))
        
        # Store metrics history
        for metric in metrics:
            self.metrics_history[metric.metric_name].append(metric)
        
        return metrics
    
    async def collect_security_metrics(self) -> List[SecurityMetric]:
        """Collect application-specific security metrics."""
        metrics = []
        timestamp = datetime.utcnow()
        
        # These would be collected from your application state
        # For now, we'll simulate with placeholder values
        
        # Failed login attempts (last hour)
        failed_logins = await self._count_failed_logins()
        metrics.append(SecurityMetric(
            metric_name="failed_login_attempts",
            value=failed_logins,
            timestamp=timestamp,
            threshold_breached=failed_logins > self.thresholds["failed_login_attempts"]["critical"]
        ))
        
        # Blocked IPs
        blocked_ips = await self._count_blocked_ips()
        metrics.append(SecurityMetric(
            metric_name="blocked_ips",
            value=blocked_ips,
            timestamp=timestamp,
            threshold_breached=blocked_ips > self.thresholds["blocked_ips"]["critical"]
        ))
        
        # Suspicious requests
        suspicious_requests = await self._count_suspicious_requests()
        metrics.append(SecurityMetric(
            metric_name="suspicious_requests",
            value=suspicious_requests,
            timestamp=timestamp,
            threshold_breached=suspicious_requests > self.thresholds["suspicious_requests"]["critical"]
        ))
        
        # Store metrics history
        for metric in metrics:
            self.metrics_history[metric.metric_name].append(metric)
        
        return metrics
    
    async def _count_failed_logins(self) -> int:
        """Count failed login attempts in the last hour."""
        # This would query your authentication system
        # Placeholder implementation
        return 5
    
    async def _count_blocked_ips(self) -> int:
        """Count currently blocked IP addresses."""
        # This would query your firewall/IDS system
        # Placeholder implementation
        return 25
    
    async def _count_suspicious_requests(self) -> int:
        """Count suspicious requests in the last hour."""
        # This would query your request analysis system
        # Placeholder implementation
        return 8
    
    def get_metric_trend(self, metric_name: str, window_minutes: int = 60) -> Dict[str, Any]:
        """Get trend analysis for a metric."""
        if metric_name not in self.metrics_history:
            return {"error": "Metric not found"}
        
        # Get metrics within time window
        cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
        recent_metrics = [
            m for m in self.metrics_history[metric_name]
            if m.timestamp >= cutoff_time
        ]
        
        if len(recent_metrics) < 2:
            return {"trend": "insufficient_data"}
        
        # Calculate trend
        values = [m.value for m in recent_metrics]
        avg_value = sum(values) / len(values)
        latest_value = values[-1]
        
        # Simple trend calculation
        if latest_value > avg_value * 1.2:
            trend = "increasing"
        elif latest_value < avg_value * 0.8:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            "metric_name": metric_name,
            "trend": trend,
            "current_value": latest_value,
            "average_value": avg_value,
            "sample_count": len(recent_metrics),
            "time_window_minutes": window_minutes
        }


class AlertManager:
    """Manages security alerts and notifications."""
    
    def __init__(self):
        self.alerts: Dict[str, SecurityAlert] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.notification_channels: List[Dict[str, Any]] = []
        self.alert_history: deque = deque(maxlen=10000)
        
        # Load default alert rules
        self._load_default_alert_rules()
        
        # Load notification channels
        self._load_notification_channels()
    
    def _load_default_alert_rules(self) -> None:
        """Load default alert rules."""
        default_rules = [
            AlertRule(
                rule_id="high_cpu_usage",
                name="High CPU Usage",
                condition=lambda data: data.get("cpu_usage", 0) > 90,
                severity=AlertSeverity.HIGH,
                description="CPU usage exceeded 90%"
            ),
            AlertRule(
                rule_id="multiple_failed_logins",
                name="Multiple Failed Login Attempts",
                condition=lambda data: data.get("failed_login_attempts", 0) > 20,
                severity=AlertSeverity.MEDIUM,
                description="High number of failed login attempts detected"
            ),
            AlertRule(
                rule_id="ddos_attack",
                name="Potential DDoS Attack",
                condition=lambda data: data.get("requests_per_minute", 0) > 1000,
                severity=AlertSeverity.CRITICAL,
                description="Abnormally high request rate detected"
            ),
            AlertRule(
                rule_id="malicious_ip_activity",
                name="Malicious IP Activity",
                condition=lambda data: data.get("malicious_ip_requests", 0) > 0,
                severity=AlertSeverity.HIGH,
                description="Requests from known malicious IP addresses"
            ),
            AlertRule(
                rule_id="disk_space_low",
                name="Low Disk Space",
                condition=lambda data: data.get("disk_usage", 0) > 85,
                severity=AlertSeverity.MEDIUM,
                description="Disk usage exceeded 85%"
            ),
            AlertRule(
                rule_id="memory_usage_high",
                name="High Memory Usage",
                condition=lambda data: data.get("memory_usage", 0) > 90,
                severity=AlertSeverity.HIGH,
                description="Memory usage exceeded 90%"
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule
    
    def _load_notification_channels(self) -> None:
        """Load notification channels from configuration."""
        # Email notifications
        if settings.monitoring.email_notifications_enabled:
            self.notification_channels.append({
                "type": "email",
                "config": {
                    "smtp_server": settings.monitoring.smtp_server,
                    "smtp_port": settings.monitoring.smtp_port,
                    "username": settings.monitoring.smtp_username,
                    "password": settings.monitoring.smtp_password,
                    "recipients": settings.monitoring.alert_recipients
                }
            })
        
        # Slack notifications
        if hasattr(settings.monitoring, 'slack_webhook_url') and settings.monitoring.slack_webhook_url:
            self.notification_channels.append({
                "type": "slack",
                "config": {
                    "webhook_url": settings.monitoring.slack_webhook_url
                }
            })
        
        # PagerDuty notifications
        if hasattr(settings.monitoring, 'pagerduty_routing_key') and settings.monitoring.pagerduty_routing_key:
            self.notification_channels.append({
                "type": "pagerduty",
                "config": {
                    "routing_key": settings.monitoring.pagerduty_routing_key
                }
            })
    
    async def evaluate_rules(self, event_data: Dict[str, Any]) -> List[SecurityAlert]:
        """Evaluate alert rules against event data."""
        triggered_alerts = []
        
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            try:
                if rule.condition(event_data):
                    alert = await self._create_alert(rule, event_data)
                    triggered_alerts.append(alert)
                    
                    rule.triggered_count += 1
                    rule.last_triggered = datetime.utcnow()
                    
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.rule_id}: {e}")
        
        return triggered_alerts
    
    async def _create_alert(self, rule: AlertRule, event_data: Dict[str, Any]) -> SecurityAlert:
        """Create a new security alert."""
        alert_id = f"{rule.rule_id}_{int(time.time())}"
        
        alert = SecurityAlert(
            alert_id=alert_id,
            title=rule.name,
            description=rule.description,
            severity=rule.severity,
            source=rule.rule_id,
            event_data=event_data,
            tags=["automated", "rule-based"]
        )
        
        self.alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        await self._send_notifications(alert)
        
        # Log alert creation
        await audit_logger.log_security_event(
            event_type="security_alert_created",
            details={
                "alert_id": alert_id,
                "rule_id": rule.rule_id,
                "severity": rule.severity.value,
                "title": rule.name
            },
            severity=rule.severity.value
        )
        
        return alert
    
    async def _send_notifications(self, alert: SecurityAlert) -> None:
        """Send alert notifications through configured channels."""
        for channel in self.notification_channels:
            try:
                if channel["type"] == "email":
                    await self._send_email_notification(alert, channel["config"])
                elif channel["type"] == "slack":
                    await self._send_slack_notification(alert, channel["config"])
                elif channel["type"] == "pagerduty":
                    await self._send_pagerduty_notification(alert, channel["config"])
                    
            except Exception as e:
                logger.error(f"Failed to send {channel['type']} notification: {e}")
    
    async def _send_email_notification(self, alert: SecurityAlert, config: Dict[str, Any]) -> None:
        """Send email notification for alert."""
        subject = f"[{alert.severity.value.upper()}] Security Alert: {alert.title}"
        
        body = f"""
Security Alert Details:

Alert ID: {alert.alert_id}
Title: {alert.title}
Severity: {alert.severity.value.upper()}
Description: {alert.description}
Timestamp: {alert.timestamp.isoformat()}
Source: {alert.source}

Event Data:
{json.dumps(alert.event_data, indent=2)}

Please investigate this alert promptly.
        """
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = config["username"]
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        # Send to all recipients
        for recipient in config["recipients"]:
            msg['To'] = recipient
            
            try:
                server = smtplib.SMTP(config["smtp_server"], config["smtp_port"])
                server.starttls()
                server.login(config["username"], config["password"])
                server.sendmail(config["username"], recipient, msg.as_string())
                server.quit()
                
            except Exception as e:
                logger.error(f"Failed to send email to {recipient}: {e}")
    
    async def _send_slack_notification(self, alert: SecurityAlert, config: Dict[str, Any]) -> None:
        """Send Slack notification for alert."""
        # Color coding based on severity
        color_map = {
            AlertSeverity.LOW: "good",
            AlertSeverity.MEDIUM: "warning", 
            AlertSeverity.HIGH: "danger",
            AlertSeverity.CRITICAL: "#ff0000"
        }
        
        payload = {
            "text": f"Security Alert: {alert.title}",
            "attachments": [
                {
                    "color": color_map.get(alert.severity, "warning"),
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert.severity.value.upper(),
                            "short": True
                        },
                        {
                            "title": "Alert ID",
                            "value": alert.alert_id,
                            "short": True
                        },
                        {
                            "title": "Description",
                            "value": alert.description,
                            "short": False
                        },
                        {
                            "title": "Timestamp",
                            "value": alert.timestamp.isoformat(),
                            "short": True
                        }
                    ]
                }
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            await session.post(config["webhook_url"], json=payload)
    
    async def _send_pagerduty_notification(self, alert: SecurityAlert, config: Dict[str, Any]) -> None:
        """Send PagerDuty notification for alert."""
        # Only send critical and high severity alerts to PagerDuty
        if alert.severity not in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
            return
        
        payload = {
            "routing_key": config["routing_key"],
            "event_action": "trigger",
            "dedup_key": alert.alert_id,
            "payload": {
                "summary": f"{alert.title} - {alert.description}",
                "severity": alert.severity.value,
                "source": alert.source,
                "timestamp": alert.timestamp.isoformat(),
                "custom_details": alert.event_data
            }
        }
        
        async with aiohttp.ClientSession() as session:
            await session.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.utcnow()
            
            await audit_logger.log_security_event(
                event_type="alert_acknowledged",
                details={
                    "alert_id": alert_id,
                    "acknowledged_by": acknowledged_by
                },
                severity="info"
            )
            
            return True
        
        return False
    
    async def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """Resolve an alert."""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.resolved = True
            alert.resolved_by = resolved_by
            alert.resolved_at = datetime.utcnow()
            
            await audit_logger.log_security_event(
                event_type="alert_resolved",
                details={
                    "alert_id": alert_id,
                    "resolved_by": resolved_by
                },
                severity="info"
            )
            
            return True
        
        return False
    
    def get_active_alerts(self, severity_filter: Optional[AlertSeverity] = None) -> List[SecurityAlert]:
        """Get currently active (unresolved) alerts."""
        active_alerts = [
            alert for alert in self.alerts.values()
            if not alert.resolved
        ]
        
        if severity_filter:
            active_alerts = [
                alert for alert in active_alerts
                if alert.severity == severity_filter
            ]
        
        return sorted(active_alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get alert statistics for the specified period."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Filter alerts from the specified period
        recent_alerts = [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff_date
        ]
        
        # Calculate statistics
        total_alerts = len(recent_alerts)
        
        severity_counts = defaultdict(int)
        for alert in recent_alerts:
            severity_counts[alert.severity.value] += 1
        
        resolved_count = sum(1 for alert in recent_alerts if alert.resolved)
        acknowledged_count = sum(1 for alert in recent_alerts if alert.acknowledged)
        
        return {
            "period_days": days,
            "total_alerts": total_alerts,
            "resolved_alerts": resolved_count,
            "acknowledged_alerts": acknowledged_count,
            "resolution_rate": (resolved_count / total_alerts * 100) if total_alerts > 0 else 0,
            "severity_breakdown": dict(severity_counts),
            "active_alerts": len(self.get_active_alerts())
        }


class IncidentManager:
    """Manages security incidents and response coordination."""
    
    def __init__(self):
        self.incidents: Dict[str, Incident] = {}
        self.incident_history: deque = deque(maxlen=1000)
    
    async def create_incident(self, title: str, description: str, 
                            severity: AlertSeverity, alerts: List[str] = None) -> str:
        """Create a new security incident."""
        incident_id = f"INC-{int(time.time())}"
        
        incident = Incident(
            incident_id=incident_id,
            title=title,
            description=description,
            severity=severity,
            status=IncidentStatus.OPEN,
            alerts=alerts or []
        )
        
        # Add creation event to timeline
        incident.timeline.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "incident_created",
            "description": f"Incident created: {title}",
            "user": "system"
        })
        
        self.incidents[incident_id] = incident
        self.incident_history.append(incident)
        
        # Log incident creation
        await audit_logger.log_security_event(
            event_type="incident_created",
            details={
                "incident_id": incident_id,
                "title": title,
                "severity": severity.value,
                "alert_count": len(alerts) if alerts else 0
            },
            severity=severity.value
        )
        
        return incident_id
    
    async def update_incident_status(self, incident_id: str, status: IncidentStatus, 
                                   user: str, notes: str = "") -> bool:
        """Update incident status."""
        if incident_id not in self.incidents:
            return False
        
        incident = self.incidents[incident_id]
        old_status = incident.status
        incident.status = status
        incident.updated_at = datetime.utcnow()
        
        # Add to timeline
        incident.timeline.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "status_changed",
            "description": f"Status changed from {old_status.value} to {status.value}",
            "user": user,
            "notes": notes
        })
        
        await audit_logger.log_security_event(
            event_type="incident_status_updated",
            details={
                "incident_id": incident_id,
                "old_status": old_status.value,
                "new_status": status.value,
                "user": user
            },
            severity="info"
        )
        
        return True
    
    async def assign_incident(self, incident_id: str, assigned_to: str, 
                            assigned_by: str) -> bool:
        """Assign incident to a user."""
        if incident_id not in self.incidents:
            return False
        
        incident = self.incidents[incident_id]
        old_assignee = incident.assigned_to
        incident.assigned_to = assigned_to
        incident.updated_at = datetime.utcnow()
        
        # Add to timeline
        incident.timeline.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "incident_assigned",
            "description": f"Assigned to {assigned_to}",
            "user": assigned_by,
            "previous_assignee": old_assignee
        })
        
        return True
    
    async def add_incident_note(self, incident_id: str, note: str, user: str) -> bool:
        """Add a note to incident timeline."""
        if incident_id not in self.incidents:
            return False
        
        incident = self.incidents[incident_id]
        incident.updated_at = datetime.utcnow()
        
        # Add to timeline
        incident.timeline.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "note_added",
            "description": note,
            "user": user
        })
        
        return True
    
    async def resolve_incident(self, incident_id: str, resolution: str, 
                             root_cause: str, resolved_by: str) -> bool:
        """Resolve an incident."""
        if incident_id not in self.incidents:
            return False
        
        incident = self.incidents[incident_id]
        incident.status = IncidentStatus.RESOLVED
        incident.resolution = resolution
        incident.root_cause = root_cause
        incident.updated_at = datetime.utcnow()
        
        # Add to timeline
        incident.timeline.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "incident_resolved",
            "description": f"Incident resolved: {resolution}",
            "user": resolved_by,
            "root_cause": root_cause
        })
        
        await audit_logger.log_security_event(
            event_type="incident_resolved",
            details={
                "incident_id": incident_id,
                "resolution": resolution,
                "root_cause": root_cause,
                "resolved_by": resolved_by
            },
            severity="info"
        )
        
        return True
    
    def get_open_incidents(self) -> List[Incident]:
        """Get all open incidents."""
        return [
            incident for incident in self.incidents.values()
            if incident.status in [IncidentStatus.OPEN, IncidentStatus.ACKNOWLEDGED, IncidentStatus.INVESTIGATING]
        ]
    
    def get_incident_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get incident statistics for the specified period."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        recent_incidents = [
            incident for incident in self.incident_history
            if incident.created_at >= cutoff_date
        ]
        
        total_incidents = len(recent_incidents)
        
        # Calculate statistics
        status_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for incident in recent_incidents:
            status_counts[incident.status.value] += 1
            severity_counts[incident.severity.value] += 1
        
        resolved_incidents = [
            incident for incident in recent_incidents
            if incident.status == IncidentStatus.RESOLVED
        ]
        
        # Calculate mean time to resolution
        if resolved_incidents:
            resolution_times = [
                (incident.updated_at - incident.created_at).total_seconds() / 3600
                for incident in resolved_incidents
            ]
            mttr_hours = sum(resolution_times) / len(resolution_times)
        else:
            mttr_hours = 0
        
        return {
            "period_days": days,
            "total_incidents": total_incidents,
            "open_incidents": len(self.get_open_incidents()),
            "resolved_incidents": len(resolved_incidents),
            "mean_time_to_resolution_hours": mttr_hours,
            "status_breakdown": dict(status_counts),
            "severity_breakdown": dict(severity_counts)
        }


class SecurityMonitoringManager:
    """Main security monitoring manager coordinating all components."""
    
    def __init__(self):
        self.metrics_collector = SecurityMetricsCollector()
        self.alert_manager = AlertManager()
        self.incident_manager = IncidentManager()
        
        # Start background monitoring tasks
        self.monitoring_task = None
        self.start_monitoring()
    
    def start_monitoring(self) -> None:
        """Start background monitoring tasks."""
        if self.monitoring_task is None or self.monitoring_task.done():
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring tasks."""
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while True:
            try:
                # Collect metrics
                system_metrics = await self.metrics_collector.collect_system_metrics()
                security_metrics = await self.metrics_collector.collect_security_metrics()
                
                all_metrics = system_metrics + security_metrics
                
                # Convert metrics to event data for rule evaluation
                event_data = {}
                for metric in all_metrics:
                    event_data[metric.metric_name] = metric.value
                
                # Evaluate alert rules
                triggered_alerts = await self.alert_manager.evaluate_rules(event_data)
                
                # Auto-create incidents for critical alerts
                for alert in triggered_alerts:
                    if alert.severity == AlertSeverity.CRITICAL:
                        await self._auto_create_incident(alert)
                
                # Wait before next iteration
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)  # Wait 30 seconds on error
    
    async def _auto_create_incident(self, alert: SecurityAlert) -> None:
        """Automatically create incident for critical alerts."""
        # Check if similar incident already exists
        existing_incidents = self.incident_manager.get_open_incidents()
        
        for incident in existing_incidents:
            if alert.alert_id in incident.alerts:
                # Alert already associated with incident
                return
        
        # Create new incident
        incident_id = await self.incident_manager.create_incident(
            title=f"Critical Alert: {alert.title}",
            description=f"Auto-generated incident for critical alert: {alert.description}",
            severity=alert.severity,
            alerts=[alert.alert_id]
        )
        
        logger.info(f"Auto-created incident {incident_id} for critical alert {alert.alert_id}")
    
    async def process_security_event(self, event_data: Dict[str, Any]) -> None:
        """Process a security event and trigger appropriate responses."""
        # Evaluate alert rules
        triggered_alerts = await self.alert_manager.evaluate_rules(event_data)
        
        # Log the event
        await audit_logger.log_security_event(
            event_type="security_event_processed",
            details={
                "triggered_alerts": len(triggered_alerts),
                "event_data": event_data
            },
            severity="info"
        )
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get overall monitoring system status."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "monitoring_active": self.monitoring_task and not self.monitoring_task.done(),
            "active_alerts": len(self.alert_manager.get_active_alerts()),
            "open_incidents": len(self.incident_manager.get_open_incidents()),
            "alert_rules": len(self.alert_manager.alert_rules),
            "notification_channels": len(self.alert_manager.notification_channels),
            "metrics_collected": len(self.metrics_collector.metrics_history)
        }


# Global security monitoring manager instance
security_monitoring = SecurityMonitoringManager()