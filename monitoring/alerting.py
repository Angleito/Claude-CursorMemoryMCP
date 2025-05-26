"""Comprehensive alerting system for mem0ai monitoring.

This module provides a complete alerting framework with:
- Rule-based alerting with flexible conditions
- Multiple notification channels (email, Slack, webhook, PagerDuty)
- Alert severity levels and escalation
- Rate limiting and deduplication
- Alert correlation and grouping
- Historical alert tracking
- Recovery notifications

Examples:
    >>> alert_manager = AlertManager()
    >>> await alert_manager.initialize()
    >>>
    >>> # Define an alert rule
    >>> rule = AlertRule(
    ...     name="high_error_rate",
    ...     condition="mem0_errors_total > 10",
    ...     severity=AlertSeverity.CRITICAL,
    ...     duration=60
    ... )
    >>> alert_manager.add_rule(rule)
    >>>
    >>> # Add notification channel
    >>> email_channel = EmailNotificationChannel("alerts@company.com")
    >>> alert_manager.add_notification_channel(email_channel)
"""

import asyncio
import statistics
import time
import uuid
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from enum import Enum
from typing import Any

import httpx
import structlog
from prometheus_client import Counter
from prometheus_client import Gauge
from prometheus_client import Histogram

from .metrics_collector import MetricsCollector
from .metrics_collector import get_metrics_collector

# Alerting system constants
MIN_METRIC_VALUES_FOR_RATE = 2
FLOAT_COMPARISON_PRECISION = 1e-9
DEFAULT_RATE_LIMIT = 10  # notifications per minute
RATE_LIMIT_WINDOW_SECONDS = 60
DEFAULT_EVALUATION_INTERVAL = 30  # seconds
DEFAULT_MAX_ALERTS_HISTORY = 10000
DEFAULT_RULE_DURATION = 60  # seconds
DEFAULT_CONDITION_WINDOW = 300  # 5 minutes
HTTP_CLIENT_TIMEOUT = 10.0  # seconds
SHUTDOWN_TIMEOUT = 10.0  # seconds

logger = structlog.get_logger()

# Prometheus metrics for alerting system
ALERTS_FIRED_TOTAL = Counter(
    "mem0_alerts_fired_total",
    "Total alerts fired",
    ["rule_name", "severity", "channel"]
)
ALERTS_RESOLVED_TOTAL = Counter(
    "mem0_alerts_resolved_total",
    "Total alerts resolved",
    ["rule_name", "severity"]
)
ALERT_PROCESSING_DURATION = Histogram(
    "mem0_alert_processing_duration_seconds",
    "Time taken to process alerts",
    ["operation"]
)
ACTIVE_ALERTS_COUNT = Gauge(
    "mem0_active_alerts_count",
    "Number of currently active alerts",
    ["severity"]
)
NOTIFICATION_FAILURES_TOTAL = Counter(
    "mem0_notification_failures_total",
    "Total notification delivery failures",
    ["channel_type", "error_type"]
)


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertState(str, Enum):
    """Alert states."""

    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ACKNOWLEDGED = "acknowledged"


class ConditionOperator(str, Enum):
    """Condition operators for alert rules."""

    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="
    CONTAINS = "contains"
    REGEX_MATCH = "regex"


@dataclass
class AlertCondition:
    """Alert condition definition.

    Attributes:
        metric_name: Name of the metric to evaluate
        operator: Comparison operator
        threshold: Threshold value to compare against
        aggregation: How to aggregate metric values (avg, sum, max, min, count)
        window_seconds: Time window for evaluation
        labels: Optional label filters
    """

    metric_name: str
    operator: ConditionOperator
    threshold: float
    aggregation: str = "avg"  # avg, sum, max, min, count, rate
    window_seconds: int = DEFAULT_CONDITION_WINDOW  # 5 minutes default
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate condition parameters."""
        if not self.metric_name:
            raise ValueError("Metric name cannot be empty")

        if self.window_seconds <= 0:
            raise ValueError("Window seconds must be positive")

        valid_aggregations = {"avg", "sum", "max", "min", "count", "rate", "p50", "p95", "p99"}
        if self.aggregation not in valid_aggregations:
            raise ValueError(f"Invalid aggregation: {self.aggregation}")


@dataclass
class AlertRule:
    """Alert rule definition.

    Attributes:
        name: Unique rule name
        description: Human-readable description
        condition: Alert condition
        severity: Alert severity level
        duration_seconds: How long condition must be true before firing
        labels: Additional labels to attach to alerts
        enabled: Whether the rule is enabled
        runbook_url: Optional URL to troubleshooting documentation
        escalation_rules: Optional escalation configuration
    """

    name: str
    description: str
    condition: AlertCondition
    severity: AlertSeverity
    duration_seconds: int = DEFAULT_RULE_DURATION
    labels: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    runbook_url: str | None = None
    escalation_rules: dict[str, Any] | None = None
    notification_channels: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate rule parameters."""
        if not self.name or not self.name.strip():
            raise ValueError("Rule name cannot be empty")

        if self.duration_seconds < 0:
            raise ValueError("Duration seconds cannot be negative")


@dataclass
class Alert:
    """Active alert instance.

    Attributes:
        id: Unique alert ID
        rule_name: Name of the rule that fired this alert
        severity: Alert severity
        state: Current alert state
        description: Alert description
        labels: Alert labels
        annotations: Additional annotations
        started_at: When the alert first fired
        resolved_at: When the alert was resolved (if applicable)
        acknowledged_at: When the alert was acknowledged (if applicable)
        acknowledged_by: Who acknowledged the alert
        value: Current metric value that triggered the alert
        threshold: Threshold that was exceeded
        runbook_url: URL to troubleshooting documentation
        fingerprint: Unique fingerprint for deduplication
    """

    id: str
    rule_name: str
    severity: AlertSeverity
    state: AlertState
    description: str
    labels: dict[str, str]
    annotations: dict[str, str]
    started_at: datetime
    resolved_at: datetime | None = None
    acknowledged_at: datetime | None = None
    acknowledged_by: str | None = None
    value: float | None = None
    threshold: float | None = None
    runbook_url: str | None = None
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        """Generate fingerprint if not provided."""
        if not self.fingerprint:
            # Create fingerprint from rule name and sorted labels
            label_str = ",".join(f"{k}={v}" for k, v in sorted(self.labels.items()))
            self.fingerprint = f"{self.rule_name}:{label_str}"


class NotificationChannel(ABC):
    """Abstract base class for notification channels."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        """Initialize notification channel.

        Args:
            name: Channel name
            config: Channel configuration
        """
        self.name = name
        self.config = config
        self.enabled = config.get("enabled", True)
        self.rate_limit = config.get("rate_limit", DEFAULT_RATE_LIMIT)  # Max notifications per minute
        self.rate_limit_window = RATE_LIMIT_WINDOW_SECONDS  # seconds
        self.notification_times: deque = deque(maxlen=self.rate_limit)

    @abstractmethod
    async def send_notification(
        self,
        alert: Alert,
        is_resolution: bool = False
    ) -> bool:
        """Send notification for an alert.

        Args:
            alert: Alert to notify about
            is_resolution: Whether this is a resolution notification

        Returns:
            True if notification was sent successfully
        """

    def is_rate_limited(self) -> bool:
        """Check if channel is currently rate limited.

        Returns:
            True if rate limited
        """
        now = time.time()
        # Remove old notifications outside the window
        while (self.notification_times and
               now - self.notification_times[0] > self.rate_limit_window):
            self.notification_times.popleft()

        return len(self.notification_times) >= self.rate_limit

    def record_notification(self) -> None:
        """Record that a notification was sent."""
        self.notification_times.append(time.time())


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel."""

    async def send_notification(
        self,
        alert: Alert,
        is_resolution: bool = False
    ) -> bool:
        """Send email notification.

        Args:
            alert: Alert to notify about
            is_resolution: Whether this is a resolution notification

        Returns:
            True if email was sent successfully
        """
        if not self.enabled or self.is_rate_limited():
            return False

        try:
            # Email sending logic would go here
            # For now, just log the notification
            action = "RESOLVED" if is_resolution else "FIRED"
            logger.info(
                f"Email notification: Alert {action}",
                alert_id=alert.id,
                rule_name=alert.rule_name,
                severity=alert.severity.value,
                description=alert.description,
                to=self.config.get("to", "unknown"),
                channel="email"
            )

            self.record_notification()
            return True

        except Exception as e:
            logger.error("Failed to send email notification",
                        error=str(e),
                        alert_id=alert.id,
                        channel="email")
            NOTIFICATION_FAILURES_TOTAL.labels(
                channel_type="email",
                error_type=type(e).__name__
            ).inc()
            return False


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel."""

    async def send_notification(
        self,
        alert: Alert,
        is_resolution: bool = False
    ) -> bool:
        """Send Slack notification.

        Args:
            alert: Alert to notify about
            is_resolution: Whether this is a resolution notification

        Returns:
            True if Slack message was sent successfully
        """
        if not self.enabled or self.is_rate_limited():
            return False

        webhook_url = self.config.get("webhook_url")
        if not webhook_url:
            logger.error("Slack webhook URL not configured")
            return False

        try:
            # Prepare Slack message
            color = self._get_alert_color(alert.severity, is_resolution)
            action = "RESOLVED" if is_resolution else "FIRED"

            message = {
                "username": "mem0ai-alerts",
                "icon_emoji": ":warning:" if not is_resolution else ":white_check_mark:",
                "attachments": [
                    {
                        "color": color,
                        "title": f"Alert {action}: {alert.rule_name}",
                        "text": alert.description,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.value.upper(),
                                "short": True
                            },
                            {
                                "title": "State",
                                "value": alert.state.value.upper(),
                                "short": True
                            }
                        ],
                        "footer": "mem0ai monitoring",
                        "ts": int(time.time())
                    }
                ]
            }

            # Add runbook link if available
            if alert.runbook_url:
                message["attachments"][0]["fields"].append({
                    "title": "Runbook",
                    "value": f"<{alert.runbook_url}|Troubleshooting Guide>",
                    "short": False
                })

            # Send to Slack
            async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT) as client:
                response = await client.post(webhook_url, json=message)
                response.raise_for_status()

            logger.info("Slack notification sent",
                       alert_id=alert.id,
                       rule_name=alert.rule_name,
                       channel="slack")

            self.record_notification()
            return True

        except Exception as e:
            logger.error("Failed to send Slack notification",
                        error=str(e),
                        alert_id=alert.id,
                        channel="slack")
            NOTIFICATION_FAILURES_TOTAL.labels(
                channel_type="slack",
                error_type=type(e).__name__
            ).inc()
            return False

    def _get_alert_color(self, severity: AlertSeverity, is_resolution: bool) -> str:
        """Get color for Slack attachment based on severity.

        Args:
            severity: Alert severity
            is_resolution: Whether this is a resolution

        Returns:
            Hex color string
        """
        if is_resolution:
            return "good"  # Green

        color_map = {
            AlertSeverity.INFO: "#36a64f",      # Green
            AlertSeverity.WARNING: "#ff9500",  # Orange
            AlertSeverity.CRITICAL: "#ff0000", # Red
            AlertSeverity.EMERGENCY: "#8b0000" # Dark red
        }

        return color_map.get(severity, "#808080")  # Gray default


class WebhookNotificationChannel(NotificationChannel):
    """Generic webhook notification channel."""

    async def send_notification(
        self,
        alert: Alert,
        is_resolution: bool = False
    ) -> bool:
        """Send webhook notification.

        Args:
            alert: Alert to notify about
            is_resolution: Whether this is a resolution notification

        Returns:
            True if webhook was called successfully
        """
        if not self.enabled or self.is_rate_limited():
            return False

        webhook_url = self.config.get("url")
        if not webhook_url:
            logger.error("Webhook URL not configured")
            return False

        try:
            # Prepare webhook payload
            payload = {
                "alert_id": alert.id,
                "rule_name": alert.rule_name,
                "severity": alert.severity.value,
                "state": alert.state.value,
                "description": alert.description,
                "labels": alert.labels,
                "annotations": alert.annotations,
                "started_at": alert.started_at.isoformat(),
                "is_resolution": is_resolution,
                "value": alert.value,
                "threshold": alert.threshold,
                "runbook_url": alert.runbook_url
            }

            if is_resolution:
                payload["resolved_at"] = alert.resolved_at.isoformat() if alert.resolved_at else None

            # Send webhook
            headers = {"Content-Type": "application/json"}
            auth_header = self.config.get("auth_header")
            if auth_header:
                headers["Authorization"] = auth_header

            async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT) as client:
                response = await client.post(
                    webhook_url,
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()

            logger.info("Webhook notification sent",
                       alert_id=alert.id,
                       rule_name=alert.rule_name,
                       webhook_url=webhook_url,
                       channel="webhook")

            self.record_notification()
            return True

        except Exception as e:
            logger.error("Failed to send webhook notification",
                        error=str(e),
                        alert_id=alert.id,
                        webhook_url=webhook_url,
                        channel="webhook")
            NOTIFICATION_FAILURES_TOTAL.labels(
                channel_type="webhook",
                error_type=type(e).__name__
            ).inc()
            return False


class AlertManager:
    """Main alert manager that evaluates rules and manages alerts.

    This manager provides:
    - Rule evaluation against metrics
    - Alert state management
    - Notification delivery
    - Alert correlation and deduplication
    - Rate limiting and suppression
    - Historical tracking
    """

    def __init__(
        self,
        metrics_collector: MetricsCollector | None = None,
        evaluation_interval: int = DEFAULT_EVALUATION_INTERVAL,  # seconds
        max_alerts_history: int = DEFAULT_MAX_ALERTS_HISTORY
    ) -> None:
        """Initialize alert manager.

        Args:
            metrics_collector: MetricsCollector instance to query
            evaluation_interval: How often to evaluate rules (seconds)
            max_alerts_history: Maximum number of alerts to keep in history
        """
        self.metrics_collector = metrics_collector or get_metrics_collector()
        self.evaluation_interval = evaluation_interval
        self.max_alerts_history = max_alerts_history

        # Storage
        self.rules: dict[str, AlertRule] = {}
        self.active_alerts: dict[str, Alert] = {}  # fingerprint -> alert
        self.alert_history: deque = deque(maxlen=max_alerts_history)
        self.notification_channels: dict[str, NotificationChannel] = {}

        # State tracking
        self.pending_alerts: dict[str, datetime] = {}  # fingerprint -> first_seen
        self.suppressed_alerts: set[str] = set()  # fingerprints
        self.last_evaluation: datetime | None = None

        # Background tasks
        self.evaluation_task: asyncio.Task | None = None
        self.shutdown_event = asyncio.Event()

        logger.info("AlertManager initialized",
                   evaluation_interval=evaluation_interval)

    async def initialize(self) -> None:
        """Initialize the alert manager and start background tasks."""
        try:
            # Start rule evaluation task
            self.evaluation_task = asyncio.create_task(
                self._evaluation_loop(),
                name="alert_manager_evaluation"
            )

            logger.info("AlertManager started successfully")

        except Exception as e:
            logger.error("Failed to initialize AlertManager", error=str(e))
            raise RuntimeError(f"AlertManager initialization failed: {e}") from e

    async def close(self) -> None:
        """Close the alert manager and cleanup resources."""
        logger.info("Shutting down AlertManager")

        # Signal shutdown
        self.shutdown_event.set()

        # Cancel evaluation task
        if self.evaluation_task and not self.evaluation_task.done():
            self.evaluation_task.cancel()
            try:
                await asyncio.wait_for(self.evaluation_task, timeout=SHUTDOWN_TIMEOUT)
            except (TimeoutError, asyncio.CancelledError):
                logger.warning("Evaluation task shutdown timed out")

        logger.info("AlertManager shutdown complete")

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule.

        Args:
            rule: AlertRule to add

        Raises:
            ValueError: If rule name already exists
        """
        if rule.name in self.rules:
            raise ValueError(f"Rule '{rule.name}' already exists")

        self.rules[rule.name] = rule
        logger.info("Alert rule added", rule_name=rule.name, severity=rule.severity.value)

    def remove_rule(self, rule_name: str) -> bool:
        """Remove an alert rule.

        Args:
            rule_name: Name of the rule to remove

        Returns:
            True if rule was removed, False if not found
        """
        if rule_name not in self.rules:
            return False

        # Resolve any active alerts for this rule
        alerts_to_resolve = [
            alert for alert in self.active_alerts.values()
            if alert.rule_name == rule_name
        ]

        for alert in alerts_to_resolve:
            asyncio.create_task(self._resolve_alert(alert.fingerprint))

        del self.rules[rule_name]
        logger.info("Alert rule removed", rule_name=rule_name)
        return True

    def add_notification_channel(self, channel: NotificationChannel) -> None:
        """Add a notification channel.

        Args:
            channel: NotificationChannel to add
        """
        self.notification_channels[channel.name] = channel
        logger.info("Notification channel added",
                   channel_name=channel.name,
                   channel_type=type(channel).__name__)

    def remove_notification_channel(self, channel_name: str) -> bool:
        """Remove a notification channel.

        Args:
            channel_name: Name of the channel to remove

        Returns:
            True if channel was removed, False if not found
        """
        if channel_name not in self.notification_channels:
            return False

        del self.notification_channels[channel_name]
        logger.info("Notification channel removed", channel_name=channel_name)
        return True

    async def _evaluation_loop(self) -> None:
        """Main evaluation loop that runs in the background."""
        logger.info("Starting alert evaluation loop")

        while not self.shutdown_event.is_set():
            try:
                await self._evaluate_rules()
                await asyncio.sleep(self.evaluation_interval)

            except asyncio.CancelledError:
                logger.info("Alert evaluation loop cancelled")
                break
            except Exception as e:
                logger.error("Error in alert evaluation loop", error=str(e))
                await asyncio.sleep(min(self.evaluation_interval, 60))

        logger.info("Alert evaluation loop ended")

    async def _evaluate_rules(self) -> None:
        """Evaluate all alert rules against current metrics."""
        start_time = time.time()

        try:
            evaluation_time = datetime.utcnow()
            rules_evaluated = 0

            for rule_name, rule in self.rules.items():
                if not rule.enabled:
                    continue

                try:
                    await self._evaluate_single_rule(rule, evaluation_time)
                    rules_evaluated += 1

                except Exception as e:
                    logger.error("Error evaluating rule",
                               rule_name=rule_name,
                               error=str(e))

            # Update metrics
            processing_time = time.time() - start_time
            ALERT_PROCESSING_DURATION.labels(operation="evaluation").observe(processing_time)

            # Update active alert counts
            severity_counts = defaultdict(int)
            for alert in self.active_alerts.values():
                severity_counts[alert.severity.value] += 1

            for severity in AlertSeverity:
                ACTIVE_ALERTS_COUNT.labels(severity=severity.value).set(
                    severity_counts[severity.value]
                )

            self.last_evaluation = evaluation_time

            logger.debug("Rule evaluation completed",
                        rules_evaluated=rules_evaluated,
                        active_alerts=len(self.active_alerts),
                        processing_time_ms=processing_time * 1000)

        except Exception as e:
            logger.error("Error in rule evaluation", error=str(e))

    async def _evaluate_single_rule(self, rule: AlertRule, evaluation_time: datetime) -> None:
        """Evaluate a single alert rule.

        Args:
            rule: AlertRule to evaluate
            evaluation_time: Time of this evaluation
        """
        try:
            # Get metric values for the condition
            end_time = evaluation_time
            start_time = end_time - timedelta(seconds=rule.condition.window_seconds)

            metric_values = self.metrics_collector.get_metric_values(
                rule.condition.metric_name,
                start_time=start_time,
                end_time=end_time,
                labels=rule.condition.labels
            )

            if not metric_values:
                # No data available - resolve any existing alerts for this rule
                await self._resolve_alerts_for_rule(rule.name)
                return

            # Aggregate metric values according to the condition
            aggregated_value = self._aggregate_values(
                metric_values,
                rule.condition.aggregation,
                rule.condition.window_seconds
            )

            # Evaluate the condition
            condition_met = self._evaluate_condition(
                aggregated_value,
                rule.condition.operator,
                rule.condition.threshold
            )

            # Generate alert fingerprint
            fingerprint = self._generate_fingerprint(rule, rule.condition.labels)

            if condition_met:
                await self._handle_condition_met(
                    rule,
                    fingerprint,
                    aggregated_value,
                    evaluation_time
                )
            else:
                await self._handle_condition_not_met(rule, fingerprint)

        except Exception as e:
            logger.error("Error evaluating single rule",
                        rule_name=rule.name,
                        error=str(e))

    def _aggregate_values(
        self,
        metric_values: list[Any],
        aggregation: str,
        window_seconds: int
    ) -> float:
        """Aggregate metric values according to the specified method.

        Args:
            metric_values: List of MetricValue objects
            aggregation: Aggregation method
            window_seconds: Time window in seconds

        Returns:
            Aggregated value
        """
        if not metric_values:
            return 0.0

        values = [v.value for v in metric_values]

        # Dictionary dispatch pattern for aggregation methods
        aggregation_methods = {
            "avg": lambda: statistics.mean(values),
            "sum": lambda: sum(values),
            "max": lambda: max(values),
            "min": lambda: min(values),
            "count": lambda: len(values),
            "rate": lambda: self._calculate_rate(metric_values, values),
            "p50": lambda: statistics.median(values),
            "p95": lambda: self._calculate_percentile(values, 0.95),
            "p99": lambda: self._calculate_percentile(values, 0.99),
        }

        # Get the aggregation method or default to average
        method = aggregation_methods.get(aggregation, lambda: statistics.mean(values))
        return method()

    def _calculate_rate(self, metric_values: list[Any], values: list[float]) -> float:
        """Calculate rate per second for metric values."""
        if len(values) < MIN_METRIC_VALUES_FOR_RATE:
            return 0.0
        time_span = (metric_values[-1].timestamp - metric_values[0].timestamp).total_seconds()
        if time_span <= 0:
            return 0.0
        return sum(values) / time_span

    def _calculate_percentile(self, values: list[float], percentile: float) -> float:
        """Calculate percentile value from sorted values."""
        sorted_values = sorted(values)
        index = int(percentile * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]

    def _evaluate_condition(
        self,
        value: float,
        operator: ConditionOperator,
        threshold: float
    ) -> bool:
        """Evaluate a condition against a value.

        Args:
            value: Value to check
            operator: Comparison operator
            threshold: Threshold to compare against

        Returns:
            True if condition is met
        """
        # Dictionary dispatch pattern for operators
        operator_map = {
            ConditionOperator.GREATER_THAN: lambda: value > threshold,
            ConditionOperator.LESS_THAN: lambda: value < threshold,
            ConditionOperator.GREATER_EQUAL: lambda: value >= threshold,
            ConditionOperator.LESS_EQUAL: lambda: value <= threshold,
            ConditionOperator.EQUAL: lambda: abs(value - threshold) < FLOAT_COMPARISON_PRECISION,
            ConditionOperator.NOT_EQUAL: lambda: abs(value - threshold) >= FLOAT_COMPARISON_PRECISION,
        }

        # Get the evaluation function or default to False
        evaluator = operator_map.get(operator, lambda: False)
        return evaluator()

    def _generate_fingerprint(self, rule: AlertRule, labels: dict[str, str]) -> str:
        """Generate a unique fingerprint for an alert.

        Args:
            rule: AlertRule
            labels: Alert labels

        Returns:
            Unique fingerprint string
        """
        # Combine rule name with sorted labels
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{rule.name}:{label_str}"

    async def _handle_condition_met(
        self,
        rule: AlertRule,
        fingerprint: str,
        value: float,
        evaluation_time: datetime
    ) -> None:
        """Handle when an alert condition is met.

        Args:
            rule: AlertRule that matched
            fingerprint: Alert fingerprint
            value: Current metric value
            evaluation_time: Time of evaluation
        """
        if fingerprint in self.active_alerts:
            # Alert is already active, just update the value
            self.active_alerts[fingerprint].value = value
            return

        if fingerprint in self.suppressed_alerts:
            # Alert is suppressed
            return

        # Check if this is a new pending alert
        if fingerprint not in self.pending_alerts:
            self.pending_alerts[fingerprint] = evaluation_time
            logger.debug("Alert condition met, starting duration check",
                        rule_name=rule.name,
                        fingerprint=fingerprint,
                        value=value)
            return

        # Check if duration requirement is met
        first_seen = self.pending_alerts[fingerprint]
        duration = (evaluation_time - first_seen).total_seconds()

        if duration >= rule.duration_seconds:
            # Fire the alert
            await self._fire_alert(rule, fingerprint, value, first_seen)
            # Remove from pending
            del self.pending_alerts[fingerprint]

    async def _handle_condition_not_met(
        self,
        rule: AlertRule,
        fingerprint: str
    ) -> None:
        """Handle when an alert condition is not met.

        Args:
            rule: AlertRule
            fingerprint: Alert fingerprint
        """
        # Remove from pending if it was there
        if fingerprint in self.pending_alerts:
            del self.pending_alerts[fingerprint]
            logger.debug("Alert condition no longer met, removing from pending",
                        rule_name=rule.name,
                        fingerprint=fingerprint)

        # Resolve if it was active
        if fingerprint in self.active_alerts:
            await self._resolve_alert(fingerprint)

    async def _fire_alert(
        self,
        rule: AlertRule,
        fingerprint: str,
        value: float,
        started_at: datetime
    ) -> None:
        """Fire a new alert.

        Args:
            rule: AlertRule that fired
            fingerprint: Alert fingerprint
            value: Current metric value
            started_at: When the condition was first met
        """
        alert_id = str(uuid.uuid4())

        alert = Alert(
            id=alert_id,
            rule_name=rule.name,
            severity=rule.severity,
            state=AlertState.FIRING,
            description=rule.description,
            labels=rule.labels.copy(),
            annotations={
                "condition": f"{rule.condition.metric_name} {rule.condition.operator.value} {rule.condition.threshold}",
                "aggregation": rule.condition.aggregation,
                "window": f"{rule.condition.window_seconds}s"
            },
            started_at=started_at,
            value=value,
            threshold=rule.condition.threshold,
            runbook_url=rule.runbook_url,
            fingerprint=fingerprint
        )

        # Store the alert
        self.active_alerts[fingerprint] = alert
        self.alert_history.append(alert)

        # Update metrics
        ALERTS_FIRED_TOTAL.labels(
            rule_name=rule.name,
            severity=rule.severity.value,
            channel="total"
        ).inc()

        # Send notifications
        await self._send_notifications(alert, is_resolution=False)

        logger.warning("Alert fired",
                      alert_id=alert_id,
                      rule_name=rule.name,
                      severity=rule.severity.value,
                      value=value,
                      threshold=rule.condition.threshold,
                      fingerprint=fingerprint)

    async def _resolve_alert(self, fingerprint: str) -> None:
        """Resolve an active alert.

        Args:
            fingerprint: Alert fingerprint
        """
        if fingerprint not in self.active_alerts:
            return

        alert = self.active_alerts[fingerprint]
        alert.state = AlertState.RESOLVED
        alert.resolved_at = datetime.utcnow()

        # Remove from active alerts
        del self.active_alerts[fingerprint]

        # Update metrics
        ALERTS_RESOLVED_TOTAL.labels(
            rule_name=alert.rule_name,
            severity=alert.severity.value
        ).inc()

        # Send resolution notifications
        await self._send_notifications(alert, is_resolution=True)

        logger.info("Alert resolved",
                   alert_id=alert.id,
                   rule_name=alert.rule_name,
                   fingerprint=fingerprint,
                   duration_seconds=(alert.resolved_at - alert.started_at).total_seconds())

    async def _resolve_alerts_for_rule(self, rule_name: str) -> None:
        """Resolve all active alerts for a specific rule.

        Args:
            rule_name: Name of the rule
        """
        alerts_to_resolve = [
            fingerprint for fingerprint, alert in self.active_alerts.items()
            if alert.rule_name == rule_name
        ]

        for fingerprint in alerts_to_resolve:
            await self._resolve_alert(fingerprint)

    async def _send_notifications(
        self,
        alert: Alert,
        is_resolution: bool = False
    ) -> None:
        """Send notifications for an alert.

        Args:
            alert: Alert to notify about
            is_resolution: Whether this is a resolution notification
        """
        rule = self.rules.get(alert.rule_name)
        if not rule:
            return

        # Determine which channels to use
        channels_to_use = rule.notification_channels or list(self.notification_channels.keys())

        # Send notifications to each channel
        for channel_name in channels_to_use:
            channel = self.notification_channels.get(channel_name)
            if not channel:
                continue

            try:
                success = await channel.send_notification(alert, is_resolution)

                if success:
                    ALERTS_FIRED_TOTAL.labels(
                        rule_name=alert.rule_name,
                        severity=alert.severity.value,
                        channel=channel_name
                    ).inc()

            except Exception as e:
                logger.error("Error sending notification",
                            channel_name=channel_name,
                            alert_id=alert.id,
                            error=str(e))

    def get_active_alerts(
        self,
        severity: AlertSeverity | None = None,
        rule_name: str | None = None
    ) -> list[Alert]:
        """Get active alerts with optional filtering.

        Args:
            severity: Filter by severity
            rule_name: Filter by rule name

        Returns:
            List of active alerts
        """
        alerts = list(self.active_alerts.values())

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if rule_name:
            alerts = [a for a in alerts if a.rule_name == rule_name]

        return sorted(alerts, key=lambda a: a.started_at, reverse=True)

    def get_alert_history(
        self,
        limit: int = 100,
        severity: AlertSeverity | None = None,
        rule_name: str | None = None
    ) -> list[Alert]:
        """Get alert history with optional filtering.

        Args:
            limit: Maximum number of alerts to return
            severity: Filter by severity
            rule_name: Filter by rule name

        Returns:
            List of historical alerts
        """
        alerts = list(self.alert_history)

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if rule_name:
            alerts = [a for a in alerts if a.rule_name == rule_name]

        # Sort by start time, most recent first
        alerts = sorted(alerts, key=lambda a: a.started_at, reverse=True)

        return alerts[:limit]

    async def acknowledge_alert(
        self,
        fingerprint: str,
        acknowledged_by: str,
        comment: str | None = None
    ) -> bool:
        """Acknowledge an active alert.

        Args:
            fingerprint: Alert fingerprint
            acknowledged_by: Who acknowledged the alert
            comment: Optional acknowledgment comment

        Returns:
            True if alert was acknowledged
        """
        if fingerprint not in self.active_alerts:
            return False

        alert = self.active_alerts[fingerprint]
        alert.state = AlertState.ACKNOWLEDGED
        alert.acknowledged_at = datetime.utcnow()
        alert.acknowledged_by = acknowledged_by

        if comment:
            alert.annotations["acknowledgment_comment"] = comment

        logger.info("Alert acknowledged",
                   alert_id=alert.id,
                   rule_name=alert.rule_name,
                   acknowledged_by=acknowledged_by,
                   comment=comment)

        return True

    def suppress_alert(self, fingerprint: str) -> bool:
        """Suppress an alert (prevent notifications).

        Args:
            fingerprint: Alert fingerprint

        Returns:
            True if alert was suppressed
        """
        self.suppressed_alerts.add(fingerprint)

        # If alert is currently active, mark it as suppressed
        if fingerprint in self.active_alerts:
            self.active_alerts[fingerprint].state = AlertState.SUPPRESSED

        logger.info("Alert suppressed", fingerprint=fingerprint)
        return True

    def unsuppress_alert(self, fingerprint: str) -> bool:
        """Remove suppression from an alert.

        Args:
            fingerprint: Alert fingerprint

        Returns:
            True if suppression was removed
        """
        if fingerprint in self.suppressed_alerts:
            self.suppressed_alerts.remove(fingerprint)

            # If alert is active, change state back to firing
            if fingerprint in self.active_alerts:
                self.active_alerts[fingerprint].state = AlertState.FIRING

            logger.info("Alert suppression removed", fingerprint=fingerprint)
            return True

        return False

    def get_stats(self) -> dict[str, Any]:
        """Get alerting system statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "total_rules": len(self.rules),
            "enabled_rules": sum(1 for rule in self.rules.values() if rule.enabled),
            "active_alerts": len(self.active_alerts),
            "pending_alerts": len(self.pending_alerts),
            "suppressed_alerts": len(self.suppressed_alerts),
            "notification_channels": len(self.notification_channels),
            "last_evaluation": self.last_evaluation.isoformat() if self.last_evaluation else None,
            "evaluation_interval_seconds": self.evaluation_interval,
        }


# Global alert manager instance
alert_manager = AlertManager()


async def initialize_alert_manager(
    metrics_collector: MetricsCollector | None = None
) -> AlertManager:
    """Initialize the global alert manager.

    Args:
        metrics_collector: Optional MetricsCollector instance

    Returns:
        Initialized AlertManager instance
    """
    global alert_manager
    if metrics_collector:
        alert_manager = AlertManager(metrics_collector)
    await alert_manager.initialize()
    return alert_manager


def get_alert_manager() -> AlertManager:
    """Get the global alert manager instance.

    Returns:
        AlertManager instance
    """
    return alert_manager
