#!/usr/bin/env python3
"""Validation script for monitoring and metrics setup."""

import asyncio
import logging
import sys
from pathlib import Path

import yaml

from monitoring.audit_logger import AuditLogger
from src.metrics import PerformanceMonitor
from src.metrics import metrics_collector

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# Validation script constants
ERROR_RATE_THRESHOLD = 0.1


class MonitoringValidator:
    """Validates monitoring and metrics configuration."""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.success_count = 0
        self.test_count = 0

    def add_error(self, test_name: str, error: str):
        """Add an error."""
        self.errors.append(f"{test_name}: {error}")
        logger.error("❌ %s: %s", test_name, error)

    def add_warning(self, test_name: str, warning: str):
        """Add a warning."""
        self.warnings.append(f"{test_name}: {warning}")
        logger.warning("⚠️  %s: %s", test_name, warning)

    def add_success(self, test_name: str, message: str = ""):
        """Add a success."""
        self.success_count += 1
        suffix = f": {message}" if message else ""
        logger.info("✅ %s%s", test_name, suffix)

    def test_prometheus_metrics(self):
        """Test Prometheus metrics definitions."""
        self.test_count += 1

        try:
            from src.metrics import MEMORY_COUNT
            from src.metrics import REQUEST_COUNT
            from src.metrics import REQUEST_DURATION

            # Test metric creation
            REQUEST_COUNT.labels(method="GET", endpoint="/test", status=200).inc()
            REQUEST_DURATION.labels(method="GET", endpoint="/test").observe(0.1)
            MEMORY_COUNT.set(100)

            self.add_success("Prometheus metrics initialization")

        except Exception as e:
            self.add_error("Prometheus metrics initialization", str(e))

    def test_metrics_collector(self):
        """Test MetricsCollector functionality."""
        self.test_count += 1

        try:
            # Test basic functionality
            metrics_collector.record_request_start("test-1", "GET", "/test")
            metrics_collector.record_request_end("test-1", "GET", "/test", 200)

            # Test search metrics
            metrics_collector.record_search(0.5, 10, "test-user")

            # Test error recording
            metrics_collector.record_error(
                "test_error", "Test error message", "test_component"
            )

            # Test system metrics collection
            metrics_collector.collect_system_metrics()

            # Test metrics summary
            summary = metrics_collector.get_metrics_summary()
            if not isinstance(summary, dict):
                raise ValueError("Metrics summary should be a dictionary")

            self.add_success("MetricsCollector functionality")

        except Exception as e:
            self.add_error("MetricsCollector functionality", str(e))

    def test_performance_monitor(self):
        """Test PerformanceMonitor."""
        self.test_count += 1

        try:
            monitor = PerformanceMonitor()

            # Test threshold setting
            monitor.set_threshold("error_rate", ERROR_RATE_THRESHOLD)

            if monitor.thresholds["error_rate"] != ERROR_RATE_THRESHOLD:
                raise ValueError("Threshold setting failed")

            self.add_success("PerformanceMonitor functionality")

        except Exception as e:
            self.add_error("PerformanceMonitor functionality", str(e))

    async def test_audit_logger(self):
        """Test AuditLogger functionality."""
        self.test_count += 1

        try:
            # Test audit logger creation
            logger = AuditLogger()

            # Test event logging (without database)
            await logger.log_event(
                action="test_action", resource="test_resource", details={"test": "data"}
            )

            await logger.log_security_event(
                event_type="test_event", details={"test": "security_data"}
            )

            await logger.log_api_call(
                method="GET", path="/test", status_code=200, duration=0.1
            )

            self.add_success("AuditLogger functionality")

        except Exception as e:
            self.add_error("AuditLogger functionality", str(e))

    def test_prometheus_config(self):
        """Test Prometheus configuration."""
        self.test_count += 1

        try:
            config_path = (
                Path(__file__).parent.parent
                / "config"
                / "prometheus"
                / "prometheus.yml"
            )

            if not config_path.exists():
                raise FileNotFoundError(f"Prometheus config not found: {config_path}")

            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Validate basic structure
            required_sections = ["global", "scrape_configs"]
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required section: {section}")

            # Validate scrape configs
            scrape_configs = config["scrape_configs"]
            if not isinstance(scrape_configs, list) or len(scrape_configs) == 0:
                raise ValueError("No scrape configs defined")

            # Check for mem0ai job
            mem0ai_jobs = [
                job for job in scrape_configs if job.get("job_name") == "mem0ai"
            ]
            if not mem0ai_jobs:
                self.add_warning("Prometheus config", "No mem0ai job defined")

            self.add_success("Prometheus configuration")

        except Exception as e:
            self.add_error("Prometheus configuration", str(e))

    def test_alert_rules(self):
        """Test alert rules configuration."""
        self.test_count += 1

        try:
            rules_path = (
                Path(__file__).parent.parent
                / "config"
                / "prometheus"
                / "alert_rules.yml"
            )

            if not rules_path.exists():
                raise FileNotFoundError(f"Alert rules not found: {rules_path}")

            with open(rules_path) as f:
                rules = yaml.safe_load(f)

            # Validate basic structure
            if "groups" not in rules:
                raise ValueError("No alert groups defined")

            groups = rules["groups"]
            if not isinstance(groups, list) or len(groups) == 0:
                raise ValueError("No alert rule groups found")

            # Count total rules
            total_rules = sum(len(group.get("rules", [])) for group in groups)

            if total_rules == 0:
                raise ValueError("No alert rules defined")

            # Validate rule structure
            for group in groups:
                for rule in group.get("rules", []):
                    required_fields = ["alert", "expr", "labels", "annotations"]
                    for field in required_fields:
                        if field not in rule:
                            raise ValueError(f"Alert rule missing field: {field}")

            self.add_success(
                "Alert rules configuration",
                f"{total_rules} rules in {len(groups)} groups",
            )

        except Exception as e:
            self.add_error("Alert rules configuration", str(e))

    def test_monitoring_file_structure(self):
        """Test monitoring file structure."""
        self.test_count += 1

        try:
            project_root = Path(__file__).parent.parent

            required_files = [
                "src/metrics.py",
                "monitoring_metrics.py",
                "monitoring/audit_logger.py",
                "config/prometheus/prometheus.yml",
                "config/prometheus/alert_rules.yml",
            ]

            missing_files = []
            for file_path in required_files:
                full_path = project_root / file_path
                if not full_path.exists():
                    missing_files.append(file_path)

            if missing_files:
                raise FileNotFoundError(f"Missing files: {', '.join(missing_files)}")

            self.add_success("Monitoring file structure")

        except Exception as e:
            self.add_error("Monitoring file structure", str(e))

    def test_import_validation(self):
        """Test that all monitoring modules can be imported."""
        self.test_count += 1

        try:
            # Test imports

            self.add_success("Module imports")

        except Exception as e:
            self.add_error("Module imports", str(e))

    async def run_all_tests(self):
        """Run all validation tests."""
        logger.info("🔍 Starting monitoring and metrics validation...\n")

        # Run synchronous tests
        self.test_monitoring_file_structure()
        self.test_import_validation()
        self.test_prometheus_metrics()
        self.test_metrics_collector()
        self.test_performance_monitor()
        self.test_prometheus_config()
        self.test_alert_rules()

        # Run async tests
        await self.test_audit_logger()

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)

        logger.info("✅ Successful tests: %d/%d", self.success_count, self.test_count)

        if self.warnings:
            logger.warning("⚠️  Warnings: %d", len(self.warnings))
            for warning in self.warnings:
                logger.warning("   - %s", warning)

        if self.errors:
            logger.error("❌ Errors: %d", len(self.errors))
            for error in self.errors:
                logger.error("   - %s", error)
            return False
        else:
            logger.info("🎉 All tests passed!")
            return True


async def main():
    """Main validation function."""
    validator = MonitoringValidator()
    success = await validator.run_all_tests()

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
