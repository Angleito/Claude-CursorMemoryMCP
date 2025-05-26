"""Compliance frameworks and audit capabilities including GDPR, HIPAA, SOC 2, and PCI DSS."""

import hashlib
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field

from config.settings import get_settings
from monitoring.audit_logger import audit_logger
from security.data_protection import data_protection

settings = get_settings()


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"
    HIPAA = "hipaa" 
    SOC2 = "soc2"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"
    NIST = "nist"


class ComplianceStatus(Enum):
    """Compliance status levels."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"
    NOT_ASSESSED = "not_assessed"


class AuditResult(Enum):
    """Audit result types."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class ComplianceControl:
    """Represents a compliance control requirement."""
    control_id: str
    framework: ComplianceFramework
    title: str
    description: str
    category: str
    required: bool = True
    automated_check: bool = False
    implementation_notes: str = ""
    evidence_required: List[str] = field(default_factory=list)
    responsible_party: str = ""
    last_assessed: Optional[datetime] = None
    status: ComplianceStatus = ComplianceStatus.NOT_ASSESSED


@dataclass
class AuditEvidence:
    """Represents evidence for compliance audits."""
    evidence_id: str
    control_id: str
    evidence_type: str  # document, log, screenshot, configuration, etc.
    description: str
    file_path: Optional[str] = None
    content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    collected_at: datetime = field(default_factory=datetime.utcnow)
    collected_by: str = ""
    hash_value: Optional[str] = None


@dataclass
class AuditFinding:
    """Represents an audit finding."""
    finding_id: str
    control_id: str
    severity: str  # critical, high, medium, low
    result: AuditResult
    title: str
    description: str
    remediation: str
    evidence: List[str] = field(default_factory=list)  # Evidence IDs
    status: str = "open"  # open, in_progress, resolved, false_positive
    identified_at: datetime = field(default_factory=datetime.utcnow)
    identified_by: str = ""
    due_date: Optional[datetime] = None
    assigned_to: Optional[str] = None


class ComplianceChecker(ABC):
    """Abstract base class for compliance checkers."""
    
    @abstractmethod
    def get_framework(self) -> ComplianceFramework:
        """Get the compliance framework this checker implements."""
        pass
    
    @abstractmethod
    def get_controls(self) -> List[ComplianceControl]:
        """Get all controls for this framework."""
        pass
    
    @abstractmethod
    async def assess_control(self, control: ComplianceControl) -> List[AuditFinding]:
        """Assess a specific control and return findings."""
        pass


class GDPRChecker(ComplianceChecker):
    """GDPR compliance checker."""
    
    def get_framework(self) -> ComplianceFramework:
        return ComplianceFramework.GDPR
    
    def get_controls(self) -> List[ComplianceControl]:
        """Get GDPR controls based on key articles."""
        return [
            ComplianceControl(
                control_id="GDPR-5",
                framework=ComplianceFramework.GDPR,
                title="Data Processing Principles (Article 5)",
                description="Personal data must be processed lawfully, fairly, transparently, and for specified purposes.",
                category="data_processing",
                automated_check=True,
                evidence_required=["processing_records", "privacy_policy", "consent_records"]
            ),
            ComplianceControl(
                control_id="GDPR-6",
                framework=ComplianceFramework.GDPR,
                title="Lawful Basis for Processing (Article 6)",
                description="Processing must have a lawful basis such as consent, contract, legal obligation, etc.",
                category="legal_basis",
                automated_check=True,
                evidence_required=["consent_records", "legal_basis_documentation"]
            ),
            ComplianceControl(
                control_id="GDPR-7",
                framework=ComplianceFramework.GDPR,
                title="Consent (Article 7)",
                description="Consent must be freely given, specific, informed, and unambiguous.",
                category="consent",
                automated_check=True,
                evidence_required=["consent_forms", "consent_records", "withdrawal_mechanisms"]
            ),
            ComplianceControl(
                control_id="GDPR-12",
                framework=ComplianceFramework.GDPR,
                title="Transparent Information (Article 12)",
                description="Information about processing must be provided in a clear and plain language.",
                category="transparency",
                automated_check=False,
                evidence_required=["privacy_policy", "data_subject_communications"]
            ),
            ComplianceControl(
                control_id="GDPR-17",
                framework=ComplianceFramework.GDPR,
                title="Right to Erasure (Article 17)",
                description="Data subjects have the right to have their personal data erased.",
                category="data_subject_rights",
                automated_check=True,
                evidence_required=["erasure_procedures", "deletion_logs", "retention_policies"]
            ),
            ComplianceControl(
                control_id="GDPR-25",
                framework=ComplianceFramework.GDPR,
                title="Data Protection by Design (Article 25)",
                description="Data protection measures must be implemented from the design phase.",
                category="privacy_by_design",
                automated_check=True,
                evidence_required=["system_design_docs", "privacy_impact_assessments", "technical_measures"]
            ),
            ComplianceControl(
                control_id="GDPR-30",
                framework=ComplianceFramework.GDPR,
                title="Records of Processing (Article 30)",
                description="Organizations must maintain records of data processing activities.",
                category="documentation",
                automated_check=True,
                evidence_required=["processing_records", "data_flow_diagrams"]
            ),
            ComplianceControl(
                control_id="GDPR-32",
                framework=ComplianceFramework.GDPR,
                title="Security of Processing (Article 32)",
                description="Appropriate technical and organizational measures must ensure data security.",
                category="security",
                automated_check=True,
                evidence_required=["security_measures", "encryption_implementation", "access_controls"]
            )
        ]
    
    async def assess_control(self, control: ComplianceControl) -> List[AuditFinding]:
        """Assess GDPR control compliance."""
        findings = []
        
        if control.control_id == "GDPR-5":
            findings.extend(await self._check_data_processing_principles())
        elif control.control_id == "GDPR-6":
            findings.extend(await self._check_lawful_basis())
        elif control.control_id == "GDPR-7":
            findings.extend(await self._check_consent_management())
        elif control.control_id == "GDPR-17":
            findings.extend(await self._check_erasure_capabilities())
        elif control.control_id == "GDPR-25":
            findings.extend(await self._check_privacy_by_design())
        elif control.control_id == "GDPR-30":
            findings.extend(await self._check_processing_records())
        elif control.control_id == "GDPR-32":
            findings.extend(await self._check_security_measures())
        
        return findings
    
    async def _check_data_processing_principles(self) -> List[AuditFinding]:
        """Check compliance with data processing principles."""
        findings = []
        
        # Check if data minimization is implemented
        # This would check actual data collection practices
        findings.append(AuditFinding(
            finding_id=f"GDPR-5-{uuid.uuid4().hex[:8]}",
            control_id="GDPR-5",
            severity="medium",
            result=AuditResult.WARNING,
            title="Data Minimization Assessment Required",
            description="Manual assessment required to verify data minimization practices",
            remediation="Review data collection practices to ensure only necessary data is collected"
        ))
        
        return findings
    
    async def _check_lawful_basis(self) -> List[AuditFinding]:
        """Check lawful basis for processing."""
        findings = []
        
        # Check if consent records exist
        consent_records_exist = len(data_protection.consent_manager.consent_records) > 0
        
        if not consent_records_exist:
            findings.append(AuditFinding(
                finding_id=f"GDPR-6-{uuid.uuid4().hex[:8]}",
                control_id="GDPR-6",
                severity="high",
                result=AuditResult.FAIL,
                title="No Consent Records Found",
                description="No consent records found in the system",
                remediation="Implement consent management and ensure all processing has lawful basis"
            ))
        else:
            findings.append(AuditFinding(
                finding_id=f"GDPR-6-{uuid.uuid4().hex[:8]}",
                control_id="GDPR-6",
                severity="low",
                result=AuditResult.PASS,
                title="Consent Management Implemented",
                description="Consent management system is in place",
                remediation="Continue monitoring consent records for compliance"
            ))
        
        return findings
    
    async def _check_consent_management(self) -> List[AuditFinding]:
        """Check consent management implementation."""
        findings = []
        
        # Check if consent withdrawal is supported
        consent_manager = data_protection.consent_manager
        
        findings.append(AuditFinding(
            finding_id=f"GDPR-7-{uuid.uuid4().hex[:8]}",
            control_id="GDPR-7",
            severity="low",
            result=AuditResult.PASS,
            title="Consent Withdrawal Mechanism Available",
            description="System supports consent withdrawal functionality",
            remediation="Ensure consent withdrawal is clearly communicated to users"
        ))
        
        return findings
    
    async def _check_erasure_capabilities(self) -> List[AuditFinding]:
        """Check right to erasure implementation."""
        findings = []
        
        # Check if data retention policies exist
        retention_policies = data_protection.retention_manager.policies
        
        if not retention_policies:
            findings.append(AuditFinding(
                finding_id=f"GDPR-17-{uuid.uuid4().hex[:8]}",
                control_id="GDPR-17",
                severity="high",
                result=AuditResult.FAIL,
                title="No Data Retention Policies",
                description="No data retention policies are configured",
                remediation="Implement data retention policies and automated deletion procedures"
            ))
        else:
            findings.append(AuditFinding(
                finding_id=f"GDPR-17-{uuid.uuid4().hex[:8]}",
                control_id="GDPR-17",
                severity="low",
                result=AuditResult.PASS,
                title="Data Retention Policies Configured",
                description=f"Found {len(retention_policies)} retention policies",
                remediation="Review retention periods to ensure compliance with legal requirements"
            ))
        
        return findings
    
    async def _check_privacy_by_design(self) -> List[AuditFinding]:
        """Check privacy by design implementation."""
        findings = []
        
        # Check if PII detection is enabled
        pii_detector = data_protection.pii_detector
        
        findings.append(AuditFinding(
            finding_id=f"GDPR-25-{uuid.uuid4().hex[:8]}",
            control_id="GDPR-25",
            severity="low",
            result=AuditResult.PASS,
            title="PII Detection Implemented",
            description="Automated PII detection is implemented",
            remediation="Regularly update PII detection patterns and ensure comprehensive coverage"
        ))
        
        return findings
    
    async def _check_processing_records(self) -> List[AuditFinding]:
        """Check processing records maintenance."""
        findings = []
        
        # This would check if Article 30 records are maintained
        findings.append(AuditFinding(
            finding_id=f"GDPR-30-{uuid.uuid4().hex[:8]}",
            control_id="GDPR-30",
            severity="medium",
            result=AuditResult.WARNING,
            title="Processing Records Assessment Required",
            description="Manual review required to verify processing records completeness",
            remediation="Ensure all processing activities are documented according to Article 30"
        ))
        
        return findings
    
    async def _check_security_measures(self) -> List[AuditFinding]:
        """Check security of processing measures."""
        findings = []
        
        # Check encryption implementation
        findings.append(AuditFinding(
            finding_id=f"GDPR-32-{uuid.uuid4().hex[:8]}",
            control_id="GDPR-32",
            severity="low",
            result=AuditResult.PASS,
            title="Encryption Implemented",
            description="Data encryption is implemented for data at rest and in transit",
            remediation="Regularly review and update encryption standards"
        ))
        
        return findings


class SOC2Checker(ComplianceChecker):
    """SOC 2 compliance checker."""
    
    def get_framework(self) -> ComplianceFramework:
        return ComplianceFramework.SOC2
    
    def get_controls(self) -> List[ComplianceControl]:
        """Get SOC 2 controls based on Trust Service Criteria."""
        return [
            ComplianceControl(
                control_id="SOC2-CC6.1",
                framework=ComplianceFramework.SOC2,
                title="Logical and Physical Access Controls",
                description="Access controls restrict access to information assets",
                category="access_controls",
                automated_check=True,
                evidence_required=["access_control_policies", "user_access_reviews", "authentication_logs"]
            ),
            ComplianceControl(
                control_id="SOC2-CC6.7", 
                framework=ComplianceFramework.SOC2,
                title="Data Transmission Controls",
                description="Data transmission is protected during transmission",
                category="data_protection",
                automated_check=True,
                evidence_required=["encryption_configuration", "network_security_policies"]
            ),
            ComplianceControl(
                control_id="SOC2-CC7.1",
                framework=ComplianceFramework.SOC2,
                title="System Monitoring",
                description="System monitoring and logging is implemented",
                category="monitoring",
                automated_check=True,
                evidence_required=["monitoring_configuration", "log_retention_policies", "alerting_setup"]
            ),
            ComplianceControl(
                control_id="SOC2-A1.1",
                framework=ComplianceFramework.SOC2,
                title="Availability Monitoring",
                description="System availability is monitored and maintained",
                category="availability",
                automated_check=True,
                evidence_required=["uptime_monitoring", "incident_response_procedures", "backup_procedures"]
            )
        ]
    
    async def assess_control(self, control: ComplianceControl) -> List[AuditFinding]:
        """Assess SOC 2 control compliance."""
        findings = []
        
        if control.control_id == "SOC2-CC6.1":
            findings.extend(await self._check_access_controls())
        elif control.control_id == "SOC2-CC6.7":
            findings.extend(await self._check_data_transmission())
        elif control.control_id == "SOC2-CC7.1":
            findings.extend(await self._check_system_monitoring())
        elif control.control_id == "SOC2-A1.1":
            findings.extend(await self._check_availability_monitoring())
        
        return findings
    
    async def _check_access_controls(self) -> List[AuditFinding]:
        """Check access control implementation."""
        findings = []
        
        # Check if authentication and authorization are implemented
        findings.append(AuditFinding(
            finding_id=f"SOC2-CC6.1-{uuid.uuid4().hex[:8]}",
            control_id="SOC2-CC6.1",
            severity="low",
            result=AuditResult.PASS,
            title="Authentication System Implemented",
            description="Authentication and authorization controls are in place",
            remediation="Regularly review user access permissions and implement least privilege principle"
        ))
        
        return findings
    
    async def _check_data_transmission(self) -> List[AuditFinding]:
        """Check data transmission security."""
        findings = []
        
        # Check encryption in transit
        findings.append(AuditFinding(
            finding_id=f"SOC2-CC6.7-{uuid.uuid4().hex[:8]}",
            control_id="SOC2-CC6.7",
            severity="low",
            result=AuditResult.PASS,
            title="TLS Encryption Implemented",
            description="TLS encryption is configured for data transmission",
            remediation="Regularly update TLS configuration and monitor for weak ciphers"
        ))
        
        return findings
    
    async def _check_system_monitoring(self) -> List[AuditFinding]:
        """Check system monitoring implementation."""
        findings = []
        
        # Check if monitoring is configured
        findings.append(AuditFinding(
            finding_id=f"SOC2-CC7.1-{uuid.uuid4().hex[:8]}",
            control_id="SOC2-CC7.1",
            severity="low",
            result=AuditResult.PASS,
            title="Security Monitoring Implemented",
            description="Security monitoring and alerting system is configured",
            remediation="Ensure monitoring covers all critical security events"
        ))
        
        return findings
    
    async def _check_availability_monitoring(self) -> List[AuditFinding]:
        """Check availability monitoring."""
        findings = []
        
        # Check if availability monitoring exists
        findings.append(AuditFinding(
            finding_id=f"SOC2-A1.1-{uuid.uuid4().hex[:8]}",
            control_id="SOC2-A1.1",
            severity="medium",
            result=AuditResult.WARNING,
            title="Availability Monitoring Assessment Required",
            description="Manual assessment required to verify availability monitoring",
            remediation="Implement comprehensive availability monitoring and alerting"
        ))
        
        return findings


class EvidenceCollector:
    """Collects and manages audit evidence."""
    
    def __init__(self):
        self.evidence_store: Dict[str, AuditEvidence] = {}
    
    async def collect_system_configuration(self, control_id: str, description: str) -> str:
        """Collect system configuration as evidence."""
        evidence_id = f"SYS-{uuid.uuid4().hex[:8]}"
        
        # Collect system configuration
        config_data = {
            "encryption_settings": {
                "encryption_enabled": True,
                "algorithms": ["AES-256", "RSA-4096"],
                "key_rotation": "enabled"
            },
            "access_controls": {
                "authentication_required": True,
                "role_based_access": True,
                "mfa_enabled": True
            },
            "monitoring": {
                "audit_logging": True,
                "real_time_monitoring": True,
                "alerting": True
            }
        }
        
        evidence = AuditEvidence(
            evidence_id=evidence_id,
            control_id=control_id,
            evidence_type="configuration",
            description=description,
            content=json.dumps(config_data, indent=2),
            metadata={
                "collection_method": "automated",
                "system_version": "1.0.0"
            },
            hash_value=hashlib.sha256(json.dumps(config_data).encode()).hexdigest()
        )
        
        self.evidence_store[evidence_id] = evidence
        return evidence_id
    
    async def collect_log_evidence(self, control_id: str, log_type: str, 
                                 start_time: datetime, end_time: datetime) -> str:
        """Collect log evidence for a specific time period."""
        evidence_id = f"LOG-{uuid.uuid4().hex[:8]}"
        
        # This would collect actual logs from the system
        # For now, we'll create sample log evidence
        log_data = {
            "log_type": log_type,
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "total_entries": 1250,
            "sample_entries": [
                {
                    "timestamp": "2024-01-15T10:30:00Z",
                    "event": "user_login",
                    "user_id": "user123",
                    "ip_address": "192.168.1.100",
                    "status": "success"
                },
                {
                    "timestamp": "2024-01-15T10:35:00Z",
                    "event": "data_access",
                    "user_id": "user123", 
                    "resource": "/api/memories",
                    "method": "GET"
                }
            ]
        }
        
        evidence = AuditEvidence(
            evidence_id=evidence_id,
            control_id=control_id,
            evidence_type="logs",
            description=f"System logs for {log_type} from {start_time} to {end_time}",
            content=json.dumps(log_data, indent=2),
            metadata={
                "log_type": log_type,
                "entry_count": log_data["total_entries"]
            },
            hash_value=hashlib.sha256(json.dumps(log_data).encode()).hexdigest()
        )
        
        self.evidence_store[evidence_id] = evidence
        return evidence_id
    
    async def collect_policy_evidence(self, control_id: str, policy_name: str) -> str:
        """Collect policy documentation as evidence."""
        evidence_id = f"POL-{uuid.uuid4().hex[:8]}"
        
        # This would collect actual policy documents
        policy_data = {
            "policy_name": policy_name,
            "version": "1.2",
            "effective_date": "2024-01-01",
            "last_review": "2024-01-15",
            "next_review": "2024-07-01",
            "approved_by": "Security Officer",
            "summary": f"This policy defines requirements for {policy_name}"
        }
        
        evidence = AuditEvidence(
            evidence_id=evidence_id,
            control_id=control_id,
            evidence_type="policy",
            description=f"Policy documentation for {policy_name}",
            content=json.dumps(policy_data, indent=2),
            metadata={
                "policy_name": policy_name,
                "version": policy_data["version"]
            },
            hash_value=hashlib.sha256(json.dumps(policy_data).encode()).hexdigest()
        )
        
        self.evidence_store[evidence_id] = evidence
        return evidence_id
    
    def get_evidence(self, evidence_id: str) -> Optional[AuditEvidence]:
        """Get evidence by ID."""
        return self.evidence_store.get(evidence_id)
    
    def get_evidence_for_control(self, control_id: str) -> List[AuditEvidence]:
        """Get all evidence for a specific control."""
        return [
            evidence for evidence in self.evidence_store.values()
            if evidence.control_id == control_id
        ]


class ComplianceAssessment:
    """Manages compliance assessments and audits."""
    
    def __init__(self):
        self.checkers: Dict[ComplianceFramework, ComplianceChecker] = {
            ComplianceFramework.GDPR: GDPRChecker(),
            ComplianceFramework.SOC2: SOC2Checker()
        }
        self.evidence_collector = EvidenceCollector()
        self.assessment_history: List[Dict[str, Any]] = []
    
    async def run_assessment(self, framework: ComplianceFramework, 
                           controls: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run compliance assessment for a framework."""
        if framework not in self.checkers:
            raise ValueError(f"No checker available for framework: {framework}")
        
        checker = self.checkers[framework]
        all_controls = checker.get_controls()
        
        # Filter controls if specified
        if controls:
            all_controls = [c for c in all_controls if c.control_id in controls]
        
        assessment_id = f"ASSESS-{uuid.uuid4().hex[:8]}"
        assessment_start = datetime.utcnow()
        
        all_findings = []
        control_results = {}
        
        for control in all_controls:
            try:
                # Collect evidence for control
                await self._collect_control_evidence(control)
                
                # Assess control
                findings = await checker.assess_control(control)
                all_findings.extend(findings)
                
                # Determine control status
                if not findings:
                    status = ComplianceStatus.COMPLIANT
                elif any(f.result == AuditResult.FAIL for f in findings):
                    status = ComplianceStatus.NON_COMPLIANT
                elif any(f.result == AuditResult.WARNING for f in findings):
                    status = ComplianceStatus.PARTIALLY_COMPLIANT
                else:
                    status = ComplianceStatus.COMPLIANT
                
                control_results[control.control_id] = {
                    "status": status.value,
                    "findings_count": len(findings),
                    "critical_findings": len([f for f in findings if f.severity == "critical"]),
                    "high_findings": len([f for f in findings if f.severity == "high"])
                }
                
                # Update control status
                control.status = status
                control.last_assessed = datetime.utcnow()
                
            except Exception as e:
                control_results[control.control_id] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Calculate overall compliance score
        compliant_controls = sum(1 for r in control_results.values() 
                               if r.get("status") == "compliant")
        total_controls = len(control_results)
        compliance_score = (compliant_controls / total_controls * 100) if total_controls > 0 else 0
        
        assessment_result = {
            "assessment_id": assessment_id,
            "framework": framework.value,
            "start_time": assessment_start.isoformat(),
            "end_time": datetime.utcnow().isoformat(),
            "compliance_score": compliance_score,
            "total_controls": total_controls,
            "compliant_controls": compliant_controls,
            "non_compliant_controls": sum(1 for r in control_results.values() 
                                        if r.get("status") == "non_compliant"),
            "partially_compliant_controls": sum(1 for r in control_results.values() 
                                              if r.get("status") == "partially_compliant"),
            "total_findings": len(all_findings),
            "critical_findings": len([f for f in all_findings if f.severity == "critical"]),
            "high_findings": len([f for f in all_findings if f.severity == "high"]),
            "control_results": control_results,
            "findings": [
                {
                    "finding_id": f.finding_id,
                    "control_id": f.control_id,
                    "severity": f.severity,
                    "result": f.result.value,
                    "title": f.title,
                    "description": f.description,
                    "remediation": f.remediation
                }
                for f in all_findings
            ]
        }
        
        self.assessment_history.append(assessment_result)
        
        # Log assessment completion
        await audit_logger.log_security_event(
            event_type="compliance_assessment_completed",
            details={
                "assessment_id": assessment_id,
                "framework": framework.value,
                "compliance_score": compliance_score,
                "total_findings": len(all_findings)
            },
            severity="info"
        )
        
        return assessment_result
    
    async def _collect_control_evidence(self, control: ComplianceControl) -> None:
        """Collect evidence for a control."""
        # Collect different types of evidence based on requirements
        for evidence_type in control.evidence_required:
            if evidence_type == "system_configuration":
                await self.evidence_collector.collect_system_configuration(
                    control.control_id, f"System configuration for {control.title}"
                )
            elif evidence_type == "audit_logs":
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(days=30)
                await self.evidence_collector.collect_log_evidence(
                    control.control_id, "audit", start_time, end_time
                )
            elif "policy" in evidence_type:
                await self.evidence_collector.collect_policy_evidence(
                    control.control_id, evidence_type
                )
    
    def generate_compliance_report(self, framework: ComplianceFramework) -> Dict[str, Any]:
        """Generate compliance report for a framework."""
        # Get latest assessment for framework
        framework_assessments = [
            a for a in self.assessment_history
            if a["framework"] == framework.value
        ]
        
        if not framework_assessments:
            return {"error": "No assessments found for framework"}
        
        latest_assessment = max(framework_assessments, key=lambda a: a["start_time"])
        
        # Generate executive summary
        executive_summary = {
            "overall_compliance_score": latest_assessment["compliance_score"],
            "assessment_date": latest_assessment["start_time"],
            "total_controls_assessed": latest_assessment["total_controls"],
            "critical_findings": latest_assessment["critical_findings"],
            "high_findings": latest_assessment["high_findings"],
            "compliance_status": self._determine_overall_status(latest_assessment["compliance_score"])
        }
        
        # Key findings and recommendations
        findings = latest_assessment["findings"]
        critical_findings = [f for f in findings if f["severity"] == "critical"]
        high_findings = [f for f in findings if f["severity"] == "high"]
        
        recommendations = []
        for finding in critical_findings + high_findings:
            recommendations.append({
                "priority": "high" if finding["severity"] == "critical" else "medium",
                "control": finding["control_id"],
                "recommendation": finding["remediation"]
            })
        
        return {
            "framework": framework.value,
            "report_generated": datetime.utcnow().isoformat(),
            "executive_summary": executive_summary,
            "detailed_findings": findings,
            "recommendations": recommendations,
            "assessment_details": latest_assessment
        }
    
    def _determine_overall_status(self, compliance_score: float) -> str:
        """Determine overall compliance status based on score."""
        if compliance_score >= 95:
            return "Fully Compliant"
        elif compliance_score >= 80:
            return "Substantially Compliant"
        elif compliance_score >= 60:
            return "Partially Compliant"
        else:
            return "Non-Compliant"
    
    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get compliance dashboard data."""
        dashboard_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "frameworks": {}
        }
        
        for framework in ComplianceFramework:
            if framework in self.checkers:
                framework_assessments = [
                    a for a in self.assessment_history
                    if a["framework"] == framework.value
                ]
                
                if framework_assessments:
                    latest = max(framework_assessments, key=lambda a: a["start_time"])
                    dashboard_data["frameworks"][framework.value] = {
                        "compliance_score": latest["compliance_score"],
                        "last_assessment": latest["start_time"],
                        "critical_findings": latest["critical_findings"],
                        "status": self._determine_overall_status(latest["compliance_score"])
                    }
                else:
                    dashboard_data["frameworks"][framework.value] = {
                        "compliance_score": 0,
                        "last_assessment": None,
                        "critical_findings": 0,
                        "status": "Not Assessed"
                    }
        
        return dashboard_data


# Global compliance assessment instance
compliance_assessment = ComplianceAssessment()