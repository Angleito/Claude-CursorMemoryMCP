"""GDPR Compliance Implementation for Memory Vector Database."""

import uuid
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from database.secure_supabase import SecureSupabaseClient
from monitoring.audit_logger import audit_logger


class DataProcessingPurpose(str, Enum):
    """Legal basis for data processing under GDPR."""

    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


class DataCategory(str, Enum):
    """Categories of personal data."""

    IDENTITY = "identity"  # Name, username, ID numbers
    CONTACT = "contact"  # Email, phone, address
    TECHNICAL = "technical"  # IP addresses, device info, logs
    BEHAVIORAL = "behavioral"  # Usage patterns, preferences
    BIOMETRIC = "biometric"  # Biometric identifiers
    SPECIAL = "special"  # Health, political, religious data


@dataclass
class DataProcessingRecord:
    """Record of data processing activities."""

    id: uuid.UUID
    user_id: uuid.UUID
    purpose: DataProcessingPurpose
    legal_basis: str
    data_categories: List[DataCategory]
    recipients: List[str]
    retention_period: timedelta
    created_at: datetime
    consent_given: bool = False
    consent_withdrawn: Optional[datetime] = None


@dataclass
class DataSubjectRequest:
    """Data subject request (access, rectification, deletion, etc.)."""

    id: uuid.UUID
    user_id: uuid.UUID
    request_type: str  # access, rectification, deletion, portability, restriction
    status: str  # pending, processing, completed, rejected
    requested_at: datetime
    completed_at: Optional[datetime] = None
    data_provided: Optional[Dict[str, Any]] = None
    rejection_reason: Optional[str] = None


class GDPRComplianceManager:
    """GDPR compliance management system."""

    def __init__(self, db_client: SecureSupabaseClient):
        self.db = db_client

        # Data retention periods (in days)
        self.retention_periods = {
            DataCategory.IDENTITY: 365 * 7,  # 7 years
            DataCategory.CONTACT: 365 * 3,  # 3 years after last contact
            DataCategory.TECHNICAL: 365,  # 1 year
            DataCategory.BEHAVIORAL: 365 * 2,  # 2 years
            DataCategory.BIOMETRIC: 365,  # 1 year
            DataCategory.SPECIAL: 365 * 10,  # 10 years (if legally required)
        }

        # Default data processing purposes
        self.default_processing_purposes = [
            {
                "purpose": DataProcessingPurpose.CONTRACT,
                "legal_basis": "Performance of contract for memory storage services",
                "data_categories": [
                    DataCategory.IDENTITY,
                    DataCategory.CONTACT,
                    DataCategory.TECHNICAL,
                ],
                "recipients": ["Memory Database Service"],
                "retention_period": timedelta(days=365 * 3),
            },
            {
                "purpose": DataProcessingPurpose.LEGITIMATE_INTERESTS,
                "legal_basis": "Security monitoring and fraud prevention",
                "data_categories": [DataCategory.TECHNICAL, DataCategory.BEHAVIORAL],
                "recipients": ["Security Team", "System Administrators"],
                "retention_period": timedelta(days=365),
            },
        ]

    async def initialize(self):
        """Initialize GDPR compliance tables."""
        await self._create_compliance_tables()

    async def _create_compliance_tables(self):
        """Create GDPR compliance database tables."""
        # Data processing records
        await self.db.client.rpc(
            "create_table_if_not_exists",
            {
                "table_name": "data_processing_records",
                "table_schema": """
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                purpose VARCHAR(50) NOT NULL,
                legal_basis TEXT NOT NULL,
                data_categories JSONB NOT NULL,
                recipients JSONB NOT NULL,
                retention_period_days INTEGER NOT NULL,
                consent_given BOOLEAN DEFAULT FALSE,
                consent_withdrawn_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            """,
            },
        )

        # Data subject requests
        await self.db.client.rpc(
            "create_table_if_not_exists",
            {
                "table_name": "data_subject_requests",
                "table_schema": """
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                request_type VARCHAR(20) NOT NULL,
                status VARCHAR(20) DEFAULT 'pending',
                requested_at TIMESTAMPTZ DEFAULT NOW(),
                completed_at TIMESTAMPTZ,
                data_provided JSONB,
                rejection_reason TEXT,
                processed_by UUID REFERENCES users(id)
            """,
            },
        )

        # Consent history
        await self.db.client.rpc(
            "create_table_if_not_exists",
            {
                "table_name": "consent_history",
                "table_schema": """
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                purpose VARCHAR(50) NOT NULL,
                consent_given BOOLEAN NOT NULL,
                consent_text TEXT,
                ip_address INET,
                user_agent TEXT,
                timestamp TIMESTAMPTZ DEFAULT NOW()
            """,
            },
        )

        # Data retention schedule
        await self.db.client.rpc(
            "create_table_if_not_exists",
            {
                "table_name": "data_retention_schedule",
                "table_schema": """
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                data_category VARCHAR(20) NOT NULL,
                retention_until TIMESTAMPTZ NOT NULL,
                deletion_scheduled BOOLEAN DEFAULT FALSE,
                deleted_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ DEFAULT NOW()
            """,
            },
        )

    # Consent Management
    async def record_consent(
        self,
        user_id: uuid.UUID,
        purpose: DataProcessingPurpose,
        consent_given: bool,
        consent_text: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ):
        """Record user consent for data processing."""
        consent_record = {
            "user_id": str(user_id),
            "purpose": purpose.value,
            "consent_given": consent_given,
            "consent_text": consent_text,
            "ip_address": ip_address,
            "user_agent": user_agent,
        }

        await self.db.secure_insert("consent_history", consent_record)

        # Update processing record if exists
        processing_records = await self.db.secure_select(
            "data_processing_records",
            {"user_id": str(user_id), "purpose": purpose.value},
        )

        if processing_records:
            update_data = {
                "consent_given": consent_given,
                "consent_withdrawn_at": (
                    datetime.utcnow().isoformat() if not consent_given else None
                ),
            }
            await self.db.secure_update(
                "data_processing_records", processing_records[0]["id"], update_data
            )

        # Log consent action
        await audit_logger.log_event(
            action="consent_recorded",
            resource="gdpr_consent",
            user_id=user_id,
            ip_address=ip_address,
            details={"purpose": purpose.value, "consent_given": consent_given},
        )

    async def get_user_consents(self, user_id: uuid.UUID) -> List[Dict[str, Any]]:
        """Get all consents for a user."""
        return await self.db.secure_select("consent_history", {"user_id": str(user_id)})

    async def withdraw_consent(
        self, user_id: uuid.UUID, purpose: DataProcessingPurpose, ip_address: Optional[str] = None
    ):
        """Withdraw consent for data processing."""
        await self.record_consent(
            user_id=user_id,
            purpose=purpose,
            consent_given=False,
            consent_text="Consent withdrawn by user",
            ip_address=ip_address,
        )

        # Schedule data deletion if consent was the only legal basis
        await self._schedule_data_deletion_on_consent_withdrawal(user_id, purpose)

    # Data Subject Rights
    async def create_data_subject_request(
        self, user_id: uuid.UUID, request_type: str, details: Optional[Dict[str, Any]] = None
    ) -> uuid.UUID:
        """Create a data subject request."""
        request_data = {
            "user_id": str(user_id),
            "request_type": request_type,
            "status": "pending",
        }

        if details:
            request_data["details"] = details

        result = await self.db.secure_insert("data_subject_requests", request_data)
        request_id = result["id"]

        # Log request creation
        await audit_logger.log_event(
            action="data_subject_request_created",
            resource="gdpr_request",
            resource_id=str(request_id),
            user_id=user_id,
            details={"request_type": request_type},
        )

        # Auto-process certain types of requests
        if request_type in ["access", "portability"]:
            await self._auto_process_data_request(request_id, user_id, request_type)

        return request_id

    async def process_access_request(self, user_id: uuid.UUID) -> Dict[str, Any]:
        """Process data access request (Article 15)."""
        user_data = {}

        # Collect all user data
        tables_to_export = [
            "users",
            "memories",
            "api_keys",
            "sessions",
            "audit_logs",
            "data_processing_records",
            "consent_history",
        ]

        for table in tables_to_export:
            try:
                data = await self.db.secure_select(table, {"user_id": str(user_id)})
                if data:
                    user_data[table] = data
            except Exception as e:
                # Log error but continue
                await audit_logger.log_event(
                    action="data_export_error",
                    resource=f"table.{table}",
                    user_id=user_id,
                    details={"error": str(e)},
                )

        # Add processing information
        user_data["data_processing_info"] = {
            "purposes": [
                record["purpose"]
                for record in user_data.get("data_processing_records", [])
            ],
            "retention_periods": self.retention_periods,
            "last_updated": datetime.utcnow().isoformat(),
        }

        return user_data

    async def process_deletion_request(
        self, user_id: uuid.UUID, verified: bool = False
    ):
        """Process data deletion request (Article 17 - Right to be forgotten)."""
        if not verified:
            raise ValueError("Deletion request must be verified before processing")

        # Check if deletion is allowed (no legal obligations to retain)
        can_delete = await self._can_delete_user_data(user_id)
        if not can_delete:
            raise ValueError("Cannot delete data due to legal obligations")

        # Perform cascading deletion
        deletion_results = {}
        tables_to_delete = [
            "consent_history",
            "data_subject_requests",
            "data_processing_records",
            "audit_logs",
            "sessions",
            "api_keys",
            "memories",
        ]

        for table in tables_to_delete:
            try:
                records = await self.db.secure_select(table, {"user_id": str(user_id)})
                for record in records:
                    await self.db.secure_delete(table, record["id"])
                deletion_results[table] = len(records)
            except Exception as e:
                deletion_results[f"{table}_error"] = str(e)

        # Anonymize user record instead of deleting (for audit trail)
        await self._anonymize_user_record(user_id)

        # Log deletion
        await audit_logger.log_event(
            action="user_data_deleted",
            resource="user_account",
            user_id=user_id,
            details={"deletion_results": deletion_results, "anonymized": True},
        )

        return deletion_results

    async def process_portability_request(self, user_id: uuid.UUID) -> bytes:
        """Process data portability request (Article 20)."""
        user_data = await self.process_access_request(user_id)

        # Convert to JSON and return as bytes
        import json

        json_data = json.dumps(user_data, indent=2, default=str)

        # Log portability request
        await audit_logger.log_event(
            action="data_portability_export",
            resource="user_data",
            user_id=user_id,
            details={"data_size_bytes": len(json_data.encode())},
        )

        return json_data.encode()

    # Data Retention Management
    async def setup_user_data_retention(self, user_id: uuid.UUID):
        """Set up data retention schedule for a new user."""
        for category, retention_days in self.retention_periods.items():
            retention_record = {
                "user_id": str(user_id),
                "data_category": category.value,
                "retention_until": (
                    datetime.utcnow() + timedelta(days=retention_days)
                ).isoformat(),
            }
            await self.db.secure_insert("data_retention_schedule", retention_record)

    async def cleanup_expired_data(self) -> Dict[str, int]:
        """Clean up expired user data (automated retention)."""
        cleanup_results = {}

        # Get expired data records
        expired_records = (
            await self.db.client.table("data_retention_schedule")
            .select("*")
            .lte("retention_until", datetime.utcnow().isoformat())
            .eq("deletion_scheduled", False)
            .execute()
        )

        for record in expired_records.data:
            user_id = record["user_id"]
            category = record["data_category"]

            try:
                # Delete data based on category
                deleted_count = await self._delete_data_by_category(user_id, category)
                cleanup_results[f"{category}_deleted"] = deleted_count

                # Mark as deleted
                await self.db.secure_update(
                    "data_retention_schedule",
                    record["id"],
                    {
                        "deletion_scheduled": True,
                        "deleted_at": datetime.utcnow().isoformat(),
                    },
                )

                # Log cleanup
                await audit_logger.log_event(
                    action="automated_data_cleanup",
                    resource="data_retention",
                    user_id=uuid.UUID(user_id),
                    details={"category": category, "deleted_count": deleted_count},
                )

            except Exception as e:
                cleanup_results[f"{category}_error"] = str(e)

        return cleanup_results

    # Privacy Impact Assessment
    async def generate_privacy_impact_assessment(self) -> Dict[str, Any]:
        """Generate privacy impact assessment report."""
        # Data processing statistics
        processing_stats = await self.db.client.rpc("get_processing_statistics")

        # Consent statistics
        consent_stats = await self.db.client.rpc("get_consent_statistics")

        # Request statistics
        request_stats = await self.db.client.rpc("get_request_statistics")

        pia_report = {
            "generated_at": datetime.utcnow().isoformat(),
            "data_processing": {
                "active_users": processing_stats.get("active_users", 0),
                "total_processing_records": processing_stats.get("total_records", 0),
                "consent_rate": consent_stats.get("consent_rate", 0),
                "purposes": processing_stats.get("purposes", []),
            },
            "data_subject_rights": {
                "total_requests": request_stats.get("total_requests", 0),
                "access_requests": request_stats.get("access_requests", 0),
                "deletion_requests": request_stats.get("deletion_requests", 0),
                "average_response_time_hours": request_stats.get(
                    "avg_response_time", 0
                ),
            },
            "data_retention": {
                "retention_policies": len(self.retention_periods),
                "expired_data_cleaned": request_stats.get("expired_cleaned", 0),
            },
            "security_measures": {
                "encryption_enabled": True,
                "access_controls": True,
                "audit_logging": True,
                "pseudonymization": True,
            },
            "recommendations": await self._generate_privacy_recommendations(),
        }

        return pia_report

    # Helper methods
    async def _can_delete_user_data(self, user_id: uuid.UUID) -> bool:
        """Check if user data can be deleted."""
        # Check for legal obligations to retain data
        legal_holds = await self.db.secure_select(
            "legal_holds", {"user_id": str(user_id), "active": True}
        )

        return len(legal_holds) == 0

    async def _anonymize_user_record(self, user_id: uuid.UUID):
        """Anonymize user record while preserving audit trail."""
        anonymized_data = {
            "email": f"deleted-user-{user_id}@anonymized.local",
            "username": f"deleted-{user_id}",
            "full_name": "DELETED USER",
            "status": "deleted",
            "data_processing_consent": False,
            "anonymized_at": datetime.utcnow().isoformat(),
        }

        await self.db.secure_update("users", str(user_id), anonymized_data)

    async def _delete_data_by_category(self, user_id: str, category: str) -> int:
        """Delete user data by category."""
        deleted_count = 0

        category_mappings = {
            DataCategory.TECHNICAL.value: ["audit_logs", "sessions"],
            DataCategory.BEHAVIORAL.value: ["usage_analytics"],
            DataCategory.CONTACT.value: [],  # Handled in user record anonymization
        }

        tables = category_mappings.get(category, [])
        for table in tables:
            records = await self.db.secure_select(table, {"user_id": user_id})
            for record in records:
                await self.db.secure_delete(table, record["id"])
                deleted_count += 1

        return deleted_count

    async def _schedule_data_deletion_on_consent_withdrawal(
        self, user_id: uuid.UUID, purpose: DataProcessingPurpose
    ):
        """Schedule data deletion when consent is withdrawn."""
        # Check if there are other legal bases for processing
        other_legal_bases = await self.db.secure_select(
            "data_processing_records",
            {
                "user_id": str(user_id),
                "purpose": {"neq": purpose.value},
                "consent_given": True,
            },
        )

        if not other_legal_bases:
            # No other legal basis, schedule deletion
            await self.create_data_subject_request(
                user_id=user_id,
                request_type="deletion",
                details={"reason": "consent_withdrawn", "purpose": purpose.value},
            )

    async def _auto_process_data_request(
        self, request_id: uuid.UUID, user_id: uuid.UUID, request_type: str
    ):
        """Auto-process certain types of data requests."""
        try:
            if request_type == "access":
                data = await self.process_access_request(user_id)
                update_data = {
                    "status": "completed",
                    "completed_at": datetime.utcnow().isoformat(),
                    "data_provided": data,
                }
            elif request_type == "portability":
                data_bytes = await self.process_portability_request(user_id)
                update_data = {
                    "status": "completed",
                    "completed_at": datetime.utcnow().isoformat(),
                    "data_provided": {
                        "download_available": True,
                        "size_bytes": len(data_bytes),
                    },
                }
            else:
                return  # Manual processing required

            await self.db.secure_update(
                "data_subject_requests", str(request_id), update_data
            )

        except Exception as e:
            # Mark as failed
            await self.db.secure_update(
                "data_subject_requests",
                str(request_id),
                {"status": "failed", "rejection_reason": str(e)},
            )

    async def _generate_privacy_recommendations(self) -> List[str]:
        """Generate privacy improvement recommendations."""
        recommendations = []

        # Check consent rates
        consent_stats = await self.db.client.rpc("get_consent_statistics")
        if consent_stats.get("consent_rate", 1.0) < 0.8:
            recommendations.append(
                "Improve consent mechanisms - current consent rate is below 80%"
            )

        # Check data retention
        expired_data = (
            await self.db.client.table("data_retention_schedule")
            .select("count")
            .lte("retention_until", datetime.utcnow().isoformat())
            .eq("deletion_scheduled", False)
            .execute()
        )

        if expired_data.count > 100:
            recommendations.append(
                "Schedule cleanup of expired data - large backlog detected"
            )

        # Default recommendations
        recommendations.extend(
            [
                "Regularly review and update privacy notices",
                "Conduct staff training on GDPR requirements",
                "Implement privacy by design in new features",
                "Consider appointing a Data Protection Officer",
                "Review third-party data sharing agreements",
            ]
        )

        return recommendations


# Global GDPR compliance manager
async def get_gdpr_manager(db_client: SecureSupabaseClient) -> GDPRComplianceManager:
    """Get GDPR compliance manager instance."""
    manager = GDPRComplianceManager(db_client)
    await manager.initialize()
    return manager
