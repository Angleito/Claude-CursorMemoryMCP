"""Data protection and privacy controls including PII detection, data masking, and retention policies."""

import hashlib
import json
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from cryptography.fernet import Fernet
from pydantic import BaseModel, Field

from config.settings import get_settings
from monitoring.audit_logger import audit_logger
from security.encryption import encryption_manager

settings = get_settings()


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class PIIType(Enum):
    """Types of Personally Identifiable Information."""
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    EMAIL = "email"
    PHONE = "phone"
    IP_ADDRESS = "ip_address"
    DRIVER_LICENSE = "driver_license"
    PASSPORT = "passport"
    DATE_OF_BIRTH = "date_of_birth"
    ADDRESS = "address"
    NAME = "name"
    BANK_ACCOUNT = "bank_account"
    MEDICAL_ID = "medical_id"


@dataclass
class PIIMatch:
    """Represents a detected PII match in data."""
    pii_type: PIIType
    value: str
    start_pos: int
    end_pos: int
    confidence: float
    context: str = ""


@dataclass
class DataRetentionPolicy:
    """Data retention policy configuration."""
    policy_id: str
    name: str
    data_types: List[str]
    retention_period_days: int
    auto_delete: bool = True
    archive_before_delete: bool = True
    notification_before_days: int = 30
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DataProcessingRecord:
    """Record of data processing activities (GDPR Article 30)."""
    record_id: str
    processing_purpose: str
    data_categories: List[str]
    data_subjects: List[str]
    recipients: List[str]
    legal_basis: str
    retention_period: str
    security_measures: List[str]
    controller: str
    processor: Optional[str] = None
    third_country_transfers: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


class PIIDetector:
    """Advanced PII detection using patterns and machine learning."""
    
    def __init__(self):
        self.patterns = self._load_pii_patterns()
        self.custom_patterns = {}
        
    def _load_pii_patterns(self) -> Dict[PIIType, List[Dict[str, Any]]]:
        """Load PII detection patterns."""
        return {
            PIIType.SSN: [
                {
                    "pattern": r"\b\d{3}-?\d{2}-?\d{4}\b",
                    "confidence": 0.9,
                    "validator": self._validate_ssn
                }
            ],
            PIIType.CREDIT_CARD: [
                {
                    "pattern": r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
                    "confidence": 0.95,
                    "validator": self._validate_credit_card
                }
            ],
            PIIType.EMAIL: [
                {
                    "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                    "confidence": 0.98,
                    "validator": None
                }
            ],
            PIIType.PHONE: [
                {
                    "pattern": r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
                    "confidence": 0.85,
                    "validator": None
                }
            ],
            PIIType.IP_ADDRESS: [
                {
                    "pattern": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
                    "confidence": 0.9,
                    "validator": self._validate_ip_address
                }
            ],
            PIIType.DRIVER_LICENSE: [
                {
                    "pattern": r"\b[A-Z]{1,2}[0-9]{6,8}\b",
                    "confidence": 0.7,
                    "validator": None
                }
            ],
            PIIType.DATE_OF_BIRTH: [
                {
                    "pattern": r"\b(?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12][0-9]|3[01])[-/](?:19|20)[0-9]{2}\b",
                    "confidence": 0.8,
                    "validator": None
                }
            ],
            PIIType.BANK_ACCOUNT: [
                {
                    "pattern": r"\b[0-9]{8,17}\b",
                    "confidence": 0.6,
                    "validator": None
                }
            ]
        }
    
    def detect_pii(self, text: str, context: str = "") -> List[PIIMatch]:
        """Detect PII in the given text."""
        matches = []
        
        for pii_type, pattern_configs in self.patterns.items():
            for config in pattern_configs:
                pattern = config["pattern"]
                confidence = config["confidence"]
                validator = config.get("validator")
                
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    value = match.group()
                    
                    # Apply validator if available
                    if validator and not validator(value):
                        continue
                    
                    matches.append(PIIMatch(
                        pii_type=pii_type,
                        value=value,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=confidence,
                        context=context
                    ))
        
        # Check custom patterns
        for name, custom_pattern in self.custom_patterns.items():
            for match in re.finditer(custom_pattern["pattern"], text, re.IGNORECASE):
                matches.append(PIIMatch(
                    pii_type=PIIType.NAME,  # Default to NAME for custom patterns
                    value=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=custom_pattern.get("confidence", 0.5),
                    context=context
                ))
        
        return matches
    
    def add_custom_pattern(self, name: str, pattern: str, confidence: float = 0.5) -> None:
        """Add a custom PII detection pattern."""
        self.custom_patterns[name] = {
            "pattern": pattern,
            "confidence": confidence
        }
    
    def _validate_ssn(self, ssn: str) -> bool:
        """Validate SSN using basic rules."""
        # Remove separators
        clean_ssn = re.sub(r'[^0-9]', '', ssn)
        
        if len(clean_ssn) != 9:
            return False
        
        # Invalid SSNs
        invalid_ssns = {
            "000000000", "111111111", "222222222", "333333333",
            "444444444", "555555555", "666666666", "777777777",
            "888888888", "999999999"
        }
        
        return clean_ssn not in invalid_ssns
    
    def _validate_credit_card(self, cc: str) -> bool:
        """Validate credit card using Luhn algorithm."""
        # Remove spaces and dashes
        clean_cc = re.sub(r'[^0-9]', '', cc)
        
        # Luhn algorithm
        def luhn_check(card_num):
            total = 0
            reverse_digits = card_num[::-1]
            
            for i, char in enumerate(reverse_digits):
                digit = int(char)
                if i % 2 == 1:
                    digit *= 2
                    if digit > 9:
                        digit -= 9
                total += digit
            
            return total % 10 == 0
        
        return luhn_check(clean_cc)
    
    def _validate_ip_address(self, ip: str) -> bool:
        """Validate IP address format."""
        try:
            parts = ip.split('.')
            return (len(parts) == 4 and 
                    all(0 <= int(part) <= 255 for part in parts))
        except ValueError:
            return False


class DataMasker:
    """Data masking and anonymization utilities."""
    
    def __init__(self):
        self.masking_rules = {
            PIIType.SSN: self._mask_ssn,
            PIIType.CREDIT_CARD: self._mask_credit_card,
            PIIType.EMAIL: self._mask_email,
            PIIType.PHONE: self._mask_phone,
            PIIType.IP_ADDRESS: self._mask_ip,
            PIIType.NAME: self._mask_name,
        }
    
    def mask_pii_matches(self, text: str, pii_matches: List[PIIMatch], 
                        mask_char: str = "*") -> str:
        """Mask PII matches in text."""
        # Sort matches by position (reverse order to maintain positions)
        sorted_matches = sorted(pii_matches, key=lambda m: m.start_pos, reverse=True)
        
        result = text
        for match in sorted_matches:
            masked_value = self.mask_value(match.value, match.pii_type, mask_char)
            result = result[:match.start_pos] + masked_value + result[match.end_pos:]
        
        return result
    
    def mask_value(self, value: str, pii_type: PIIType, mask_char: str = "*") -> str:
        """Mask a specific value based on its PII type."""
        masker = self.masking_rules.get(pii_type, self._default_mask)
        return masker(value, mask_char)
    
    def _mask_ssn(self, ssn: str, mask_char: str = "*") -> str:
        """Mask SSN showing only last 4 digits."""
        clean_ssn = re.sub(r'[^0-9]', '', ssn)
        if len(clean_ssn) == 9:
            return f"{mask_char * 5}-{clean_ssn[-4:]}"
        return mask_char * len(ssn)
    
    def _mask_credit_card(self, cc: str, mask_char: str = "*") -> str:
        """Mask credit card showing only last 4 digits."""
        clean_cc = re.sub(r'[^0-9]', '', cc)
        if len(clean_cc) >= 13:
            return f"{mask_char * (len(clean_cc) - 4)}{clean_cc[-4:]}"
        return mask_char * len(cc)
    
    def _mask_email(self, email: str, mask_char: str = "*") -> str:
        """Mask email showing first char and domain."""
        parts = email.split('@')
        if len(parts) == 2:
            username, domain = parts
            if len(username) > 1:
                masked_username = username[0] + mask_char * (len(username) - 1)
            else:
                masked_username = mask_char
            return f"{masked_username}@{domain}"
        return mask_char * len(email)
    
    def _mask_phone(self, phone: str, mask_char: str = "*") -> str:
        """Mask phone number showing only last 4 digits."""
        digits = re.sub(r'[^0-9]', '', phone)
        if len(digits) >= 10:
            return f"({mask_char * 3}) {mask_char * 3}-{digits[-4:]}"
        return mask_char * len(phone)
    
    def _mask_ip(self, ip: str, mask_char: str = "*") -> str:
        """Mask IP address showing only first octet."""
        parts = ip.split('.')
        if len(parts) == 4:
            return f"{parts[0]}.{mask_char * 3}.{mask_char * 3}.{mask_char * 3}"
        return mask_char * len(ip)
    
    def _mask_name(self, name: str, mask_char: str = "*") -> str:
        """Mask name showing only first character."""
        if len(name) > 1:
            return name[0] + mask_char * (len(name) - 1)
        return mask_char
    
    def _default_mask(self, value: str, mask_char: str = "*") -> str:
        """Default masking for unknown PII types."""
        if len(value) <= 2:
            return mask_char * len(value)
        return value[0] + mask_char * (len(value) - 2) + value[-1]


class DataAnonymizer:
    """Advanced data anonymization techniques."""
    
    def __init__(self):
        self.pseudonym_mapping: Dict[str, str] = {}
        self.k_anonymity_processor = KAnonymityProcessor()
    
    def pseudonymize(self, identifier: str, salt: str = "") -> str:
        """Create consistent pseudonym for identifier."""
        # Use hash to create deterministic pseudonym
        combined = f"{identifier}{salt}"
        hash_value = hashlib.sha256(combined.encode()).hexdigest()
        
        # Create readable pseudonym
        pseudonym = f"user_{hash_value[:8]}"
        self.pseudonym_mapping[identifier] = pseudonym
        
        return pseudonym
    
    def anonymize_dataset(self, data: List[Dict[str, Any]], 
                         sensitive_attributes: List[str],
                         quasi_identifiers: List[str],
                         k: int = 5) -> List[Dict[str, Any]]:
        """Anonymize dataset using k-anonymity."""
        return self.k_anonymity_processor.process(
            data, sensitive_attributes, quasi_identifiers, k
        )
    
    def differential_privacy_noise(self, value: float, epsilon: float = 1.0, 
                                 sensitivity: float = 1.0) -> float:
        """Add differential privacy noise to numeric value."""
        import numpy as np
        
        # Laplace mechanism
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale)
        
        return value + noise


class KAnonymityProcessor:
    """K-anonymity implementation for dataset anonymization."""
    
    def process(self, data: List[Dict[str, Any]], 
               sensitive_attributes: List[str],
               quasi_identifiers: List[str], 
               k: int) -> List[Dict[str, Any]]:
        """Apply k-anonymity to dataset."""
        # Group records by quasi-identifier combinations
        groups = self._group_by_quasi_identifiers(data, quasi_identifiers)
        
        # Suppress or generalize groups with size < k
        anonymized_data = []
        
        for group_key, records in groups.items():
            if len(records) >= k:
                # Group satisfies k-anonymity
                anonymized_data.extend(records)
            else:
                # Apply generalization or suppression
                generalized_records = self._generalize_group(
                    records, quasi_identifiers, k
                )
                anonymized_data.extend(generalized_records)
        
        return anonymized_data
    
    def _group_by_quasi_identifiers(self, data: List[Dict[str, Any]], 
                                   quasi_identifiers: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Group records by quasi-identifier values."""
        groups = {}
        
        for record in data:
            # Create key from quasi-identifier values
            key_parts = []
            for qi in quasi_identifiers:
                key_parts.append(str(record.get(qi, "")))
            
            group_key = "|".join(key_parts)
            
            if group_key not in groups:
                groups[group_key] = []
            
            groups[group_key].append(record)
        
        return groups
    
    def _generalize_group(self, records: List[Dict[str, Any]], 
                         quasi_identifiers: List[str], k: int) -> List[Dict[str, Any]]:
        """Generalize small groups to satisfy k-anonymity."""
        # Simple generalization: replace specific values with ranges or "*"
        generalized_records = []
        
        for record in records:
            generalized_record = record.copy()
            
            for qi in quasi_identifiers:
                if qi in generalized_record:
                    # Replace with generalized value
                    generalized_record[qi] = self._generalize_value(
                        generalized_record[qi]
                    )
            
            generalized_records.append(generalized_record)
        
        return generalized_records
    
    def _generalize_value(self, value: Any) -> str:
        """Generalize a single value."""
        if isinstance(value, int):
            # Age generalization example
            if value < 20:
                return "< 20"
            elif value < 30:
                return "20-29"
            elif value < 40:
                return "30-39"
            elif value < 50:
                return "40-49"
            else:
                return "50+"
        elif isinstance(value, str):
            # String generalization
            if len(value) > 3:
                return value[:2] + "*"
            else:
                return "*"
        else:
            return "*"


class DataRetentionManager:
    """Manage data retention policies and automatic deletion."""
    
    def __init__(self):
        self.policies: Dict[str, DataRetentionPolicy] = {}
        self.scheduled_deletions: List[Dict[str, Any]] = []
    
    def add_policy(self, policy: DataRetentionPolicy) -> None:
        """Add a data retention policy."""
        self.policies[policy.policy_id] = policy
        
        # Log policy creation
        audit_logger.log_security_event(
            event_type="retention_policy_created",
            details={
                "policy_id": policy.policy_id,
                "name": policy.name,
                "retention_days": policy.retention_period_days
            },
            severity="info"
        )
    
    def check_retention_compliance(self, data_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check which records violate retention policies."""
        violations = []
        
        for record in data_records:
            # Determine applicable policies
            applicable_policies = self._get_applicable_policies(record)
            
            for policy in applicable_policies:
                created_date = record.get('created_at')
                if created_date:
                    if isinstance(created_date, str):
                        created_date = datetime.fromisoformat(created_date)
                    
                    age_days = (datetime.utcnow() - created_date).days
                    
                    if age_days > policy.retention_period_days:
                        violations.append({
                            "record_id": record.get("id"),
                            "policy_id": policy.policy_id,
                            "age_days": age_days,
                            "retention_days": policy.retention_period_days,
                            "action_required": "delete" if policy.auto_delete else "review"
                        })
        
        return violations
    
    def schedule_deletion(self, record_id: str, deletion_date: datetime, 
                         policy_id: str) -> None:
        """Schedule a record for deletion."""
        self.scheduled_deletions.append({
            "record_id": record_id,
            "deletion_date": deletion_date,
            "policy_id": policy_id,
            "scheduled_at": datetime.utcnow()
        })
    
    def _get_applicable_policies(self, record: Dict[str, Any]) -> List[DataRetentionPolicy]:
        """Get retention policies applicable to a record."""
        applicable = []
        
        record_type = record.get("type", "unknown")
        
        for policy in self.policies.values():
            if record_type in policy.data_types or "all" in policy.data_types:
                applicable.append(policy)
        
        return applicable


class ConsentManager:
    """Manage user consent for data processing (GDPR compliance)."""
    
    def __init__(self):
        self.consent_records: Dict[str, Dict[str, Any]] = {}
    
    def record_consent(self, user_id: str, purpose: str, consent_given: bool,
                      legal_basis: str = "consent", metadata: Optional[Dict] = None) -> str:
        """Record user consent for data processing."""
        consent_id = str(uuid.uuid4())
        
        consent_record = {
            "consent_id": consent_id,
            "user_id": user_id,
            "purpose": purpose,
            "consent_given": consent_given,
            "legal_basis": legal_basis,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
            "withdrawn_at": None
        }
        
        self.consent_records[consent_id] = consent_record
        
        # Log consent event
        audit_logger.log_security_event(
            event_type="consent_recorded",
            user_id=user_id,
            details={
                "consent_id": consent_id,
                "purpose": purpose,
                "consent_given": consent_given,
                "legal_basis": legal_basis
            },
            severity="info"
        )
        
        return consent_id
    
    def withdraw_consent(self, consent_id: str) -> bool:
        """Withdraw previously given consent."""
        if consent_id in self.consent_records:
            self.consent_records[consent_id]["withdrawn_at"] = datetime.utcnow().isoformat()
            
            # Log withdrawal
            audit_logger.log_security_event(
                event_type="consent_withdrawn",
                details={"consent_id": consent_id},
                severity="info"
            )
            
            return True
        
        return False
    
    def check_consent(self, user_id: str, purpose: str) -> bool:
        """Check if user has given valid consent for purpose."""
        for consent_record in self.consent_records.values():
            if (consent_record["user_id"] == user_id and 
                consent_record["purpose"] == purpose and
                consent_record["consent_given"] and
                consent_record["withdrawn_at"] is None):
                return True
        
        return False
    
    def get_user_consents(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all consent records for a user."""
        return [
            record for record in self.consent_records.values()
            if record["user_id"] == user_id
        ]


class DataProtectionManager:
    """Main data protection manager coordinating all components."""
    
    def __init__(self):
        self.pii_detector = PIIDetector()
        self.data_masker = DataMasker()
        self.data_anonymizer = DataAnonymizer()
        self.retention_manager = DataRetentionManager()
        self.consent_manager = ConsentManager()
        
        # Load default retention policies
        self._load_default_policies()
    
    def _load_default_policies(self) -> None:
        """Load default data retention policies."""
        default_policies = [
            DataRetentionPolicy(
                policy_id="user_data",
                name="User Personal Data",
                data_types=["user_profile", "personal_info"],
                retention_period_days=2555,  # 7 years
                auto_delete=False  # Require manual review
            ),
            DataRetentionPolicy(
                policy_id="access_logs",
                name="Access and Activity Logs",
                data_types=["access_log", "activity_log"],
                retention_period_days=365,  # 1 year
                auto_delete=True
            ),
            DataRetentionPolicy(
                policy_id="memory_data",
                name="AI Memory Data",
                data_types=["memory", "conversation"],
                retention_period_days=1095,  # 3 years
                auto_delete=False
            )
        ]
        
        for policy in default_policies:
            self.retention_manager.add_policy(policy)
    
    def scan_for_pii(self, data: Union[str, Dict[str, Any]], context: str = "") -> List[PIIMatch]:
        """Scan data for PII and return matches."""
        if isinstance(data, dict):
            # Scan all string values in dictionary
            text = json.dumps(data, ensure_ascii=False)
        else:
            text = str(data)
        
        return self.pii_detector.detect_pii(text, context)
    
    def protect_data(self, data: Union[str, Dict[str, Any]], 
                    protection_level: str = "mask") -> Union[str, Dict[str, Any]]:
        """Apply data protection based on detected PII."""
        is_dict = isinstance(data, dict)
        text = json.dumps(data, ensure_ascii=False) if is_dict else str(data)
        
        # Detect PII
        pii_matches = self.pii_detector.detect_pii(text)
        
        if not pii_matches:
            return data
        
        # Apply protection based on level
        if protection_level == "mask":
            protected_text = self.data_masker.mask_pii_matches(text, pii_matches)
        elif protection_level == "encrypt":
            # Encrypt detected PII values
            protected_text = self._encrypt_pii_matches(text, pii_matches)
        elif protection_level == "remove":
            # Remove PII entirely
            protected_text = self._remove_pii_matches(text, pii_matches)
        else:
            protected_text = text
        
        # Log PII detection
        audit_logger.log_security_event(
            event_type="pii_detected",
            details={
                "pii_types": list(set(match.pii_type.value for match in pii_matches)),
                "protection_level": protection_level,
                "match_count": len(pii_matches)
            },
            severity="medium"
        )
        
        # Convert back to original format
        if is_dict:
            try:
                return json.loads(protected_text)
            except json.JSONDecodeError:
                return {"protected_data": protected_text}
        
        return protected_text
    
    def _encrypt_pii_matches(self, text: str, pii_matches: List[PIIMatch]) -> str:
        """Encrypt PII matches in text."""
        # Sort matches by position (reverse order)
        sorted_matches = sorted(pii_matches, key=lambda m: m.start_pos, reverse=True)
        
        result = text
        for match in sorted_matches:
            encrypted_value = encryption_manager.encrypt_field(
                match.value, f"pii_{match.pii_type.value}"
            )
            placeholder = f"[ENCRYPTED_{match.pii_type.value.upper()}]"
            result = result[:match.start_pos] + placeholder + result[match.end_pos:]
        
        return result
    
    def _remove_pii_matches(self, text: str, pii_matches: List[PIIMatch]) -> str:
        """Remove PII matches from text."""
        # Sort matches by position (reverse order)
        sorted_matches = sorted(pii_matches, key=lambda m: m.start_pos, reverse=True)
        
        result = text
        for match in sorted_matches:
            placeholder = f"[REMOVED_{match.pii_type.value.upper()}]"
            result = result[:match.start_pos] + placeholder + result[match.end_pos:]
        
        return result
    
    def create_processing_record(self, purpose: str, data_categories: List[str],
                               legal_basis: str, controller: str) -> str:
        """Create data processing record for compliance."""
        record = DataProcessingRecord(
            record_id=str(uuid.uuid4()),
            processing_purpose=purpose,
            data_categories=data_categories,
            data_subjects=["users", "customers"],
            recipients=["internal_systems"],
            legal_basis=legal_basis,
            retention_period="As per retention policy",
            security_measures=[
                "encryption_at_rest",
                "encryption_in_transit", 
                "access_controls",
                "audit_logging"
            ],
            controller=controller
        )
        
        # Log processing record creation
        audit_logger.log_security_event(
            event_type="processing_record_created",
            details={
                "record_id": record.record_id,
                "purpose": purpose,
                "legal_basis": legal_basis
            },
            severity="info"
        )
        
        return record.record_id
    
    def get_protection_status(self) -> Dict[str, Any]:
        """Get overall data protection status."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "pii_detection_patterns": len(self.pii_detector.patterns),
            "custom_patterns": len(self.pii_detector.custom_patterns),
            "retention_policies": len(self.retention_manager.policies),
            "scheduled_deletions": len(self.retention_manager.scheduled_deletions),
            "consent_records": len(self.consent_manager.consent_records)
        }


# Global data protection manager instance
data_protection = DataProtectionManager()