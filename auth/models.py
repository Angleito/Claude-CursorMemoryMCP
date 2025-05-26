"""Authentication and user models."""

import os
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel
from pydantic import EmailStr
from pydantic import validator

# Validation constants
MIN_PASSWORD_LENGTH = 8
MIN_USERNAME_LENGTH = 3
MIN_NAME_LENGTH = 3
TOTP_CODE_LENGTH = 6


class UserRole(str, Enum):
    """User roles enum."""

    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"
    SERVICE = "service"


class UserStatus(str, Enum):
    """User status enum."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"


class User(BaseModel):
    """User model."""

    id: uuid.UUID
    email: EmailStr
    username: str
    full_name: str | None = None
    role: UserRole = UserRole.USER
    status: UserStatus = UserStatus.ACTIVE
    is_verified: bool = False
    created_at: datetime
    updated_at: datetime
    last_login: datetime | None = None
    failed_login_attempts: int = 0
    locked_until: datetime | None = None

    # Security fields
    password_changed_at: datetime | None = None
    mfa_enabled: bool = False
    mfa_secret: str | None = None

    # Compliance
    data_processing_consent: bool = False
    data_processing_consent_date: datetime | None = None

    class Config:
        """Pydantic configuration settings."""

        orm_mode = True


class UserCreate(BaseModel):
    """User creation model."""

    email: EmailStr
    username: str
    password: str
    full_name: str | None = None
    role: UserRole = UserRole.USER

    @validator("password")
    def validate_password(self, v):
        """Validate password strength requirements."""
        if len(v) < MIN_PASSWORD_LENGTH:
            raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v

    @validator("username")
    def validate_username(self, v):
        """Validate username format and length."""
        if len(v) < MIN_USERNAME_LENGTH:
            raise ValueError(f"Username must be at least {MIN_USERNAME_LENGTH} characters long")
        if not v.isalnum():
            raise ValueError("Username must be alphanumeric")
        return v


class UserUpdate(BaseModel):
    """User update model."""

    email: EmailStr | None = None
    full_name: str | None = None
    role: UserRole | None = None
    status: UserStatus | None = None


class UserLogin(BaseModel):
    """User login model."""

    email: EmailStr
    password: str
    remember_me: bool = False


class Token(BaseModel):
    """Token model."""

    access_token: str
    refresh_token: str
    token_type: str = os.getenv("TOKEN_TYPE", "bearer")
    expires_in: int


class TokenData(BaseModel):
    """Token data model."""

    user_id: uuid.UUID
    email: str
    role: UserRole
    permissions: list[str]
    exp: datetime
    iat: datetime


class APIKey(BaseModel):
    """API Key model."""

    id: uuid.UUID
    user_id: uuid.UUID
    name: str
    key_prefix: str
    key_hash: str
    permissions: list[str]
    is_active: bool = True
    created_at: datetime
    last_used: datetime | None = None
    expires_at: datetime | None = None

    class Config:
        """Pydantic configuration settings."""

        orm_mode = True


class APIKeyCreate(BaseModel):
    """API Key creation model."""

    name: str
    permissions: list[str] = []
    expires_in_days: int | None = None

    @validator("name")
    def validate_name(self, v):
        """Validate API key name length."""
        if len(v) < MIN_NAME_LENGTH:
            raise ValueError(f"API key name must be at least {MIN_NAME_LENGTH} characters long")
        return v


class APIKeyResponse(BaseModel):
    """API Key response model (includes actual key)."""

    id: uuid.UUID
    name: str
    key: str  # Only returned on creation
    permissions: list[str]
    expires_at: datetime | None


class Session(BaseModel):
    """User session model."""

    id: uuid.UUID
    user_id: uuid.UUID
    token_jti: str
    ip_address: str
    user_agent: str
    created_at: datetime
    expires_at: datetime
    is_active: bool = True

    class Config:
        """Pydantic configuration settings."""

        orm_mode = True


class AuditLog(BaseModel):
    """Audit log model."""

    id: uuid.UUID
    user_id: uuid.UUID | None = None
    api_key_id: uuid.UUID | None = None
    action: str
    resource: str
    resource_id: str | None = None
    ip_address: str
    user_agent: str
    details: dict[str, Any] = {}
    timestamp: datetime

    class Config:
        """Pydantic configuration settings."""

        orm_mode = True


class SecurityEvent(BaseModel):
    """Security event model."""

    id: uuid.UUID
    event_type: str  # failed_login, suspicious_activity, etc.
    user_id: uuid.UUID | None = None
    ip_address: str
    details: dict[str, Any] = {}
    severity: str  # low, medium, high, critical
    timestamp: datetime
    resolved: bool = False

    class Config:
        """Pydantic configuration settings."""

        orm_mode = True


class RateLimitInfo(BaseModel):
    """Rate limit information."""

    limit: int
    remaining: int
    reset_time: datetime
    window_seconds: int


class PasswordReset(BaseModel):
    """Password reset model."""

    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation model."""

    token: str
    new_password: str

    @validator("new_password")
    def validate_password(self, v):
        """Validate new password length."""
        if len(v) < MIN_PASSWORD_LENGTH:
            raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long")
        return v


class MFASetup(BaseModel):
    """MFA setup model."""

    secret: str
    qr_code: str


class MFAVerify(BaseModel):
    """MFA verification model."""

    token: str

    @validator("token")
    def validate_token(self, v):
        """Validate MFA token format."""
        if len(v) != TOTP_CODE_LENGTH or not v.isdigit():
            raise ValueError(f"MFA token must be {TOTP_CODE_LENGTH} digits")
        return v
