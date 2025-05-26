"""
Authentication and user models
"""
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, EmailStr, validator
from enum import Enum
import uuid


class UserRole(str, Enum):
    """User roles enum"""
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"
    SERVICE = "service"


class UserStatus(str, Enum):
    """User status enum"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"


class User(BaseModel):
    """User model"""
    id: uuid.UUID
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    role: UserRole = UserRole.USER
    status: UserStatus = UserStatus.ACTIVE
    is_verified: bool = False
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    
    # Security fields
    password_changed_at: Optional[datetime] = None
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    
    # Compliance
    data_processing_consent: bool = False
    data_processing_consent_date: Optional[datetime] = None
    
    class Config:
        orm_mode = True


class UserCreate(BaseModel):
    """User creation model"""
    email: EmailStr
    username: str
    password: str
    full_name: Optional[str] = None
    role: UserRole = UserRole.USER
    
    @validator("password")
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v
    
    @validator("username")
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError("Username must be at least 3 characters long")
        if not v.isalnum():
            raise ValueError("Username must be alphanumeric")
        return v


class UserUpdate(BaseModel):
    """User update model"""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    role: Optional[UserRole] = None
    status: Optional[UserStatus] = None


class UserLogin(BaseModel):
    """User login model"""
    email: EmailStr
    password: str
    remember_me: bool = False


class Token(BaseModel):
    """Token model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token data model"""
    user_id: uuid.UUID
    email: str
    role: UserRole
    permissions: List[str]
    exp: datetime
    iat: datetime


class APIKey(BaseModel):
    """API Key model"""
    id: uuid.UUID
    user_id: uuid.UUID
    name: str
    key_prefix: str
    key_hash: str
    permissions: List[str]
    is_active: bool = True
    created_at: datetime
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True


class APIKeyCreate(BaseModel):
    """API Key creation model"""
    name: str
    permissions: List[str] = []
    expires_in_days: Optional[int] = None
    
    @validator("name")
    def validate_name(cls, v):
        if len(v) < 3:
            raise ValueError("API key name must be at least 3 characters long")
        return v


class APIKeyResponse(BaseModel):
    """API Key response model (includes actual key)"""
    id: uuid.UUID
    name: str
    key: str  # Only returned on creation
    permissions: List[str]
    expires_at: Optional[datetime]


class Session(BaseModel):
    """User session model"""
    id: uuid.UUID
    user_id: uuid.UUID
    token_jti: str
    ip_address: str
    user_agent: str
    created_at: datetime
    expires_at: datetime
    is_active: bool = True
    
    class Config:
        orm_mode = True


class AuditLog(BaseModel):
    """Audit log model"""
    id: uuid.UUID
    user_id: Optional[uuid.UUID] = None
    api_key_id: Optional[uuid.UUID] = None
    action: str
    resource: str
    resource_id: Optional[str] = None
    ip_address: str
    user_agent: str
    details: Dict[str, Any] = {}
    timestamp: datetime
    
    class Config:
        orm_mode = True


class SecurityEvent(BaseModel):
    """Security event model"""
    id: uuid.UUID
    event_type: str  # failed_login, suspicious_activity, etc.
    user_id: Optional[uuid.UUID] = None
    ip_address: str
    details: Dict[str, Any] = {}
    severity: str  # low, medium, high, critical
    timestamp: datetime
    resolved: bool = False
    
    class Config:
        orm_mode = True


class RateLimitInfo(BaseModel):
    """Rate limit information"""
    limit: int
    remaining: int
    reset_time: datetime
    window_seconds: int


class PasswordReset(BaseModel):
    """Password reset model"""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation model"""
    token: str
    new_password: str
    
    @validator("new_password")
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v


class MFASetup(BaseModel):
    """MFA setup model"""
    secret: str
    qr_code: str


class MFAVerify(BaseModel):
    """MFA verification model"""
    token: str
    
    @validator("token")
    def validate_token(cls, v):
        if len(v) != 6 or not v.isdigit():
            raise ValueError("MFA token must be 6 digits")
        return v