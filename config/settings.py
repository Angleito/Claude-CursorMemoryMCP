"""Security and application settings configuration."""

import secrets
from functools import lru_cache
from typing import List
from typing import Optional

from pydantic import BaseSettings
from pydantic import validator


class SecuritySettings(BaseSettings):
    """Security-related configuration."""

    # JWT Configuration
    secret_key: str = secrets.token_urlsafe(32)
    jwt_secret_key: str = secrets.token_urlsafe(32)
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # API Keys
    api_key_prefix: str = "mk_"
    api_key_length: int = 32
    max_api_keys_per_user: int = 10

    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    rate_limit_storage: str = "redis"

    # Encryption
    encryption_key: Optional[str] = None
    encryption_algorithm: str = "AES-256-GCM"

    # Security Headers
    cors_origins: List[str] = ["http://localhost:3000"]
    trusted_hosts: List[str] = ["localhost"]

    @validator("encryption_key", pre=True, always=True)
    def generate_encryption_key(self, v):
        if not v:
            return secrets.token_urlsafe(32)
        return v

    class Config:
        env_prefix = ""
        case_sensitive = False


class DatabaseSettings(BaseSettings):
    """Database configuration."""

    # Supabase
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str

    # PostgreSQL
    database_url: str

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_password: Optional[str] = None

    class Config:
        env_prefix = ""
        case_sensitive = False


class ServerSettings(BaseSettings):
    """Server configuration."""

    server_host: str = "0.0.0.0"
    server_port: int = 8000
    workers: int = 4

    # SSL/TLS
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None

    class Config:
        env_prefix = ""
        case_sensitive = False


class MonitoringSettings(BaseSettings):
    """Monitoring and logging configuration."""

    sentry_dsn: Optional[str] = None
    log_level: str = "INFO"
    prometheus_port: int = 8080

    class Config:
        env_prefix = ""
        case_sensitive = False


class ComplianceSettings(BaseSettings):
    """Compliance and privacy configuration."""

    gdpr_enabled: bool = True
    data_retention_days: int = 365
    audit_log_retention_days: int = 2555  # 7 years

    class Config:
        env_prefix = ""
        case_sensitive = False


class Settings(BaseSettings):
    """Main application settings."""

    # Environment
    environment: str = "development"
    debug: bool = False

    # Sub-configurations
    security: SecuritySettings = SecuritySettings()
    database: DatabaseSettings = DatabaseSettings()
    server: ServerSettings = ServerSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    compliance: ComplianceSettings = ComplianceSettings()

    @validator("debug", pre=True, always=True)
    def set_debug(self, v, values):
        return values.get("environment", "development") == "development"

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Security constants
ALGORITHM = "HS256"
MIN_PASSWORD_LENGTH = 8
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 30

# Role permissions
ROLE_PERMISSIONS = {
    "admin": [
        "users:read",
        "users:write",
        "users:delete",
        "memories:read",
        "memories:write",
        "memories:delete",
        "api_keys:read",
        "api_keys:write",
        "api_keys:delete",
        "audit:read",
        "system:manage",
    ],
    "user": [
        "memories:read",
        "memories:write",
        "api_keys:read",
        "api_keys:write",
        "profile:read",
        "profile:write",
    ],
    "readonly": ["memories:read", "profile:read"],
    "service": ["memories:read", "memories:write", "api_keys:read"],
}

# API rate limits by role
ROLE_RATE_LIMITS = {
    "admin": {"requests": 1000, "window": 60},
    "user": {"requests": 100, "window": 60},
    "readonly": {"requests": 50, "window": 60},
    "service": {"requests": 500, "window": 60},
}
