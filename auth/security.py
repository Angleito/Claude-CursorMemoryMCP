"""Core security utilities and encryption functions."""

import base64
import hashlib
import hmac
import os
import secrets
from datetime import datetime
from datetime import timedelta
from io import BytesIO
from typing import Any

import pyotp
import qrcode
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from jose import JWTError
from jose import jwt
from passlib.context import CryptContext

from auth.models import TokenData
from auth.models import UserRole
from config.settings import get_settings

settings = get_settings()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# Encryption setup
def get_fernet_key(password: str, salt: bytes | None = None) -> bytes:
    """Generate Fernet key from password."""
    if salt is None:
        salt = os.urandom(16)

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key


class SecurityManager:
    """Main security manager class."""

    def __init__(self):
        self.settings = get_settings()
        self._fernet = None

    @property
    def fernet(self) -> Fernet:
        """Lazy-loaded Fernet instance."""
        if self._fernet is None:
            key = get_fernet_key(self.settings.security.encryption_key)
            self._fernet = Fernet(key)
        return self._fernet

    # Password utilities
    def hash_password(self, password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception:
            return False

    def generate_password(self, length: int = 16) -> str:
        """Generate a secure random password."""
        alphabet = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
        )
        return "".join(secrets.choice(alphabet) for _ in range(length))

    # JWT tokens
    def create_access_token(
        self, data: dict[str, Any], expires_delta: timedelta | None = None
    ) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=self.settings.security.jwt_expire_minutes
            )

        to_encode.update({"exp": expire, "iat": datetime.utcnow(), "type": "access"})

        encoded_jwt = jwt.encode(
            to_encode,
            self.settings.security.jwt_secret_key,
            algorithm=self.settings.security.jwt_algorithm,
        )
        return encoded_jwt

    def create_refresh_token(
        self, data: dict[str, Any], expires_delta: timedelta | None = None
    ) -> str:
        """Create a JWT refresh token."""
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                days=self.settings.security.refresh_token_expire_days
            )

        to_encode.update({"exp": expire, "iat": datetime.utcnow(), "type": "refresh"})

        encoded_jwt = jwt.encode(
            to_encode,
            self.settings.security.jwt_secret_key,
            algorithm=self.settings.security.jwt_algorithm,
        )
        return encoded_jwt

    def verify_token(self, token: str) -> TokenData | None:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.settings.security.jwt_secret_key,
                algorithms=[self.settings.security.jwt_algorithm],
            )

            user_id = payload.get("sub")
            email = payload.get("email")
            role = payload.get("role")
            permissions = payload.get("permissions", [])

            if user_id is None or email is None:
                return None

            return TokenData(
                user_id=user_id,
                email=email,
                role=UserRole(role) if role else UserRole.USER,
                permissions=permissions,
                exp=datetime.fromtimestamp(payload.get("exp", 0)),
                iat=datetime.fromtimestamp(payload.get("iat", 0)),
            )
        except JWTError:
            return None

    # API Keys
    def generate_api_key(self) -> tuple[str, str]:
        """Generate API key and return (key, hash)."""
        key = f"{self.settings.security.api_key_prefix}{secrets.token_urlsafe(self.settings.security.api_key_length)}"
        key_hash = self.hash_api_key(key)
        return key, key_hash

    def hash_api_key(self, api_key: str) -> str:
        """Hash an API key."""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def verify_api_key(self, api_key: str, stored_hash: str) -> bool:
        """Verify an API key against its hash."""
        computed_hash = self.hash_api_key(api_key)
        return hmac.compare_digest(computed_hash, stored_hash)

    # Data encryption
    def encrypt_data(self, data: str | bytes) -> str:
        """Encrypt sensitive data."""
        if isinstance(data, str):
            data = data.encode()
        encrypted = self.fernet.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception:
            raise ValueError("Failed to decrypt data") from None

    # MFA (Multi-Factor Authentication)
    def generate_mfa_secret(self) -> str:
        """Generate MFA secret."""
        return pyotp.random_base32()

    def generate_mfa_qr_code(
        self, secret: str, email: str, issuer: str = "MemoryDB"
    ) -> str:
        """Generate MFA QR code."""
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=email, issuer_name=issuer
        )

        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        return base64.b64encode(buf.getvalue()).decode()

    def verify_mfa_token(self, secret: str, token: str) -> bool:
        """Verify MFA token."""
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=1)

    # Security tokens for password reset, etc.
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate a secure random token."""
        return secrets.token_urlsafe(length)

    def generate_signed_token(
        self, data: dict[str, Any], expires_minutes: int = 60
    ) -> str:
        """Generate a signed token for secure operations."""
        expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
        data.update({"exp": expire.timestamp()})

        return jwt.encode(
            data,
            self.settings.security.secret_key,
            algorithm=self.settings.security.jwt_algorithm,
        )

    def verify_signed_token(self, token: str) -> dict[str, Any] | None:
        """Verify a signed token."""
        try:
            payload = jwt.decode(
                token,
                self.settings.security.secret_key,
                algorithms=[self.settings.security.jwt_algorithm],
            )
            return payload
        except JWTError:
            return None

    # Rate limiting helpers
    def generate_rate_limit_key(self, identifier: str, window: str) -> str:
        """Generate rate limit key."""
        return f"rate_limit:{identifier}:{window}"

    # Security headers
    def get_security_headers(self) -> dict[str, str]:
        """Get security headers for HTTP responses."""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        }

    # Input validation and sanitization
    def sanitize_input(self, input_str: str) -> str:
        """Sanitize user input."""
        if not isinstance(input_str, str):
            return str(input_str)

        # Remove null bytes and control characters
        sanitized = input_str.replace("\x00", "").replace("\r", "").replace("\n", " ")

        # Limit length
        max_length = 1000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        return sanitized.strip()

    def is_safe_redirect_url(self, url: str) -> bool:
        """Check if a redirect URL is safe."""
        if not url:
            return False

        # Only allow relative URLs or URLs from trusted hosts
        if url.startswith("/"):
            return True

        from urllib.parse import urlparse

        parsed = urlparse(url)

        return parsed.hostname in self.settings.security.trusted_hosts


# Global security manager instance
security_manager = SecurityManager()
