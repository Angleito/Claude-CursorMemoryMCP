"""Authentication and authorization system"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

from .config import Settings
from .models import UserResponse

logger = structlog.get_logger()
security = HTTPBearer()


class AuthManager:
    """Handles JWT token creation and validation"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.secret_key = settings.secret_key
        self.algorithm = settings.algorithm
        self.access_token_expire_minutes = settings.access_token_expire_minutes
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create a JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        
        token = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.JWTError as e:
            logger.warning("JWT validation failed", error=str(e))
            return None


# Global auth manager instance
auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """Get the global auth manager instance"""
    global auth_manager
    if not auth_manager:
        from .config import Settings
        auth_manager = AuthManager(Settings())
    return auth_manager


def create_access_token(data: Dict[str, Any]) -> str:
    """Create an access token"""
    return get_auth_manager().create_access_token(data)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get the current authenticated user from JWT token"""
    auth_manager = get_auth_manager()
    
    payload = auth_manager.verify_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return {"id": user_id, "payload": payload}


async def get_optional_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[Dict[str, Any]]:
    """Get the current user if authenticated, otherwise return None"""
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


class RoleChecker:
    """Check user roles and permissions"""
    
    def __init__(self, required_roles: list = None):
        self.required_roles = required_roles or []
    
    def __call__(self, current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        if self.required_roles:
            user_roles = current_user.get("payload", {}).get("roles", [])
            if not any(role in user_roles for role in self.required_roles):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
        return current_user


def require_roles(*roles):
    """Decorator to require specific roles"""
    return RoleChecker(list(roles))


# Common role checkers
require_admin = require_roles("admin")
require_moderator = require_roles("admin", "moderator")