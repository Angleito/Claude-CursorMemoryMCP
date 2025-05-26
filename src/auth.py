"""Authentication and authorization system.

This module provides JWT-based authentication and authorization for the Mem0 AI
MCP Server. It includes token creation, validation, user dependency injection,
and role-based access control.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import jwt
import structlog
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import Settings

logger = structlog.get_logger()
security = HTTPBearer()


class AuthManager:
    """Handles JWT token creation and validation.
    
    This class manages JWT tokens for user authentication, including token
    creation with expiration and token validation with proper error handling.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.secret_key = settings.secret_key
        self.algorithm = settings.algorithm
        self.access_token_expire_minutes = settings.access_token_expire_minutes

    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create a JWT access token.
        
        Args:
            data: Dictionary containing token payload data
            
        Returns:
            Encoded JWT token string
        """
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})

        token = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return token

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token.
        
        Args:
            token: JWT token string to verify
            
        Returns:
            Decoded token payload if valid, None otherwise
        """
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
    """Get the global auth manager instance.
    
    Returns:
        Global AuthManager instance with default settings
    """
    global auth_manager
    if not auth_manager:
        from .config import Settings

        auth_manager = AuthManager(Settings())
    return auth_manager


def create_access_token(data: Dict[str, Any]) -> str:
    """Create an access token using the global auth manager.
    
    Args:
        data: Dictionary containing token payload data
        
    Returns:
        Encoded JWT token string
    """
    return get_auth_manager().create_access_token(data)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    """Get the current authenticated user from JWT token.
    
    Args:
        credentials: HTTP authorization credentials from request header
        
    Returns:
        Dictionary containing user ID and payload
        
    Raises:
        HTTPException: If token is invalid or expired
    """
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


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[Dict[str, Any]]:
    """Get the current user if authenticated, otherwise return None.
    
    Args:
        credentials: Optional HTTP authorization credentials
        
    Returns:
        User information if authenticated, None otherwise
    """
    if not credentials:
        return None

    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


class RoleChecker:
    """Check user roles and permissions.
    
    This class provides role-based access control by validating user roles
    against required roles for specific operations.
    """

    def __init__(self, required_roles: Optional[List[str]] = None) -> None:
        self.required_roles = required_roles or []

    def __call__(
        self, current_user: Dict[str, Any] = Depends(get_current_user)
    ) -> Dict[str, Any]:
        if self.required_roles:
            user_roles = current_user.get("payload", {}).get("roles", [])
            if not any(role in user_roles for role in self.required_roles):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions",
                )
        return current_user


def require_roles(*roles: str) -> RoleChecker:
    """Decorator to require specific roles.
    
    Args:
        *roles: Variable number of role names required
        
    Returns:
        RoleChecker instance configured with required roles
    """
    return RoleChecker(list(roles))


# Common role checkers
require_admin = require_roles("admin")
require_moderator = require_roles("admin", "moderator")
