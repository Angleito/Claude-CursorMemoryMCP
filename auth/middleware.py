"""
Authentication middleware and dependencies
"""
from typing import Optional, Annotated
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uuid
from datetime import datetime

from auth.models import User, TokenData, APIKey
from auth.security import security_manager
from auth.rbac import rbac_manager
from database.user_repository import UserRepository
from database.api_key_repository import APIKeyRepository
from monitoring.audit_logger import audit_logger


security = HTTPBearer(auto_error=False)


class AuthenticationError(HTTPException):
    """Custom authentication error"""
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class AuthorizationError(HTTPException):
    """Custom authorization error"""
    def __init__(self, detail: str = "Access denied"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
        )


async def get_current_user_from_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    request: Request = None
) -> Optional[User]:
    """Get current user from JWT token"""
    if not credentials:
        return None
    
    token_data = security_manager.verify_token(credentials.credentials)
    if not token_data:
        return None
    
    # Get user from database
    user_repo = UserRepository()
    user = await user_repo.get_by_id(token_data.user_id)
    
    if not user:
        return None
    
    # Check if user is active
    if user.status != "active":
        return None
    
    # Check if account is locked
    if user.locked_until and user.locked_until > datetime.utcnow():
        return None
    
    # Update last login
    if request:
        await user_repo.update_last_login(user.id, request.client.host)
    
    return user


async def get_current_user_from_api_key(
    request: Request,
    api_key_repo: APIKeyRepository = Depends()
) -> Optional[User]:
    """Get current user from API key"""
    # Check for API key in headers
    api_key = None
    
    # Try Authorization header with "ApiKey" scheme
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("ApiKey "):
        api_key = auth_header[7:]  # Remove "ApiKey " prefix
    
    # Try X-API-Key header
    if not api_key:
        api_key = request.headers.get("X-API-Key")
    
    if not api_key:
        return None
    
    # Validate API key format
    if not api_key.startswith("mk_"):
        return None
    
    # Get API key from database
    api_key_obj = await api_key_repo.get_by_key(api_key)
    if not api_key_obj:
        return None
    
    # Check if API key is active
    if not api_key_obj.is_active:
        return None
    
    # Check if API key is expired
    if api_key_obj.expires_at and api_key_obj.expires_at < datetime.utcnow():
        return None
    
    # Get user
    user_repo = UserRepository()
    user = await user_repo.get_by_id(api_key_obj.user_id)
    
    if not user or user.status != "active":
        return None
    
    # Update API key last used
    await api_key_repo.update_last_used(api_key_obj.id)
    
    # Add API key info to request state
    request.state.api_key = api_key_obj
    
    return user


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> User:
    """Get current authenticated user (required)"""
    user = None
    auth_method = None
    
    # Try JWT token first
    if credentials:
        user = await get_current_user_from_token(credentials, request)
        if user:
            auth_method = "jwt"
    
    # Try API key if JWT failed
    if not user:
        user = await get_current_user_from_api_key(request)
        if user:
            auth_method = "api_key"
    
    if not user:
        raise AuthenticationError("Invalid or missing authentication credentials")
    
    # Store auth method in request state
    request.state.auth_method = auth_method
    request.state.current_user = user
    
    # Log authentication event
    await audit_logger.log_event(
        action="authenticate",
        resource="auth",
        user_id=user.id,
        ip_address=request.client.host,
        user_agent=request.headers.get("User-Agent", ""),
        details={"method": auth_method}
    )
    
    return user


async def get_current_user_optional(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[User]:
    """Get current authenticated user (optional)"""
    try:
        return await get_current_user(request, credentials)
    except AuthenticationError:
        return None


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user"""
    if current_user.status != "active":
        raise AuthenticationError("Account is not active")
    
    return current_user


async def get_current_admin_user(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Get current admin user"""
    if current_user.role != "admin":
        raise AuthorizationError("Admin access required")
    
    return current_user


async def get_current_user_permissions(
    current_user: User = Depends(get_current_active_user)
) -> list[str]:
    """Get current user's permissions"""
    return list(rbac_manager.get_user_permissions(current_user))


# Type annotations for dependency injection
CurrentUser = Annotated[User, Depends(get_current_user)]
CurrentUserOptional = Annotated[Optional[User], Depends(get_current_user_optional)]
CurrentActiveUser = Annotated[User, Depends(get_current_active_user)]
CurrentAdminUser = Annotated[User, Depends(get_current_admin_user)]
CurrentUserPermissions = Annotated[list[str], Depends(get_current_user_permissions)]


class RateLimitMiddleware:
    """Rate limiting middleware"""
    
    def __init__(self, app, redis_client):
        self.app = app
        self.redis = redis_client
    
    async def __call__(self, request: Request, call_next):
        # Get user or use IP address for rate limiting
        user = getattr(request.state, 'current_user', None)
        identifier = str(user.id) if user else request.client.host
        
        # Get rate limit for user role or default
        if user:
            from config.settings import ROLE_RATE_LIMITS
            rate_limit = ROLE_RATE_LIMITS.get(user.role.value, {"requests": 100, "window": 60})
        else:
            rate_limit = {"requests": 50, "window": 60}  # Stricter for unauthenticated
        
        # Check rate limit
        current_requests = await self.check_rate_limit(
            identifier, 
            rate_limit["requests"], 
            rate_limit["window"]
        )
        
        if current_requests > rate_limit["requests"]:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit": str(rate_limit["requests"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(rate_limit["window"])
                }
            )
        
        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(rate_limit["requests"])
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, rate_limit["requests"] - current_requests - 1)
        )
        response.headers["X-RateLimit-Reset"] = str(rate_limit["window"])
        
        return response
    
    async def check_rate_limit(self, identifier: str, limit: int, window: int) -> int:
        """Check and update rate limit counter"""
        key = f"rate_limit:{identifier}:{window}"
        
        try:
            # Get current count
            current = await self.redis.get(key)
            if current is None:
                # First request in window
                await self.redis.setex(key, window, 1)
                return 1
            else:
                # Increment counter
                new_count = await self.redis.incr(key)
                return int(new_count)
        except Exception:
            # If Redis is down, allow the request
            return 0


class SecurityHeadersMiddleware:
    """Security headers middleware"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        security_headers = security_manager.get_security_headers()
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response


class AuditMiddleware:
    """Audit logging middleware"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, request: Request, call_next):
        start_time = datetime.utcnow()
        
        # Process request
        response = await call_next(request)
        
        # Log API call
        user = getattr(request.state, 'current_user', None)
        api_key = getattr(request.state, 'api_key', None)
        
        await audit_logger.log_api_call(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            user_id=user.id if user else None,
            api_key_id=api_key.id if api_key else None,
            ip_address=request.client.host,
            user_agent=request.headers.get("User-Agent", ""),
            duration=(datetime.utcnow() - start_time).total_seconds()
        )
        
        return response