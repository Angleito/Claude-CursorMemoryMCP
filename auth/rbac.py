"""Role-Based Access Control (RBAC) implementation."""

import re
from enum import Enum
from functools import wraps

from fastapi import HTTPException
from fastapi import status

from auth.models import User
from auth.models import UserRole
from config.settings import ROLE_PERMISSIONS


class Permission(str, Enum):
    """System permissions."""

    # User management
    USERS_READ = "users:read"
    USERS_WRITE = "users:write"
    USERS_DELETE = "users:delete"

    # Memory management
    MEMORIES_READ = "memories:read"
    MEMORIES_WRITE = "memories:write"
    MEMORIES_DELETE = "memories:delete"

    # API key management
    API_KEYS_READ = "api_keys:read"
    API_KEYS_WRITE = "api_keys:write"
    API_KEYS_DELETE = "api_keys:delete"

    # Profile management
    PROFILE_READ = "profile:read"
    PROFILE_WRITE = "profile:write"

    # Audit logs
    AUDIT_READ = "audit:read"

    # System management
    SYSTEM_MANAGE = "system:manage"


class Resource(str, Enum):
    """System resources."""

    USER = "user"
    MEMORY = "memory"
    API_KEY = "api_key"
    AUDIT_LOG = "audit_log"
    SYSTEM = "system"


class Action(str, Enum):
    """RBAC actions."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    MANAGE = "manage"


class RBACManager:
    """Role-Based Access Control Manager."""

    def __init__(self):
        self.role_permissions = ROLE_PERMISSIONS

    def get_user_permissions(self, user: User) -> set[str]:
        """Get all permissions for a user."""
        base_permissions = set(self.role_permissions.get(user.role.value, []))

        # Add any custom permissions if needed
        # This could be extended to support user-specific permissions

        return base_permissions

    def has_permission(self, user: User, permission: str) -> bool:
        """Check if user has a specific permission."""
        user_permissions = self.get_user_permissions(user)
        return permission in user_permissions

    def has_any_permission(self, user: User, permissions: list[str]) -> bool:
        """Check if user has any of the specified permissions."""
        user_permissions = self.get_user_permissions(user)
        return any(perm in user_permissions for perm in permissions)

    def has_all_permissions(self, user: User, permissions: list[str]) -> bool:
        """Check if user has all of the specified permissions."""
        user_permissions = self.get_user_permissions(user)
        return all(perm in user_permissions for perm in permissions)

    def can_access_resource(
        self,
        user: User,
        resource: str,
        action: str,
        resource_owner_id: str | None = None,
    ) -> bool:
        """Check if user can perform action on resource."""
        permission = f"{resource}:{action}"

        # Check basic permission
        if not self.has_permission(user, permission):
            # Check if user owns the resource (for profile, own memories, etc.)
            return bool(resource_owner_id and str(user.id) == resource_owner_id and resource in ["profile", "memory"] and action in ["read", "write"])

        return True

    def filter_permissions_by_role(self, role: UserRole) -> list[str]:
        """Get permissions for a specific role."""
        return self.role_permissions.get(role.value, [])

    def get_resource_permissions(self, user: User, resource: str) -> list[str]:
        """Get all permissions for a resource that the user has."""
        user_permissions = self.get_user_permissions(user)
        resource_permissions = [
            perm for perm in user_permissions if perm.startswith(f"{resource}:")
        ]
        return resource_permissions

    def validate_permission_format(self, permission: str) -> bool:
        """Validate permission format (resource:action)."""
        pattern = r"^[a-z_]+:[a-z_]+$"
        return bool(re.match(pattern, permission))

    def get_hierarchical_permissions(self, user: User) -> dict[str, list[str]]:
        """Get permissions organized by resource."""
        user_permissions = self.get_user_permissions(user)
        hierarchical = {}

        for permission in user_permissions:
            if ":" in permission:
                resource, action = permission.split(":", 1)
                if resource not in hierarchical:
                    hierarchical[resource] = []
                hierarchical[resource].append(action)

        return hierarchical


# Global RBAC manager instance
rbac_manager = RBACManager()


# Decorators for permission checking
def require_permission(permission: str):
    """Decorator to require a specific permission."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract current user from kwargs (set by auth middleware)
            current_user = kwargs.get("current_user")
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )

            if not rbac_manager.has_permission(current_user, permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied: {permission} required",
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_any_permission(permissions: list[str]):
    """Decorator to require any of the specified permissions."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get("current_user")
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )

            if not rbac_manager.has_any_permission(current_user, permissions):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied: One of {permissions} required",
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_role(roles: list[UserRole]):
    """Decorator to require specific roles."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get("current_user")
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )

            if current_user.role not in roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role denied: One of {[r.value for r in roles]} required",
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_resource_access(resource: str, action: str, owner_id_param: str | None = None):
    """Decorator to require access to a specific resource."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get("current_user")
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )

            # Get resource owner ID if specified
            resource_owner_id = None
            if owner_id_param:
                resource_owner_id = kwargs.get(owner_id_param)

            if not rbac_manager.can_access_resource(
                current_user, resource, action, resource_owner_id
            ):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied: Cannot {action} {resource}",
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


# Admin-only decorator
require_admin = require_role([UserRole.ADMIN])

# Common permission decorators
require_user_management = require_permission(Permission.USERS_READ)
require_memory_read = require_permission(Permission.MEMORIES_READ)
require_memory_write = require_permission(Permission.MEMORIES_WRITE)
require_api_key_management = require_permission(Permission.API_KEYS_READ)
require_audit_access = require_permission(Permission.AUDIT_READ)
require_system_management = require_permission(Permission.SYSTEM_MANAGE)
