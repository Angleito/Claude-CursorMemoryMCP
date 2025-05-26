"""
API Key management for MCP clients and external integrations
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import HTTPException, status
import uuid
import secrets
import hashlib

from auth.models import APIKey, APIKeyCreate, APIKeyResponse, User, UserRole
from auth.security import security_manager
from auth.rbac import rbac_manager, Permission
from database.api_key_repository import APIKeyRepository
from monitoring.audit_logger import audit_logger


class APIKeyManager:
    """API Key management class"""
    
    def __init__(self):
        self.api_key_repo = APIKeyRepository()
    
    async def create_api_key(
        self, 
        user: User, 
        key_data: APIKeyCreate,
        created_by_user_id: Optional[uuid.UUID] = None
    ) -> APIKeyResponse:
        """Create a new API key for a user"""
        
        # Check if user can create API keys
        if not rbac_manager.has_permission(user, Permission.API_KEYS_WRITE):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied: Cannot create API keys"
            )
        
        # Check API key limit
        existing_keys = await self.api_key_repo.get_user_api_keys(user.id, active_only=True)
        if len(existing_keys) >= 10:  # Max 10 API keys per user
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum number of API keys reached"
            )
        
        # Validate permissions
        user_permissions = rbac_manager.get_user_permissions(user)
        invalid_permissions = set(key_data.permissions) - user_permissions
        if invalid_permissions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid permissions: {list(invalid_permissions)}"
            )
        
        # Generate API key
        api_key, key_hash = security_manager.generate_api_key()
        
        # Calculate expiration
        expires_at = None
        if key_data.expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=key_data.expires_in_days)
        
        # Create API key record
        api_key_obj = APIKey(
            id=uuid.uuid4(),
            user_id=user.id,
            name=key_data.name,
            key_prefix=api_key[:8] + "...",
            key_hash=key_hash,
            permissions=key_data.permissions or list(user_permissions),
            expires_at=expires_at,
            created_at=datetime.utcnow()
        )
        
        # Save to database
        await self.api_key_repo.create(api_key_obj)
        
        # Log creation
        await audit_logger.log_event(
            action="create_api_key",
            resource="api_key",
            resource_id=str(api_key_obj.id),
            user_id=created_by_user_id or user.id,
            details={
                "key_name": key_data.name,
                "permissions": key_data.permissions,
                "expires_in_days": key_data.expires_in_days
            }
        )
        
        return APIKeyResponse(
            id=api_key_obj.id,
            name=api_key_obj.name,
            key=api_key,  # Only returned on creation
            permissions=api_key_obj.permissions,
            expires_at=api_key_obj.expires_at
        )
    
    async def get_user_api_keys(
        self, 
        user: User, 
        requesting_user: User
    ) -> List[APIKey]:
        """Get API keys for a user"""
        
        # Check permissions
        if user.id != requesting_user.id:
            if not rbac_manager.has_permission(requesting_user, Permission.API_KEYS_READ):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permission denied: Cannot view API keys"
                )
        
        return await self.api_key_repo.get_user_api_keys(user.id)
    
    async def get_api_key(
        self, 
        key_id: uuid.UUID, 
        requesting_user: User
    ) -> APIKey:
        """Get a specific API key"""
        
        api_key = await self.api_key_repo.get_by_id(key_id)
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found"
            )
        
        # Check permissions
        if api_key.user_id != requesting_user.id:
            if not rbac_manager.has_permission(requesting_user, Permission.API_KEYS_READ):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permission denied: Cannot view API key"
                )
        
        return api_key
    
    async def update_api_key(
        self, 
        key_id: uuid.UUID, 
        updates: Dict[str, Any],
        requesting_user: User
    ) -> APIKey:
        """Update an API key"""
        
        api_key = await self.get_api_key(key_id, requesting_user)
        
        # Check write permissions
        if api_key.user_id != requesting_user.id:
            if not rbac_manager.has_permission(requesting_user, Permission.API_KEYS_WRITE):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permission denied: Cannot update API key"
                )
        
        # Validate permission updates
        if "permissions" in updates:
            user = await self.api_key_repo.get_key_user(api_key.id)
            user_permissions = rbac_manager.get_user_permissions(user)
            invalid_permissions = set(updates["permissions"]) - user_permissions
            if invalid_permissions:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid permissions: {list(invalid_permissions)}"
                )
        
        # Update API key
        updated_key = await self.api_key_repo.update(key_id, updates)
        
        # Log update
        await audit_logger.log_event(
            action="update_api_key",
            resource="api_key",
            resource_id=str(key_id),
            user_id=requesting_user.id,
            details={"updates": updates}
        )
        
        return updated_key
    
    async def revoke_api_key(
        self, 
        key_id: uuid.UUID, 
        requesting_user: User
    ) -> bool:
        """Revoke (deactivate) an API key"""
        
        api_key = await self.get_api_key(key_id, requesting_user)
        
        # Check delete permissions
        if api_key.user_id != requesting_user.id:
            if not rbac_manager.has_permission(requesting_user, Permission.API_KEYS_DELETE):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permission denied: Cannot revoke API key"
                )
        
        # Deactivate the key
        await self.api_key_repo.update(key_id, {"is_active": False})
        
        # Log revocation
        await audit_logger.log_event(
            action="revoke_api_key",
            resource="api_key",
            resource_id=str(key_id),
            user_id=requesting_user.id,
            details={"key_name": api_key.name}
        )
        
        return True
    
    async def delete_api_key(
        self, 
        key_id: uuid.UUID, 
        requesting_user: User
    ) -> bool:
        """Permanently delete an API key"""
        
        api_key = await self.get_api_key(key_id, requesting_user)
        
        # Check delete permissions
        if api_key.user_id != requesting_user.id:
            if not rbac_manager.has_permission(requesting_user, Permission.API_KEYS_DELETE):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permission denied: Cannot delete API key"
                )
        
        # Delete the key
        await self.api_key_repo.delete(key_id)
        
        # Log deletion
        await audit_logger.log_event(
            action="delete_api_key",
            resource="api_key",
            resource_id=str(key_id),
            user_id=requesting_user.id,
            details={"key_name": api_key.name}
        )
        
        return True
    
    async def verify_api_key(self, api_key: str) -> Optional[APIKey]:
        """Verify an API key and return the key object if valid"""
        
        if not api_key.startswith("mk_"):
            return None
        
        key_hash = security_manager.hash_api_key(api_key)
        api_key_obj = await self.api_key_repo.get_by_hash(key_hash)
        
        if not api_key_obj:
            return None
        
        # Check if key is active
        if not api_key_obj.is_active:
            return None
        
        # Check if key is expired
        if api_key_obj.expires_at and api_key_obj.expires_at < datetime.utcnow():
            # Automatically deactivate expired keys
            await self.api_key_repo.update(api_key_obj.id, {"is_active": False})
            return None
        
        return api_key_obj
    
    async def rotate_api_key(
        self, 
        key_id: uuid.UUID, 
        requesting_user: User
    ) -> APIKeyResponse:
        """Rotate an API key (generate new key value)"""
        
        api_key = await self.get_api_key(key_id, requesting_user)
        
        # Check write permissions
        if api_key.user_id != requesting_user.id:
            if not rbac_manager.has_permission(requesting_user, Permission.API_KEYS_WRITE):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permission denied: Cannot rotate API key"
                )
        
        # Generate new API key
        new_key, new_hash = security_manager.generate_api_key()
        
        # Update the key
        updates = {
            "key_prefix": new_key[:8] + "...",
            "key_hash": new_hash,
            "last_used": None  # Reset last used
        }
        
        updated_key = await self.api_key_repo.update(key_id, updates)
        
        # Log rotation
        await audit_logger.log_event(
            action="rotate_api_key",
            resource="api_key",
            resource_id=str(key_id),
            user_id=requesting_user.id,
            details={"key_name": api_key.name}
        )
        
        return APIKeyResponse(
            id=updated_key.id,
            name=updated_key.name,
            key=new_key,  # Return new key
            permissions=updated_key.permissions,
            expires_at=updated_key.expires_at
        )
    
    async def get_api_key_stats(self, requesting_user: User) -> Dict[str, Any]:
        """Get API key usage statistics"""
        
        if not rbac_manager.has_permission(requesting_user, Permission.API_KEYS_READ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied: Cannot view API key statistics"
            )
        
        return await self.api_key_repo.get_usage_stats()
    
    async def cleanup_expired_keys(self) -> int:
        """Clean up expired API keys (background task)"""
        
        expired_keys = await self.api_key_repo.get_expired_keys()
        count = 0
        
        for api_key in expired_keys:
            await self.api_key_repo.update(api_key.id, {"is_active": False})
            count += 1
            
            # Log cleanup
            await audit_logger.log_event(
                action="cleanup_expired_key",
                resource="api_key",
                resource_id=str(api_key.id),
                user_id=None,  # System action
                details={"key_name": api_key.name, "expired_at": api_key.expires_at.isoformat()}
            )
        
        return count


# Global API key manager instance
api_key_manager = APIKeyManager()


# MCP-specific API key management
class MCPAPIKeyManager:
    """Specialized API key management for MCP clients"""
    
    def __init__(self):
        self.api_key_manager = api_key_manager
    
    async def create_mcp_api_key(
        self, 
        user: User, 
        client_name: str,
        client_version: Optional[str] = None,
        capabilities: List[str] = None
    ) -> APIKeyResponse:
        """Create an API key specifically for MCP clients"""
        
        # Default MCP permissions
        mcp_permissions = [
            Permission.MEMORIES_READ,
            Permission.MEMORIES_WRITE,
            Permission.PROFILE_READ
        ]
        
        # Filter permissions based on user role
        user_permissions = rbac_manager.get_user_permissions(user)
        allowed_permissions = [p for p in mcp_permissions if p in user_permissions]
        
        key_data = APIKeyCreate(
            name=f"MCP-{client_name}",
            permissions=allowed_permissions,
            expires_in_days=365  # 1 year expiration for MCP keys
        )
        
        api_key_response = await self.api_key_manager.create_api_key(user, key_data)
        
        # Log MCP key creation with additional metadata
        await audit_logger.log_event(
            action="create_mcp_api_key",
            resource="api_key",
            resource_id=str(api_key_response.id),
            user_id=user.id,
            details={
                "client_name": client_name,
                "client_version": client_version,
                "capabilities": capabilities or [],
                "mcp_specific": True
            }
        )
        
        return api_key_response
    
    async def validate_mcp_client(self, api_key: str, client_info: Dict[str, Any]) -> bool:
        """Validate MCP client and API key combination"""
        
        api_key_obj = await self.api_key_manager.verify_api_key(api_key)
        if not api_key_obj:
            return False
        
        # Additional MCP-specific validation could go here
        # e.g., client version compatibility, capability checks
        
        return True


# Global MCP API key manager instance
mcp_api_key_manager = MCPAPIKeyManager()