"""Pydantic models for Mem0 AI MCP Server.

This module contains all Pydantic models used throughout the application
for data validation, serialization, and API contracts. Models are organized
by functionality: memory management, user authentication, MCP protocol,
and plugin configuration.

All models include comprehensive validation, proper type hints, and
documentation for each field.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class MemoryType(str, Enum):
    """Memory classification types."""

    FACT = "fact"
    CONVERSATION = "conversation"
    TASK = "task"
    PREFERENCE = "preference"
    SKILL = "skill"
    CONTEXT = "context"


class MemoryPriority(str, Enum):
    """Memory priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MemoryCreate(BaseModel):
    """Model for creating a new memory."""

    content: str = Field(
        ..., description="Memory content", min_length=1, max_length=100000
    )
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    tags: Optional[List[str]] = Field(default_factory=list)
    memory_type: MemoryType = Field(MemoryType.FACT)
    priority: MemoryPriority = Field(MemoryPriority.MEDIUM)
    source: Optional[str] = Field(
        None, description="Source of the memory", max_length=1000
    )
    context: Optional[str] = Field(
        None, description="Additional context", max_length=10000
    )
    expires_at: Optional[datetime] = Field(None)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate metadata is a JSON-serializable dictionary."""
        if v is not None:
            if not isinstance(v, dict):
                raise ValueError("Metadata must be a dictionary")
            # Ensure metadata is JSON serializable
            try:
                json.dumps(v)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Metadata must be JSON serializable: {e}")
        return v

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate tags are a list of non-empty strings."""
        if v is not None:
            if not isinstance(v, list):
                raise ValueError("Tags must be a list")
            if len(v) > 50:  # Limit number of tags
                raise ValueError("Cannot have more than 50 tags")
            for tag in v:
                if not isinstance(tag, str) or not tag.strip():
                    raise ValueError("All tags must be non-empty strings")
                if len(tag) > 100:
                    raise ValueError("Tag length cannot exceed 100 characters")
        return v

    @field_validator("expires_at")
    @classmethod
    def validate_expires_at(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Validate expiration date is in the future."""
        if v is not None and v <= datetime.now():
            raise ValueError("Expiration date must be in the future")
        return v


class MemoryResponse(BaseModel):
    """Model for memory responses."""

    id: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    memory_type: MemoryType
    priority: MemoryPriority
    source: Optional[str] = None
    context: Optional[str] = None
    user_id: str = Field(..., min_length=1)
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = Field(0, ge=0)
    similarity_score: Optional[float] = Field(None, ge=0.0, le=1.0)

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """Validate embedding is a list of numbers."""
        if v is not None:
            if not isinstance(v, list):
                raise ValueError("Embedding must be a list")
            if not all(isinstance(x, (int, float)) for x in v):
                raise ValueError("Embedding values must be numbers")
            if len(v) == 0:
                raise ValueError("Embedding cannot be empty")
        return v

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        """Validate tags are a list of strings."""
        if not isinstance(v, list):
            raise ValueError("Tags must be a list")
        for tag in v:
            if not isinstance(tag, str):
                raise ValueError("All tags must be strings")
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate metadata is a dictionary."""
        if not isinstance(v, dict):
            raise ValueError("Metadata must be a dictionary")
        return v


class MemorySearch(BaseModel):
    """Model for memory search requests."""

    query: str = Field(..., description="Search query", min_length=1, max_length=10000)
    limit: int = Field(10, ge=1, le=100)
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    memory_types: Optional[List[MemoryType]] = None
    tags: Optional[List[str]] = None
    include_embeddings: bool = False
    user_id: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate tags are a list of non-empty strings."""
        if v is not None:
            if not isinstance(v, list):
                raise ValueError("Tags must be a list")
            for tag in v:
                if not isinstance(tag, str) or not tag.strip():
                    raise ValueError("All tags must be non-empty strings")
        return v

    @field_validator("memory_types")
    @classmethod
    def validate_memory_types(cls, v: Optional[List[MemoryType]]) -> Optional[List[MemoryType]]:
        """Validate memory types are valid enum values."""
        if v is not None:
            if not isinstance(v, list):
                raise ValueError("Memory types must be a list")
            for mem_type in v:
                if not isinstance(mem_type, MemoryType):
                    raise ValueError(f"Invalid memory type: {mem_type}")
        return v

    @model_validator(mode="before")
    @classmethod
    def validate_date_range(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate date range is logical."""
        date_from = values.get("date_from")
        date_to = values.get("date_to")

        if date_from and date_to and date_from > date_to:
            raise ValueError("date_from must be before date_to")

        return values


class MemoryUpdate(BaseModel):
    """Model for updating memories."""

    content: Optional[str] = Field(None, min_length=1, max_length=100000)
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    memory_type: Optional[MemoryType] = None
    priority: Optional[MemoryPriority] = None
    context: Optional[str] = Field(None, max_length=10000)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate metadata is a JSON-serializable dictionary."""
        if v is not None:
            if not isinstance(v, dict):
                raise ValueError("Metadata must be a dictionary")
            try:
                json.dumps(v)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Metadata must be JSON serializable: {e}")
        return v

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate tags are a list of non-empty strings."""
        if v is not None:
            if not isinstance(v, list):
                raise ValueError("Tags must be a list")
            if len(v) > 50:
                raise ValueError("Cannot have more than 50 tags")
            for tag in v:
                if not isinstance(tag, str) or not tag.strip():
                    raise ValueError("All tags must be non-empty strings")
                if len(tag) > 100:
                    raise ValueError("Tag length cannot exceed 100 characters")
        return v

    @model_validator(mode="before")
    @classmethod
    def validate_update_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate at least one update field is provided."""
        # At least one field must be provided for update
        update_fields = [
            "content",
            "metadata",
            "tags",
            "memory_type",
            "priority",
            "context",
        ]
        if not any(values.get(field) is not None for field in update_fields):
            raise ValueError("At least one field must be provided for update")
        return values


class UserCreate(BaseModel):
    """Model for creating users."""

    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
    email: str = Field(..., pattern=r"^[^@]+@[^@]+\.[^@]+$")
    password: str = Field(..., min_length=8, max_length=128)
    full_name: Optional[str] = Field(None, max_length=200)

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Validate username format and reserved words."""
        if not v or not v.strip():
            raise ValueError("Username cannot be empty")
        if v.lower() in ["admin", "root", "system", "api", "null", "undefined"]:
            raise ValueError("Username is reserved")
        return v.strip()

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength requirements."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserResponse(BaseModel):
    """Model for user responses."""

    id: str
    username: str
    email: str
    full_name: Optional[str]
    is_active: bool
    created_at: datetime


class Token(BaseModel):
    """Authentication token model."""

    access_token: str
    token_type: str = "bearer"


class MCPRequest(BaseModel):
    """Model Context Protocol request."""

    id: Optional[Union[str, int]] = None
    method: str
    params: Optional[Dict[str, Any]] = None

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate method is a non-empty string."""
        if not v or not isinstance(v, str):
            raise ValueError("Method must be a non-empty string")
        return v

    @field_validator("params")
    @classmethod
    def validate_params(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate params is a dictionary."""
        if v is not None and not isinstance(v, dict):
            raise ValueError("Params must be a dictionary")
        return v


class MCPResponse(BaseModel):
    """Model Context Protocol response."""

    id: Optional[Union[str, int]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None

    @model_validator(mode="before")
    @classmethod
    def validate_response(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate response has either result or error, not both."""
        result = values.get("result")
        error = values.get("error")

        # Either result or error must be present, but not both
        if result is not None and error is not None:
            raise ValueError("Response cannot have both result and error")
        if result is None and error is None:
            raise ValueError("Response must have either result or error")

        return values

    @field_validator("error")
    @classmethod
    def validate_error(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate error structure with code and message."""
        if v is not None:
            if not isinstance(v, dict):
                raise ValueError("Error must be a dictionary")
            if "code" not in v or "message" not in v:
                raise ValueError("Error must contain code and message fields")
            if not isinstance(v["code"], int):
                raise ValueError("Error code must be an integer")
            if not isinstance(v["message"], str):
                raise ValueError("Error message must be a string")
        return v


class PluginConfig(BaseModel):
    """Plugin configuration model."""

    name: str
    version: str
    description: str
    enabled: bool = True
    settings: Dict[str, Any] = Field(default_factory=dict)


class MetricsData(BaseModel):
    """Metrics data model."""

    memory_count: int = Field(..., ge=0)
    user_count: int = Field(..., ge=0)
    search_requests: int = Field(..., ge=0)
    average_response_time: float = Field(..., ge=0.0)
    uptime_seconds: int = Field(..., ge=0)
    timestamp: datetime

    @field_validator("average_response_time")
    @classmethod
    def validate_response_time(cls, v: float) -> float:
        """Validate response time is reasonable."""
        if v < 0:
            raise ValueError("Average response time cannot be negative")
        if v > 60:  # More than 60 seconds seems unreasonable
            raise ValueError("Average response time seems too high")
        return v
