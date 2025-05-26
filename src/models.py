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
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

# Constants for validation limits
MAX_TAGS = 50  # Maximum number of tags allowed
MAX_TAG_LENGTH = 100  # Maximum length of a single tag
MIN_PASSWORD_LENGTH = 8  # Minimum password length
MAX_RESPONSE_TIME_SECONDS = 60  # Maximum reasonable response time

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
    metadata: dict[str, Any] | None = Field(default_factory=dict)
    tags: list[str] | None = Field(default_factory=list)
    memory_type: MemoryType = Field(MemoryType.FACT)
    priority: MemoryPriority = Field(MemoryPriority.MEDIUM)
    source: str | None = Field(
        None, description="Source of the memory", max_length=1000
    )
    context: str | None = Field(
        None, description="Additional context", max_length=10000
    )
    expires_at: datetime | None = Field(None)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        """Validate metadata is a JSON-serializable dictionary."""
        if v is not None:
            if not isinstance(v, dict):
                raise ValueError("Metadata must be a dictionary")
            # Ensure metadata is JSON serializable
            try:
                json.dumps(v)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Metadata must be JSON serializable: {e}") from e
        return v

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: list[str] | None) -> list[str] | None:
        """Validate tags are a list of non-empty strings."""
        if v is not None:
            if not isinstance(v, list):
                raise ValueError("Tags must be a list")
            if len(v) > MAX_TAGS:  # Limit number of tags
                raise ValueError(f"Cannot have more than {MAX_TAGS} tags")
            for tag in v:
                if not isinstance(tag, str) or not tag.strip():
                    raise ValueError("All tags must be non-empty strings")
                if len(tag) > MAX_TAG_LENGTH:
                    raise ValueError(f"Tag length cannot exceed {MAX_TAG_LENGTH} characters")
        return v

    @field_validator("expires_at")
    @classmethod
    def validate_expires_at(cls, v: datetime | None) -> datetime | None:
        """Validate expiration date is in the future."""
        if v is not None and v <= datetime.now():
            raise ValueError("Expiration date must be in the future")
        return v


class MemoryResponse(BaseModel):
    """Model for memory responses."""

    id: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    memory_type: MemoryType
    priority: MemoryPriority
    source: str | None = None
    context: str | None = None
    user_id: str = Field(..., min_length=1)
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None
    access_count: int = Field(0, ge=0)
    similarity_score: float | None = Field(None, ge=0.0, le=1.0)

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: list[float] | None) -> list[float] | None:
        """Validate embedding is a list of numbers."""
        if v is not None:
            if not isinstance(v, list):
                raise ValueError("Embedding must be a list")
            if not all(isinstance(x, int | float) for x in v):
                raise ValueError("Embedding values must be numbers")
            if len(v) == 0:
                raise ValueError("Embedding cannot be empty")
        return v

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: list[str]) -> list[str]:
        """Validate tags are a list of strings."""
        if not isinstance(v, list):
            raise ValueError("Tags must be a list")
        for tag in v:
            if not isinstance(tag, str):
                raise ValueError("All tags must be strings")
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate metadata is a dictionary."""
        if not isinstance(v, dict):
            raise ValueError("Metadata must be a dictionary")
        return v


class MemorySearch(BaseModel):
    """Model for memory search requests."""

    query: str = Field(..., description="Search query", min_length=1, max_length=10000)
    limit: int = Field(10, ge=1, le=100)
    threshold: float | None = Field(None, ge=0.0, le=1.0)
    memory_types: list[MemoryType] | None = None
    tags: list[str] | None = None
    include_embeddings: bool = False
    user_id: str | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: list[str] | None) -> list[str] | None:
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
    def validate_memory_types(cls, v: list[MemoryType] | None) -> list[MemoryType] | None:
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
    def validate_date_range(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate date range is logical."""
        date_from = values.get("date_from")
        date_to = values.get("date_to")

        if date_from and date_to and date_from > date_to:
            raise ValueError("date_from must be before date_to")

        return values


class MemoryUpdate(BaseModel):
    """Model for updating memories."""

    content: str | None = Field(None, min_length=1, max_length=100000)
    metadata: dict[str, Any] | None = None
    tags: list[str] | None = None
    memory_type: MemoryType | None = None
    priority: MemoryPriority | None = None
    context: str | None = Field(None, max_length=10000)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        """Validate metadata is a JSON-serializable dictionary."""
        if v is not None:
            if not isinstance(v, dict):
                raise ValueError("Metadata must be a dictionary")
            try:
                json.dumps(v)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Metadata must be JSON serializable: {e}") from e
        return v

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: list[str] | None) -> list[str] | None:
        """Validate tags are a list of non-empty strings."""
        if v is not None:
            if not isinstance(v, list):
                raise ValueError("Tags must be a list")
            if len(v) > MAX_TAGS:
                raise ValueError(f"Cannot have more than {MAX_TAGS} tags")
            for tag in v:
                if not isinstance(tag, str) or not tag.strip():
                    raise ValueError("All tags must be non-empty strings")
                if len(tag) > MAX_TAG_LENGTH:
                    raise ValueError(f"Tag length cannot exceed {MAX_TAG_LENGTH} characters")
        return v

    @model_validator(mode="before")
    @classmethod
    def validate_update_fields(cls, values: dict[str, Any]) -> dict[str, Any]:
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
    full_name: str | None = Field(None, max_length=200)

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
        if len(v) < MIN_PASSWORD_LENGTH:
            raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long")
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
    full_name: str | None
    is_active: bool
    created_at: datetime


class Token(BaseModel):
    """Authentication token model."""

    access_token: str
    token_type: str = "bearer"  # noqa: S105


class MCPRequest(BaseModel):
    """Model Context Protocol request."""

    id: str | int | None = None
    method: str
    params: dict[str, Any] | None = None

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate method is a non-empty string."""
        if not v or not isinstance(v, str):
            raise ValueError("Method must be a non-empty string")
        return v

    @field_validator("params")
    @classmethod
    def validate_params(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        """Validate params is a dictionary."""
        if v is not None and not isinstance(v, dict):
            raise ValueError("Params must be a dictionary")
        return v


class MCPResponse(BaseModel):
    """Model Context Protocol response."""

    id: str | int | None = None
    result: Any | None = None
    error: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_response(cls, values: dict[str, Any]) -> dict[str, Any]:
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
    def validate_error(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
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
    settings: dict[str, Any] = Field(default_factory=dict)


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
        if v > MAX_RESPONSE_TIME_SECONDS:  # More than 60 seconds seems unreasonable
            raise ValueError("Average response time seems too high")
        return v
