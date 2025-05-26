"""Configuration management for Mem0 AI MCP Server.

This module provides comprehensive configuration management using Pydantic
for validation and environment variable loading. All settings are centralized
here with proper defaults and validation rules.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Settings(BaseModel):
    """Application settings with comprehensive validation.
    
    This class defines all configuration options for the Mem0 AI MCP Server,
    including database connections, authentication settings, vector search
    parameters, and various operational limits. All settings can be configured
    via environment variables.
    
    Environment variables should be prefixed with the setting name in uppercase.
    For example, SUPABASE_URL for the supabase_url setting.
    
    Attributes:
        supabase_url: Supabase project URL for database connection
        supabase_key: Supabase API key for authentication
        database_url: PostgreSQL connection URL
        redis_url: Redis connection URL for caching
        secret_key: Secret key for JWT token signing
        algorithm: JWT signing algorithm (default: HS256)
        access_token_expire_minutes: JWT token expiration time
        openai_api_key: OpenAI API key for embeddings
        host: Server host address
        port: Server port number
        debug: Enable debug mode
        log_level: Logging level (debug, info, warning, error)
        embedding_model: OpenAI embedding model to use
        vector_dimension: Dimension of embedding vectors
        similarity_threshold: Minimum similarity score for search results
        max_memory_size: Maximum size of memory content in characters
        memory_ttl_days: Memory time-to-live in days
        auto_cleanup_enabled: Enable automatic cleanup of expired memories
        plugins_dir: Directory path for plugin files
        enable_plugins: Enable plugin system
    """

    # Database Configuration
    supabase_url: str = Field(..., env="SUPABASE_URL")
    supabase_key: str = Field(..., env="SUPABASE_KEY")
    database_url: str = Field(..., env="DATABASE_URL")

    # Redis Configuration
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")

    # Authentication
    secret_key: str = Field(..., env="SECRET_KEY")
    algorithm: str = Field("HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")

    # Server Configuration
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("info", env="LOG_LEVEL")

    # Vector Search Configuration
    embedding_model: str = Field("text-embedding-ada-002", env="EMBEDDING_MODEL")
    vector_dimension: int = Field(1536, env="VECTOR_DIMENSION")
    similarity_threshold: float = Field(0.7, env="SIMILARITY_THRESHOLD")

    # Memory Configuration
    max_memory_size: int = Field(1000000, env="MAX_MEMORY_SIZE")
    memory_ttl_days: int = Field(365, env="MEMORY_TTL_DAYS")
    auto_cleanup_enabled: bool = Field(True, env="AUTO_CLEANUP_ENABLED")

    # Plugin Configuration
    plugins_dir: str = Field("./plugins", env="PLUGINS_DIR")
    enable_plugins: bool = Field(True, env="ENABLE_PLUGINS")

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
        use_enum_values=True,
    )
    
    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Validate secret key strength."""
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        return v
    
    @field_validator("access_token_expire_minutes")
    @classmethod
    def validate_token_expire_minutes(cls, v: int) -> int:
        """Validate token expiration time."""
        if v <= 0 or v > 43200:  # Max 30 days
            raise ValueError("Token expiration must be between 1 and 43200 minutes")
        return v
    
    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port number."""
        if v < 1 or v > 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    @field_validator("vector_dimension")
    @classmethod
    def validate_vector_dimension(cls, v: int) -> int:
        """Validate vector dimension."""
        if v <= 0 or v > 10000:
            raise ValueError("Vector dimension must be between 1 and 10000")
        return v
    
    @field_validator("similarity_threshold")
    @classmethod
    def validate_similarity_threshold(cls, v: float) -> float:
        """Validate similarity threshold."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
        return v
    
    @field_validator("max_memory_size")
    @classmethod
    def validate_max_memory_size(cls, v: int) -> int:
        """Validate maximum memory size."""
        if v <= 0 or v > 10000000:  # 10MB limit
            raise ValueError("Max memory size must be between 1 and 10,000,000 characters")
        return v
    
    @field_validator("memory_ttl_days")
    @classmethod
    def validate_memory_ttl_days(cls, v: int) -> int:
        """Validate memory TTL."""
        if v <= 0 or v > 3650:  # Max 10 years
            raise ValueError("Memory TTL must be between 1 and 3650 days")
        return v
