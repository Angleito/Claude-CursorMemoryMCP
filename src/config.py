"""Configuration management for Mem0 AI MCP Server"""

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class Settings(BaseModel):
    """Application settings"""
    
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
        extra="ignore"
    )