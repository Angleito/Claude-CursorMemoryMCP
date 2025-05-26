"""Database connection and management for Supabase + pgvector"""

# Standard library imports
import asyncio
import contextlib
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union, Tuple

# Third-party imports
import asyncpg
import structlog
from pgvector.asyncpg import register_vector
from supabase import create_client, Client

# Local imports
from .config import Settings

logger = structlog.get_logger()


class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.supabase: Optional[Client] = None
        self.pool: Optional[asyncpg.Pool] = None
        self._initialized = False
        self._initialization_lock = asyncio.Lock()
        self._stats = {
            'total_queries': 0,
            'failed_queries': 0,
            'connection_errors': 0,
            'last_health_check': None
        }
        
    async def initialize(self):
        """Initialize database connections with proper error handling and validation"""
        async with self._initialization_lock:
            if self._initialized:
                return
                
            try:
                # Validate settings
                if not self.settings.supabase_url or not self.settings.supabase_key:
                    raise ValueError("Supabase URL and key are required")
                if not self.settings.database_url:
                    raise ValueError("Database URL is required")
                
                # Initialize Supabase client with proper error handling
                try:
                    self.supabase = create_client(
                        self.settings.supabase_url,
                        self.settings.supabase_key
                    )
                    logger.info("Supabase client initialized")
                except Exception as e:
                    logger.error("Failed to initialize Supabase client", error=str(e))
                    raise
                
                # Initialize asyncpg connection pool with optimized settings
                try:
                    self.pool = await asyncpg.create_pool(
                        self.settings.database_url,
                        min_size=5,
                        max_size=20,
                        max_queries=50000,
                        max_inactive_connection_lifetime=300.0,
                        command_timeout=60,
                        server_settings={
                            'application_name': 'mem0ai_server',
                            'jit': 'off'  # Disable JIT for predictable performance
                        }
                    )
                    logger.info("Database connection pool initialized", 
                              min_size=5, max_size=20)
                except Exception as e:
                    logger.error("Failed to initialize connection pool", error=str(e))
                    raise
                
                # Register pgvector extension and create tables
                async with self.pool.acquire() as conn:
                    try:
                        await register_vector(conn)
                        await self.create_tables(conn)
                        await self._perform_health_check(conn)
                        logger.info("Database schema initialized")
                    except Exception as e:
                        logger.error("Failed to initialize database schema", error=str(e))
                        raise
                
                self._initialized = True
                self._stats['last_health_check'] = datetime.now()
                logger.info("Database initialized successfully")
                
            except Exception as e:
                logger.error("Failed to initialize database", error=str(e))
                await self._cleanup_on_error()
                raise
    
    async def _cleanup_on_error(self):
        """Cleanup resources on initialization error"""
        if self.pool:
            try:
                await self.pool.close()
                self.pool = None
            except Exception as e:
                logger.error("Error during cleanup", error=str(e))
        self.supabase = None
        self._initialized = False
    
    async def _perform_health_check(self, conn: asyncpg.Connection) -> bool:
        """Perform database health check"""
        try:
            await conn.fetchval("SELECT 1")
            await conn.fetchval("SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'")
            return True
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return False
    
    async def create_tables(self, conn: asyncpg.Connection):
        """Create necessary database tables with proper constraints and indexes"""
        
        # Enable required extensions
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")
        
        # Users table with proper constraints
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                username VARCHAR(50) UNIQUE NOT NULL CHECK (length(username) >= 3),
                email VARCHAR(255) UNIQUE NOT NULL CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'),
                password_hash VARCHAR(255) NOT NULL,
                full_name VARCHAR(255),
                is_active BOOLEAN DEFAULT true,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                CONSTRAINT users_username_valid CHECK (username !~ '\\s')
            )
        """)
        
        # Memories table with vector support and proper constraints
        # Validate settings to prevent SQL injection
        max_size = self.settings.max_memory_size
        vector_dim = self.settings.vector_dimension
        
        if not isinstance(max_size, int) or max_size <= 0 or max_size > 1000000:
            raise ValueError(f"Invalid max_memory_size: {max_size}")
        if not isinstance(vector_dim, int) or vector_dim <= 0 or vector_dim > 10000:
            raise ValueError(f"Invalid vector_dimension: {vector_dim}")
            
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                content TEXT NOT NULL CHECK (length(content) > 0 AND length(content) <= $1),
                embedding vector($2),
                metadata JSONB DEFAULT '{}',
                tags TEXT[] DEFAULT ARRAY[]::TEXT[],
                memory_type VARCHAR(50) DEFAULT 'fact' CHECK (memory_type IN ('fact', 'conversation', 'task', 'preference', 'skill', 'context')),
                priority VARCHAR(20) DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high', 'critical')),
                source VARCHAR(255),
                context TEXT,
                access_count INTEGER DEFAULT 0 CHECK (access_count >= 0),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                expires_at TIMESTAMP WITH TIME ZONE,
                CONSTRAINT valid_expiry CHECK (expires_at IS NULL OR expires_at > created_at)
            )
        """, max_size, vector_dim)
        
        # Memory search history
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS search_history (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                query TEXT NOT NULL,
                results_count INTEGER,
                execution_time_ms FLOAT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Plugin configurations
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS plugin_configs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                plugin_name VARCHAR(255) NOT NULL,
                config JSONB DEFAULT '{}',
                enabled BOOLEAN DEFAULT true,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(user_id, plugin_name)
            )
        """)
        
        # Create optimized indexes for better performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_memories_user_created ON memories(user_id, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_memories_user_type ON memories(user_id, memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_memories_user_priority ON memories(user_id, priority)",
            "CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories USING GIN(tags)",
            "CREATE INDEX IF NOT EXISTS idx_memories_metadata ON memories USING GIN(metadata)",
            "CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_memories_expires_at ON memories(expires_at) WHERE expires_at IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active) WHERE is_active = true",
            "CREATE INDEX IF NOT EXISTS idx_users_email_active ON users(email) WHERE is_active = true",
            "CREATE INDEX IF NOT EXISTS idx_search_history_user_created ON search_history(user_id, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_plugin_configs_user_enabled ON plugin_configs(user_id, enabled) WHERE enabled = true"
        ]
        
        for index_sql in indexes:
            await conn.execute(index_sql)
        
        # Create vector index separately with error handling and proper validation
        try:
            # Calculate optimal lists parameter safely
            lists_param = max(100, min(1000, vector_dim // 10))
            
            # Try HNSW first (better for most use cases)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw ON memories 
                USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
                WHERE embedding IS NOT NULL
            """)
            
        except Exception as e:
            logger.warning("HNSW index creation failed, trying IVFFlat fallback", error=str(e))
            # Fallback to IVFFlat index
            try:
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_embedding_ivf ON memories 
                    USING ivfflat (embedding vector_cosine_ops) WITH (lists = $1)
                    WHERE embedding IS NOT NULL
                """, lists_param)
            except Exception as e2:
                logger.error("Vector index creation failed completely", error=str(e2))
                # Continue without vector index - will impact performance but won't break functionality
            try:
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw ON memories 
                    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
                """)
            except Exception as e2:
                logger.error("Both vector index types failed", ivf_error=str(e), hnsw_error=str(e2))
        
        # Create triggers for updated_at
        await conn.execute("""
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = NOW();
                RETURN NEW;
            END;
            $$ language 'plpgsql'
        """)
        
        await conn.execute("""
            DROP TRIGGER IF EXISTS update_users_updated_at ON users
        """)
        
        await conn.execute("""
            CREATE TRIGGER update_users_updated_at 
                BEFORE UPDATE ON users 
                FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()
        """)
        
        await conn.execute("""
            DROP TRIGGER IF EXISTS update_memories_updated_at ON memories
        """)
        
        await conn.execute("""
            CREATE TRIGGER update_memories_updated_at 
                BEFORE UPDATE ON memories 
                FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()
        """)
        
        logger.info("Database tables created successfully")
    
    def _validate_query_security(self, query: str, args: tuple) -> None:
        """Comprehensive security validation for SQL queries"""
        if not isinstance(query, str):
            raise ValueError("Query must be a string")
            
        if len(query) > 10000:  # Prevent extremely long queries
            raise ValueError("Query too long (max 10,000 characters)")
            
        if not query.strip():
            raise ValueError("Query cannot be empty")
            
        # Normalize query for analysis
        query_upper = query.upper().strip()
        
        # Check for dangerous SQL patterns
        dangerous_patterns = [
            'DROP TABLE', 'DROP DATABASE', 'DROP SCHEMA', 'DROP INDEX',
            'TRUNCATE', 'DELETE FROM users', 'DELETE FROM memories',
            'ALTER TABLE', 'ALTER DATABASE', 'CREATE USER', 'DROP USER',
            'GRANT', 'REVOKE', 'SET SESSION', 'SET GLOBAL',
            'LOAD_FILE', 'INTO OUTFILE', 'INTO DUMPFILE',
            'UNION SELECT', '--', '/*', '*/',
            'EXEC(', 'EXECUTE(', 'SP_', 'XP_'
        ]
        
        for pattern in dangerous_patterns:
            if pattern in query_upper:
                # Allow safe patterns in specific contexts
                if pattern == 'DROP TABLE' and 'CREATE TABLE IF NOT EXISTS' in query_upper:
                    continue
                if pattern == 'DROP TRIGGER' and 'CREATE TRIGGER' in query_upper:
                    continue
                if pattern == '--' and query_upper.startswith('--'):
                    continue  # Allow comment-only queries
                    
                raise ValueError(f"Potentially dangerous SQL pattern detected: {pattern}")
        
        # Validate that parameterized queries are used properly
        if len(args) > 0 and not any(f'${i+1}' in query for i, _ in enumerate(args)):
            logger.warning("Query parameter mismatch detected", query=query[:100])
            
        # Check for SQL injection patterns in parameters
        for i, arg in enumerate(args):
            if isinstance(arg, str):
                if len(arg) > 1000000:  # 1MB limit per parameter
                    raise ValueError(f"Parameter {i} too large")
                    
                # Check for SQL injection attempts in string parameters
                dangerous_in_params = ["';", '";', '/*', '*/', '--', 'UNION', 'SELECT']
                arg_upper = arg.upper()
                for dangerous in dangerous_in_params:
                    if dangerous in arg_upper:
                        logger.warning(f"Potentially dangerous content in parameter {i}", content=arg[:50])
                        # Don't raise exception as this might be legitimate content
                        
        # Ensure query starts with expected operations
        allowed_start_patterns = [
            'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH',
            'CREATE TABLE', 'CREATE INDEX', 'CREATE TRIGGER', 'CREATE FUNCTION',
            'DROP TRIGGER', 'DROP INDEX',  # Only if followed by CREATE
            'EXPLAIN', 'ANALYZE',
            '--'  # Comments
        ]
        
        if not any(query_upper.startswith(pattern) for pattern in allowed_start_patterns):
            raise ValueError(f"Query starts with disallowed operation: {query_upper[:20]}")
            
        # Additional validation for specific operations
        if query_upper.startswith('DELETE'):
            if 'WHERE' not in query_upper:
                raise ValueError("DELETE queries must include WHERE clause")
                
        if query_upper.startswith('UPDATE'):
            if 'WHERE' not in query_upper:
                raise ValueError("UPDATE queries must include WHERE clause")
                
        # Limit number of parameters to prevent DoS
        if len(args) > 100:
            raise ValueError("Too many parameters (max 100)")
    
    @contextlib.asynccontextmanager
    async def get_connection(self):
        """Get a database connection with proper resource management"""
        if not self._initialized or not self.pool:
            raise RuntimeError("Database not initialized")
        
        conn = None
        try:
            conn = await self.pool.acquire()
            yield conn
        except Exception as e:
            self._stats['connection_errors'] += 1
            logger.error("Connection error", error=str(e))
            raise
        finally:
            if conn and self.pool:
                try:
                    await self.pool.release(conn)
                except Exception as e:
                    logger.error("Error releasing connection", error=str(e))
    
    @contextlib.asynccontextmanager
    async def get_transaction(self):
        """Get a database transaction with proper rollback handling"""
        async with self.get_connection() as conn:
            transaction = conn.transaction()
            try:
                await transaction.start()
                yield conn
                await transaction.commit()
            except Exception as e:
                await transaction.rollback()
                logger.error("Transaction rolled back", error=str(e))
                raise
    
    async def execute_query(self, query: str, *args, timeout: Optional[float] = None) -> List[asyncpg.Record]:
        """Execute a query and return the result with comprehensive security validation"""
        # Comprehensive input validation
        self._validate_query_security(query, args)
        
        start_time = time.time()
        try:
            async with self.get_connection() as conn:
                result = await asyncio.wait_for(
                    conn.fetch(query, *args),
                    timeout=timeout or 30.0
                )
                self._stats['total_queries'] += 1
                execution_time = (time.time() - start_time) * 1000
                if execution_time > 1000:  # Log slow queries
                    logger.warning("Slow query detected", 
                                 query=query[:100], 
                                 execution_time_ms=execution_time)
                return result
        except Exception as e:
            self._stats['failed_queries'] += 1
            logger.error("Query execution failed", 
                        query=query[:100], 
                        error=str(e), 
                        args_count=len(args))
            raise
    
    async def execute_one(self, query: str, *args, timeout: Optional[float] = None) -> Optional[asyncpg.Record]:
        """Execute a query and return a single result with security validation"""
        # Comprehensive input validation
        self._validate_query_security(query, args)
        
        start_time = time.time()
        try:
            async with self.get_connection() as conn:
                result = await asyncio.wait_for(
                    conn.fetchrow(query, *args),
                    timeout=timeout or 30.0
                )
                self._stats['total_queries'] += 1
                execution_time = (time.time() - start_time) * 1000
                if execution_time > 1000:
                    logger.warning("Slow query detected", 
                                 query=query[:100], 
                                 execution_time_ms=execution_time)
                return result
        except Exception as e:
            self._stats['failed_queries'] += 1
            logger.error("Query execution failed", 
                        query=query[:100], 
                        error=str(e))
            raise
    
    async def execute_command(self, query: str, *args, timeout: Optional[float] = None) -> str:
        """Execute a command and return status"""
        start_time = time.time()
        try:
            async with self.get_connection() as conn:
                result = await asyncio.wait_for(
                    conn.execute(query, *args),
                    timeout=timeout or 30.0
                )
                self._stats['total_queries'] += 1
                execution_time = (time.time() - start_time) * 1000
                if execution_time > 1000:
                    logger.warning("Slow command detected", 
                                 query=query[:100], 
                                 execution_time_ms=execution_time)
                return result
        except Exception as e:
            self._stats['failed_queries'] += 1
            logger.error("Command execution failed", 
                        query=query[:100], 
                        error=str(e))
            raise
    
    async def execute_batch(self, queries: List[Tuple[str, tuple]], timeout: Optional[float] = None) -> List[str]:
        """Execute multiple commands in a transaction"""
        try:
            async with self.get_transaction() as conn:
                results = []
                for query, args in queries:
                    result = await asyncio.wait_for(
                        conn.execute(query, *args),
                        timeout=timeout or 30.0
                    )
                    results.append(result)
                self._stats['total_queries'] += len(queries)
                return results
        except Exception as e:
            self._stats['failed_queries'] += len(queries)
            logger.error("Batch execution failed", error=str(e), batch_size=len(queries))
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health = {
            'database_connected': False,
            'pool_size': 0,
            'pool_max_size': 0,
            'stats': self._stats.copy(),
            'last_check': datetime.now().isoformat()
        }
        
        try:
            if self.pool:
                health['pool_size'] = self.pool.get_size()
                health['pool_max_size'] = self.pool.get_max_size()
                
                async with self.get_connection() as conn:
                    health['database_connected'] = await self._perform_health_check(conn)
                    
                self._stats['last_health_check'] = datetime.now()
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            health['error'] = str(e)
        
        return health
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = self._stats.copy()
        if self.pool:
            stats.update({
                'pool_size': self.pool.get_size(),
                'pool_max_size': self.pool.get_max_size(),
                'pool_idle_size': self.pool.get_idle_size()
            })
        return stats
    
    async def close(self):
        """Close database connections with proper cleanup"""
        if self.pool:
            try:
                # Wait for all connections to be returned
                await asyncio.wait_for(self.pool.close(), timeout=10.0)
                logger.info("Database connections closed gracefully", 
                          final_stats=self._stats)
            except asyncio.TimeoutError:
                logger.warning("Database pool close timed out")
            except Exception as e:
                logger.error("Error closing database pool", error=str(e))
            finally:
                self.pool = None
        
        self.supabase = None
        self._initialized = False