"""Secure Supabase communication layer with encryption and monitoring."""

import ssl
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import asyncpg
import structlog
from supabase import Client
from supabase import create_client

from config.settings import get_settings
from monitoring.audit_logger import audit_logger
from security.encryption import database_encryption
from security.encryption import encryption_manager

logger = structlog.get_logger()

settings = get_settings()


class SecureSupabaseClient:
    """Secure wrapper for Supabase client with encryption and auditing."""

    def __init__(self) -> None:
        self.settings = settings
        self.client: Optional[Client] = None
        self.pool: Optional[asyncpg.Pool] = None
        self.encryption = encryption_manager
        self.db_encryption = database_encryption
        self._initialized: bool = False

        # Encrypted fields configuration
        self.encrypted_fields: Dict[str, List[str]] = {
            "users": ["email", "full_name"],
            "memories": ["content", "metadata"],
            "api_keys": ["key_hash"],
            "sessions": ["user_agent"],
            "audit_logs": ["details"],
        }

    async def initialize(self) -> None:
        """Initialize Supabase client and connection pool."""
        if self._initialized:
            return
            
        try:
            if not self.client:
                # Validate settings first
                if not self.settings.database.supabase_url or not self.settings.database.supabase_anon_key:
                    raise ValueError("Supabase URL and key are required")
                if not self.settings.database.database_url:
                    raise ValueError("Database URL is required")
                    
                # Create Supabase client
                self.client = create_client(
                    self.settings.database.supabase_url,
                    self.settings.database.supabase_anon_key,
                )
                logger.info("Supabase client initialized")

                # Create direct PostgreSQL connection pool for advanced operations
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

                self.pool = await asyncpg.create_pool(
                    self.settings.database.database_url,
                    ssl=ssl_context,
                    min_size=5,
                    max_size=20,
                    command_timeout=60,
                    server_settings={
                        "application_name": "mem0ai_secure",
                    },
                )
                logger.info("Database connection pool initialized")

                # Ensure pgvector extension
                await self._ensure_extensions()
                self._initialized = True
                logger.info("Secure database initialized successfully")
                
        except Exception as e:
            logger.error("Failed to initialize secure database", error=str(e))
            await self._cleanup_on_error()
            raise

    async def close(self) -> None:
        """Close connections with proper cleanup."""
        try:
            if self.pool:
                await self.pool.close()
                logger.info("Database connections closed")
        except Exception as e:
            logger.error("Error closing database connections", error=str(e))
        finally:
            self.pool = None
            self.client = None
            self._initialized = False

    async def _ensure_extensions(self) -> None:
        """Ensure required PostgreSQL extensions are enabled."""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")
            
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
                logger.info("Database extensions ensured")
        except Exception as e:
            logger.error("Failed to ensure database extensions", error=str(e))
            raise
            
    async def _cleanup_on_error(self) -> None:
        """Cleanup resources on initialization error."""
        if self.pool:
            try:
                await self.pool.close()
            except Exception as e:
                logger.error("Error during cleanup", error=str(e))
            finally:
                self.pool = None
        self.client = None
        self._initialized = False

    # Secure CRUD operations
    async def secure_insert(
        self, table: str, data: Dict[str, Any], user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Insert data with encryption and auditing."""
        # Encrypt sensitive fields
        encrypted_data = self._encrypt_data(table, data)

        # Add metadata
        encrypted_data.update(
            {
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }
        )

        # Insert via Supabase
        result = self.client.table(table).insert(encrypted_data).execute()

        if result.data:
            # Decrypt result for return
            decrypted_result = self._decrypt_data(table, result.data[0])

            # Log operation
            await self._log_operation(
                "insert", table, decrypted_result.get("id"), user_id, encrypted_data
            )

            return decrypted_result
        else:
            raise RuntimeError(f"Failed to insert into {table}: {result}")

    async def secure_select(
        self,
        table: str,
        filters: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Select data with decryption."""
        query = self.client.table(table).select(
            "*" if not columns else ",".join(columns)
        )

        # Apply filters (encrypt filter values if needed) with validation
        if filters:
            if len(filters) > 10:  # Limit number of filters to prevent complex queries
                raise ValueError("Too many filters (max 10)")

            encrypted_filters = self._encrypt_filters(table, filters)
            for key, value in encrypted_filters.items():
                # Additional validation for filter values
                if value is None:
                    query = query.is_(key, None)
                elif isinstance(value, (str, int, float, bool)):
                    query = query.eq(key, value)
                else:
                    raise ValueError(
                        f"Unsupported filter value type for {key}: {type(value)}"
                    )

        result = query.execute()

        if result.data:
            # Decrypt results
            decrypted_results = [self._decrypt_data(table, row) for row in result.data]

            # Log operation
            await self._log_operation(
                "select",
                table,
                None,
                user_id,
                {"filters": filters, "count": len(decrypted_results)},
            )

            return decrypted_results

        return []

    async def secure_update(
        self,
        table: str,
        record_id: str,
        data: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update data with encryption and auditing."""
        # Encrypt sensitive fields
        encrypted_data = self._encrypt_data(table, data)
        encrypted_data["updated_at"] = datetime.utcnow().isoformat()

        # Update via Supabase
        result = (
            self.client.table(table)
            .update(encrypted_data)
            .eq("id", record_id)
            .execute()
        )

        if result.data:
            # Decrypt result for return
            decrypted_result = self._decrypt_data(table, result.data[0])

            # Log operation
            await self._log_operation(
                "update", table, record_id, user_id, encrypted_data
            )

            return decrypted_result
        else:
            raise RuntimeError(f"Failed to update {table} record {record_id}")

    async def secure_delete(
        self, table: str, record_id: str, user_id: Optional[str] = None
    ) -> bool:
        """Delete data with auditing."""
        # Get record before deletion for audit
        existing = await self.secure_select(table, {"id": record_id}, user_id=user_id)

        # Delete via Supabase
        result = self.client.table(table).delete().eq("id", record_id).execute()

        # Log operation
        await self._log_operation(
            "delete",
            table,
            record_id,
            user_id,
            {"deleted_record": existing[0] if existing else None},
        )

        return len(result.data) > 0

    # Vector operations
    async def secure_vector_search(
        self,
        table: str,
        vector: List[float],
        limit: int = 10,
        threshold: float = 0.8,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Perform encrypted vector similarity search."""
        # Encrypt the query vector
        self.encryption.encrypt_vector(vector)

        # Use parameterized SQL for vector operations (no table name interpolation)
        if table not in [
            "memories",
            "mem0_vectors.memories",
        ]:  # whitelist allowed tables
            raise ValueError(f"Invalid table name: {table}")

        if not isinstance(vector, list) or len(vector) != 1536:  # validate vector
            raise ValueError("Vector must be a list of 1536 floats")

        if not isinstance(limit, int) or limit <= 0 or limit > 1000:
            raise ValueError("Limit must be between 1 and 1000")

        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
            raise ValueError("Threshold must be between 0 and 1")

        async with self.pool.acquire() as conn:
            # Use safe table reference - never interpolate table names directly
            if table == "memories":
                table_name = "public.memories"
            else:
                table_name = "mem0_vectors.memories"

            query = f"""
            SELECT id, content, embedding <-> $1::vector as distance, metadata
            FROM {table_name}
            WHERE embedding <-> $1::vector < $2
                AND embedding IS NOT NULL
            ORDER BY embedding <-> $1::vector
            LIMIT $3
            """

            # Note: For production, you'd want to encrypt vectors in a way that preserves
            # similarity relationships, which requires specialized techniques like
            # homomorphic encryption or searchable encryption schemes

            try:
                rows = await conn.fetch(query, vector, 1.0 - threshold, limit)
            except Exception as e:
                logger.error(f"Vector search query failed: {e}")
                raise RuntimeError(f"Vector search failed: {str(e)}")

            results = []
            for row in rows:
                result = dict(row)

                # Decrypt sensitive fields
                if result.get("content"):
                    result["content"] = self.encryption.decrypt_data(result["content"])

                if result.get("metadata"):
                    result["metadata"] = self.encryption.decrypt_json(
                        result["metadata"]
                    )

                results.append(result)

        # Log operation
        await self._log_operation(
            "vector_search",
            table,
            None,
            user_id,
            {
                "vector_length": len(vector),
                "limit": limit,
                "results_count": len(results),
            },
        )

        return results

    async def secure_vector_insert(
        self,
        table: str,
        content: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Insert data with vector embedding."""
        data = {"content": content, "embedding": vector, "metadata": metadata or {}}

        return await self.secure_insert(table, data, user_id)

    # Backup and restore
    async def create_encrypted_backup(
        self, tables: List[str], user_id: Optional[str] = None
    ) -> str:
        """Create encrypted backup of specified tables."""
        backup_data = {}

        for table in tables:
            table_data = await self.secure_select(table, user_id=user_id)
            backup_data[table] = table_data

        # Encrypt backup
        from security.encryption import backup_encryption

        encrypted_backup = backup_encryption.encrypt_backup(backup_data)

        # Log backup creation
        await self._log_operation(
            "backup",
            "system",
            None,
            user_id,
            {
                "tables": tables,
                "record_count": sum(len(data) for data in backup_data.values()),
            },
        )

        return encrypted_backup

    async def restore_encrypted_backup(
        self, encrypted_backup: str, user_id: Optional[str] = None
    ) -> bool:
        """Restore from encrypted backup."""
        try:
            from security.encryption import backup_encryption

            backup_data = backup_encryption.decrypt_backup(encrypted_backup)

            # Restore each table
            for table, records in backup_data.items():
                for record in records:
                    # Remove auto-generated fields
                    clean_record = {
                        k: v
                        for k, v in record.items()
                        if k not in ["id", "created_at", "updated_at"]
                    }
                    await self.secure_insert(table, clean_record, user_id)

            # Log restore
            await self._log_operation(
                "restore",
                "system",
                None,
                user_id,
                {
                    "tables": list(backup_data.keys()),
                    "total_records": sum(
                        len(records) for records in backup_data.values()
                    ),
                },
            )

            return True
        except Exception as e:
            # Log error
            await self._log_operation(
                "restore_error", "system", None, user_id, {"error": str(e)}
            )
            return False

    # Private helper methods
    def _encrypt_data(self, table: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive fields in data."""
        encrypted_fields = self.encrypted_fields.get(table, [])
        return self.db_encryption.encrypt_row(data, encrypted_fields)

    def _decrypt_data(self, table: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive fields in data."""
        encrypted_fields = self.encrypted_fields.get(table, [])
        return self.db_encryption.decrypt_row(data, encrypted_fields)

    def _encrypt_filters(self, table: str, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt filter values for searchable encrypted fields with validation."""
        if not isinstance(table, str) or not table:
            raise ValueError("Table name must be a non-empty string")

        if not isinstance(filters, dict):
            raise ValueError("Filters must be a dictionary")

        # Validate table name against whitelist
        allowed_tables = set(self.encrypted_fields.keys())
        if table not in allowed_tables:
            raise ValueError(f"Table '{table}' not in allowed tables: {allowed_tables}")

        encrypted_filters = {}
        encrypted_fields = self.encrypted_fields.get(table, [])

        for key, value in filters.items():
            # Validate column names to prevent injection
            if (
                not isinstance(key, str)
                or not key.replace("_", "").replace("-", "").isalnum()
            ):
                raise ValueError(f"Invalid column name: {key}")

            # Limit value length to prevent DoS
            if isinstance(value, str) and len(value) > 1000:
                raise ValueError(f"Filter value too long for {key}")

            if key in encrypted_fields:
                # For exact matches on encrypted fields, we need searchable encryption
                # This is a simplified approach - in production, you'd use deterministic encryption
                try:
                    encrypted_filters[f"{key}_hash"] = (
                        self.db_encryption.get_encrypted_search_hash(str(value), key)
                    )
                except Exception as e:
                    logger.error(f"Encryption failed for {key}: {e}")
                    raise ValueError(f"Failed to encrypt filter for {key}")
            else:
                encrypted_filters[key] = value

        return encrypted_filters

    async def _log_operation(
        self,
        operation: str,
        table: str,
        record_id: Optional[str],
        user_id: Optional[str],
        details: Dict[str, Any],
    ):
        """Log database operation for audit with input validation."""
        # Validate inputs to prevent log injection
        if not isinstance(operation, str) or not operation.replace("_", "").isalpha():
            raise ValueError(f"Invalid operation: {operation}")

        if (
            not isinstance(table, str)
            or not table.replace("_", "").replace(".", "").isalnum()
        ):
            raise ValueError(f"Invalid table name: {table}")

        if record_id is not None and (
            not isinstance(record_id, str) or len(record_id) > 100
        ):
            raise ValueError(f"Invalid record_id: {record_id}")

        if user_id is not None and (not isinstance(user_id, str) or len(user_id) > 100):
            raise ValueError(f"Invalid user_id: {user_id}")

        # Sanitize details to prevent injection in logs
        sanitized_details = {}
        for key, value in details.items():
            if (
                isinstance(key, str)
                and len(key) <= 50
                and key.replace("_", "").isalnum()
            ):
                # Truncate large values and sanitize
                if isinstance(value, str):
                    sanitized_details[key] = (
                        value[:500]
                        if len(value) <= 500
                        else value[:500] + "...[truncated]"
                    )
                elif isinstance(value, (int, float, bool, type(None))):
                    sanitized_details[key] = value
                else:
                    sanitized_details[key] = str(value)[:100]

        try:
            await audit_logger.log_event(
                action=f"db_{operation}",
                resource=f"database.{table}",
                resource_id=record_id,
                user_id=user_id,
                details=sanitized_details,
            )
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            # Don't raise - logging failure shouldn't break operations

    # Connection management
    @asynccontextmanager
    async def transaction(self):
        """Database transaction context manager."""
        async with self.pool.acquire() as conn, conn.transaction():
            yield conn

    # Health check
    async def health_check(self) -> Dict[str, Any]:
        """Check database connectivity and health."""
        try:
            # Test Supabase client
            self.client.table("health_check").select("*").limit(1).execute()
            supabase_healthy = True
        except Exception as e:
            supabase_healthy = False
            supabase_error = str(e)

        try:
            # Test direct connection pool
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            pool_healthy = True
        except Exception as e:
            pool_healthy = False
            pool_error = str(e)

        return {
            "supabase_healthy": supabase_healthy,
            "pool_healthy": pool_healthy,
            "supabase_error": supabase_error if not supabase_healthy else None,
            "pool_error": pool_error if not pool_healthy else None,
            "timestamp": datetime.utcnow().isoformat(),
        }


class SupabaseConnectionManager:
    """Manages Supabase connections with connection pooling and failover."""

    def __init__(self):
        self.clients: Dict[str, SecureSupabaseClient] = {}
        self.primary_client: Optional[SecureSupabaseClient] = None

    async def get_client(self, database: str = "primary") -> SecureSupabaseClient:
        """Get or create secure Supabase client."""
        if database not in self.clients:
            client = SecureSupabaseClient()
            await client.initialize()
            self.clients[database] = client

            if database == "primary":
                self.primary_client = client

        return self.clients[database]

    async def close_all(self):
        """Close all connections."""
        for client in self.clients.values():
            await client.close()
        self.clients.clear()
        self.primary_client = None

    async def health_check_all(self) -> Dict[str, Any]:
        """Health check for all clients."""
        results = {}
        for name, client in self.clients.items():
            results[name] = await client.health_check()
        return results


# Global connection manager
connection_manager = SupabaseConnectionManager()


# Utility functions
async def get_secure_supabase() -> SecureSupabaseClient:
    """Dependency to get secure Supabase client."""
    return await connection_manager.get_client()


# Database migration utilities
class EncryptionMigrator:
    """Utility for migrating existing data to encrypted format."""

    def __init__(self, client: SecureSupabaseClient):
        self.client = client

    async def migrate_table_to_encrypted(self, table: str, batch_size: int = 100):
        """Migrate existing table data to encrypted format."""
        # Get all records
        async with self.client.pool.acquire() as conn:
            total_records = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")

            for offset in range(0, total_records, batch_size):
                records = await conn.fetch(
                    f"SELECT * FROM {table} ORDER BY id LIMIT $1 OFFSET $2",
                    batch_size,
                    offset,
                )

                # Process each record
                for record in records:
                    record_dict = dict(record)
                    record_id = record_dict["id"]

                    # Encrypt sensitive fields
                    encrypted_data = self.client._encrypt_data(table, record_dict)

                    # Update record
                    update_fields = []
                    values = []
                    param_count = 1

                    for field, value in encrypted_data.items():
                        if field != "id":  # Don't update ID
                            update_fields.append(f"{field} = ${param_count}")
                            values.append(value)
                            param_count += 1

                    if update_fields:
                        query = f"UPDATE {table} SET {', '.join(update_fields)} WHERE id = ${param_count}"
                        values.append(record_id)
                        await conn.execute(query, *values)



# Create global secure client instance
async def initialize_secure_database() -> SecureSupabaseClient:
    """Initialize the secure database connection."""
    try:
        client = await connection_manager.get_client()
        logger.info("Secure database initialized successfully")
        return client
    except Exception as e:
        logger.error("Failed to initialize secure database", error=str(e))
        raise
